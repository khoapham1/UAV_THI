#!/usr/bin/env python3
# landing_H.py
# PID-based landing on H-marker using YOLOv5 detections (NO x_ang/y_ang, NO LANDING_TARGET).
#
# Idea:
#   - Keep vehicle in GUIDED.
#   - Continuously detect H bbox center (cx, cy) from the camera frame.
#   - Compute pixel error ex, ey to image center.
#   - Use PID to generate body-frame velocity commands (vx, vy) to center the marker.
#   - Simultaneously descend with vz (down) while correcting.
#   - Near ground, switch to LAND to finish.
#
# Notes:
#   - Uses MAV_FRAME_BODY_NED velocity setpoints (vx forward, vy right, vz down).
#   - If your drone climbs instead of descends, flip the sign of descend_vz.
#   - This file keeps backward-compatibility: it accepts extra kwargs (old params like horizontal_fov_deg, etc.)
#     so dropball.py won't crash if it still passes them.

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from dronekit import VehicleMode
from pymavlink import mavutil


@dataclass
class LandingResult:
    ok: bool
    reason: str


class _PID:
    def __init__(self, kp: float, ki: float, kd: float, out_max: float, i_max: float):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.out_max = float(out_max)
        self.i_max = float(i_max)
        self.reset()

    def reset(self):
        self._i = 0.0
        self._prev_e = None
        self._prev_t = None

    def update(self, e: float) -> float:
        t = time.time()
        if self._prev_t is None:
            dt = 0.0
        else:
            dt = t - self._prev_t
            if dt <= 1e-6:
                dt = 0.0

        # integral
        if dt > 0.0:
            self._i += e * dt
            if self._i > self.i_max:
                self._i = self.i_max
            elif self._i < -self.i_max:
                self._i = -self.i_max

        # derivative
        if self._prev_e is None or dt == 0.0:
            de = 0.0
        else:
            de = (e - self._prev_e) / dt

        self._prev_e = e
        self._prev_t = t

        u = self.kp * e + self.ki * self._i + self.kd * de

        # clamp
        if u > self.out_max:
            u = self.out_max
        elif u < -self.out_max:
            u = -self.out_max
        return u


class HPrecisionLander:
    """
    PID landing controller using YOLO H detections.

    Usage (same as before):
      lander = HPrecisionLander(vehicle=..., detector_h=..., get_frame_bgr=..., logger=...)
      res = lander.land_with_h(timeout_sec=90.0)

    detector_h.detect_all(frame_bgr) must return list of:
      {"bbox":(x1,y1,x2,y2), "center":(cx,cy), "conf":float, "cls":int}
    """

    def __init__(
        self,
        vehicle,
        detector_h,
        get_frame_bgr: Callable[[], Optional[Any]],
        # loop / detection
        send_rate_hz: float = 15.0,
        conf_th: float = 0.25,
        # pixel tolerance / hold
        center_tol_px: int = 8,
        center_hold_sec: float = 1.0,
        # descent
        descend_vz: float = 0.25,        # vz down in BODY_NED (if climbs, set negative)
        descend_vz_slow: float = 0.10,   # when not centered
        # velocity limits
        max_vxy: float = 0.60,
        # altitude thresholds
        alt_land_m: float = 0.70,        # switch to LAND under this altitude
        alt_done_m: float = 0.12,        # consider done
        # loss handling
        lost_timeout_sec: float = 1.5,   # if H lost too long => fallback LAND
        # axis mapping (if your ex/ey mapping is reversed)
        flip_ex: bool = False,
        flip_ey: bool = False,
        # PID gains (default tuned conservatively; adjust as needed)
        kp_x: float = 0.0055,
        ki_x: float = 0.00005,
        kd_x: float = 0.00030,
        kp_y: float = 0.0060,
        ki_y: float = 0.00005,
        kd_y: float = 0.00035,
        i_max: float = 300.0,
        # logging
        logger=None,
        log_period_sec: float = 0.5,
        # accept old/extra kwargs to avoid TypeError from older dropball.py
        **_ignored,
    ):
        self.vehicle = vehicle
        self.detector_h = detector_h
        self.get_frame_bgr = get_frame_bgr

        self.dt = 1.0 / max(1.0, float(send_rate_hz))
        self.conf_th = float(conf_th)

        self.center_tol_px = int(center_tol_px)
        self.center_hold_sec = float(center_hold_sec)

        self.descend_vz = float(descend_vz)
        self.descend_vz_slow = float(descend_vz_slow)
        self.max_vxy = float(max_vxy)

        self.alt_land_m = float(alt_land_m)
        self.alt_done_m = float(alt_done_m)

        self.lost_timeout_sec = float(lost_timeout_sec)

        self.flip_ex = bool(flip_ex)
        self.flip_ey = bool(flip_ey)

        self.pid_x = _PID(kp_x, ki_x, kd_x, out_max=self.max_vxy, i_max=float(i_max))
        self.pid_y = _PID(kp_y, ki_y, kd_y, out_max=self.max_vxy, i_max=float(i_max))

        self.logger = logger
        self.log_period_sec = float(log_period_sec)
        self._last_log_t = 0.0

    def _log(self, s: str):
        if self.logger is not None:
            try:
                self.logger.info(s)
                return
            except Exception:
                pass
        print(s)

    def _should_log(self) -> bool:
        return (time.time() - self._last_log_t) >= self.log_period_sec

    def _ensure_guided(self) -> bool:
        if self.vehicle.mode.name != "GUIDED":
            self.vehicle.mode = VehicleMode("GUIDED")
            t0 = time.time()
            while self.vehicle.mode.name != "GUIDED":
                if time.time() - t0 > 6.0:
                    return False
                time.sleep(0.1)
        return True

    def _set_land_mode(self) -> bool:
        if self.vehicle.mode.name != "LAND":
            self.vehicle.mode = VehicleMode("LAND")
            t0 = time.time()
            while self.vehicle.mode.name != "LAND":
                if time.time() - t0 > 6.0:
                    return False
                time.sleep(0.1)
        return True

    def _send_body_ned_velocity(self, vx: float, vy: float, vz: float):
        """
        Same style as your dropball.send_local_ned_velocity:
          MAV_FRAME_BODY_NED + type_mask=1479 (vel only)
        """
        # clamp
        if vx > self.max_vxy:
            vx = self.max_vxy
        elif vx < -self.max_vxy:
            vx = -self.max_vxy
        if vy > self.max_vxy:
            vy = self.max_vxy
        elif vy < -self.max_vxy:
            vy = -self.max_vxy

        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            self.vehicle._master.target_system,
            self.vehicle._master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            1479,
            0,
            0,
            0,
            float(vx),
            float(vy),
            float(vz),
            0,
            0,
            0,
            0.0,
            0,
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def _best_h(self, frame_bgr) -> Optional[Tuple[int, int, float]]:
        dets = self.detector_h.detect_all(frame_bgr)
        if not dets:
            return None
        best = dets[0]
        conf = float(best.get("conf", 0.0))
        if conf < self.conf_th:
            return None
        cx, cy = best["center"]
        return int(cx), int(cy), conf

    def land_with_h(self, timeout_sec: float = 90.0) -> LandingResult:
        """
        PID-based "land while centering" using H bbox.
        """
        if not self._ensure_guided():
            return LandingResult(False, "Failed to set GUIDED mode")

        self.pid_x.reset()
        self.pid_y.reset()

        start_t = time.time()
        last_seen_t = 0.0
        hold_t0 = None

        while True:
            # read altitude
            try:
                alt = float(self.vehicle.location.global_relative_frame.alt)
            except Exception:
                alt = 999.0

            # done conditions
            if alt <= self.alt_done_m:
                # stop XY command; switch LAND to finish/disarm
                self._send_body_ned_velocity(0.0, 0.0, 0.0)
                self._set_land_mode()
                return LandingResult(True, f"Altitude <= {self.alt_done_m}m")

            if hasattr(self.vehicle, "armed") and (not self.vehicle.armed):
                return LandingResult(True, "Vehicle disarmed")

            if (time.time() - start_t) > float(timeout_sec):
                # fallback LAND
                self._set_land_mode()
                return LandingResult(False, "Timeout")

            frame = self.get_frame_bgr()
            if frame is None:
                # no frame -> descend slowly, no XY
                self._send_body_ned_velocity(0.0, 0.0, self.descend_vz_slow)
                time.sleep(self.dt)
                continue

            h, w = frame.shape[:2]
            icx, icy = w // 2, h // 2
            best = self._best_h(frame)

            if best is None:
                # H lost: hover XY, descend very slowly; fallback to LAND if lost too long
                if last_seen_t > 0.0 and (time.time() - last_seen_t) > self.lost_timeout_sec:
                    self._send_body_ned_velocity(0.0, 0.0, 0.0)
                    self._set_land_mode()
                    return LandingResult(False, "Lost H too long -> fallback LAND")

                self._send_body_ned_velocity(0.0, 0.0, self.descend_vz_slow)
                if self._should_log():
                    self._log(f"[H-PID-LAND] LOST alt={alt:.2f}")
                    self._last_log_t = time.time()
                time.sleep(self.dt)
                continue

            cx, cy, conf = best
            last_seen_t = time.time()

            # pixel errors
            ex = float(cx - icx)
            ey = float(cy - icy)

            if self.flip_ex:
                ex = -ex
            if self.flip_ey:
                ey = -ey

            # PID mapping same style as your control_drone_to_center:
            #   vy = PID_X(ex)
            #   vx = PID_Y(-ey)
            vx = self.pid_y.update(-ey)
            vy = self.pid_x.update(ex)

            # descend policy
            centered = (abs(ex) <= self.center_tol_px and abs(ey) <= self.center_tol_px)

            if centered:
                if hold_t0 is None:
                    hold_t0 = time.time()
                # descend faster while centered
                vz = self.descend_vz
            else:
                hold_t0 = None
                # still descend, but slower
                vz = self.descend_vz_slow

            # Near ground: switch to LAND to let autopilot finish touchdown
            # but only if centered-hold is achieved OR alt already very low
            hold_ok = False
            if hold_t0 is not None and (time.time() - hold_t0) >= self.center_hold_sec:
                hold_ok = True

            if alt <= self.alt_land_m and (hold_ok or centered):
                self._send_body_ned_velocity(vx, vy, 0.0)  # stabilize lateral just before LAND
                ok = self._set_land_mode()
                if not ok:
                    return LandingResult(False, "Failed to set LAND mode at end")
                return LandingResult(True, f"Switch LAND at alt={alt:.2f}")

            self._send_body_ned_velocity(vx, vy, vz)

            if self._should_log():
                self._log(
                    f"[H-PID-LAND] alt={alt:.2f} conf={conf:.2f} "
                    f"ex={ex:.1f} ey={ey:.1f} vx={vx:.3f} vy={vy:.3f} vz={vz:.3f} hold={int(hold_ok)}"
                )
                self._last_log_t = time.time()

            time.sleep(self.dt)