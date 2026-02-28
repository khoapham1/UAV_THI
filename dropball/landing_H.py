#!/usr/bin/env python3
# landing_H.py
#
# This module provides H-marker landing controllers.
#
# - HCenteringLander: the "acquire / track / reacquire" visual-servo landing used in dropball.py
#   (descend in GUIDED using BODY_NED velocities, with safe fallback to LAND when needed).
#
# - HPrecisionLander: a simpler PID-only controller kept for backward compatibility.
#
# Notes:
# - BODY_NED: vx forward, vy right, vz down (positive is DOWN).
# - If your vehicle climbs instead of descends, check the sign convention of vz in your stack.
#
# dropball.py integration:
#   from landing_H import HCenteringLander, LandResult

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from dronekit import VehicleMode
from pymavlink import mavutil

# Optional ROS2 guard (so this file can be imported outside ROS2 for testing)
try:
    import rclpy  # type: ignore
    def _ros_ok() -> bool:
        try:
            return bool(rclpy.ok())
        except Exception:
            return True
except Exception:  # pragma: no cover
    rclpy = None  # type: ignore
    def _ros_ok() -> bool:
        return True

# Optional PIDController (your project)
try:
    from PID_control import PIDController  # type: ignore
except Exception:  # pragma: no cover
    PIDController = None  # type: ignore


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------
@dataclass
class LandingResult:
    ok: bool
    reason: str

# Alias used by dropball.py (older code uses LandResult)
LandResult = LandingResult


PID_X = PIDController(0.0050, 0.00005, 0.00035, max_output=0.3, integral_limit=300)
PID_Y = PIDController(0.0050, 0.00005, 0.00030, max_output=0.3, integral_limit=300)



# ----------------------------------------------------------------------
# HCenteringLander (ported from dropball.py)
# ----------------------------------------------------------------------
class HCenteringLander:
    """
    Visual-servo landing xuống chữ H:
    - Dựa trên detection bbox center của H.
    - Vừa hạ độ cao (vz > 0 trong BODY_NED) vừa chỉnh vx/vy để đưa tâm H về tâm ảnh.
    - Nếu mất H quá lâu -> fallback sang LAND của autopilot hoặc GUIDED touchdown khi rất thấp.

    get_h_dets() phải trả về list các dict:
      {"bbox": (x1,y1,x2,y2), "center": (cx,cy), "conf": float, "cls": int}
    """

    def __init__(
        self,
        vehicle,
        get_frame_bgr: Callable[[], Optional[Any]],
        get_h_dets: Callable[[], list],
        logger,
        pid_x=None,
        pid_y=None,
        conf_th: float = 0.25,
        tol_px: int = 10,
        hold_center_sec: float = 0.8,
        hold_last_sec: float = 1.0,
        loop_hz: float = 20.0,
        # descent profile
        descent_vz: float = 0.35,         # m/s (down)
        min_vz: float = 0.10,             # m/s (down) when slightly off-center
        stop_descend_err_px: int = 220,   # if error >= this -> hover (vz=0)
        err_full_px: int = 180,           # error where descent becomes close to min_vz
        # altitude thresholds
        switch_to_land_alt_m: float = 0.8,  # kept for future gating (not used in this version)
        alt_done_m: float = 0.15,
        max_vxy: float = 0.30,
        ema_alpha: float = 0.25,
    ):
        self.vehicle = vehicle
        self.get_frame_bgr = get_frame_bgr
        self.get_h_dets = get_h_dets
        self.log = logger

        # allow sharing PID objects with the rest of the mission if desired
        self.pid_x = pid_x if pid_x is not None else PID_X
        self.pid_y = pid_y if pid_y is not None else PID_Y

        self.conf_th = float(conf_th)
        self.tol_px = int(tol_px)
        self.hold_center_sec = float(hold_center_sec)
        self.hold_last_sec = float(hold_last_sec)
        self.loop_hz = float(loop_hz)

        self.descent_vz = float(descent_vz)
        self.min_vz = float(min_vz)
        self.stop_descend_err_px = int(stop_descend_err_px)
        self.err_full_px = int(err_full_px)

        self.switch_to_land_alt_m = float(switch_to_land_alt_m)
        self.alt_done_m = float(alt_done_m)

        self.max_vxy = float(max_vxy)
        self.ema_alpha = float(ema_alpha)

        self._ema_ex = None
        self._ema_ey = None

    # --- MAVLink velocity (BODY_NED) ---
    def send_local_ned_velocity(self, vx: float, vy: float, vz: float):
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            self.vehicle._master.target_system,
            self.vehicle._master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            1479,  # only velocity
            0, 0, 0,
            float(vx), float(vy), float(vz),
            0, 0, 0,
            0.0, 0,
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def _best_h_center(self, dets) -> Optional[Tuple[int, int]]:
        if not dets:
            return None
        best = None
        best_conf = -1.0
        for d in dets:
            c = float(d.get("conf", 0.0))
            if c >= self.conf_th and c > best_conf:
                best_conf = c
                best = d
        return best.get("center") if best is not None else None

    def _ema(self, ex: float, ey: float) -> Tuple[float, float]:
        if self._ema_ex is None:
            self._ema_ex, self._ema_ey = float(ex), float(ey)
        else:
            a = self.ema_alpha
            self._ema_ex = (1 - a) * self._ema_ex + a * float(ex)
            self._ema_ey = (1 - a) * self._ema_ey + a * float(ey)
        return float(self._ema_ex), float(self._ema_ey)

    def land_on_h(self, timeout_sec: float = 90.0) -> LandResult:
        """
        Descend while trying to acquire and center on the H marker.

        Behavior:
        - ACQUIRE: if H not seen yet, descend slowly until H is acquired (or fallback LAND).
        - TRACK: PID centering + adaptive descent.
        - REACQUIRE: if H lost after being seen, allow brief reacquire window; otherwise fallback.
        - Very low altitude: perform GUIDED touchdown smoothly (no LAND switch).
        """
        # reset PID giống seek_target_and_center()
        try:
            self.pid_x.reset()
            self.pid_y.reset()
        except Exception:
            pass

        # local helpers
        def _switch_to_land_and_wait(reason: str, wait_sec: float = 30.0) -> LandResult:
            try:
                self.log.error(f"[H-LAND] {reason} -> switching to LAND")
                self.vehicle.mode = VehicleMode("LAND")
            except Exception:
                pass
            try:
                self.send_local_ned_velocity(0.0, 0.0, 0.0)
            except Exception:
                pass

            t_land = time.time()
            while _ros_ok() and (time.time() - t_land) < float(wait_sec):
                try:
                    alt2 = float(self.vehicle.location.global_relative_frame.alt)
                    if (not bool(self.vehicle.armed)) or alt2 <= self.alt_done_m:
                        return LandResult(True, f"LAND ok alt={alt2:.2f}m")
                except Exception:
                    pass
                time.sleep(0.2)

            return LandResult(True, f"LAND switched ({reason})")

        def _guided_touchdown(reason: str, max_sec: float = 25.0) -> LandResult:
            """Keep GUIDED and descend smoothly to ground without switching to LAND."""
            try:
                # rclpy logger uses warn(); keep best-effort
                if hasattr(self.log, "warn"):
                    self.log.warn(f"[H-LAND] {reason} -> GUIDED touchdown")
                else:
                    self.log.info(f"[H-LAND] {reason} -> GUIDED touchdown")
            except Exception:
                pass

            t_td = time.time()
            while _ros_ok() and (time.time() - t_td) < float(max_sec):
                try:
                    alt_td = float(self.vehicle.location.global_relative_frame.alt)
                except Exception:
                    alt_td = 999.0

                try:
                    if (not bool(self.vehicle.armed)) or alt_td <= self.alt_done_m:
                        break
                except Exception:
                    break

                # flare: reduce vertical speed near ground
                if alt_td < 0.12:
                    vz_td = 0.05
                elif alt_td < 0.20:
                    vz_td = 0.07
                elif alt_td < 0.35:
                    vz_td = 0.10
                elif alt_td < 0.60:
                    vz_td = 0.18
                else:
                    vz_td = 0.22

                try:
                    self.send_local_ned_velocity(0.0, 0.0, float(vz_td))
                except Exception:
                    pass
                time.sleep(0.05)

            try:
                self.send_local_ned_velocity(0.0, 0.0, 0.0)
            except Exception:
                pass

            # try disarm (best-effort)
            try:
                self.vehicle.armed = False
            except Exception:
                pass
            try:
                msg = self.vehicle.message_factory.command_long_encode(
                    0,
                    self.vehicle._master.target_system,
                    self.vehicle._master.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0,
                    0, 0, 0, 0, 0, 0, 0,
                )
                self.vehicle.send_mavlink(msg)
                self.vehicle.flush()
            except Exception:
                pass

            try:
                alt_final = float(self.vehicle.location.global_relative_frame.alt)
            except Exception:
                alt_final = -1.0

            return LandResult(True, f"GUIDED touchdown ({reason}) alt={alt_final:.2f}m")

        t0 = time.time()
        dt = 1.0 / max(1.0, self.loop_hz)

        last_seen_t = None
        last_center = None
        center_hold_t0 = None

        # acquisition / reacquisition behavior (safe defaults)
        acquire_vz = min(0.35, max(0.10, float(getattr(self, "descent_vz", 0.35))))
        reacquire_vz = min(0.20, acquire_vz)
        acquire_floor_alt_m = 0.55
        acquire_grace_sec = 35.0
        reacquire_grace_sec = 4.0

        seen_once = False
        acquire_t0 = time.time()
        lost_t0 = None

        # ensure GUIDED for velocity control
        try:
            if self.vehicle.mode.name != "GUIDED":
                self.vehicle.mode = VehicleMode("GUIDED")
                t_mode = time.time()
                while self.vehicle.mode.name != "GUIDED" and (time.time() - t_mode) < 8.0:
                    time.sleep(0.2)
        except Exception:
            pass

        try:
            self.log.info("[H-LAND] Start visual-servo landing (GUIDED velocities).")
        except Exception:
            pass

        while _ros_ok():
            # global timeout
            if timeout_sec is not None and (time.time() - t0) > float(timeout_sec):
                try:
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass
                return LandResult(False, f"timeout {timeout_sec}s")

            # altitude
            try:
                alt = float(self.vehicle.location.global_relative_frame.alt)
            except Exception:
                alt = 999.0

            # finish condition (already landed/disarmed)
            try:
                if not bool(self.vehicle.armed):
                    try:
                        self.send_local_ned_velocity(0.0, 0.0, 0.0)
                    except Exception:
                        pass
                    return LandResult(True, f"done disarmed alt={alt:.2f}m")

                if alt <= self.alt_done_m:
                    return _guided_touchdown(reason=f"finalize at alt={alt:.2f}m", max_sec=12.0)
            except Exception:
                pass

            # frame
            try:
                frame = self.get_frame_bgr()
            except Exception:
                frame = None

            if frame is None:
                try:
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass
                time.sleep(dt)
                continue

            h_img, w_img = frame.shape[:2]
            icx, icy = w_img // 2, h_img // 2

            # detection
            try:
                dets = self.get_h_dets()
            except Exception:
                dets = []

            center = self._best_h_center(dets)
            now = time.time()

            if center is not None:
                last_center = center
                last_seen_t = now
                lost_t0 = None

                if not seen_once:
                    seen_once = True
                    self._ema_ex = None
                    self._ema_ey = None
                    try:
                        self.pid_x.reset()
                        self.pid_y.reset()
                    except Exception:
                        pass
                    center_hold_t0 = None

            else:
                # keep last center briefly (helps through 1-2 missed frames)
                if last_center is not None and last_seen_t is not None and (now - last_seen_t) <= self.hold_last_sec:
                    center = last_center
                else:
                    # ---- ACQUIRE PHASE (never seen H yet) ----
                    if not seen_once:
                        if alt > acquire_floor_alt_m and (now - acquire_t0) <= acquire_grace_sec:
                            try:
                                self.send_local_ned_velocity(0.0, 0.0, acquire_vz)
                            except Exception:
                                pass
                            time.sleep(dt)
                            continue

                        return _switch_to_land_and_wait(
                            reason=f"H not acquired (alt={alt:.2f}m, t={now - acquire_t0:.1f}s)"
                        )

                    # ---- REACQUIRE PHASE (H was seen before) ----
                    if lost_t0 is None:
                        lost_t0 = now

                    if (now - lost_t0) <= reacquire_grace_sec and alt > acquire_floor_alt_m:
                        try:
                            self.send_local_ned_velocity(0.0, 0.0, reacquire_vz)
                        except Exception:
                            pass
                        time.sleep(dt)
                        continue

                    # lost too long:
                    if alt <= 0.35:
                        return _guided_touchdown(reason=f"lost H low-alt {now - lost_t0:.1f}s (alt={alt:.2f}m)")
                    return _switch_to_land_and_wait(
                        reason=f"lost H for {now - lost_t0:.1f}s (alt={alt:.2f}m)"
                    )

            # from here: we have a center (fresh or held)
            tx, ty = center
            ex = float(tx - icx)
            ey = float(ty - icy)

            # smooth error a bit (reduce jitter)
            ex, ey = self._ema(ex, ey)

            # compute vx/vy using same mapping as control_drone_to_center()
            vx = float(self.pid_y.update(-ey))
            vy = float(self.pid_x.update(ex))

            # clamp lateral speeds
            vx = max(min(vx, self.max_vxy), -self.max_vxy)
            vy = max(min(vy, self.max_vxy), -self.max_vxy)

            # descent profile: descend faster when near center, slower when off
            err = max(abs(ex), abs(ey))

            if err >= float(self.stop_descend_err_px):
                vz = 0.0
                center_hold_t0 = None
            else:
                if err <= float(self.tol_px):
                    if center_hold_t0 is None:
                        center_hold_t0 = now
                    held = (now - center_hold_t0) >= self.hold_center_sec
                    vz = self.descent_vz if held else max(self.min_vz, 0.7 * self.descent_vz)
                else:
                    center_hold_t0 = None
                    scale = 1.0 - (err / float(max(1, self.err_full_px)))
                    scale = max(0.0, min(1.0, scale))
                    vz = self.min_vz + (self.descent_vz - self.min_vz) * scale
                    if scale <= 0.05:
                        vz = 0.0

            # Cap descent rate near ground for smoother contact, and reduce lateral authority.
            if alt < 0.12:
                vz_cap = 0.05
            elif alt < 0.20:
                vz_cap = 0.07
            elif alt < 0.35:
                vz_cap = 0.10
            elif alt < 0.60:
                vz_cap = 0.18
            elif alt < 0.90:
                vz_cap = 0.25
            else:
                vz_cap = float(self.descent_vz)

            vz = float(max(0.0, min(float(vz), float(vz_cap))))

            # always make progress to touchdown when very low (avoid hover-forever)
            if alt < 0.35:
                vz = float(max(vz, 0.05))

            # reduce lateral speeds near ground (avoid skid / oscillation)
            if alt < 0.30:
                vx = float(max(min(vx, 0.15), -0.15))
                vy = float(max(min(vy, 0.15), -0.15))
            elif alt < 0.60:
                vx = float(max(min(vx, 0.22), -0.22))
                vy = float(max(min(vy, 0.22), -0.22))

            # send velocity command
            try:
                self.send_local_ned_velocity(vx, vy, vz)
            except Exception:
                pass

            time.sleep(dt)


# ----------------------------------------------------------------------
# HPrecisionLander (kept from your previous landing_H.py)
# ----------------------------------------------------------------------
class HPrecisionLander:
    """
    PID landing controller using YOLO H detections (simple version).

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
        descend_vz: float = 0.25,
        descend_vz_slow: float = 0.10,
        # velocity limits
        max_vxy: float = 0.60,
        # altitude thresholds
        alt_land_m: float = 0.70,
        alt_done_m: float = 0.12,
        # loss handling
        lost_timeout_sec: float = 1.5,
        # axis mapping
        flip_ex: bool = False,
        flip_ey: bool = False,
        # PID gains
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

        self.pid_x = PID_X(kp_x, ki_x, kd_x, out_max=self.max_vxy, i_max=float(i_max))
        self.pid_y = PID_Y(kp_y, ki_y, kd_y, out_max=self.max_vxy, i_max=float(i_max))

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
            0, 0, 0,
            float(vx), float(vy), float(vz),
            0, 0, 0,
            0.0, 0,
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
        if not self._ensure_guided():
            return LandingResult(False, "Failed to set GUIDED mode")

        self.pid_x.reset()
        self.pid_y.reset()

        start_t = time.time()
        last_seen_t = 0.0
        hold_t0 = None

        while True:
            try:
                alt = float(self.vehicle.location.global_relative_frame.alt)
            except Exception:
                alt = 999.0

            if alt <= self.alt_done_m:
                self._send_body_ned_velocity(0.0, 0.0, 0.0)
                self._set_land_mode()
                return LandingResult(True, f"Altitude <= {self.alt_done_m}m")

            if hasattr(self.vehicle, "armed") and (not self.vehicle.armed):
                return LandingResult(True, "Vehicle disarmed")

            if (time.time() - start_t) > float(timeout_sec):
                self._set_land_mode()
                return LandingResult(False, "Timeout")

            frame = self.get_frame_bgr()
            if frame is None:
                self._send_body_ned_velocity(0.0, 0.0, self.descend_vz_slow)
                time.sleep(self.dt)
                continue

            h, w = frame.shape[:2]
            icx, icy = w // 2, h // 2
            best = self._best_h(frame)

            if best is None:
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

            ex = float(cx - icx)
            ey = float(cy - icy)

            if self.flip_ex:
                ex = -ex
            if self.flip_ey:
                ey = -ey

            vx = self.pid_y.update(-ey)
            vy = self.pid_x.update(ex)

            centered = (abs(ex) <= self.center_tol_px and abs(ey) <= self.center_tol_px)

            if centered:
                if hold_t0 is None:
                    hold_t0 = time.time()
                vz = self.descend_vz
            else:
                hold_t0 = None
                vz = self.descend_vz_slow

            hold_ok = False
            if hold_t0 is not None and (time.time() - hold_t0) >= self.center_hold_sec:
                hold_ok = True

            if alt <= self.alt_land_m and (hold_ok or centered):
                self._send_body_ned_velocity(vx, vy, 0.0)
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
