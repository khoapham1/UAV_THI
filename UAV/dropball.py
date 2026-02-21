#!/usr/bin/env python3
# dropball.py
# ROS2 Humble + DroneKit + 2x YOLOv5 (torch.hub local repo)
#  - Model 1: circle detect (best.pt) + HSV color label (RED/YELLOW/BLUE) + RED1/RED2 tracking (NO PROMOTE)
#  - Model 2: H detect (best_H.pt) for visual-servo landing
#
# NOTE:
# - cv_bridge imgmsg_to_cv2(..., "bgr8") => OpenCV BGR frame
# - HSVColorClassifier.classify() expects BGR ROI (it uses cv2.COLOR_BGR2HSV internally)

import os
import time
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from dronekit import connect, VehicleMode
from pymavlink import mavutil

from hsv_color import HSVColorClassifier
from PID_control import PIDController
from CenterTracker import CenterTracker  # optional (kept)

warnings.filterwarnings("ignore", message=".*torch\\.cuda\\.amp\\.autocast.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Globals for image subscriber
# -----------------------------
_image_node = None
_image_executor = None
_image_thread = None
_image_running = False

_latest_frame_jpeg = None
_latest_frame_lock = threading.Lock()

# -----------------------------
# Paths / Params
# -----------------------------
MODEL_CIRCLE_PATH = "/home/legion/ai/yolov5/runs/train/circle_yolov5n_4162/weights/best.pt"
MODEL_H_PATH = "/home/legion/ai/yolov5/runs/train/hmarker_yolov5n_4162/weights/best_H.pt"
YOLOV5_REPO_DIR = "/home/legion/ai/yolov5"

# PID (giữ theo file bạn đang dùng)
PID_X = PIDController(0.0060, 0.00005, 0.00035, max_output=0.3, integral_limit=300)
PID_Y = PIDController(0.0055, 0.00005, 0.00030, max_output=0.3, integral_limit=300)


# ==============================
# ROS2 Image stream Node
# ==============================
class ImageSubcriber(Node):
    def __init__(self, topic_name="/UAV/forward/image_raw", queue_size=2):
        super().__init__("image_subscriber")
        self.topic_name = topic_name
        self.sub = self.create_subscription(Image, self.topic_name, self.cb_image, queue_size)
        self.bridge = CvBridge()

        self.pub = self.create_publisher(Image, self.topic_name + "_debug", queue_size)

        # NOTE: BGR frame
        self.last_frame_bgr = None
        self.get_logger().info(f"Subscribed to {self.topic_name}")

    def cb_image(self, msg):
        global _latest_frame_jpeg, _latest_frame_lock

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Error in cb_image: {e}")
            return

        self.last_frame_bgr = frame_bgr

        try:
            ok, buf = cv2.imencode(".jpg", frame_bgr)
            if ok:
                with _latest_frame_lock:
                    _latest_frame_jpeg = buf.tobytes()
        except Exception:
            pass

        if rclpy.ok():
            try:
                out_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
                out_msg.header = msg.header
                self.pub.publish(out_msg)
            except Exception:
                pass


def start_image_subscriber(topic_name="/UAV/forward/image_raw"):
    global _image_node, _image_executor, _image_thread, _image_running

    if _image_node is not None:
        return

    if not rclpy.ok():
        raise RuntimeError("rclpy is not initialized. Call rclpy.init() first.")

    _image_node = ImageSubcriber(topic_name=topic_name)
    _image_executor = SingleThreadedExecutor()
    _image_executor.add_node(_image_node)

    _image_running = True

    def spin_loop():
        global _image_running, _image_node, _image_executor
        try:
            while _image_running and rclpy.ok():
                _image_executor.spin_once(timeout_sec=0.1)
                time.sleep(0.01)
        except Exception:
            pass
        finally:
            try:
                if _image_executor is not None and _image_node is not None:
                    _image_executor.remove_node(_image_node)
                    _image_node.destroy_node()
            except Exception:
                pass

    _image_thread = threading.Thread(target=spin_loop, daemon=True)
    _image_thread.start()


def stop_image_subscriber():
    global _image_node, _image_executor, _image_thread, _image_running
    if _image_node is None:
        return

    _image_running = False

    try:
        if _image_executor is not None:
            _image_executor.shutdown()
    except Exception:
        pass

    try:
        if _image_thread is not None:
            _image_thread.join(timeout=1.0)
    except Exception:
        pass

    _image_node = None
    _image_executor = None
    _image_thread = None


def get_latest_frame_jpeg():
    global _latest_frame_jpeg, _latest_frame_lock
    with _latest_frame_lock:
        return _latest_frame_jpeg


# ==============================
# YOLOv5 torch.hub detector wrapper
# ==============================
class YOLOv5TorchHubDetector:
    def __init__(
        self,
        weights_path: str,
        imgsz: int = 320,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 10,
        device: str = "",
    ):
        self.weights_path = str(Path(weights_path).expanduser().resolve())
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"weights not found: {self.weights_path}")

        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)

        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")

        hubconf = os.path.join(YOLOV5_REPO_DIR, "hubconf.py")
        if not os.path.exists(hubconf):
            raise FileNotFoundError(f"YOLOv5 repo not found/invalid: {YOLOV5_REPO_DIR}")

        # Force reload to avoid stale hub cache
        self.model = torch.hub.load(
            YOLOV5_REPO_DIR,
            "custom",
            path=self.weights_path,
            source="local",
            force_reload=True,
        )
        self.model.to(self.device).eval()

        self.model.conf = self.conf
        self.model.iou = self.iou
        self.model.max_det = self.max_det
        self.model.classes = None  # or [0] if single-class

    @torch.no_grad()
    def detect_all(self, frame_bgr):
        results = self.model(frame_bgr, size=self.imgsz)
        det = results.xyxy[0]
        if det is None or len(det) == 0:
            return []

        det = det.detach().cpu().numpy()
        h, w = frame_bgr.shape[:2]

        out = []
        for (x1, y1, x2, y2, conf, cls_id) in det:
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w - 1, x2)))
            y2 = int(max(0, min(h - 1, y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            cx = int((x1 + x2) * 0.5)
            cy = int((y1 + y2) * 0.5)
            out.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "conf": float(conf),
                    "cls": int(cls_id),
                }
            )

        out.sort(key=lambda d: d["conf"], reverse=True)
        return out


# ==============================
# H landing: descend while centering to H bbox center
# ==============================
@dataclass
class LandResult:
    ok: bool
    reason: str


class HCenteringLander:
    """
    Visual-servo landing xuống chữ H:
    - Dựa trên detection bbox center của H (model best_H.pt).
    - Vừa hạ độ cao (vz > 0 trong BODY_NED) vừa chỉnh vx/vy để đưa tâm H về tâm ảnh.
    - Nếu mất H quá lâu -> fail để fallback sang LAND của autopilot.

    Mapping vx/vy dùng lại đúng như seek_target_and_center()/control_drone_to_center() hiện tại.
    """

    def __init__(
        self,
        vehicle,
        get_frame_bgr,
        get_h_dets,
        logger,
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
        switch_to_land_alt_m: float = 0.8,
        alt_done_m: float = 0.15,
        max_vxy: float = 0.30,
        ema_alpha: float = 0.25,
    ):
        self.vehicle = vehicle
        self.get_frame_bgr = get_frame_bgr
        self.get_h_dets = get_h_dets
        self.log = logger

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
    def send_local_ned_velocity(self, vx, vy, vz):
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

    def _best_h_center(self, dets):
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

    def _ema(self, ex, ey):
        if self._ema_ex is None:
            self._ema_ex, self._ema_ey = float(ex), float(ey)
        else:
            a = self.ema_alpha
            self._ema_ex = (1 - a) * self._ema_ex + a * float(ex)
            self._ema_ey = (1 - a) * self._ema_ey + a * float(ey)
        return self._ema_ex, self._ema_ey

    def land_on_h(self, timeout_sec: float = 90.0) -> LandResult:
        # reset PID giống seek_target_and_center()
        try:
            PID_X.reset()
            PID_Y.reset()
        except Exception:
            pass

        t0 = time.time()
        dt = 1.0 / max(1.0, self.loop_hz)

        last_seen_t = None
        last_center = None
        center_hold_t0 = None

        # ensure GUIDED for velocity control
        try:
            if self.vehicle.mode.name != "GUIDED":
                self.vehicle.mode = VehicleMode("GUIDED")
                t_mode = time.time()
                while self.vehicle.mode.name != "GUIDED" and (time.time() - t_mode) < 8.0:
                    time.sleep(0.2)
        except Exception:
            pass

        self.log.info("[H-LAND] Start visual-servo landing (GUIDED velocities).")

        while rclpy.ok():
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
                if (not bool(self.vehicle.armed)) or alt <= self.alt_done_m:
                    try:
                        self.send_local_ned_velocity(0.0, 0.0, 0.0)
                    except Exception:
                        pass
                    return LandResult(True, f"done alt={alt:.2f}m armed={self.vehicle.armed}")
            except Exception:
                pass

            frame = None
            try:
                frame = self.get_frame_bgr()
            except Exception:
                frame = None

            if frame is None:
                # no camera -> hover, wait
                try:
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass
                time.sleep(dt)
                continue

            h_img, w_img = frame.shape[:2]
            icx, icy = w_img // 2, h_img // 2

            try:
                dets = self.get_h_dets()
            except Exception:
                dets = []

            center = self._best_h_center(dets)
            now = time.time()

            if center is not None:
                last_center = center
                last_seen_t = now
            else:
                # if just lost -> keep last center for a short time
                if last_center is not None and last_seen_t is not None and (now - last_seen_t) <= self.hold_last_sec:
                    center = last_center
                else:
                    # lost too long -> stop and fail for fallback LAND
                    try:
                        self.send_local_ned_velocity(0.0, 0.0, 0.0)
                    except Exception:
                        pass
                    return LandResult(False, "lost H")

            tx, ty = center
            ex = float(tx - icx)
            ey = float(ty - icy)

            # smooth error a bit (reduce jitter)
            ex, ey = self._ema(ex, ey)

            # compute vx/vy using same mapping as control_drone_to_center()
            vx = float(PID_Y.update(-ey))
            vy = float(PID_X.update(ex))

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
                    # once held centered for some time -> allow full descent
                    held = (now - center_hold_t0) >= self.hold_center_sec
                    vz = self.descent_vz if held else max(self.min_vz, 0.7 * self.descent_vz)
                else:
                    center_hold_t0 = None
                    # linear blend from descent_vz -> min_vz
                    scale = 1.0 - (err / float(max(1, self.err_full_px)))
                    scale = max(0.0, min(1.0, scale))
                    vz = self.min_vz + (self.descent_vz - self.min_vz) * scale
                    # if too far, effectively hover (still correcting laterally)
                    if scale <= 0.05:
                        vz = 0.0

            # if low altitude -> switch to LAND mode (autopilot handles flare/disarm)
            if alt <= self.switch_to_land_alt_m:
                try:
                    self.log.info(f"[H-LAND] Switch to LAND at alt={alt:.2f}m")
                    self.vehicle.mode = VehicleMode("LAND")
                    # stop sending guided velocities (LAND ignores)
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass

                # wait until disarm or altitude done or timeout
                t_land = time.time()
                while rclpy.ok() and (time.time() - t_land) < 30.0:
                    try:
                        alt2 = float(self.vehicle.location.global_relative_frame.alt)
                        if (not bool(self.vehicle.armed)) or alt2 <= self.alt_done_m:
                            return LandResult(True, f"LAND ok alt={alt2:.2f}m")
                    except Exception:
                        pass
                    time.sleep(0.2)

                return LandResult(True, "LAND switched (wait timeout)")

            # send velocity command
            try:
                self.send_local_ned_velocity(vx, vy, vz)
            except Exception:
                pass

            time.sleep(dt)


# ==============================
# Drone Controller
# ==============================
class DroneController(Node):
    def __init__(self, connection_str="tcp:127.0.0.1:5763", takeoff_height=4.5):
        super().__init__("drone_controller")
        self._stop_flag = False

        self.connection_str = connection_str
        self.takeoff_height = float(takeoff_height)

        self.get_logger().info(f"[Connecting to vehicle on]: {self.connection_str}")
        self.vehicle = connect(self.connection_str, wait_ready=True, timeout=60)

        # ArduPilot precision landing params (best-effort)
        try:
            self.vehicle.parameters["PLND_ENABLED"] = 1
            self.vehicle.parameters["PLND_TYPE"] = 1
            self.vehicle.parameters["PLND_EST_TYPE"] = 1
            self.vehicle.parameters["LAND_SPEED"] = 30
        except Exception:
            pass

        # Start image subscriber first
        self.start_image("/UAV/forward/image_raw")

        # 2 detectors
        self.detector_circle = YOLOv5TorchHubDetector(
            MODEL_CIRCLE_PATH, imgsz=320, conf=0.35, iou=0.35, max_det=10, device="cuda:0"
        )
        self.detector_h = YOLOv5TorchHubDetector(
            MODEL_H_PATH, imgsz=416, conf=0.25, iou=0.45, max_det=5, device="cuda:0"
        )

        # HSV + tracker (tracker optional)
        self.color_classifier = HSVColorClassifier(color_threshold=0.05, debug=False)
        self.tracker = CenterTracker(offset=(0, 0), circle_detect=250)

        # detection buffers (single lock for both read/write)
        self._det_lock = threading.Lock()
        self._last_circle = []
        self._last_h = []

        # viewer/record
        self._show_camera = True
        self._window_name = "UAV Camera (circle+HSV + H)"
        self._recording = False
        self._video_writer = None
        self._record_fps = 30.0
        self._record_dir = os.path.join(os.path.dirname(__file__), "records")
        os.makedirs(self._record_dir, exist_ok=True)

        # RED1/RED2 tracking (NO PROMOTE)
        self.prev_red1 = None
        self.prev_red2 = None
        self.red1_miss = 0
        self.red2_miss = 0
        self.RED_MISS_MAX = 15
        self.RED_MATCH_DIST = 120  # px

        # yellow left/right disambiguation
        self.YELLOW_SIDE_MARGIN = 40
        self.YELLOW_LOCK_DIST = 250

        # mission state
        self.mission_step = 0

        # H landing controller (descend while centering)
        self.h_lander = HCenteringLander(
            vehicle=self.vehicle,
            get_frame_bgr=lambda: getattr(_image_node, "last_frame_bgr", None) if _image_node is not None else None,
            get_h_dets=self._get_last_h_dets_copy,
            logger=self.get_logger(),
            conf_th=0.25,
            tol_px=10,
            hold_center_sec=0.8,
            hold_last_sec=1.0,
            loop_hz=20.0,
            descent_vz=0.35,
            min_vz=0.10,
            stop_descend_err_px=220,
            err_full_px=180,
            switch_to_land_alt_m=0.8,
            alt_done_m=0.15,
            max_vxy=0.30,
            ema_alpha=0.25,
        )

        # threads/timers
        self._viz_timer = self.create_timer(1.0 / 30.0, self._viz_timer_cb)
        self._det_thread = threading.Thread(target=self._det_loop, daemon=True)
        self._det_thread.start()

        self._mission_thread = threading.Thread(target=self.mission_complete, daemon=True)
        self._mission_thread.start()

    def _get_last_h_dets_copy(self):
        with self._det_lock:
            return list(self._last_h) if self._last_h else []

    # --------------------------
    # Detect both models continuously (so step7 is not "cold")
    # --------------------------
    def _det_loop(self):
        det_hz = 12.0
        det_dt = 1.0 / det_hz
        last_t = 0.0

        while rclpy.ok() and not self._stop_flag:
            t0 = time.time()
            if t0 - last_t < det_dt:
                time.sleep(0.002)
                continue
            last_t = t0

            global _image_node
            if _image_node is None:
                time.sleep(0.05)
                continue

            frame_bgr = getattr(_image_node, "last_frame_bgr", None)
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            try:
                circle_dets = self.detector_circle.detect_all(frame_bgr)
            except Exception as e:
                try:
                    self.get_logger().error(f"[DETECT circle] {e}")
                except Exception:
                    pass
                circle_dets = []

            try:
                h_dets = self.detector_h.detect_all(frame_bgr)
            except Exception as e:
                try:
                    self.get_logger().error(f"[DETECT H] {e}")
                except Exception:
                    pass
                h_dets = []

            with self._det_lock:
                self._last_circle = circle_dets
                self._last_h = h_dets

    # --------------------------
    # RED1/RED2 association (NO PROMOTE)
    # --------------------------
    def _assign_red_ids_no_promote(self, red_dets):
        if not red_dets:
            return None, None

        def dist2(p, q):
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            return dx * dx + dy * dy

        gate2 = self.RED_MATCH_DIST * self.RED_MATCH_DIST

        if self.prev_red1 is None and self.prev_red2 is None:
            if len(red_dets) == 1:
                return red_dets[0], None
            red_sorted = sorted(red_dets, key=lambda d: d["center"][1])
            return red_sorted[0], red_sorted[1]

        if len(red_dets) == 1:
            d = red_dets[0]
            c = d["center"]
            d1 = dist2(c, self.prev_red1) if self.prev_red1 is not None else 1e18
            d2v = dist2(c, self.prev_red2) if self.prev_red2 is not None else 1e18
            ok1 = d1 <= gate2
            ok2 = d2v <= gate2

            if ok1 and ok2:
                return (d, None) if d1 <= d2v else (None, d)
            if ok2:
                return None, d
            if ok1:
                return d, None

            # cannot match -> fill missing track WITHOUT promote
            if self.prev_red2 is None:
                return None, d
            if self.prev_red1 is None:
                return d, None

            return (d, None) if d1 <= d2v else (None, d)

        red_dets = sorted(red_dets, key=lambda x: x["conf"], reverse=True)
        cand = red_dets[:2]
        c0 = cand[0]["center"]
        c1 = cand[1]["center"]

        d00 = dist2(c0, self.prev_red1) if self.prev_red1 is not None else 1e18
        d01 = dist2(c0, self.prev_red2) if self.prev_red2 is not None else 1e18
        d10 = dist2(c1, self.prev_red1) if self.prev_red1 is not None else 1e18
        d11 = dist2(c1, self.prev_red2) if self.prev_red2 is not None else 1e18

        costA = d00 + d11
        costB = d10 + d01

        A_ok = ((d00 <= gate2) or (self.prev_red1 is None)) and ((d11 <= gate2) or (self.prev_red2 is None))
        B_ok = ((d10 <= gate2) or (self.prev_red1 is None)) and ((d01 <= gate2) or (self.prev_red2 is None))

        if A_ok and (not B_ok or costA <= costB):
            return cand[0], cand[1]
        if B_ok:
            return cand[1], cand[0]

        red_sorted = sorted(cand, key=lambda d: d["center"][1])
        return red_sorted[0], red_sorted[1]

    # --------------------------
    # Visualization: draw circles+HSV and H detections
    # --------------------------
    def _viz_timer_cb(self):
        if self._stop_flag or not self._show_camera:
            return

        global _image_node
        if _image_node is None:
            return

        frame_bgr = getattr(_image_node, "last_frame_bgr", None)
        if frame_bgr is None:
            return

        vis = frame_bgr.copy()
        h_img, w_img = vis.shape[:2]
        icx, icy = w_img // 2, h_img // 2

        with self._det_lock:
            circle_dets = list(self._last_circle) if self._last_circle else []
            h_dets = list(self._last_h) if self._last_h else []

        # ---- HSV classify for circle detections
        colored = []
        for det in circle_dets:
            x1, y1, x2, y2 = det["bbox"]
            roi = vis[y1:y2, x1:x2]  # BGR ROI
            color_name, draw_bgr = "UNK", (0, 255, 0)
            if roi is not None and roi.size > 0:
                info = self.color_classifier.classify(roi)
                if info is not None:
                    color_name, draw_bgr = info

            d2 = dict(det)
            d2["color_name"] = color_name
            d2["draw_bgr"] = draw_bgr
            colored.append(d2)

        # ---- RED1/RED2 tracking
        red_dets = [d for d in colored if d["color_name"] == "RED"]
        red1, red2 = self._assign_red_ids_no_promote(red_dets)

        if red1 is not None:
            self.prev_red1 = red1["center"]
            self.red1_miss = 0
        else:
            self.red1_miss += 1
            if self.red1_miss >= self.RED_MISS_MAX:
                self.prev_red1 = None

        if red2 is not None:
            self.prev_red2 = red2["center"]
            self.red2_miss = 0
        else:
            self.red2_miss += 1
            if self.red2_miss >= self.RED_MISS_MAX:
                self.prev_red2 = None

        red1_center = red1["center"] if red1 is not None else None
        red2_center = red2["center"] if red2 is not None else None

        def near(a, b, tol=3):
            if a is None or b is None:
                return False
            return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol

        # ---- draw circle detections
        for d in colored:
            x1, y1, x2, y2 = d["bbox"]
            cx, cy = d["center"]
            conf = d["conf"]
            color_name = d["color_name"]
            draw_bgr = d["draw_bgr"]

            tag = color_name
            if color_name == "RED":
                if near((cx, cy), red1_center):
                    tag = "RED1"
                elif near((cx, cy), red2_center):
                    tag = "RED2"
                else:
                    tag = "RED"
            elif color_name == "YELLOW":
                tag = "YELLOW_LEFT" if cx < icx else "YELLOW_RIGHT"

            cv2.rectangle(vis, (x1, y1), (x2, y2), draw_bgr, 2)
            cv2.circle(vis, (cx, cy), 4, draw_bgr, -1)
            cv2.putText(
                vis,
                f"{tag} {conf:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                draw_bgr,
                2,
            )

        # ---- draw H detections (cyan)
        best_h_conf = 0.0
        for hd in h_dets:
            x1, y1, x2, y2 = hd["bbox"]
            cx, cy = hd["center"]
            conf = float(hd["conf"])
            best_h_conf = max(best_h_conf, conf)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 4, (255, 255, 0), -1)
            cv2.putText(
                vis,
                f"H {conf:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2,
            )

        # ---- center marker + status
        cv2.circle(vis, (icx, icy), 4, (255, 0, 0), -1)
        cv2.putText(
            vis,
            f"circle={len(circle_dets)}  H={len(h_dets)}(best={best_h_conf:.2f})  step={self.mission_step}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        # ---- show/record
        try:
            if self._recording:
                cv2.putText(vis, "REC", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow(self._window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                if not self._recording:
                    self._start_recording(vis)
                else:
                    self._stop_recording()

            if key in (27, ord("q")):
                if self._recording:
                    self._stop_recording()
                self._show_camera = False
                cv2.destroyWindow(self._window_name)
                return

            if self._recording and self._video_writer is not None:
                self._video_writer.write(vis)

        except Exception as e:
            try:
                self.get_logger().error(f"cv2.imshow failed: {e}")
            except Exception:
                pass
            self._show_camera = False
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # --------------------------
    # Video recording
    # --------------------------
    def _start_recording(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self._record_dir, f"record_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, float(self._record_fps), (w, h))
        if not vw.isOpened():
            raise RuntimeError("VideoWriter open failed (try XVID + .avi).")
        self._video_writer = vw
        self._recording = True
        self.get_logger().info(f"[REC] START -> {out_path}")

    def _stop_recording(self):
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
        self._video_writer = None
        self._recording = False
        self.get_logger().info("[REC] STOP")

    # --------------------------
    # Mission helpers: SEEK + CENTER + HOLD
    # --------------------------
    def _direction_to_vxvy(self, direction: str, speed: float):
        d = (direction or "").lower()
        if d == "forward":
            return float(speed), 0.0
        if d == "backward":
            return -float(speed), 0.0
        if d == "left":
            return 0.0, -float(speed)
        if d == "right":
            return 0.0, float(speed)
        return 0.0, 0.0

    def _colorize_dets(self, frame_bgr, dets):
        colored = []
        if frame_bgr is None or dets is None:
            return colored
        h, w = frame_bgr.shape[:2]

        for det in dets:
            try:
                x1, y1, x2, y2 = det["bbox"]
                x1 = int(max(0, min(w - 1, x1)))
                x2 = int(max(0, min(w, x2)))
                y1 = int(max(0, min(h - 1, y1)))
                y2 = int(max(0, min(h, y2)))
                roi = frame_bgr[y1:y2, x1:x2]
            except Exception:
                continue

            color_name = "UNK"
            if roi is not None and roi.size > 0:
                try:
                    info = self.color_classifier.classify(roi)
                    if info is not None:
                        color_name, _ = info
                except Exception:
                    pass

            d2 = dict(det)
            d2["color_name"] = color_name
            colored.append(d2)

        return colored

    def _pick_target_center(self, colored_dets, target_tag: str, img_cx: int = None):
        tag = (target_tag or "").upper()

        if tag in ("RED1", "RED2", "RED"):
            red_dets = [d for d in colored_dets if d.get("color_name") == "RED"]
            if not red_dets:
                return None

            if tag == "RED":
                red_dets.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
                return red_dets[0].get("center", None)

            red1, red2 = self._assign_red_ids_no_promote(red_dets)
            if tag == "RED1":
                return red1.get("center") if red1 is not None else None
            return red2.get("center") if red2 is not None else None

        if tag in ("YELLOW", "YELLOW_LEFT", "YELLOW_RIGHT"):
            ys = [d for d in colored_dets if d.get("color_name") == "YELLOW"]
            if not ys:
                return None

            if tag == "YELLOW":
                ys.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
                return ys[0].get("center", None)

            ys_sorted = sorted(ys, key=lambda d: d.get("center", (10**9, 0))[0])
            margin = int(getattr(self, "YELLOW_SIDE_MARGIN", 40))

            if tag == "YELLOW_LEFT":
                cand = ys_sorted[0]
                if img_cx is not None and cand.get("center", (0, 0))[0] > int(img_cx + margin):
                    return None
                return cand.get("center", None)

            cand = ys_sorted[-1]  # right
            if img_cx is not None and cand.get("center", (0, 0))[0] < int(img_cx - margin):
                return None
            return cand.get("center", None)

        if tag == "BLUE":
            bs = [d for d in colored_dets if d.get("color_name") == "BLUE"]
            if not bs:
                return None
            bs.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
            return bs[0].get("center", None)

        if colored_dets:
            colored_dets = sorted(colored_dets, key=lambda d: d.get("conf", 0.0), reverse=True)
            return colored_dets[0].get("center", None)

        return None

    def seek_target_and_center(
        self,
        target_tag: str,
        direction: str,
        search_speed: float = 1.0,
        tol_px: int = 5,
        hold_s: float = 3.0,
        timeout_s: float = 30.0,
        pre_delay_s: float = 0.0,
        loop_hz: float = 20.0,
    ):
        try:
            PID_X.reset()
            PID_Y.reset()
        except Exception:
            pass

        if pre_delay_s and float(pre_delay_s) > 0.0:
            vx0, vy0 = self._direction_to_vxvy(direction, search_speed)
            t_pre = time.time()
            while rclpy.ok() and not self._stop_flag and (time.time() - t_pre) < float(pre_delay_s):
                self.send_local_ned_velocity(vx0, vy0, 0.0)
                time.sleep(0.1)
            try:
                PID_X.reset()
                PID_Y.reset()
            except Exception:
                pass

        global _image_node
        start_t = time.time()
        hold_start = None
        dt_sleep = 1.0 / float(loop_hz)

        yellow_lock = None
        lock_dist = int(getattr(self, "YELLOW_LOCK_DIST", 250))
        lock_gate2 = lock_dist * lock_dist
        margin = int(getattr(self, "YELLOW_SIDE_MARGIN", 40))

        while rclpy.ok() and not self._stop_flag:
            if timeout_s is not None and (time.time() - start_t) > float(timeout_s):
                self.get_logger().error(f"[SEEK] Timeout target={target_tag} dir={direction}")
                self.send_local_ned_velocity(0.0, 0.0, 0.0)
                return False

            if _image_node is None:
                time.sleep(0.05)
                continue

            frame_bgr = getattr(_image_node, "last_frame_bgr", None)
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            with self._det_lock:
                dets = list(self._last_circle) if self._last_circle else []

            h, w = frame_bgr.shape[:2]
            icx, icy = w // 2, h // 2

            colored = self._colorize_dets(frame_bgr, dets)

            tagU = (target_tag or "").upper()
            center = None

            if tagU in ("YELLOW_LEFT", "YELLOW_RIGHT"):
                ys = [d for d in colored if d.get("color_name") == "YELLOW"]
                if ys:
                    # keep tracking nearest to lock
                    if yellow_lock is not None:
                        def d2(det):
                            cx, cy = det.get("center", (0, 0))
                            dx = cx - yellow_lock[0]
                            dy = cy - yellow_lock[1]
                            return dx * dx + dy * dy

                        ys_nn = sorted(ys, key=d2)
                        if d2(ys_nn[0]) <= lock_gate2:
                            center = ys_nn[0].get("center", None)
                            yellow_lock = center
                        else:
                            yellow_lock = None

                    # acquire new lock only if on desired side
                    if yellow_lock is None:
                        if tagU == "YELLOW_LEFT":
                            cand = min(ys, key=lambda d: d.get("center", (10**9, 0))[0])
                            cx = cand.get("center", (0, 0))[0]
                            if cx < icx - margin:
                                center = cand.get("center", None)
                                yellow_lock = center
                        else:
                            cand = max(ys, key=lambda d: d.get("center", (-10**9, 0))[0])
                            cx = cand.get("center", (0, 0))[0]
                            if cx > icx + margin:
                                center = cand.get("center", None)
                                yellow_lock = center
            else:
                center = self._pick_target_center(colored, target_tag, img_cx=icx)

            if center is None:
                hold_start = None
                vx, vy = self._direction_to_vxvy(direction, search_speed)
                self.send_local_ned_velocity(vx, vy, 0.0)
            else:
                tx, ty = center
                ex = float(tx - icx)
                ey = float(ty - icy)

                if abs(ex) <= tol_px and abs(ey) <= tol_px:
                    if hold_start is None:
                        hold_start = time.time()
                        self.get_logger().info(f"[HOLD] start target={target_tag} ex={ex:.1f} ey={ey:.1f}")
                    elif (time.time() - hold_start) >= float(hold_s):
                        self.get_logger().info(f"[HOLD] done target={target_tag} hold={hold_s}s")
                        self.send_local_ned_velocity(0.0, 0.0, 0.0)
                        return True
                else:
                    hold_start = None

                self.control_drone_to_center(ex, ey)

            time.sleep(dt_sleep)

        try:
            self.send_local_ned_velocity(0.0, 0.0, 0.0)
        except Exception:
            pass
        return False

    # --------------------------
    # Mission
    # --------------------------
    def mission_complete(self):
        try:
            self.arm_and_takeoff(self.takeoff_height)
            self.get_logger().info("[Mission Started]")
            self.mission_step = 1

            while rclpy.ok() and not self._stop_flag:
                if self.mission_step == 1:
                    self.get_logger().info("[Step 1] FORWARD -> RED1 center hold 3s")
                    ok = self.seek_target_and_center(
                        target_tag="RED1",
                        direction="forward",
                        search_speed=1.0,
                        tol_px=5,
                        hold_s=3.0,
                        timeout_s=30.0,
                    )
                    if not ok:
                        break
                    self.mission_step = 2

                elif self.mission_step == 2:
                    self.get_logger().info("[Step 2] FORWARD -> RED2 center hold 3s")
                    ok = self.seek_target_and_center(
                        target_tag="RED2",
                        direction="forward",
                        search_speed=1.0,
                        pre_delay_s=1.0,
                        tol_px=5,
                        hold_s=3.0,
                        timeout_s=20.0,
                    )
                    if not ok:
                        break
                    self.mission_step = 3

                elif self.mission_step == 3:
                    self.get_logger().info("[Step 3] LEFT -> YELLOW_LEFT center hold 3s")
                    ok = self.seek_target_and_center(
                        target_tag="YELLOW_LEFT",
                        direction="left",
                        search_speed=1.0,
                        pre_delay_s=1.0,
                        tol_px=5,
                        hold_s=3.0,
                        timeout_s=25.0,
                    )
                    if not ok:
                        break
                    self.mission_step = 4

                elif self.mission_step == 4:
                    self.get_logger().info("[Step 4] RIGHT -> YELLOW_RIGHT center hold 3s")
                    ok = self.seek_target_and_center(
                        target_tag="YELLOW_RIGHT",
                        direction="right",
                        search_speed=1.0,
                        pre_delay_s=1.0,
                        tol_px=5,
                        hold_s=3.0,
                        timeout_s=25.0,
                    )
                    if not ok:
                        break
                    self.mission_step = 5

                elif self.mission_step == 5:
                    self.get_logger().info("[Step 5] Original move: left 2.5s then forward 2.5s")
                    ok = self.move_with_timer(direction="left", duration=2.5, speed=1.0)
                    time.sleep(5)
                    if not ok:
                        break
                    ok = self.move_with_timer(direction="forward", duration=2.5, speed=1.0)
                    time.sleep(5)
                    if not ok:
                        break
                    self.mission_step = 6

                elif self.mission_step == 6:
                    self.get_logger().info("[Step 6] Backward 13.5s")
                    ok = self.move_with_timer(direction="backward", duration=13.5, speed=1.0)
                    time.sleep(1)
                    if not ok:
                        break
                    self.mission_step = 7

                elif self.mission_step == 7:
                    self.get_logger().info("[Step 7] LAND on H (visual-servo)")
                    res = self.h_lander.land_on_h(timeout_sec=90.0)
                    if not res.ok:
                        self.get_logger().error(f"[H-LAND] failed: {res.reason} -> fallback LAND")
                        try:
                            self.vehicle.mode = VehicleMode("LAND")
                        except Exception:
                            pass
                    else:
                        self.get_logger().info(f"[H-LAND] ok: {res.reason}")
                    self.mission_step = 8

                elif self.mission_step == 8:
                    self.get_logger().info("[STOP]")
                    self.destroy_node()
                    break

                time.sleep(0.1)

        except Exception as e:
            try:
                self.get_logger().error(f"[Mission Error] {e}")
            except Exception:
                pass

    # --------------------------
    # Image subscriber control
    # --------------------------
    def start_image(self, topic_name="/UAV/forward/image_raw"):
        try:
            start_image_subscriber(topic_name=topic_name)
            self.get_logger().info(f"[Image Subscriber Started]: {topic_name}")
        except Exception as e:
            self.get_logger().error(f"Error starting image subscriber: {e}")

    # --------------------------
    # DroneKit helpers
    # --------------------------
    def send_local_ned_velocity(self, vx, vy, vz):
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

    def arm_and_takeoff(self, targetHeight):
        self.vehicle.mode = VehicleMode("GUIDED")
        while self.vehicle.mode.name != "GUIDED" and rclpy.ok() and not self._stop_flag:
            self.get_logger().info(" Waiting for mode change ...")
            time.sleep(1)

        self.vehicle.armed = True
        while not self.vehicle.armed and rclpy.ok() and not self._stop_flag:
            self.get_logger().info(" Waiting for arming...")
            time.sleep(1)

        self.vehicle.simple_takeoff(targetHeight)
        while rclpy.ok() and not self._stop_flag:
            alt = self.vehicle.location.global_relative_frame.alt
            self.get_logger().info(f" Altitude: {alt:.1f} m")
            if alt >= targetHeight * 0.95:
                break
            time.sleep(1)

        self.get_logger().info("Target altitude reached.")

    def move_with_timer(self, direction, duration, speed=0.5):
        vx, vy = self._direction_to_vxvy(direction, speed)
        self.get_logger().info(f"[MOVE] {direction} for {duration}s at {speed} m/s")
        t0 = time.time()
        while (time.time() - t0) < duration and rclpy.ok() and not self._stop_flag:
            self.send_local_ned_velocity(vx, vy, 0.0)
            time.sleep(0.1)
        self.send_local_ned_velocity(0.0, 0.0, 0.0)
        return True

    def control_drone_to_center(self, ex, ey):
        if abs(ex) < 5 and abs(ey) < 5:
            self.send_local_ned_velocity(0.0, 0.0, 0.0)
            return

        vx = PID_Y.update(-ey)
        vy = PID_X.update(ex)

        vx = max(min(vx, 0.3), -0.3)
        vy = max(min(vy, 0.3), -0.3)

        self.send_local_ned_velocity(vx, vy, 0.0)

    # --------------------------
    # Clean shutdown
    # --------------------------
    def destroy_node(self):
        self._stop_flag = True

        try:
            if self._recording:
                self._stop_recording()
        except Exception:
            pass

        try:
            stop_image_subscriber()
        except Exception:
            pass

        try:
            if self.vehicle is not None:
                self.vehicle.close()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DroneController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
