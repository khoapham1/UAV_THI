#!/usr/bin/env python3
# dropball.py (UPDATED)
# Pure Python (NO ROS2) + Webcam VideoCapture(0)
# - YOLOv5 (torch.hub local repo) for circle detection (best.pt) + HSV color classify
# - YOLOv5 (torch.hub local repo) for H detection (best_H.pt) for landing
# - HOLD logic UPDATED: When IMAGE CENTER enters target circle BBOX -> start HOLD.
#   Hold continuously for 5s -> step complete. If center leaves bbox -> reset hold.
# - UI UPDATED: show HOLD timer on cv2.imshow

import os
import time
import threading
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

# ---- Windows PosixPath cache fix (prevents: cannot instantiate 'PosixPath' on your system) ----
import pathlib
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

# Optional: isolate torch cache per project (recommended)
TORCH_HOME = os.path.join(os.path.dirname(__file__), ".torch_cache")
os.environ["TORCH_HOME"] = TORCH_HOME

from dronekit import connect, VehicleMode
from pymavlink import mavutil

from hsv_color import HSVColorClassifier
from PID_control import PIDController
from CenterTracker import CenterTracker  # kept for compatibility (optional usage)

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("UAV")

# =========================
# Helpers: find YOLOv5 repo
# =========================
def find_yolov5_repo(start_dir: str) -> str:
    """
    Find a local YOLOv5 repo folder that contains hubconf.py.
    Search order:
      1) start_dir
      2) common subfolders: yolov5/, YOLOv5/, UAV_Contest_2026-main/.../yolov5
      3) recursive walk up to depth 5
    """
    start = Path(start_dir).resolve()

    # 1) direct
    if (start / "hubconf.py").exists():
        return str(start)

    # 2) common guesses
    common = [
        start / "yolov5",
        start / "YOLOv5",
        start / "UAV_Contest_2026-main" / "yolov5",
        start / "UAV_Contest_2026-main" / "YOLOv5",
        start / "models" / "yolov5",
        start / "models" / "YOLOv5",
    ]
    for p in common:
        if (p / "hubconf.py").exists():
            return str(p.resolve())

    # 3) walk (depth-limited)
    max_depth = 5
    for root, dirs, files in os.walk(start):
        root_path = Path(root)
        try:
            depth = len(root_path.relative_to(start).parts)
        except Exception:
            depth = 999
        if depth > max_depth:
            dirs[:] = []
            continue
        if "hubconf.py" in files:
            return str(root_path.resolve())

    raise FileNotFoundError(
        "Không tìm thấy YOLOv5 repo (hubconf.py). "
        "Hãy đảm bảo bạn có folder yolov5/ chứa hubconf.py (vd: git clone https://github.com/ultralytics/yolov5.git)."
    )

def resolve_weights(preferred_abs: str, fallback_filename: str) -> str:
    """
    Use preferred absolute path if exists, else fallback to file next to this script.
    """
    p = Path(preferred_abs)
    if p.exists():
        return str(p.resolve())
    fb = Path(os.path.dirname(__file__)) / fallback_filename
    if fb.exists():
        return str(fb.resolve())
    raise FileNotFoundError(f"Không tìm thấy weights: {preferred_abs} và cũng không thấy {fb}")

# =========================
# UPDATED model paths (robust)
# =========================
# If your old absolute paths exist, we will use them; otherwise use local D:\UAV\best.pt style.
PREFERRED_BEST_PT = r"D:\UAV\best.pt"
PREFERRED_BEST_H_PT = r"D:\UAV\best_H.pt"

MODEL_CIRCLE_PATH = resolve_weights(PREFERRED_BEST_PT, "best.pt")
MODEL_H_PATH = resolve_weights(PREFERRED_BEST_H_PT, "best_H.pt")

YOLOV5_REPO_DIR = find_yolov5_repo(os.path.dirname(__file__))
log.info(f"[YOLOv5] Using repo: {YOLOV5_REPO_DIR}")
log.info(f"[WEIGHTS] circle={MODEL_CIRCLE_PATH}")
log.info(f"[WEIGHTS] H={MODEL_H_PATH}")

# =========================
# PID controllers
# =========================
PID_X = PIDController(0.0060, 0.00005, 0.00035, max_output=0.3, integral_limit=300)
PID_Y = PIDController(0.0055, 0.00005, 0.00030, max_output=0.3, integral_limit=300)

# =========================
# Camera stream (threaded)
# =========================
class CameraStream:
    def __init__(self, cam_index=0, width=None, height=None, fps=None):
        self.cam_index = int(cam_index)
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)  # Windows friendly
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam VideoCapture({self.cam_index})")

        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))

        self._lock = threading.Lock()
        self._last_frame = None
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._last_frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self._lock:
            if self._last_frame is None:
                return None
            return self._last_frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        try:
            self.cap.release()
        except Exception:
            pass
        self._thread = None

# =========================
# YOLOv5 detector wrapper
# =========================
class YOLOv5TorchHubDetector:
    def __init__(self, weights_path: str, imgsz=320, conf=0.25, iou=0.45, max_det=10, device=""):
        self.weights_path = str(Path(weights_path).resolve())
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"weights not found: {self.weights_path}")

        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")

        hubconf = os.path.join(YOLOV5_REPO_DIR, "hubconf.py")
        if not os.path.exists(hubconf):
            raise FileNotFoundError(f"YOLOv5 repo not found/invalid (missing hubconf.py): {YOLOV5_REPO_DIR}")

        # force_reload=True helps avoid stale cache issues across machines
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
        self.model.classes = None

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


# =========================
# H visual landing controller
# =========================
@dataclass
class LandResult:
    ok: bool
    reason: str

class HCenteringLander:
    def __init__(
        self,
        vehicle,
        get_frame_bgr,
        get_h_dets,
        logger,
        stop_flag_getter,
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
    ):
        self.vehicle = vehicle
        self.get_frame_bgr = get_frame_bgr
        self.get_h_dets = get_h_dets
        self.log = logger
        self.stop_flag_getter = stop_flag_getter

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

    def send_local_ned_velocity(self, vx, vy, vz):
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

    def land_on_h(self, timeout_sec=90.0) -> LandResult:
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

        try:
            if self.vehicle.mode.name != "GUIDED":
                self.vehicle.mode = VehicleMode("GUIDED")
                t_mode = time.time()
                while self.vehicle.mode.name != "GUIDED" and (time.time() - t_mode) < 8.0:
                    if self.stop_flag_getter():
                        return LandResult(False, "stopped")
                    time.sleep(0.2)
        except Exception:
            pass

        self.log.info("[H-LAND] Start visual-servo landing (GUIDED velocities).")

        while not self.stop_flag_getter():
            if timeout_sec is not None and (time.time() - t0) > float(timeout_sec):
                try:
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass
                return LandResult(False, f"timeout {timeout_sec}s")

            try:
                alt = float(self.vehicle.location.global_relative_frame.alt)
            except Exception:
                alt = 999.0

            try:
                if (not bool(self.vehicle.armed)) or alt <= self.alt_done_m:
                    try:
                        self.send_local_ned_velocity(0.0, 0.0, 0.0)
                    except Exception:
                        pass
                    return LandResult(True, f"done alt={alt:.2f}m armed={self.vehicle.armed}")
            except Exception:
                pass

            frame = self.get_frame_bgr()
            if frame is None:
                try:
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass
                time.sleep(dt)
                continue

            h_img, w_img = frame.shape[:2]
            icx, icy = w_img // 2, h_img // 2

            dets = self.get_h_dets() or []
            center = self._best_h_center(dets)
            now = time.time()

            if center is not None:
                last_center = center
                last_seen_t = now
            else:
                if last_center is not None and last_seen_t is not None and (now - last_seen_t) <= self.hold_last_sec:
                    center = last_center
                else:
                    try:
                        self.send_local_ned_velocity(0.0, 0.0, 0.0)
                    except Exception:
                        pass
                    return LandResult(False, "lost H")

            tx, ty = center
            ex = float(tx - icx)
            ey = float(ty - icy)
            ex, ey = self._ema(ex, ey)

            vx = float(PID_Y.update(-ey))
            vy = float(PID_X.update(ex))
            vx = max(min(vx, self.max_vxy), -self.max_vxy)
            vy = max(min(vy, self.max_vxy), -self.max_vxy)

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

            if alt <= self.switch_to_land_alt_m:
                try:
                    self.log.info(f"[H-LAND] Switch to LAND at alt={alt:.2f}m")
                    self.vehicle.mode = VehicleMode("LAND")
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass

                t_land = time.time()
                while not self.stop_flag_getter() and (time.time() - t_land) < 30.0:
                    try:
                        alt2 = float(self.vehicle.location.global_relative_frame.alt)
                        if (not bool(self.vehicle.armed)) or alt2 <= self.alt_done_m:
                            return LandResult(True, f"LAND ok alt={alt2:.2f}m")
                    except Exception:
                        pass
                    time.sleep(0.2)

                return LandResult(True, "LAND switched (wait timeout)")

            try:
                self.send_local_ned_velocity(vx, vy, vz)
            except Exception:
                pass

            time.sleep(dt)

        return LandResult(False, "stopped")

# =========================
# DroneController
# =========================
class DroneController:
    def __init__(self, connection_str="tcp:127.0.0.1:5763", takeoff_height=4.5, cam_index=0):
        self._stop_flag = False

        self.connection_str = connection_str
        self.takeoff_height = float(takeoff_height)

        # UI hold status
        self._ui_lock = threading.Lock()
        self._ui_hold_target = None
        self._ui_hold_in_zone = False
        self._ui_hold_elapsed = 0.0
        self._ui_hold_required = 0.0

        log.info(f"[Connecting to vehicle on]: {self.connection_str}")
        self.vehicle = connect(self.connection_str, wait_ready=True, timeout=60)

        # Param set (best-effort)
        try:
            self.vehicle.parameters["PLND_ENABLED"] = 1
            self.vehicle.parameters["PLND_TYPE"] = 1
            self.vehicle.parameters["PLND_EST_TYPE"] = 1
            self.vehicle.parameters["LAND_SPEED"] = 30
        except Exception:
            pass

        # Camera
        self.camera = CameraStream(cam_index=cam_index)
        self.camera.start()

        # Detectors
        self.detector_circle = YOLOv5TorchHubDetector(
            MODEL_CIRCLE_PATH, imgsz=320, conf=0.65, iou=0.35, max_det=10, device=""
        )
        self.detector_h = YOLOv5TorchHubDetector(
            MODEL_H_PATH, imgsz=416, conf=0.65, iou=0.45, max_det=5, device=""
        )

        # HSV + tracker
        self.color_classifier = HSVColorClassifier(color_threshold=0.05, debug=False)
        self.tracker = CenterTracker(offset=(0, 0), circle_detect=250)

        # Detection buffers
        self._det_lock = threading.Lock()
        self._last_circle = []
        self._last_h = []

        # Video window/record
        
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

        # yellow disambiguation
        self.YELLOW_SIDE_MARGIN = 40
        self.YELLOW_LOCK_DIST = 250

        # mission state
        self.mission_step = 0

        # H lander
        self.h_lander = HCenteringLander(
            vehicle=self.vehicle,
            get_frame_bgr=self.camera.get_frame,
            get_h_dets=self._get_last_h_dets_copy,
            logger=log,
            stop_flag_getter=lambda: self._stop_flag,
        )

        # Threads
        self._det_thread = threading.Thread(target=self._det_loop, daemon=True)
        self._det_thread.start()

        self._mission_thread = threading.Thread(target=self.mission_complete, daemon=True)
        self._mission_thread.start()

    # ---------- UI hold helpers ----------
    def _ui_set_hold(self, target, in_zone, elapsed, required):
        with self._ui_lock:
            self._ui_hold_target = target
            self._ui_hold_in_zone = bool(in_zone)
            self._ui_hold_elapsed = float(elapsed)
            self._ui_hold_required = float(required)

    def _ui_clear_hold(self):
        with self._ui_lock:
            self._ui_hold_target = None
            self._ui_hold_in_zone = False
            self._ui_hold_elapsed = 0.0
            self._ui_hold_required = 0.0

    def stop(self):
        self._stop_flag = True

    def _get_last_h_dets_copy(self):
        with self._det_lock:
            return list(self._last_h) if self._last_h else []

    # ---------- Detection loop ----------
    def _det_loop(self):
        det_hz = 12.0
        det_dt = 1.0 / det_hz
        last_t = 0.0

        while not self._stop_flag:
            t0 = time.time()
            if t0 - last_t < det_dt:
                time.sleep(0.002)
                continue
            last_t = t0

            frame_bgr = self.camera.get_frame()
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            try:
                circle_dets = self.detector_circle.detect_all(frame_bgr)
            except Exception as e:
                log.error(f"[DETECT circle] {e}")
                circle_dets = []

            try:
                h_dets = self.detector_h.detect_all(frame_bgr)
            except Exception as e:
                log.error(f"[DETECT H] {e}")
                h_dets = []

            with self._det_lock:
                self._last_circle = circle_dets
                self._last_h = h_dets

    # --------------------------
    # RED1/RED2 association (NO PROMOTE)
    # --------------------------
    def _assign_red_ids_no_promote(self, red_dets):
        """
        Phân biệt được RED 1 và RED 2
        Dán nhãn ổn định “RED1” và “RED2” cho các detection
        màu đỏ ở frame hiện tại, dựa trên vị trí các RED ở frame trước

        Khi co 2 detection: chon cach gan co tong khoang cach nho nhat
        |------------------------|   |----------------------------|    |------------------|
        |           Input        | ->|        Prev                | -> |    Output        |
        |(red_dets: center, conf)|   |prev_red1, prev_red2        |    |det_red1, det_red2|
        |------------------------|   |Tam RED1, RED2 o frame truoc|    |------------------|

        """
        if not red_dets:
            return None, None
        # Tinh toan khoang cach Euclid giua 2 diem center RED 1 vaf RED 2
        def dist2(p, q):
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            return dx * dx + dy * dy

        gate2 = self.RED_MATCH_DIST * self.RED_MATCH_DIST


        if self.prev_red1 is None and self.prev_red2 is None:
            #Neu co 1 RED => RED1 , RED2 = None
            if len(red_dets) == 1:
                return red_dets[0], None
            # Neu >= 2 RED se sort theo truc Y Frame (center[1]) de lay lan luot RED1 RED2
            red_sorted = sorted(red_dets, key=lambda d: d["center"][1])
            return red_sorted[0], red_sorted[1]
        
        #===== TH1: Chi? co' 1 RED dc Detect =====
        if len(red_dets) == 1:
            d = red_dets[0]
            c = d["center"]
            d1 = dist2(c, self.prev_red1) if self.prev_red1 is not None else 1e18
            d2v = dist2(c, self.prev_red2) if self.prev_red2 is not None else 1e18
            ok1 = d1 <= gate2
            ok2 = d2v <= gate2
            
            # Neu bat duoc ca 2 RED => gan vao track gan hon
            # Match RED 1 => (d, None)
            # Match Red 2 => (None, d) 
            if ok1 and ok2:
                return (d, None) if d1 <= d2v else (None, d)
            if ok2:
                return None, d
            if ok1:
                return d, None

            # cannot match -> fill missing track WITHOUT promote
            # Neu Frame truoc RED 2 chua ton tai => Detect moi coi nhu RED 2
            # Neu Frame truoc RED 1 chua ton tai => Detect moi coi nhu RED 1
            # Neu ca hai prev ko match  => gan vao track nao gan hon 
            if self.prev_red2 is None:
                return None, d
            if self.prev_red1 is None:
                return d, None

            return (d, None)  if d1 <= d2v else (None, d)
        #======= TH2 Detect duoc >=2 RED ========
        # Lay top-2 Conf de dan nhan RED1, RED2
        red_dets = sorted(red_dets, key=lambda x: x["conf"], reverse=True)
        cand = red_dets[:2]
        c0 = cand[0]["center"]
        c1 = cand[1]["center"]

        # Tinh 4 khoang cach 2 Object toi 2 track cu:
        # d00: cand0 ↔ prev_red1
        # d01: cand0 ↔ prev_red2
        # d10: cand1 ↔ prev_red1
        # d11: cand1 ↔ prev_red2

        d00 = dist2(c0, self.prev_red1) if self.prev_red1 is not None else 1e18
        d01 = dist2(c0, self.prev_red2) if self.prev_red2 is not None else 1e18
        d10 = dist2(c1, self.prev_red1) if self.prev_red1 is not None else 1e18
        d11 = dist2(c1, self.prev_red2) if self.prev_red2 is not None else 1e18

        # Xet 2 cach gan:
        # A: cand0 -> RED1 and cand1 -> RED2, costA = d00 + d11
        # B: cand1 -> RED1 and cand0 -> RED2, costB = d10 + d01 
        costA = d00 + d11
        costB = d10 + d01

        A_ok = ((d00 <= gate2) or (self.prev_red1 is None)) and ((d11 <= gate2) or (self.prev_red2 is None))
        B_ok = ((d10 <= gate2) or (self.prev_red1 is None)) and ((d01 <= gate2) or (self.prev_red2 is None))

        # Nếu A hợp lệ và (B không hợp lệ hoặc A rẻ hơn) ⇒ chọn A
        # Else nếu B hợp lệ ⇒ chọn B
        # Nếu cả A và B đều không hợp lệ ⇒ fallback: 
        # sort theo y để gán RED1/RED2 theo vị trí như lúc khởi tạo.
        if A_ok and (not B_ok or costA <= costB):
            return cand[0], cand[1]
        if B_ok:
            return cand[1], cand[0]

        red_sorted = sorted(cand, key=lambda d: d["center"][1])
        return red_sorted[0], red_sorted[1]
    # ---------- Visualization ----------
    def run_viewer_loop(self):
        """
        Qui tac dat tag:
        Neu "RED":
            - Tam center gan red1_center -> tag = "RED1"
            - Gan red2_center            -> tag = "RED2"
            - Khong -> "RED"
            - KT bang ham near() sai so 3 px
        Neu "YELLOW":
            - cx < icx -> "YELLOW_LEFT"
            - cx > icx -> "YELLOW_RIGHT"
        FRAME:

        0-------------------------->x
        |
        |    
        |            0(icx, icy)
        |
        |
        y
        """
        fps_dt = 1.0 / 30.0
        while not self._stop_flag:
            if not self._show_camera:
                time.sleep(0.05)
                continue

            frame_bgr = self.camera.get_frame()
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            vis = frame_bgr.copy()
            h_img, w_img = vis.shape[:2]
            icx, icy = w_img // 2, h_img // 2 # lay tam frame (icx, icy)

            with self._det_lock:
                circle_dets = list(self._last_circle) if self._last_circle else []
                h_dets = list(self._last_h) if self._last_h else []

            # HSV classify for circle detections
            colored = []
            for det in circle_dets:
                x1, y1, x2, y2 = det["bbox"]
                roi = vis[y1:y2, x1:x2]
                color_name, draw_bgr = "UNK", (0, 255, 0)
                if roi is not None and roi.size > 0:
                    info = self.color_classifier.classify(roi)
                    if info is not None:
                        color_name, draw_bgr = info

                d2 = dict(det)
                d2["color_name"] = color_name
                d2["draw_bgr"] = draw_bgr
                colored.append(d2)

            # RED1/RED2 tracking
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
            
            # draw circle detections
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

            # draw H detections
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

            # center marker + status
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

            # HOLD timer overlay
            with self._ui_lock:
                tgt = self._ui_hold_target
                inz = self._ui_hold_in_zone
                elp = self._ui_hold_elapsed
                req = self._ui_hold_required

            if tgt is not None and req > 0:
                txt = f"HOLD {tgt}: {elp:.1f}/{req:.1f}s" if inz else f"WAIT {tgt}: {elp:.1f}/{req:.1f}s"
                color = (0, 255, 0) if inz else (0, 255, 255)
                cv2.putText(vis, txt, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # recording flag
            if self._recording:
                cv2.putText(vis, "REC", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            try:
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

                if self._recording and self._video_writer is not None:
                    self._video_writer.write(vis)

            except Exception as e:
                log.error(f"cv2.imshow failed: {e}")
                self._show_camera = False
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            time.sleep(fps_dt)

    # ---------- Recording ----------
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
        log.info(f"[REC] START -> {out_path}")

    def _stop_recording(self):
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
        self._video_writer = None
        self._recording = False
        log.info("[REC] STOP")

    # ---------- Movement helpers ----------
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
        """
        - Lay bbox -> cat ROI trong anh -> phan loai mau -> tra list detection color_name
        
        |------------------------------|      |------------------|
        |           Input              | ->   |    Output        |
        |(frame_bgr, dets[x1,y1,x2,y2])|      |colored, dict(det)|
        |------------------------------|      |------------------|

        """
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

    # ---------- UPDATED: HOLD when image center enters bbox (but still PID to true center) ----------
    def seek_target_and_center(
        self,
        target_tag: str,
        direction: str,
        search_speed: float = 1.0,
        tol_px: int = 5,        # deadband for "really centered"
        hold_s: float = 5.0,    # Hold time while image center stays inside target bbox
        timeout_s: float = 30.0,
        pre_delay_s: float = 0.0,
        loop_hz: float = 20.0,
        hold_bbox_pad: int = 0,  # optional expand bbox zone
    ):
        """
        Logic (FIXED):
        - Nếu chưa thấy target: bay theo direction để "seek".
        - Nếu thấy target: luôn chạy PID để kéo tâm target (circle center) về tâm ảnh.
        - Khi tâm ảnh (icx,icy) đi vào trong bbox (có thể pad): bắt đầu đếm hold_s,
        NHƯNG KHÔNG đứng im — vẫn tiếp tục PID để về đúng tâm circle.
        - Nếu tâm ảnh ra khỏi bbox: reset hold timer.
        """
        # reset PID
        try:
            PID_X.reset()
            PID_Y.reset()
        except Exception:
            pass

        # Drone bay theo huong direction 1 khoảng thời gian pre_delay_s -> reset PID
        # Mục đích giảm giật và tăng khả năng target
        if pre_delay_s and float(pre_delay_s) > 0.0:
            vx0, vy0 = self._direction_to_vxvy(direction, search_speed)
            t_pre = time.time()
            while not self._stop_flag and (time.time() - t_pre) < float(pre_delay_s):
                self.send_local_ned_velocity(vx0, vy0, 0.0)
                time.sleep(0.1)
            try:
                PID_X.reset()
                PID_Y.reset()
            except Exception:
                pass

        start_t = time.time()
        hold_start = None
        dt_sleep = 1.0 / float(max(1e-6, loop_hz))

        # small smoothing to reduce jitter
        ema_alpha = 0.25
        ex_f = 0.0
        ey_f = 0.0
        ema_inited = False

        # speed policy
        max_v_seek = 0.30
        max_v_hold = 0.18  # slower when already inside bbox (fine centering)

        # yellow disambiguation
        yellow_lock = None
        lock_dist = int(getattr(self, "YELLOW_LOCK_DIST", 250))
        lock_gate2 = lock_dist * lock_dist
        margin = int(getattr(self, "YELLOW_SIDE_MARGIN", 40))

        def pid_drive_to_center(ex_px: float, ey_px: float, max_vxy: float):
            """Compute and send BODY_NED vx/vy from pixel errors using PID (same mapping as control_drone_to_center).
            |------------------------|   |----------------------------|    |--------------------------|
            |           Input        | ->|        PID                 | -> |    Output                |
            |(ex_px, ey_px, max_vxy )|   |       vx, vy               |    |ex_px = 0, ey_px=0        |
            |(Trái/phải),(Lên/xuống),|   |                            |    |tâm Frame trùng tâm circle|
            |------------------------|   |----------------------------|    |--------------------------|
            """
            #Lọc nhiễu bằng EMA 
            nonlocal ex_f, ey_f, ema_inited
            if not ema_inited:
                ex_f, ey_f = float(ex_px), float(ey_px)
                ema_inited = True
            else:
                a = float(ema_alpha)
                ex_f = a * float(ex_px) + (1.0 - a) * ex_f
                ey_f = a * float(ey_px) + (1.0 - a) * ey_f

            # mapping:
            #   vx = PID_Y.update(-ey)
            #   vy = PID_X.update(ex)

            try:
                vx = float(PID_Y.update(-ey_f)) #đảo dấu -ey_f để đúng chiều tiến lùi theo trục ảnh
            except Exception:
                vx = 0.0
            try:
                vy = float(PID_X.update(ex_f))
            except Exception:
                vy = 0.0

            # Giới hạn tốc độ
            vx = float(np.clip(vx, -max_vxy, max_vxy))
            vy = float(np.clip(vy, -max_vxy, max_vxy))

            # Tránh rung micro quanh điểm cân bằng
            if abs(ex_f) <= float(tol_px) and abs(ey_f) <= float(tol_px):
                vx, vy = 0.0, 0.0

            self.send_local_ned_velocity(vx, vy, 0.0)
            return ex_f, ey_f, vx, vy
        
        # =========== Chọn màu và phân biết RED1, RED2, YELLOW_LEFT, YELLOW_RIGHT VÀ BLUE    ================
        while not self._stop_flag:
            # timeout
            if timeout_s is not None and (time.time() - start_t) > float(timeout_s):
                log.error(f"[SEEK] Timeout target={target_tag} dir={direction}")
                self.send_local_ned_velocity(0.0, 0.0, 0.0)
                self._ui_clear_hold()
                return False

            frame_bgr = self.camera.get_frame()
            if frame_bgr is None:
                time.sleep(0.01)
                continue
            # Lấy snapshot detections(circle detection) mới nhất
            with self._det_lock:
                dets = list(self._last_circle) if self._last_circle else []

            h, w = frame_bgr.shape[:2]
            icx, icy = w // 2, h // 2

            # “Colorize” detection + chuẩn hoá tag + chuẩn bị biến target
            colored = self._colorize_dets(frame_bgr, dets)
            tagU = (target_tag or "").upper() #về uppercase để so sánh dễ
            target_det = None

            # ---- choose target_det (includes bbox + center) ----
            if tagU in ("RED1", "RED2", "RED"):
                red_dets = [d for d in colored if d.get("color_name") == "RED"]
                if red_dets:
                    if tagU == "RED":
                        red_dets.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
                        target_det = red_dets[0]
                    else:
                        r1, r2 = self._assign_red_ids_no_promote(red_dets)
                        target_det = r1 if tagU == "RED1" else r2

            elif tagU in ("YELLOW", "YELLOW_LEFT", "YELLOW_RIGHT"):
                ys = [d for d in colored if d.get("color_name") == "YELLOW"]
                if ys:
                    if tagU == "YELLOW":
                        ys.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
                        target_det = ys[0]
                    else:
                        # lock-based selection but returns det
                        if yellow_lock is not None:
                            def d2(det):
                                cx, cy = det.get("center", (0, 0))
                                dx = cx - yellow_lock[0]
                                dy = cy - yellow_lock[1]
                                return dx * dx + dy * dy

                            ys_nn = sorted(ys, key=d2)
                            if d2(ys_nn[0]) <= lock_gate2:
                                target_det = ys_nn[0]
                                yellow_lock = target_det.get("center", None)
                            else:
                                yellow_lock = None

                        if yellow_lock is None:
                            if tagU == "YELLOW_LEFT":
                                cand = min(ys, key=lambda d: d.get("center", (10**9, 0))[0])
                                cx = cand.get("center", (0, 0))[0]
                                if cx < icx - margin:
                                    target_det = cand
                                    yellow_lock = target_det.get("center", None)
                            else:
                                cand = max(ys, key=lambda d: d.get("center", (-10**9, 0))[0])
                                cx = cand.get("center", (0, 0))[0]
                                if cx > icx + margin:
                                    target_det = cand
                                    yellow_lock = target_det.get("center", None)

            elif tagU == "BLUE":
                bs = [d for d in colored if d.get("color_name") == "BLUE"]
                if bs:
                    bs.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
                    target_det = bs[0]

            else:
                if colored:
                    colored_sorted = sorted(colored, key=lambda d: d.get("conf", 0.0), reverse=True)
                    target_det = colored_sorted[0]
            #===========================================================================================
            
            #========= Tìm -> bám -> vào vùng -> hold -> hoàn thành ====================================
            """
            Không thấy target → bay seek. Thấy target → PID bám tâm. 
            Khi tâm ảnh vào bbox → bắt đầu hold và vẫn PID chậm. Giữ đủ hold_s → True; timeout/stop → False.
            """
            # TH1: khống thấy target
            if target_det is None:
                # not found -> Drone keep searching in given direction
                hold_start = None
                self._ui_set_hold(target_tag, False, 0.0, hold_s)

                vx, vy = self._direction_to_vxvy(direction, search_speed)
                self.send_local_ned_velocity(vx, vy, 0.0)
                time.sleep(dt_sleep)
                continue

            # TH2: thấy target
            # target found -> Tính sai số pixel từ tâm target về tâm ảnh
            tx, ty = target_det.get("center", (icx, icy))
            ex = float(tx - icx)
            ey = float(ty - icy)

            x1, y1, x2, y2 = target_det.get("bbox", (0, 0, 0, 0))
            pad = int(hold_bbox_pad)
            if pad > 0:
                x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
            
            # Xác định tâm ảnh có nằm trong bbox không
            in_zone = (x1 <= icx <= x2) and (y1 <= icy <= y2)

            now = time.time()
            if in_zone:
                if hold_start is None:
                    hold_start = now
                    # reset integrators when entering hold zone for cleaner final centering
                    try:
                        PID_X.reset()
                        PID_Y.reset()
                    except Exception:
                        pass
                    # Re-init EMA (ex_f, ey_f = ex, ey): tránh EMA nhảy mạnh lúc chuyển trạng thái
                    ex_f, ey_f = ex, ey
                    ema_inited = True
                    log.info(f"[HOLD-ZONE] start target={target_tag}")

                elapsed = now - hold_start
                self._ui_set_hold(target_tag, True, elapsed, hold_s)

                # IMPORTANT FIX: still run PID to drive to true center (do NOT hover)
                pid_drive_to_center(ex, ey, max_v_hold)

                if elapsed >= float(hold_s):
                    log.info(f"[HOLD-ZONE] done target={target_tag} hold={hold_s}s")
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                    self._ui_clear_hold()
                    return True
            else:
                hold_start = None
                self._ui_set_hold(target_tag, False, 0.0, hold_s)

                # steer to center until we enter bbox
                pid_drive_to_center(ex, ey, max_v_seek)

            time.sleep(dt_sleep)

        # stop flag
        self.send_local_ned_velocity(0.0, 0.0, 0.0)
        self._ui_clear_hold()
        return False
    
    def land_to_markerH(
        self,
        timeout_s: float = 90.0,
        conf_th: float = 0.25,
        tol_px: int = 10,
        hold_center_sec: float = 0.8,
        loop_hz: float = 20.0,
        descent_vz: float = 0.35,         # m/s (down, BODY_NED => vz>0 là xuống)
        min_vz: float = 0.10,             # m/s (down tối thiểu khi đã thấy H)
        stop_descend_err_px: float = 220, # nếu lệch quá lớn => dừng xuống (vz=0)
        err_full_px: float = 180,         # scale tốc độ xuống theo sai số
        max_vxy: float = 0.30,            # giới hạn vx/vy
        ema_alpha: float = 0.25,          # làm mượt ex/ey
        switch_to_land_alt_m: float = 1.0,
        alt_done_m: float = 0.05,
        lost_h_timeout_s: float = 2.0,
    ):
        """
        Visual-servo landing xuống marker H (PURE PYTHON):
        - Lấy H detection mới nhất (self._last_h), chọn bbox có conf cao nhất.
        - Tính ex, ey so với tâm ảnh (icx, icy).
        - PID để sinh vx/vy (BODY_NED): vx tiến/lùi, vy trái/phải.
        - Đồng thời tạo vz>0 để hạ cao độ, giảm vz nếu lệch nhiều, dừng vz nếu lệch quá lớn.
        - Khi alt <= switch_to_land_alt_m: chuyển sang LAND để đáp nốt.
        """

        # reset PID
        try:
            PID_X.reset()
            PID_Y.reset()
        except Exception:
            pass

        # best-effort: đảm bảo GUIDED trước khi gửi velocity setpoint
        try:
            if getattr(self.vehicle, "mode", None) and self.vehicle.mode.name != "GUIDED":
                self.vehicle.mode = VehicleMode("GUIDED")
                t_mode = time.time()
                while (
                    not self._stop_flag
                    and (time.time() - t_mode) < 8.0
                    and getattr(self.vehicle, "mode", None)
                    and self.vehicle.mode.name != "GUIDED"
                ):
                    time.sleep(0.2)
        except Exception:
            pass

        t0 = time.time()
        last_seen_h = time.time()

        ex_f = 0.0
        ey_f = 0.0
        hold_start = None

        dt = 1.0 / float(max(1e-6, loop_hz))
        log_t = 0.0

        while not self._stop_flag:
            # timeout
            if (time.time() - t0) > float(timeout_s):
                try:
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass
                return LandResult(False, f"timeout {timeout_s}s")

            # lấy frame để biết tâm ảnh
            frame_bgr = None
            try:
                frame_bgr = self.camera.get_frame()
            except Exception:
                frame_bgr = None

            if frame_bgr is None:
                time.sleep(0.02)
                continue

            h_img, w_img = frame_bgr.shape[:2]
            icx, icy = w_img // 2, h_img // 2

            # lấy dets H mới nhất
            with self._det_lock:
                h_dets = list(self._last_h) if self._last_h else []

            # chọn best H theo conf (>=conf_th)
            best = None
            best_conf = -1.0
            for d in h_dets:
                cf = float(d.get("conf", 0.0))
                if cf >= float(conf_th) and cf > best_conf:
                    best_conf = cf
                    best = d

            if best is None:
                # mất H
                if (time.time() - last_seen_h) > float(lost_h_timeout_s):
                    try:
                        self.send_local_ned_velocity(0.0, 0.0, 0.0)
                    except Exception:
                        pass
                    return LandResult(False, f"lost H > {lost_h_timeout_s}s")

                # hover chờ H xuất hiện lại
                try:
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass
                time.sleep(dt)
                continue

            last_seen_h = time.time()

            # --- tính ex, ey theo tâm ảnh
            cx, cy = best.get("center", (icx, icy))
            ex = float(cx - icx)
            ey = float(cy - icy)

            # làm mượt để giảm rung (EMA)
            a = float(ema_alpha)
            ex_f = a * ex + (1.0 - a) * ex_f
            ey_f = a * ey + (1.0 - a) * ey_f

            # --- PID -> vx/vy (giữ mapping như control_drone_to_center())
            #   vx = PID_Y.update(-ey)  (ey dương => target dưới tâm ảnh)
            #   vy = PID_X.update(ex)   (ex dương => target bên phải tâm ảnh)
            try:
                vx = float(PID_Y.update(-ey_f))
            except Exception:
                vx = 0.0
            try:
                vy = float(PID_X.update(ex_f))
            except Exception:
                vy = 0.0

            vx = float(np.clip(vx, -max_vxy, max_vxy))
            vy = float(np.clip(vy, -max_vxy, max_vxy))

            # --- kiểm tra hold khi đã vào tâm
            if abs(ex_f) <= float(tol_px) and abs(ey_f) <= float(tol_px):
                if hold_start is None:
                    hold_start = time.time()
                centered = (time.time() - hold_start) >= float(hold_center_sec)
            else:
                hold_start = None
                centered = False

            # --- vz: xuống nhanh khi gần tâm, giảm/stop khi lệch lớn
            err = float(np.hypot(ex_f, ey_f))
            if err >= float(stop_descend_err_px):
                vz = 0.0
            else:
                scale = 1.0 - min(err / float(max(1e-6, err_full_px)), 1.0)
                vz = float(min_vz + (descent_vz - min_vz) * max(scale, 0.0))
                if centered:
                    vz = float(descent_vz)

            # --- nếu đủ thấp thì chuyển LAND để đáp nốt
            alt = None
            try:
                alt = float(self.vehicle.location.global_relative_frame.alt)
            except Exception:
                alt = None

            if alt is not None and alt <= float(switch_to_land_alt_m):
                # dừng guid-control trước khi chuyển LAND
                try:
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass

                try:
                    self.vehicle.mode = VehicleMode("LAND")
                except Exception:
                    pass

                # đợi chạm đất / disarm (best-effort)
                t_land = time.time()
                while not self._stop_flag and (time.time() - t_land) < 30.0:
                    try:
                        alt2 = float(self.vehicle.location.global_relative_frame.alt)
                    except Exception:
                        alt2 = None
                    try:
                        armed = bool(self.vehicle.armed)
                    except Exception:
                        armed = True

                    if (alt2 is not None and alt2 <= float(alt_done_m)) or (not armed):
                        return LandResult(True, "landed")
                    time.sleep(0.2)

                return LandResult(True, "switched to LAND (finalize)")

            # --- gửi velocity (GUIDED visual-servo)
            try:
                self.send_local_ned_velocity(vx, vy, vz)
            except Exception:
                pass

            # log nhẹ (~2Hz)
            if (time.time() - log_t) > 0.5:
                log_t = time.time()
                try:
                    log.info(
                        f"[H-LAND2] alt={alt if alt is not None else -1:.2f} conf={best_conf:.2f} "
                        f"ex={ex_f:.1f} ey={ey_f:.1f} vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}"
                    )
                except Exception:
                    pass

            time.sleep(dt)

        # stop_flag
        try:
            self.send_local_ned_velocity(0.0, 0.0, 0.0)
        except Exception:
            pass
        return LandResult(False, "stopped")

    # ---------- Mission ----------
    def mission_complete(self):
        try:
            self.arm_and_takeoff(self.takeoff_height)
            log.info("[Mission Started]")
            self.mission_step = 1

            while not self._stop_flag:
                if self.mission_step == 1:
                    log.info("[Step 1] FORWARD -> RED1 enter bbox then hold 5s")
                    ok = self.seek_target_and_center("RED1", "forward", 1.0, hold_s=5.0, timeout_s=30.0)
                    if not ok:
                        break
                    self.mission_step = 2

                elif self.mission_step == 2:
                    log.info("[Step 2] FORWARD -> RED2 enter bbox then hold 5s")
                    ok = self.seek_target_and_center("RED2", "forward", 1.0, hold_s=5.0, timeout_s=25.0, pre_delay_s=1.0)
                    if not ok:
                        break
                    self.mission_step = 3

                elif self.mission_step == 3:
                    log.info("[Step 3] LEFT -> YELLOW_LEFT enter bbox then hold 5s")
                    ok = self.seek_target_and_center("YELLOW_LEFT", "left", 1.0, hold_s=5.0, timeout_s=30.0, pre_delay_s=1.0)
                    if not ok:
                        break
                    self.mission_step = 4

                elif self.mission_step == 4:
                    log.info("[Step 4] RIGHT -> YELLOW_RIGHT enter bbox then hold 5s")
                    ok = self.seek_target_and_center("YELLOW_RIGHT", "right", 1.0, hold_s=5.0, timeout_s=30.0, pre_delay_s=1.0)
                    if not ok:
                        break
                    self.mission_step = 5

                elif self.mission_step == 5:
                    log.info("[Step 5] Move: left 2.5s then forward 2.5s -> BLUE enter bbox hold 5s")
                    ok = self.move_with_timer("left", 2.5, 1.0)
                    time.sleep(1)
                    if not ok:
                        break
                    ok = self.move_with_timer("forward", 2.5, 1.0)
                    time.sleep(1)

                    if not ok:
                        break
                    ok = self.seek_target_and_center("BLUE","", 0.0, hold_s=5.0, timeout_s=30.0)
                    self.mission_step = 6

                elif self.mission_step == 6:
                    log.info("[Step 6] Backward 13.5s")
                    ok = self.move_with_timer("backward", 13.5, 1.0)
                    time.sleep(1)
                    if not ok:
                        break
                    self.mission_step = 7

                elif self.mission_step == 7:
                    log.info("[Step 7] LAND on H (visual-servo)")
                    res = self.land_to_markerH(timeout_s=90)
                    # res = self.h_lander.land_on_h(timeout_sec=90.0)
                    if not res.ok:
                        log.error(f"[H-LAND] failed: {res.reason} -> fallback LAND")
                        try:
                            self.vehicle.mode = VehicleMode("LAND")
                        except Exception:
                            pass
                    else:
                        log.info(f"[H-LAND] ok: {res.reason}")
                    self.mission_step = 8

                elif self.mission_step == 8:
                    log.info("[STOP] Mission finished.")
                    self.stop()
                    break

                time.sleep(0.1)

        except Exception as e:
            log.error(f"[Mission Error] {e}")
            self.stop()

    # ---------- DroneKit helpers ----------
    def send_local_ned_velocity(self, vx, vy, vz):
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

    def arm_and_takeoff(self, targetHeight):
        self.vehicle.mode = VehicleMode("GUIDED")
        while self.vehicle.mode.name != "GUIDED" and not self._stop_flag:
            log.info(" Waiting for mode change ...")
            time.sleep(1)

        self.vehicle.armed = True
        while not self.vehicle.armed and not self._stop_flag:
            log.info(" Waiting for arming...")
            time.sleep(1)

        self.vehicle.simple_takeoff(targetHeight)
        while not self._stop_flag:
            alt = self.vehicle.location.global_relative_frame.alt
            log.info(f" Altitude: {alt:.1f} m")
            if alt >= targetHeight * 0.95:
                break
            time.sleep(1)

        log.info("Target altitude reached.")

    def move_with_timer(self, direction, duration, speed=0.5):
        vx, vy = self._direction_to_vxvy(direction, speed)
        log.info(f"[MOVE] {direction} for {duration}s at {speed} m/s")
        t0 = time.time()
        while (time.time() - t0) < duration and not self._stop_flag:
            self.send_local_ned_velocity(vx, vy, 0.0)
            time.sleep(0.1)
        self.send_local_ned_velocity(0.0, 0.0, 0.0)
        return True

    def control_drone_to_center(self, ex, ey):
        # ex: + means target is right of image center
        # ey: + means target is below image center
        vx = PID_Y.update(-ey)
        vy = PID_X.update(ex)

        vx = max(min(vx, 0.3), -0.3)
        vy = max(min(vy, 0.3), -0.3)

        self.send_local_ned_velocity(vx, vy, 0.0)

    # ---------- Shutdown ----------
    def shutdown(self):
        self._stop_flag = True
        try:
            if self._recording:
                self._stop_recording()
        except Exception:
            pass

        try:
            self.camera.stop()
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

# =========================
# Main
# =========================
def main():
    connection_str = "tcp:127.0.0.1:5763"
    takeoff_height = 4.5
    cam_index = 0

    dc = None
    try:
        dc = DroneController(connection_str=connection_str, takeoff_height=takeoff_height, cam_index=cam_index)
        dc.run_viewer_loop()  # OpenCV window must run on main thread
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt -> stopping")
    except Exception as e:
        log.error(f"Fatal error: {e}")
    finally:
        if dc is not None:
            dc.shutdown()

if __name__ == "__main__":
    main()