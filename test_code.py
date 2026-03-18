

import os
import time
import threading
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dronekit import connect, VehicleMode
from pymavlink import mavutil

from PID_control import PIDController
from CenterTracker import CircleMemoryTracker  # kept for compatibility (optional usage)
from common import (
    HailoPythonInferenceEngine, DetectionPostProcessor,
    scale_detections_to_original
)


warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("dropball")


# =========================
# Mock vehicle for image-only testing
# =========================
class MockMessageFactory:
    def set_position_target_local_ned_encode(self, *args, **kwargs):
        return {
            "type": "set_position_target_local_ned",
            "args": args,
            "kwargs": kwargs,
        }


class MockGlobalRelativeFrame:
    def __init__(self, alt=0.0):
        self.alt = float(alt)


class MockLocation:
    def __init__(self, alt=0.0):
        self.global_relative_frame = MockGlobalRelativeFrame(alt=alt)


class MockMaster:
    def __init__(self):
        self.target_system = 1
        self.target_component = 1


class MockVehicle:
    def __init__(self):
        self.mode = VehicleMode("GUIDED")
        self.armed = False
        self.location = MockLocation(alt=0.0)
        self._master = MockMaster()
        self.message_factory = MockMessageFactory()
        self.parameters = {}

    def send_mavlink(self, msg):
        return None

    def flush(self):
        return None

    def close(self):
        return None

    def simple_takeoff(self, target_height):
        try:
            self.location.global_relative_frame.alt = float(target_height)
        except Exception:
            self.location.global_relative_frame.alt = 0.0

# =========================
# Model / runtime config
# =========================
MODEL_PATH = "/home/pi/UAV_THI/UAV/best.hef"
INPUT_SIZE = (640, 640)

# WAIT TIME
TARGET_FPS = 30.0
FRAME_TIME = 1.0 / TARGET_FPS  # ~0.033s (33ms)

# CONFIG FOR MODEL
VERBOSE = False
NORMALIZE = False
CONFIDENCE_THRESHOLD = 0.7
NMS_IOU_THRESHOLD = 0.45
CLASS_NAMES = {
    0: "blue",
    1: "h_marker",
    2: "red",
    3: "yellow",
}

CLASS_BLUE = 0
CLASS_H_MARKER = 1
CLASS_RED = 2
CLASS_YELLOW = 3

log.info(f"[MODEL] HEF = {MODEL_PATH}")

# =========================
# PID controllers
# =========================
#trai/phai
PID_X = PIDController(0.0015, 0.0000008, 0.00032, max_output=0.115, integral_limit=300)
#tien/lui
PID_Y = PIDController(0.0013489, 0.0000008, 0.00045, max_output=0.115, integral_limit=300)


# =========================
# Camera stream (threaded)
# =========================
class CameraStream:
    def __init__(self, cam_index=0, width=None, height=None, fps=None):
        self.cam_index = int(cam_index)
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(self.cam_index, backend)
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
# Hailo detector wrapper
# =========================
class HailoHEFDetector:
    """
    Wrapper giữ output tương thích code cũ:
    detect_all(frame_bgr) -> list[{bbox, center, conf, cls}]

    Model HEF hiện tại chứa 4 class:
      - class 0: blue
      - class 1: h_marker
      - class 2: red
      - class 3: yellow
    """

    def __init__(
        self,
        hef_path: str,
        input_size=(640, 640),
        conf=0.5,
        iou=0.45,
        normalize=False,
        verbose=False,
        class_names=None,
    ):
        self.hef_path = os.path.abspath(os.path.expanduser(str(hef_path)))
        if not os.path.exists(self.hef_path):
            raise FileNotFoundError(f"HEF model not found: {self.hef_path}")

        self.input_size = tuple(input_size)
        self.conf = float(conf)
        self.iou = float(iou)
        self.normalize = bool(normalize)
        self.verbose = bool(verbose)
        self.class_names = class_names or CLASS_NAMES

        self.engine = self._build_engine()
        self.postprocessor = self._build_postprocessor()
        log.info(f"[HAILO] detector ready: {self.hef_path}")

    def _build_engine(self):
        last_err = None
        ctor_attempts = [
            ((self.hef_path,), {}),
            ((), {"hef_path": self.hef_path}),
            ((), {"model_path": self.hef_path}),
            ((), {"hef": self.hef_path}),
            ((), {"model": self.hef_path}),
            ((), {}),
        ]

        for args, kwargs in ctor_attempts:
            try:
                engine = HailoPythonInferenceEngine(*args, **kwargs)

                for load_name in ("load_model", "load_hef", "initialize", "init_model"):
                    if hasattr(engine, load_name):
                        loader = getattr(engine, load_name)
                        try:
                            loader(self.hef_path)
                            break
                        except TypeError:
                            try:
                                loader()
                                break
                            except Exception:
                                pass
                        except Exception:
                            pass
                return engine
            except Exception as e:
                last_err = e

        raise RuntimeError(f"Cannot initialize HailoPythonInferenceEngine: {last_err}")

    def _build_postprocessor(self):
        attempts = [
            ((), {}),
            ((), {"conf_threshold": self.conf}),
            ((), {"conf_threshold": self.conf, "iou_threshold": self.iou}),
            ((), {"num_classes": len(self.class_names), "conf_threshold": self.conf, "iou_threshold": self.iou}),
        ]
        for args, kwargs in attempts:
            try:
                return DetectionPostProcessor(*args, **kwargs)
            except Exception:
                pass
        return None

    def _extract_prediction_tensor(self, results):
        if results is None:
            return None

        if isinstance(results, dict):
            values = list(results.values())
        elif isinstance(results, (list, tuple)):
            values = list(results)
        else:
            values = [results]

        best = None
        best_size = -1
        for v in values:
            arr = np.asarray(v)
            if arr.size <= 0:
                continue
            if arr.ndim >= 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                continue
            if arr.shape[1] < 5 and arr.shape[0] >= 5:
                arr = arr.T
            if arr.ndim == 2 and arr.shape[1] >= 5 and arr.size > best_size:
                best = arr
                best_size = arr.size
        return best

    def _run_common_postprocess(self, results):
        # HailoPythonInferenceEngine.infer() trong common.py da tra ve list[dict]
        # voi keys: x1,y1,x2,y2,conf,cls_id,cls_name. Khong duoc postprocess them
        # lan nua, neu khong se sai format va khong ve duoc bbox.
        if results is None:
            return None

        if isinstance(results, (list, tuple)):
            if len(results) == 0:
                return []
            if isinstance(results[0], dict):
                return results

        if self.postprocessor is None:
            return None

        call_attempts = []
        pp = self.postprocessor
        if callable(pp):
            call_attempts.extend([
                lambda: pp(results),
                lambda: pp(results, conf_threshold=self.conf),
                lambda: pp(results, conf_threshold=self.conf, iou_threshold=self.iou),
            ])
        if hasattr(pp, "process"):
            call_attempts.extend([
                lambda: pp.process(results),
                lambda: pp.process(results, conf_threshold=self.conf),
                lambda: pp.process(results, conf_threshold=self.conf, iou_threshold=self.iou),
            ])
        if hasattr(pp, "postprocess"):
            call_attempts.extend([
                lambda: pp.postprocess(results),
                lambda: pp.postprocess(results, conf_threshold=self.conf),
                lambda: pp.postprocess(results, conf_threshold=self.conf, iou_threshold=self.iou),
            ])

        for fn in call_attempts:
            try:
                out = fn()
                if out is not None:
                    return out
            except Exception:
                continue
        return None

    def _normalize_common_output(self, processed, frame_shape, scale, pad_w, pad_h):
        if processed is None:
            return []

        h, w = frame_shape[:2]
        out = []

        if isinstance(processed, dict):
            processed = list(processed.values())
        elif not isinstance(processed, (list, tuple)):
            processed = [processed]

        for item in processed:
            if isinstance(item, dict):
                bbox = item.get("bbox") or item.get("box")
                if bbox is None:
                    # common.py tra ve dict dang: x1,y1,x2,y2,conf,cls_id,cls_name
                    keys_ok = all(k in item for k in ("x1", "y1", "x2", "y2"))
                    if keys_ok:
                        bbox = (item.get("x1"), item.get("y1"), item.get("x2"), item.get("y2"))

                conf = float(item.get("conf", item.get("score", 0.0)))
                cls_id = int(item.get("cls", item.get("class_id", item.get("cls_id", 0))))
                if bbox is None or len(bbox) < 4 or conf < self.conf:
                    continue
                x1, y1, x2, y2 = self._scale_one_box(bbox[:4], w, h, scale, pad_w, pad_h)
                cx = int((x1 + x2) * 0.5)
                cy = int((y1 + y2) * 0.5)
                out.append({"bbox": (x1, y1, x2, y2), "center": (cx, cy), "conf": conf, "cls": cls_id})
                continue

            arr = np.asarray(item).squeeze()
            if arr.ndim == 1 and arr.size >= 6:
                x1, y1, x2, y2, conf, cls_id = arr[:6]
                conf = float(conf)
                cls_id = int(cls_id)
                if conf < self.conf:
                    continue
                x1, y1, x2, y2 = self._scale_one_box((x1, y1, x2, y2), w, h, scale, pad_w, pad_h)
                cx = int((x1 + x2) * 0.5)
                cy = int((y1 + y2) * 0.5)
                out.append({"bbox": (x1, y1, x2, y2), "center": (cx, cy), "conf": conf, "cls": cls_id})

        out.sort(key=lambda d: d["conf"], reverse=True)
        return out

    def _scale_one_box(self, box, orig_w, orig_h, scale, pad_w, pad_h):
        try:
            scaled = scale_detections_to_original(
                [{"bbox": tuple(map(float, box)), "conf": 1.0, "cls": 0}],
                (orig_h, orig_w),
                scale,
                pad_w,
                pad_h,
            )
            if scaled and isinstance(scaled, (list, tuple)) and isinstance(scaled[0], dict):
                sb = scaled[0].get("bbox")
                if sb is not None and len(sb) >= 4:
                    x1, y1, x2, y2 = sb[:4]
                    x1 = int(max(0, min(orig_w - 1, round(x1))))
                    y1 = int(max(0, min(orig_h - 1, round(y1))))
                    x2 = int(max(0, min(orig_w - 1, round(x2))))
                    y2 = int(max(0, min(orig_h - 1, round(y2))))
                    if x2 > x1 and y2 > y1:
                        return x1, y1, x2, y2
        except Exception:
            pass

        if isinstance(scale, (tuple, list)) and len(scale) >= 2:
            sx = float(scale[0]) if float(scale[0]) != 0.0 else 1.0
            sy = float(scale[1]) if float(scale[1]) != 0.0 else 1.0
        else:
            sx = sy = float(scale) if float(scale) != 0.0 else 1.0

        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = (x1 - float(pad_w)) / sx
        y1 = (y1 - float(pad_h)) / sy
        x2 = (x2 - float(pad_w)) / sx
        y2 = (y2 - float(pad_h)) / sy

        x1 = int(max(0, min(orig_w - 1, round(x1))))
        y1 = int(max(0, min(orig_h - 1, round(y1))))
        x2 = int(max(0, min(orig_w - 1, round(x2))))
        y2 = int(max(0, min(orig_h - 1, round(y2))))
        return x1, y1, x2, y2

    def _raw_parse_with_nms(self, results, frame_shape, scale, pad_w, pad_h):
        raw = self._extract_prediction_tensor(results)
        if raw is None or raw.size == 0:
            return []

        h, w = frame_shape[:2]
        by_class = {}

        for row in raw:
            row = np.asarray(row).reshape(-1)
            if row.size < 5:
                continue

            x1, y1, x2, y2 = [float(v) for v in row[:4]]
            tail = row[4:]

            if tail.size >= 2:
                cls_id = int(np.argmax(tail))
                conf = float(tail[cls_id])
            else:
                cls_id = 0
                conf = float(tail[0])

            if conf < self.conf:
                continue

            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            if box_w <= 1.0 or box_h <= 1.0:
                continue

            by_class.setdefault(cls_id, {"boxes": [], "scores": [], "raw_boxes": []})
            by_class[cls_id]["boxes"].append([int(round(x1)), int(round(y1)), int(round(box_w)), int(round(box_h))])
            by_class[cls_id]["scores"].append(conf)
            by_class[cls_id]["raw_boxes"].append((x1, y1, x2, y2))

        out = []
        for cls_id, data in by_class.items():
            idxs = cv2.dnn.NMSBoxes(data["boxes"], data["scores"], self.conf, self.iou)
            if idxs is None or len(idxs) == 0:
                continue

            idxs = np.array(idxs).reshape(-1)
            for idx in idxs:
                x1, y1, x2, y2 = self._scale_one_box(data["raw_boxes"][int(idx)], w, h, scale, pad_w, pad_h)
                if x2 <= x1 or y2 <= y1:
                    continue
                cx = int((x1 + x2) * 0.5)
                cy = int((y1 + y2) * 0.5)
                out.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "conf": float(data["scores"][int(idx)]),
                    "cls": int(cls_id),
                })

        out.sort(key=lambda d: d["conf"], reverse=True)
        return out

    def detect_all(self, frame_bgr, class_filter=None):
        if frame_bgr is None:
            return []

        input_data, orig_size, scale, pad_w, pad_h = HailoPythonInferenceEngine.preprocess(
            frame_bgr, normalize=self.normalize
        )

        results, stats = self.engine.infer(
            input_data,
            verbose=self.verbose,
            save_output=True,
            conf_threshold=self.conf,
        )

        dets = self._normalize_common_output(
            self._run_common_postprocess(results),
            frame_bgr.shape,
            scale,
            pad_w,
            pad_h,
        )
        if not dets:
            dets = self._raw_parse_with_nms(results, frame_bgr.shape, scale, pad_w, pad_h)

        if class_filter is not None:
            class_ids = set(class_filter if isinstance(class_filter, (list, tuple, set)) else [class_filter])
            dets = [d for d in dets if int(d.get("cls", -1)) in class_ids]

        dets.sort(key=lambda d: d["conf"], reverse=True)
        return dets


# =========================
# H visual landing controller
# =========================
@dataclass
class LandResult:
    ok: bool
    reason: str




# =========================
# DroneController
# =========================
class DroneController:
    def __init__(
        self,
        connection_str="/dev/ttyACM0",
        takeoff_height=4.5,
        cam_index=0,
        enable_mission=False,
    ):
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

        # # Param set (best-effort)
        # try:
        #     self.vehicle.parameters["PLND_ENABLED"] = 1
        #     self.vehicle.parameters["PLND_TYPE"] = 1
        #     self.vehicle.parameters["PLND_EST_TYPE"] = 1
        #     self.vehicle.parameters["LAND_SPEED"] = 30
        # except Exception:
        #     pass

        # Camera
        self.camera = CameraStream(cam_index=cam_index)
        self.camera.start()

        # Detector (1 HEF model -> 4 classes: blue / h_marker / red / yellow)
        self.detector = HailoHEFDetector(
            MODEL_PATH,
            input_size=INPUT_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            normalize=NORMALIZE,
            verbose=VERBOSE,
            class_names=CLASS_NAMES,
        )

        # Tracker + class/tag maps (bo HSV, dung class tu model)
        self.tracker = CircleMemoryTracker(match_dist_px=250.0)
        self.cls_to_color_name = {
            CLASS_BLUE: "BLUE",
            CLASS_RED: "RED",
            CLASS_YELLOW: "YELLOW",
            CLASS_H_MARKER: "H_MARKER",
        }
        self.color_to_bgr = {
            "BLUE": (255, 0, 0),
            "RED": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "H_MARKER": (255, 255, 0),
            "UNK": (0, 255, 0),
        }

        # Detection buffers
        self._det_lock = threading.Lock()
        self._last_circle = []
        self._last_h = []

        # Circle memory: giu bbox/tam trong mot khoang ngan neu model hut frame
        self._circle_memory = CircleMemoryTracker(
            max_missing_frames=30,
            max_hold_sec=0.80,
            match_dist_px=140.0,
            max_tracks=8,
            conf_decay=0.92,
            predict_motion=True,
        )

        # Video window/record
        
        self._show_camera = True
        self._show_overlay = True
        self._window_name = "UAV Camera (Hailo blue/red/yellow + h_marker)"
        self._recording = False
        self._video_writer = None
        self._record_fps = 30.0
        self._record_dir = os.path.join(os.path.dirname(__file__), "records")
        os.makedirs(self._record_dir, exist_ok=True)
        self._pid_dir = os.path.join(self._record_dir, "pid_logs")
        os.makedirs(self._pid_dir, exist_ok=True)

        self._pid_log = []
        self._pid_log_lock = threading.Lock()
        self._pid_run_ts = time.strftime("%Y%m%d_%H%M%S")

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


        # Threads
        self._det_thread = threading.Thread(target=self._det_loop, daemon=True)
        self._det_thread.start()

        # self._mission_thread = None
        self._mission_thread = threading.Thread(target=self.mission_complete, daemon=True)
        self._mission_thread.start()
        log.info("[Mission] Started Mission")

        # if enable_mission:
        #     self._mission_thread = threading.Thread(target=self.mission_complete, daemon=True)
        #     self._mission_thread.start()
        # else:
        #     log.info("[Mission] Disabled at startup")


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
                all_dets = self.detector.detect_all(frame_bgr)
                circle_raw = [
                    d for d in all_dets
                    if int(d.get("cls", -1)) in (CLASS_BLUE, CLASS_RED, CLASS_YELLOW)
                ]
                h_dets = [d for d in all_dets if int(d.get("cls", -1)) == CLASS_H_MARKER]
            except Exception as e:
                log.error(f"[DETECT HAILO] {e}")
                circle_raw = []
                h_dets = []

            circle_dets = self._circle_memory.update(circle_raw, frame_bgr.shape)

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

            # Gan mau truc tiep tu class model (bo HSV)
            colored = self._colorize_dets(vis, circle_dets)

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
            
            mem_circle_count = 0
            best_h_conf = 0.0
            if self._show_overlay:
                # draw circle detections
                for d in colored:
                    x1, y1, x2, y2 = d["bbox"]
                    cx, cy = d["center"]
                    conf = d["conf"]
                    color_name = d["color_name"]
                    draw_bgr = d["draw_bgr"]
                    is_memory = bool(d.get("is_memory", False))
                    if is_memory:
                        mem_circle_count += 1
                        draw_bgr = (0, 165, 255)

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
                    label = f"{tag} {conf:.2f}"
                    if is_memory:
                        label += f" [MEM:{int(d.get('miss_count', 0))}]"

                    cv2.putText(
                        vis,
                        label,
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        draw_bgr,
                        2,
                    )

                # draw H detections
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
                    f"circle={len(circle_dets)} mem={mem_circle_count}  H={len(h_dets)}(best={best_h_conf:.2f})  step={self.mission_step}  overlay=ON",
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

                if key == ord("t"):
                    self._show_overlay = not self._show_overlay
                    log.info(f"[VIEW] show_overlay = {self._show_overlay}")

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

    def _append_pid_sample(self, phase, target, ex, ey, vx, vy):
        try:
            sample = {
                "t": time.time(),
                "phase": str(phase),
                "target": str(target),
                "ex": float(ex),
                "ey": float(ey),
                "vx": float(vx),
                "vy": float(vy),
            }
            with self._pid_log_lock:
                self._pid_log.append(sample)
        except Exception as e:
            log.debug(f"[PID_LOG] append failed: {e}")

    def _export_pid_plots(self):
        with self._pid_log_lock:
            rows = list(self._pid_log)

        if not rows:
            log.info("[PID_PLOT] no PID samples -> skip export")
            return None, None

        t0 = rows[0]["t"]
        ts = self._pid_run_ts
        csv_path = os.path.join(self._pid_dir, f"pid_trace_{ts}.csv")
        png_path = os.path.join(self._pid_dir, f"pid_plot_{ts}.png")

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t_rel_s", "phase", "target", "ex", "ey", "vx", "vy"])
                for r in rows:
                    writer.writerow([
                        f"{r['t'] - t0:.3f}",
                        r["phase"],
                        r["target"],
                        f"{r['ex']:.6f}",
                        f"{r['ey']:.6f}",
                        f"{r['vx']:.6f}",
                        f"{r['vy']:.6f}",
                    ])
        except Exception as e:
            log.error(f"[PID_PLOT] CSV export failed: {e}")
            csv_path = None

        try:
            t = [r["t"] - t0 for r in rows]
            ex = [r["ex"] for r in rows]
            ey = [r["ey"] for r in rows]
            vx = [r["vx"] for r in rows]
            vy = [r["vy"] for r in rows]

            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

            axes[0].plot(t, ex, label="ex (px)")
            axes[0].plot(t, ey, label="ey (px)")
            axes[0].axhline(0.0, linewidth=1.0, linestyle="--")
            axes[0].set_ylabel("Pixel error")
            axes[0].set_title("PID error trace")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(t, vx, label="vx (m/s)")
            axes[1].plot(t, vy, label="vy (m/s)")
            axes[1].axhline(0.0, linewidth=1.0, linestyle="--")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Velocity")
            axes[1].set_title("PID output trace")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            # Mark phase changes for easier reading
            prev_key = None
            for r in rows:
                key = (r["phase"], r["target"])
                if key != prev_key:
                    x = r["t"] - t0
                    for ax in axes:
                        ax.axvline(x, linewidth=0.8, linestyle=":", alpha=0.5)
                    axes[0].text(x, axes[0].get_ylim()[1], f"{r['phase']}:{r['target']}", fontsize=8, rotation=90, va="top", ha="right")
                    prev_key = key

            fig.tight_layout()
            fig.savefig(png_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            log.info(f"[PID_PLOT] saved plot -> {png_path}")
        except Exception as e:
            log.error(f"[PID_PLOT] PNG export failed: {e}")
            png_path = None

        return csv_path, png_path

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
        Bo HSV. Mau duoc lay truc tiep tu cls cua model:
            0 -> BLUE
            2 -> RED
            3 -> YELLOW

        Van giu ten ham cu de khong pha vo cau truc code.
        """
        colored = []
        if dets is None:
            return colored

        for det in dets:
            cls_id = int(det.get("cls", -1))
            color_name = self.cls_to_color_name.get(cls_id, "UNK")

            d2 = dict(det)
            d2["color_name"] = color_name
            d2["draw_bgr"] = self.color_to_bgr.get(color_name, self.color_to_bgr["UNK"])
            colored.append(d2)

        return colored

    # ---------- UPDATED: HOLD when image center enters bbox (but still PID to true center) ----------
    def pid_drive_to_center_H(
        self,
        ex_px: float,
        ey_px: float,
        tol_px: int = 5,
        max_vxy: float = 0.30,
        ema_alpha: float = 0.25,
        ema_state: dict = None,
        send_now: bool = False,
        vz: float = 0.0,
    ):
        """
        PID kéo tâm target về tâm ảnh.
        - ex_px > 0: target lệch sang phải
        - ey_px > 0: target lệch xuống dưới
        - BODY_NED:
            vx: tiến/lùi
            vy: trái/phải
            vz: xuống khi vz > 0
        """
        if ema_state is None:
            ema_state = {}

        if not ema_state.get("inited", False):
            ex_f = float(ex_px)
            ey_f = float(ey_px)
            ema_state["inited"] = True
        else:
            a = float(ema_alpha)
            ex_f = a * float(ex_px) + (1.0 - a) * float(ema_state.get("ex_f", 0.0))
            ey_f = a * float(ey_px) + (1.0 - a) * float(ema_state.get("ey_f", 0.0))

        ema_state["ex_f"] = ex_f
        ema_state["ey_f"] = ey_f

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

        log.info(f"[PID_CONTROL] target=H_MARKER ex={ex_f:.1f} ey={ey_f:.1f} vx={vx:.3f} vy={vy:.3f}")

        # deadband nhỏ để tránh rung
        if abs(ex_f) <= float(tol_px) and abs(ey_f) <= float(tol_px):
            vx = 0.0
            vy = 0.0

        self._append_pid_sample("land_h", "H_MARKER", ex_f, ey_f, vx, vy)

        if send_now:
            self.send_local_ned_velocity(vx, vy, float(vz))

        return ex_f, ey_f, vx, vy
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
        duration_s: float = None,
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

        duration_logged = False

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

            log.info(f"[PID_CONTROL] target={tagU} ex={ex_f:.1f} ey={ey_f:.1f} vx={vx:.3f} vy={vy:.3f}")
            # Tránh rung micro quanh điểm cân bằng
            if abs(ex_f) <= float(tol_px) and abs(ey_f) <= float(tol_px):
                vx, vy = 0.0, 0.0

            self._append_pid_sample("seek", tagU, ex_f, ey_f, vx, vy)
            self.send_local_ned_velocity(vx, vy, 0.0)
            return ex_f, ey_f, vx, vy
        
        # =========== Chọn màu và phân biết RED1, RED2, YELLOW_LEFT, YELLOW_RIGHT VÀ BLUE    ================
        while not self._stop_flag:
            now = time.time()
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
            # Duration_s
            duration_expired = (
                duration_s is not None and (now - start_t) >= float(duration_s)
            )

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
                if duration_expired:
                    if not duration_logged:
                        log.info(
                            f"[SEEK] duration_s={duration_s}s expired"
                            f"(tartget={target_tag}, dir={direction})"
                        )
                        duration_logged = True
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                else:                        
                    vx, vy = self._direction_to_vxvy(direction, search_speed)
                    self.send_local_ned_velocity(vx, vy, 0.0)

            else:
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
    
    def land_to_h(
        self,
        timeout_s: float = 90.0,
        conf_th: float = 0.8,
        tol_px: int = 10,
        hold_center_sec: float = 0.5,
        loop_hz: float = 20.0,
        descent_vz: float = 0.30,          # BODY_NED: vz > 0 là đi xuống
        min_vz: float = 0.08,
        stop_descend_err_px: float = 180.0,
        err_full_px: float = 140.0,
        max_vxy: float = 0.30,
        ema_alpha: float = 0.25,
        switch_to_land_alt_m: float = 0.5, # đúng yêu cầu của bạn
        lost_h_timeout_s: float = 2.0,
        lock_dist_px: float = 180.0,
    ):
        """
        Hạ cánh theo H_marker:
        - detect H từ self._last_h
        - PID kéo tâm drone về tâm H
        - đồng thời giảm độ cao
        - alt <= 0.5m => chuyển mode LAND

        Return:
            LandResult(ok=True/False, reason="...")
        """
        try:
            PID_X.reset()
            PID_Y.reset()
        except Exception:
            pass

        # đảm bảo đang ở GUIDED trước khi visual-servo
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
        dt = 1.0 / float(max(1e-6, loop_hz))
        log_t = 0.0

        ema_state = {"inited": False, "ex_f": 0.0, "ey_f": 0.0}
        hold_start = None
        h_lock = None
        lock_gate2 = float(lock_dist_px) * float(lock_dist_px)

        def dist2(p, q):
            dx = float(p[0]) - float(q[0])
            dy = float(p[1]) - float(q[1])
            return dx * dx + dy * dy

        while not self._stop_flag:
            now = time.time()

            # timeout mission
            if (now - t0) > float(timeout_s):
                try:
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass
                self._ui_clear_hold()
                return LandResult(False, f"timeout {timeout_s}s")

            frame_bgr = self.camera.get_frame()
            if frame_bgr is None:
                time.sleep(0.02)
                continue

            h_img, w_img = frame_bgr.shape[:2]
            icx, icy = w_img // 2, h_img // 2

            # altitude hiện tại
            try:
                alt = float(self.vehicle.location.global_relative_frame.alt)
            except Exception:
                alt = None

            # nếu đã xuống đủ thấp thì chuyển LAND
            if alt is not None and alt <= float(switch_to_land_alt_m):
                try:
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass

                try:
                    self.vehicle.mode = VehicleMode("LAND")
                except Exception:
                    pass

                self._ui_clear_hold()
                log.info(f"[H-LAND] alt={alt:.2f} -> switch LAND")
                return LandResult(True, f"switch LAND at alt={alt:.2f}m")

            # lấy H detections mới nhất
            h_dets = self._get_last_h_dets_copy()

            # lọc theo conf
            h_dets = [d for d in h_dets if float(d.get("conf", 0.0)) >= float(conf_th)]

            best = None
            if h_dets:
                # lock H gần target trước đó để tránh nhảy bbox
                if h_lock is not None:
                    nearest = min(h_dets, key=lambda d: dist2(d.get("center", (0, 0)), h_lock))
                    if dist2(nearest.get("center", (0, 0)), h_lock) <= lock_gate2:
                        best = nearest

                # nếu chưa lock được thì lấy bbox conf cao nhất
                if best is None:
                    best = max(h_dets, key=lambda d: float(d.get("conf", 0.0)))

            if best is None:
                # mất H
                self._ui_set_hold("H_MARKER", False, 0.0, hold_center_sec)

                if (now - last_seen_h) > float(lost_h_timeout_s):
                    try:
                        self.send_local_ned_velocity(0.0, 0.0, 0.0)
                    except Exception:
                        pass
                    self._ui_clear_hold()
                    return LandResult(False, f"lost H > {lost_h_timeout_s}s")

                # hover chờ thấy lại H
                try:
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass
                time.sleep(dt)
                continue

            # cập nhật lock
            h_lock = best.get("center", None)
            last_seen_h = now

            # sai số tâm H so với tâm ảnh
            tx, ty = best.get("center", (icx, icy))
            ex = float(tx - icx)
            ey = float(ty - icy)

            # PID ngang để kéo về tâm marker H
            ex_f, ey_f, vx, vy = self.pid_drive_to_center_H(
                ex_px=ex,
                ey_px=ey,
                tol_px=tol_px,
                max_vxy=max_vxy if (alt is None or alt > 1.2) else min(max_vxy, 0.18),
                ema_alpha=ema_alpha,
                ema_state=ema_state,
                send_now=False,
                vz=0.0,
            )

            # hold khi đã gần tâm
            if abs(ex_f) <= float(tol_px) and abs(ey_f) <= float(tol_px):
                if hold_start is None:
                    hold_start = now
                hold_elapsed = now - hold_start
                self._ui_set_hold("H_MARKER", True, hold_elapsed, hold_center_sec)
                centered = hold_elapsed >= float(hold_center_sec)
            else:
                hold_start = None
                self._ui_set_hold("H_MARKER", False, 0.0, hold_center_sec)
                centered = False

            # tốc độ hạ: lệch nhiều thì hạ chậm / dừng hạ
            err = float(np.hypot(ex_f, ey_f))
            if err >= float(stop_descend_err_px):
                vz = 0.0
            else:
                ratio = 1.0 - min(err / float(max(1e-6, err_full_px)), 1.0)
                vz = float(min_vz + (descent_vz - min_vz) * max(ratio, 0.0))

                # nếu đã center tốt thì cho xuống nhanh hơn một chút
                if centered:
                    vz = float(descent_vz)

                # gần đất thì xuống mềm hơn
                if alt is not None and alt <= 1.0:
                    vz = min(vz, 0.18)
                if alt is not None and alt <= 0.7:
                    vz = min(vz, 0.12)

            # gửi lệnh visual-servo: vừa chỉnh tâm vừa xuống
            try:
                self.send_local_ned_velocity(vx, vy, vz)
            except Exception:
                pass

            if (now - log_t) > 0.5:
                log_t = now
                log.info(
                    f"[LAND_H] alt={alt if alt is not None else -1:.2f} "
                    f"conf={float(best.get('conf', 0.0)):.2f} "
                    f"ex={ex_f:.1f} ey={ey_f:.1f} "
                    f"vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}"
                )

            time.sleep(dt)

        # stop_flag
        try:
            self.send_local_ned_velocity(0.0, 0.0, 0.0)
        except Exception:
            pass
        self._ui_clear_hold()
        return LandResult(False, "stopped")

    # ---------- Mission ----------
    def mission_complete(self):
        try:
            # self.arm_and_takeoff(self.takeoff_height)
            log.info("[Mission Started]")
            self.mission_step = 1

            while True:
                if self.mission_step == 1:
                    log.info("[Step 1] FORWARD -> RED1 enter bbox then hold 5s")
                    # ok = self.move_with_timer("forward", 2.5, 1.0)

                    ok = self.seek_target_and_center("RED1", "forward", 0.7, hold_s=8.0, timeout_s=30.0, duration_s= 10)
                    if not ok:
                        break
                    self.mission_step = 6

                elif self.mission_step == 2:
                    log.info("[Step 2] FORWARD -> RED2 enter bbox then hold 5s")
                    ok = self.seek_target_and_center("RED2", "forward", 1.0, hold_s=5.0, timeout_s=25.0, pre_delay_s=1.0, duration_s=2.5)
                    if not ok:
                        break
                    self.mission_step = 3

                elif self.mission_step == 3:
                    log.info("[Step 3] LEFT -> YELLOW_LEFT enter bbox then hold 5s")
                    ok = self.seek_target_and_center("YELLOW_LEFT", "left", 1.0, hold_s=5.0, timeout_s=30.0, pre_delay_s=1.0,duration_s=2.5)
                    if not ok:
                        break
                    self.mission_step = 4

                elif self.mission_step == 4:
                    log.info("[Step 4] RIGHT -> YELLOW_RIGHT enter bbox then hold 5s")
                    ok = self.seek_target_and_center("YELLOW_RIGHT", "right", 1.0, hold_s=5.0, timeout_s=30.0, pre_delay_s=1.0, duration_s=2.5)
                    if not ok:
                        break
                    self.mission_step = 5

                elif self.mission_step == 5:
                    log.info("[Step 5] Move: left 2.5s then forward 2.5s -> BLUE enter bbox hold 5s")
                    ok = self.move_with_timer("left", 2.5, 1.0)
                    time.sleep(1)
                    if not ok:
                        break
                    ok = self.seek_target_and_center("BLUE","forward", 1.0, hold_s=5.0, timeout_s=30.0, duration_s=2.5)
                    self.mission_step = 6

                elif self.mission_step == 6:
                    log.info("[Step 6] Backward 2.5s")
                    ok = self.move_with_timer("backward", 3.8, 1.0)
                    time.sleep(1)
                    if not ok:
                        break
                    self.mission_step = 7

                elif self.mission_step == 7:
                    log.info("[Step 7] LAND on H (visual-servo)")
                    self.vehicle.mode = VehicleMode("LAND")
                    # res = self.land_to_h(timeout_s=90)
                    # if not res.ok:
                    #     log.error(f"[H-LAND] failed: {res.reason} -> fallback LAND")
                    #     try:
                    #         self.vehicle.mode = VehicleMode("LAND")
                    #     except Exception:
                    #         pass
                    # else:
                    #     log.info(f"[H-LAND] ok: {res.reason}")
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

        while not self.vehicle.armed and not self._stop_flag:
            log.info(" Waiting for arming...")
            time.sleep(1)

        # self.vehicle.mode = VehicleMode("GUIDED")
        while self.vehicle.mode.name != "GUIDED" and not self._stop_flag:
            log.info(" Waiting for mode change ...")
            time.sleep(1)

        # self.vehicle.armed = True
        

        self.vehicle.simple_takeoff(targetHeight)
        while not self._stop_flag:
            alt = self.vehicle.location.global_relative_frame.alt
            log.info(f" Altitude: {alt:.1f} m")
            if alt >= targetHeight * 0.9:
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
            self._export_pid_plots()
        except Exception as e:
            log.error(f"[PID_PLOT] shutdown export failed: {e}")

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
    connection_str = "/dev/ttyACM0"
    takeoff_height = 4
    cam_index = 0
    enable_vehicle = False   # False = khong goi connect(), chi test camera + detect + viewer
    enable_mission = False   # False = khong chay mission thread, tranh logic bay khi dang test anh

    dc = None
    try:
        dc = DroneController(
            connection_str=connection_str,
            takeoff_height=takeoff_height,
            cam_index=cam_index,
            enable_mission=enable_mission,
        )
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