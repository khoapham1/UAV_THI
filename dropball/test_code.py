#!/usr/bin/env python3
# dropball.py
# ROS2 Humble + DroneKit + YOLOv5 (torch.hub local repo) circle detection
# - Fix NumPy/cv_bridge compatibility (handled by env: numpy<2)
# - Fix YOLOv5 .pt loading (do NOT use ultralytics.YOLO for YOLOv5 checkpoints)
# - Detect MULTIPLE circles per frame (returns list)
# - Clean shutdown (no publish after rclpy.shutdown)
# - Optional OpenCV viewer + recording toggle with key 'r'

import os
import time
import math
import threading
import warnings
from pathlib import Path

import numpy as np
import cv2
import torch

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from dronekit import connect, VehicleMode
from pymavlink import mavutil

# Project-local modules (must exist in same folder / PYTHONPATH)
from CenterTracker import CenterTracker
from hsv_color import HSVColorClassifier
from PID_control import PIDController

# -----------------------------
# Silence spammy FutureWarnings (torch amp deprecations in YOLOv5)
# -----------------------------
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
# Your trained YOLOv5 model
MODEL_PATH = "/home/legion/couseroot/ardupilot_gazebo/dropball/best.pt"
# Your local YOLOv5 repo (must contain hubconf.py)
YOLOV5_REPO_DIR = "/home/legion/ai/yolov5"

HOLD_TIME = 5

# PID (tune if needed)
PID_X = PIDController(0.00008, 0.0, 0.0, max_output=0.3)
PID_Y = PIDController(0.00008, 0.0, 0.0, max_output=0.3)


# ==============================
# ROS2 Image subscriber node
# ==============================
class ImageSubcriber(Node):
    def __init__(self, topic_name="/UAV/forward/image_raw", queue_size=2):
        super().__init__("image_subscriber")
        self.topic_name = topic_name
        self.sub = self.create_subscription(Image, self.topic_name, self.cb_image, queue_size)
        self.bridge = CvBridge()

        # Optional debug republish
        self.pub = self.create_publisher(Image, self.topic_name + "_debug", queue_size)

        self.last_frame_bgr = None
        self.get_logger().info(f"Subscribed to {self.topic_name}")

    def cb_image(self, msg: Image):
        global _latest_frame_jpeg, _latest_frame_lock

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cb_image decode error: {e}")
            return

        self.last_frame_bgr = frame_bgr

        # Cache latest frame as JPEG bytes (optional external streaming)
        try:
            ok, buf = cv2.imencode(".jpg", frame_bgr)
            if ok:
                with _latest_frame_lock:
                    _latest_frame_jpeg = buf.tobytes()
        except Exception:
            pass

        # Republish debug only if ROS context is still valid (avoid shutdown errors)
        if rclpy.ok():
            try:
                out_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
                out_msg.header = msg.header
                self.pub.publish(out_msg)
            except Exception:
                # During shutdown publisher context can become invalid; ignore.
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
                time.sleep(0.005)
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
# YOLOv5 detector (torch.hub local)
# ==============================
class YOLOv5TorchHubDetector:
    def __init__(
        self,
        weights_path: str,
        imgsz: int = 320,
        conf: float = 0.3,
        iou: float = 0.45,
        max_det: int = 10,
        device: str = "",
    ):
        self.weights_path = str(Path(weights_path).expanduser().resolve())
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)

        if device:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Validate YOLOv5 repo
        hubconf = os.path.join(YOLOV5_REPO_DIR, "hubconf.py")
        if not os.path.exists(hubconf):
            raise FileNotFoundError(f"Invalid YOLOv5 repo (hubconf.py missing): {YOLOV5_REPO_DIR}")

        # Load model
        self.model = torch.hub.load(YOLOV5_REPO_DIR, "custom", path=self.weights_path, source="local")
        self.model.to(self.device).eval()

        # Configure thresholds
        self.model.conf = self.conf
        self.model.iou = self.iou
        self.model.max_det = self.max_det
        self.model.classes = None  # or [0] if only one class id 0

    @torch.no_grad()
    def detect_all(self, frame_bgr):
        """
        Returns list of detections (possibly empty).
        Each det: {"bbox":(x1,y1,x2,y2), "center":(cx,cy), "conf":float, "cls":int, "area":int}
        Sorted by confidence desc.
        """
        results = self.model(frame_bgr, size=self.imgsz)
        det = results.xyxy[0]  # tensor Nx6 [x1,y1,x2,y2,conf,cls]
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
            area = int((x2 - x1) * (y2 - y1))
            out.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "conf": float(conf),
                    "cls": int(cls_id),
                    "area": area,
                }
            )

        out.sort(key=lambda d: d["conf"], reverse=True)
        return out


# ==============================
# Drone Controller Node
# ==============================
class DroneController(Node):
    def __init__(self, connection_str="tcp:127.0.0.1:5763", takeoff_height=2.5):
        super().__init__("drone_controller")

        self.connection_str = connection_str
        self.takeoff_height = float(takeoff_height)

        # Stop flag for clean shutdown
        self._stop_flag = False

        # Connect to vehicle
        self.get_logger().info(f"[Connecting to vehicle on]: {self.connection_str}")
        self.vehicle = connect(self.connection_str, wait_ready=True, timeout=60)

        # Precision landing params (optional)
        try:
            self.vehicle.parameters["PLND_ENABLED"] = 1
            self.vehicle.parameters["PLND_TYPE"] = 1
            self.vehicle.parameters["PLND_EST_TYPE"] = 1
            self.vehicle.parameters["LAND_SPEED"] = 30
        except Exception:
            pass

        # Mission state
        self.mission_step = 0
        self.hold_start_time = 0.0
        self.in_target_zone = False

        # Detector + helper modules
        # NOTE: conf=0.5 is high; if you miss second circle, reduce to 0.25~0.35
        self.detector = YOLOv5TorchHubDetector(
            MODEL_PATH, imgsz=320, conf=0.35, iou=0.45, max_det=10, device="cuda:0"
        )
        self.color_classifier = HSVColorClassifier(color_threshold=0.05, debug=True)
        self.tracker = CenterTracker(offset=(0, 0), circle_detect=250)

        # Start image subscriber
        self.start_image("/UAV/forward/image_raw")

        # Visualization + detection state
        self._show_camera = True
        self._window_name = "UAV Camera (circle detect)"
        self._last_dets = []
        self._last_det_lock = threading.Lock()

        # Video recording (toggle with 'r')
        self._recording = False
        self._video_writer = None
        self._record_fps = 30.0
        self._record_dir = os.path.join(os.path.dirname(__file__), "records")
        os.makedirs(self._record_dir, exist_ok=True)

        # Timers/threads
        self._viz_timer = self.create_timer(1.0 / 30.0, self._viz_timer_cb)
        self._det_thread = threading.Thread(target=self._det_loop, daemon=True)
        self._det_thread.start()

        # Start mission thread
        self._start_thread = threading.Thread(target=self.mission_complete, daemon=True)
        self._start_thread.start()

    # --------------------------
    # Detection background loop
    # --------------------------
    def _det_loop(self):
        det_hz = 15.0
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

            frame = getattr(_image_node, "last_frame_bgr", None)
            if frame is None:
                time.sleep(0.01)
                continue

            try:
                dets = self.detector.detect_all(frame)
            except Exception as e:
                try:
                    self.get_logger().error(f"[DETECT] {e}")
                except Exception:
                    pass
                dets = []

            with self._last_det_lock:
                self._last_dets = dets

    # --------------------------
    # Visualization timer callback
    # --------------------------
    def _viz_timer_cb(self):
        if self._stop_flag or not self._show_camera:
            return

        global _image_node
        if _image_node is None:
            return
        frame = getattr(_image_node, "last_frame_bgr", None)
        if frame is None:
            return

        vis = frame.copy()

        with self._last_det_lock:
            dets = list(self._last_dets) if self._last_dets else []

        # Draw all detections
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]
            conf = det["conf"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(
                vis,
                f"circle {conf:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Optionally compute error to image center using BEST detection
        if dets:
            best = dets[0]  # sorted by conf desc
            cx, cy = best["center"]
            h, w = vis.shape[:2]
            icx, icy = w // 2, h // 2
            cv2.circle(vis, (icx, icy), 4, (255, 0, 0), -1)
            ex, ey = cx - icx, cy - icy
            cv2.putText(
                vis,
                f"ex={ex} ey={ey} n={len(dets)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        try:
            if self._recording:
                cv2.putText(vis, "REC", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow(self._window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            # Toggle recording
            if key == ord("r"):
                if not self._recording:
                    self._start_recording(vis)
                else:
                    self._stop_recording()

            # Close viewer
            if key in (27, ord("q")):
                if self._recording:
                    self._stop_recording()
                self._show_camera = False
                cv2.destroyWindow(self._window_name)
                return

            # Write frame
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
    # Recording helpers
    # --------------------------
    def _start_recording(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self._record_dir, f"record_{ts}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, float(self._record_fps), (w, h))
        if not vw.isOpened():
            raise RuntimeError("VideoWriter open failed (try fourcc='XVID' + .avi).")

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
        try:
            self.get_logger().info("[REC] STOP")
        except Exception:
            pass

    # --------------------------
    # Mission logic
    # --------------------------
    def mission_complete(self):
        try:
            self.arm_and_takeoff(self.takeoff_height)
            self.get_logger().info("[Mission Started]")
            self.mission_step = 1

            while rclpy.ok() and not self._stop_flag:
                if self.mission_step == 1:
                    self.get_logger().info("[Step 1] Forward (detect RED)")
                    if not self.move_with_timer("forward", duration=8.3, speed=1):
                        break
                    time.sleep(3)
                    self.mission_step = 2

                elif self.mission_step == 2:
                    self.get_logger().info("[Step 2] Forward (detect RED)")
                    if not self.move_with_timer("forward", duration=2.5, speed=1):
                        break
                    time.sleep(5)
                    self.mission_step = 3

                elif self.mission_step == 3:
                    self.get_logger().info("[Step 3] Left (detect YELLOW)")
                    if not self.move_with_timer("left", duration=2.5, speed=1):
                        break
                    time.sleep(5)
                    self.mission_step = 4

                elif self.mission_step == 4:
                    self.get_logger().info("[Step 4] Right (detect YELLOW)")
                    if not self.move_with_timer("right", duration=5, speed=1):
                        break
                    time.sleep(5)
                    self.mission_step = 5

                elif self.mission_step == 5:
                    self.get_logger().info("[Step 5] Left + Forward (detect BLUE)")
                    if not self.move_with_timer("left", duration=2.5, speed=1):
                        break
                    time.sleep(5)
                    if not self.move_with_timer("forward", duration=2.5, speed=1):
                        break
                    time.sleep(5)
                    self.mission_step = 6

                elif self.mission_step == 6:
                    self.get_logger().info("[Step 6] Backward")
                    if not self.move_with_timer("backward", duration=13.5, speed=1):
                        break
                    time.sleep(1)
                    self.mission_step = 7

                elif self.mission_step == 7:
                    self.get_logger().info("[Step 7] Landing...")
                    self.land()
                    self.mission_step = 8

                elif self.mission_step == 8:
                    self.get_logger().info("[STOP]")
                    self.destroy_node()
                    break

                time.sleep(0.2)

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

    def stop_image(self):
        try:
            stop_image_subscriber()
            self.get_logger().info("[Image Subscriber Stopped]")
        except Exception as e:
            self.get_logger().error(f"Error stopping image subscriber: {e}")

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
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0,
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def arm_and_takeoff(self, targetHeight):
        self.get_logger().info("Basic pre-arm checks...")
        while not self.vehicle.is_armable and rclpy.ok() and not self._stop_flag:
            self.get_logger().info(" Waiting for vehicle to initialise...")
            time.sleep(1)

        self.get_logger().info("Arming motors...")
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

    def land(self):
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.mode.name != "LAND" and rclpy.ok() and not self._stop_flag:
            self.get_logger().info(" Waiting for mode change ...")
            time.sleep(1)
        self.get_logger().info("Landing completed.")

    def move_with_timer(self, direction, duration, speed=0.5):
        vx, vy = 0.0, 0.0
        d = (direction or "").lower()
        if d == "forward":
            vx = speed
        elif d == "backward":
            vx = -speed
        elif d == "left":
            vy = -speed
        elif d == "right":
            vy = speed
        else:
            self.get_logger().error(f"Unknown direction: {direction}")
            return False

        self.get_logger().info(f"[MOVE] {direction} for {duration}s at {speed} m/s")
        start_time = time.time()
        while (time.time() - start_time) < duration and rclpy.ok() and not self._stop_flag:
            self.send_local_ned_velocity(vx, vy, vz=0.0)
            time.sleep(0.1)

        self.send_local_ned_velocity(0.0, 0.0, 0.0)
        self.get_logger().info(f"[MOVE] Completed {direction}.")
        return True

    def control_drone_to_center(self, ex, ey):
        # Keep same behavior as your previous logic
        if abs(ex) < 5 and abs(ey) < 5:
            self.send_local_ned_velocity(0.0, 0.0, 0.0)
            return

        vx = PID_Y.update(-ey)
        vy = PID_X.update(ex)

        vx = max(min(vx, 0.3), -0.3)
        vy = max(min(vy, 0.3), -0.3)

        self.get_logger().info(f"[PID] ex={ex:.1f} ey={ey:.1f} => vx={vx:.2f} vy={vy:.2f}")
        self.send_local_ned_velocity(vx, vy, 0.0)

    # --------------------------
    # Clean destroy
    # --------------------------
    def destroy_node(self):
        self._stop_flag = True

        try:
            if hasattr(self, "_viz_timer") and self._viz_timer is not None:
                self._viz_timer.cancel()
        except Exception:
            pass

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
            if getattr(self, "vehicle", None) is not None:
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