#!/usr/bin/env python3
# drone_control_yolov5_pid_centering.py
import time
import math
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
from rclpy.executors import SingleThreadedExecutor
import requests
import os

# YOLOv5
try:
    import torch
except Exception:
    torch = None

bridge = CvBridge()

time_to_wait = 0.1
time_last = 0

_latest_frame_lock = threading.Lock()
_latest_frame_jpeg = None
_image_streamer_node = None
_image_streamer_executor = None
_image_streamer_thread = None


# =========================
# ROS2 Image Streamer Node
# =========================
class ImageStreamerNode(Node):
    """
    ROS 2 node that subscribes to annotated image topic and updates latest JPEG.
    """
    def __init__(self, topic_name='/UAV/forward/image_new', queue_size=2):
        super().__init__('image_streamer_node')
        self.topic_name = topic_name
        self.sub = self.create_subscription(Image, self.topic_name, self.cb_image, queue_size)
        self.bridge = CvBridge()
        self.get_logger().info(f"ImageStreamerNode subscribing to {self.topic_name}")

    def cb_image(self, msg):
        global _latest_frame_jpeg, _latest_frame_lock
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            ret, jpeg = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                with _latest_frame_lock:
                    _latest_frame_jpeg = jpeg.tobytes()
        except Exception as e:
            self.get_logger().error(f"Error in cb_image: {e}")


def start_image_streamer(topic_name='/UAV/forward/image_new'):
    global _image_streamer_node, _image_streamer_executor, _image_streamer_thread
    if _image_streamer_node is not None:
        return
    if not rclpy.ok():
        try:
            rclpy.init(args=None)
        except Exception:
            pass
    _image_streamer_node = ImageStreamerNode(topic_name=topic_name)
    _image_streamer_executor = SingleThreadedExecutor()
    _image_streamer_executor.add_node(_image_streamer_node)

    def spin_loop():
        try:
            while rclpy.ok():
                _image_streamer_executor.spin_once(timeout_sec=0.1)
                time.sleep(0.01)
        except Exception:
            pass
        finally:
            try:
                _image_streamer_executor.remove_node(_image_streamer_node)
                _image_streamer_node.destroy_node()
            except Exception:
                pass

    _image_streamer_thread = threading.Thread(target=spin_loop, daemon=True)
    _image_streamer_thread.start()


def stop_image_streamer():
    global _image_streamer_node, _image_streamer_executor, _image_streamer_thread
    if _image_streamer_node is None:
        return
    try:
        _image_streamer_executor.shutdown()
    except Exception:
        pass
    _image_streamer_node = None
    _image_streamer_executor = None
    _image_streamer_thread = None


def get_lastest_frame():
    global _latest_frame_jpeg, _latest_frame_lock
    with _latest_frame_lock:
        return _latest_frame_jpeg


# =========================
# PID
# =========================
class PIDController:
    def __init__(self, Kp, Ki, Kd,
                 out_limit=0.7,          # m/s
                 i_limit=0.5,            # clamp integral
                 deadband=0.02,          # error deadband (normalized)
                 err_ema_alpha=0.6):     # low-pass error
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.out_limit = float(abs(out_limit))
        self.i_limit = float(abs(i_limit))
        self.deadband = float(abs(deadband))
        self.err_ema_alpha = float(err_ema_alpha)
        self.reset()

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.err_f = 0.0

    @staticmethod
    def _clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def update(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 1e-4:
            return 0.0

        # deadband
        if abs(error) < self.deadband:
            error = 0.0

        # low-pass filter
        a = self.err_ema_alpha
        self.err_f = a * self.err_f + (1.0 - a) * float(error)

        # derivative
        de = (self.err_f - self.last_error) / dt

        # integral with clamp
        self.integral += self.err_f * dt
        self.integral = self._clamp(self.integral, -self.i_limit, self.i_limit)

        out = (self.Kp * self.err_f) + (self.Ki * self.integral) + (self.Kd * de)
        out = self._clamp(out, -self.out_limit, self.out_limit)

        self.last_error = self.err_f
        self.last_time = now
        return out


# PID theo error_norm [-1..1] -> vận tốc m/s
PID_VX = PIDController(Kp=0.60, Ki=0.00, Kd=0.12, out_limit=0.70, i_limit=0.40, deadband=0.02, err_ema_alpha=0.6)
PID_VY = PIDController(Kp=0.60, Ki=0.00, Kd=0.12, out_limit=0.70, i_limit=0.40, deadband=0.02, err_ema_alpha=0.6)


def _clamp01(x):
    return max(-1.0, min(1.0, x))


def compute_center_errors(bbox_cx, bbox_cy, frame_cx, frame_cy, w, h):
    dx = float(bbox_cx - frame_cx)
    dy = float(bbox_cy - frame_cy)

    ex = dx / (0.5 * float(w))
    ey = dy / (0.5 * float(h))
    return _clamp01(ex), _clamp01(ey), dx, dy


def pid_centering_velocity(error_x, error_y):
    """
    Mapping (BODY_NED):
      vy (right)   <- +error_x
      vx (forward) <- -error_y
    Nếu thấy ngược, đảo dấu tại đây.
    """
    vx = PID_VX.update(-error_y)
    vy = PID_VY.update(+error_x)
    return vx, vy


# =========================
# DroneController
# =========================
class DroneController:
    def __init__(self, connection_str='tcp:127.0.0.1:5763', takeoff_height=3):
        self.connection_str = connection_str
        print("Connecting to vehicle on", connection_str)
        self.vehicle = connect(connection_str, wait_ready=True, timeout=120)
        self.takeoff_height = takeoff_height

        if not rclpy.ok():
            rclpy.init(args=None)

        self.flown_path = []

        # scanning ROS2
        self.ros_node = None
        self.executor = None
        self.scan_thread = None
        self.scan_running = False

        # ===== Tracking state (CENTERING MODE) =====
        self._trk_lock = threading.Lock()
        self._trk_active = False
        self._trk_reason = None
        self._trk_start_ts = 0.0
        self._trk_last_ts = 0.0
        self._trk_last_cmd = (0.0, 0.0, 0.0)  # vx, vy, vz
        self._trk_last_err = (0.0, 0.0)       # ex, ey

        self._trk_cmd_timeout_s = float(os.getenv("TRACK_CMD_TIMEOUT_S", "0.6"))
        self._trk_max_s = float(os.getenv("TRACK_MAX_SECONDS", "20.0"))

    # ========== Image stream ==========
    def start_image_stream(self, topic_name='/UAV/forward/image_new'):
        try:
            start_image_streamer(topic_name=topic_name)
            print("Started image streamer on topic", topic_name)
        except Exception as e:
            print("Failed to start image streamer:", e)

    def stop_image_streamer(self):
        try:
            stop_image_streamer()
            print("Stopped image streamer")
        except Exception as e:
            print("Failed to stop image streamer:", e)

    # ========== Server update ==========
    def send_aruco_marker_to_server(self, markers):
        """Send detected person zones to server (keeps old method name for compatibility)."""
        try:
            response = requests.post(
                'http://localhost:5000/update_person_zones',
                json={'zones': markers},
                timeout=2
            )
            if response.status_code == 200:
                print("Successfully sent person zones to server")
            else:
                print(f"Failed to send person zones, status code: {response.status_code}")
        except Exception as e:
            print(f"Error sending person zones to server: {e}")

    # ========== MAVLink ==========
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
            0.0,
            0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def set_speed(self, speed):
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            1,
            float(speed),
            -1, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        print(f"Set speed to {speed} m/s")

    # ========== Tracking API ==========
    def start_tracking(self, reason="person_in_water"):
        now = time.time()
        with self._trk_lock:
            if self._trk_active:
                return
            self._trk_active = True
            self._trk_reason = reason
            self._trk_start_ts = now
            self._trk_last_ts = now
            self._trk_last_cmd = (0.0, 0.0, 0.0)
            self._trk_last_err = (0.0, 0.0)
        # immediate stop
        try:
            self.send_local_ned_velocity(0, 0, 0)
        except Exception:
            pass
        print(f"[TRACK] START: reason={reason}")

    def update_tracking_cmd(self, vx, vy, vz, ex, ey):
        now = time.time()
        with self._trk_lock:
            if not self._trk_active:
                self._trk_active = True
                self._trk_reason = "person_in_water"
                self._trk_start_ts = now
            self._trk_last_ts = now
            self._trk_last_cmd = (float(vx), float(vy), float(vz))
            self._trk_last_err = (float(ex), float(ey))

    def stop_tracking(self, why="centered"):
        with self._trk_lock:
            was = self._trk_active
            self._trk_active = False
            self._trk_reason = None
            self._trk_start_ts = 0.0
            self._trk_last_ts = 0.0
            self._trk_last_cmd = (0.0, 0.0, 0.0)
            self._trk_last_err = (0.0, 0.0)
        if was:
            try:
                self.send_local_ned_velocity(0, 0, 0)
            except Exception:
                pass
            print(f"[TRACK] STOP: {why}")

    def _is_tracking(self):
        now = time.time()
        with self._trk_lock:
            if not self._trk_active:
                return False
            # fail-safe max time
            if self._trk_start_ts and (now - self._trk_start_ts) > self._trk_max_s:
                return False
            # if no new cmd recently -> treat as not tracking
            if self._trk_last_ts and (now - self._trk_last_ts) > self._trk_cmd_timeout_s:
                return False
            return True

    def _tracking_elapsed(self):
        with self._trk_lock:
            if not self._trk_active or not self._trk_start_ts:
                return 0.0
            return time.time() - self._trk_start_ts

    # ========== Flight primitives ==========
    @staticmethod
    def get_distance_meters(targetLocation, currentLocation):
        dLat = targetLocation.lat - currentLocation.lat
        dLon = targetLocation.lon - currentLocation.lon
        return math.sqrt((dLon * dLon) + (dLat * dLat)) * 1.113195e5

    def goto(self, targetLocation, tolerance=2.0, timeout=60, speed=0.7):
        """
        Go to a waypoint with TRACK-aware timeout/resume.
        When tracking is active: suspend motion to waypoint (by not reissuing goto),
        keep waiting until tracking ends, then reissue goto and continue.
        """
        if speed < 0.1 or speed > 5.0:
            print(f"Tốc độ {speed} m/s không hợp lệ, đặt về 0.7 m/s")
            speed = 0.7
        if not self.vehicle:
            return False

        dist0 = self.get_distance_meters(targetLocation, self.vehicle.location.global_relative_frame)
        self.set_speed(speed)
        self.vehicle.simple_goto(targetLocation, groundspeed=speed)

        start_time = time.time()
        start_dist = dist0

        # Do not count tracking time into timeout
        track_accum = 0.0
        track_start = None
        resumed_once = False

        while self.vehicle.mode.name == "GUIDED" and self.vehicle.armed:
            now = time.time()

            # effective elapsed excluding tracking time
            elapsed = now - start_time - track_accum
            if track_start is not None:
                elapsed -= (now - track_start)
            if elapsed > timeout:
                break

            # ===== TRACK handling =====
            if self._is_tracking():
                if track_start is None:
                    track_start = now
                    resumed_once = False
                    print(f"[TRACK] Holding for centering... elapsed={self._tracking_elapsed():.1f}s")
                # During tracking, do nothing here; DroneNode keeps sending velocities.
                time.sleep(0.05)
                continue
            else:
                # if was tracking -> resume goto
                if track_start is not None:
                    track_accum += (now - track_start)
                    track_start = None
                    print("[TRACK] Resume waypoint navigation")
                    try:
                        self.set_speed(speed)
                        self.vehicle.simple_goto(targetLocation, groundspeed=speed)
                        resumed_once = True
                    except Exception:
                        pass

            # normal navigation check
            current_pos = self.vehicle.location.global_relative_frame
            if current_pos.lat and current_pos.lon:
                self.flown_path.append([current_pos.lat, current_pos.lon])

            current_dist = self.get_distance_meters(targetLocation, current_pos)
            if current_dist < max(tolerance, start_dist * 0.01):
                print("Reached target waypoint")
                return True

            time.sleep(0.02)

        print("Timeout reaching waypoint, proceeding anyway")
        return False

    def arm_and_takeoff(self, targetHeight):
        while not self.vehicle.is_armable:
            print('Waiting for vehicle to become armable')
            time.sleep(1)
        self.vehicle.mode = VehicleMode('GUIDED')
        while self.vehicle.mode != 'GUIDED':
            print('Waiting for GUIDED...')
            time.sleep(1)
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print('Arming...')
            time.sleep(1)
        self.vehicle.simple_takeoff(targetHeight)
        while True:
            alt = self.vehicle.location.global_relative_frame.alt
            print('Altitude: %.2f' % (alt if alt else 0.0))
            if alt is not None and alt >= 0.95 * targetHeight:
                break
            time.sleep(1)
        print("Reached takeoff altitude")

    def land_and_wait(self, timeout=120):
        try:
            self.vehicle.mode = VehicleMode("LAND")
        except Exception as e:
            print("Failed to set LAND mode:", e)
            return False

        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if not self.vehicle.armed:
                    print("Landed (disarmed)")
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        print("land_and_wait timeout")
        return False

    # ========== YOLO scanning ==========
    def start_scanning(self):
        if self.scan_running:
            print("Scanning already running")
            return
        self.ros_node = DroneNode(self, mode='scan')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.ros_node)
        self.scan_running = True
        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()
        print("Started scanning for YOLOv5 person_in_water")

    def _scan_loop(self):
        while rclpy.ok() and self.scan_running and self.vehicle.armed:
            self.executor.spin_once(timeout_sec=0.1)

    def stop_scanning(self):
        if not self.scan_running:
            return
        self.scan_running = False
        if self.scan_thread:
            self.scan_thread.join(timeout=2.0)
        if self.executor and self.ros_node:
            try:
                self.executor.remove_node(self.ros_node)
                self.ros_node.destroy_node()
            except Exception:
                pass
        self.ros_node = None
        self.executor = None
        print("Stopped scanning")

    def fly_and_precision_land_with_waypoints(self, waypoints, loiter_alt=3):
        if not waypoints or len(waypoints) < 2:
            raise ValueError("Invalid waypoints")

        self.flown_path = []

        print("Arming and taking off")
        self.arm_and_takeoff(loiter_alt)
        time.sleep(1)

        # Start YOLO scanning during flight
        self.start_scanning()

        for i, wp in enumerate(waypoints[1:], start=1):
            wp_loc = LocationGlobalRelative(wp[0], wp[1], loiter_alt)
            print(f"Flying to waypoint {i}: {wp[0]}, {wp[1]}")
            self.goto(wp_loc)

        self.stop_scanning()

        print("Starting landing phase...")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.mode.name != "LAND":
            print("Waiting for LAND mode...")
            time.sleep(1)

        while self.vehicle.armed:
            print("Waiting for disarming...")
            time.sleep(1)

        print("Mission complete")
        return True


# =========================
# DroneNode for YOLOv5 + Centering
# =========================
class DroneNode(Node):
    def __init__(self, controller: DroneController, mode='scan'):
        node_name = f"drone_node_for_yolo_{int(time.time()*1000)}"
        super().__init__(node_name)

        self.controller = controller
        self.vehicle = controller.vehicle
        self.mode = mode

        # publish annotated image
        self.newimg_pub = self.create_publisher(Image, '/UAV/forward/image_new', 10)
        # subscribe raw camera
        self.subscription = self.create_subscription(Image, '/UAV/forward/image_raw', self.msg_receiver, 10)

        # send interval to server
        self.last_send_time = 0.0
        self.send_interval = float(os.getenv("PERSON_SEND_INTERVAL", "2.0"))

        # store zones (map)
        self.detected_marker = {}

        names_env = os.getenv("YOLOV5_TARGET_CLASS_NAMES", "person_in_water,swimming,drowning")
        self.target_labels = [s.strip().lower() for s in names_env.split(",") if s.strip()]

        self.conf_thres = float(os.getenv("YOLOV5_CONF", "0.35"))
        self.conf_thres_water = float(os.getenv("YOLOV5_CONF_WATER", "0.20"))

        self.zone_radius_m = float(os.getenv("PERSON_ZONE_RADIUS_M", "5"))
        self.scale_zone_with_alt = os.getenv("ZONE_SCALE_WITH_ALT", "0").strip() not in ("0", "false", "False")

        # water heuristic
        self.enable_water_heuristic = os.getenv("WATER_HEURISTIC", "1").strip() not in ("0", "false", "False")
        self.water_hsv_low = tuple(int(x) for x in os.getenv("WATER_HSV_LOW", "80,40,40").split(","))
        self.water_hsv_high = tuple(int(x) for x in os.getenv("WATER_HSV_HIGH", "140,255,255").split(","))
        self.water_ratio_thres = float(os.getenv("WATER_RATIO_THRES", "0.12"))

        # temporal confirm
        self.confirm_frames_person_in_water = int(os.getenv("CONFIRM_FRAMES_PERSON_IN_WATER", "2"))
        self.confirm_frames_drowning = int(os.getenv("CONFIRM_FRAMES_DROWNING", "3"))
        self.track_timeout_s = float(os.getenv("TRACK_TIMEOUT_S", "1.0"))
        self._tracks = {}

        # centering stop condition
        self.center_err_thres = float(os.getenv("CENTER_ERR_THRES", "0.05"))     # normalized
        self.center_stable_frames = int(os.getenv("CENTER_STABLE_FRAMES", "3"))
        self._stable_cnt = 0

        # lost handling
        self.lost_count = 0
        self.lost_reset_frames = int(os.getenv("LOST_RESET_FRAMES", "5"))

        self.yolo = self._load_yolov5()
        self.get_logger().info("DroneNode YOLOv5 ready")

    def _load_yolov5(self):
        if torch is None:
            raise RuntimeError("PyTorch (torch) is not installed in this environment.")
        repo = os.getenv("YOLOV5_REPO", os.path.expanduser("~/ai/yolov5"))
        weights = os.getenv("YOLOV5_WEIGHTS", "yolov5s.pt")
        try:
            if os.path.isdir(repo):
                model = torch.hub.load(repo, 'custom', path=weights, source='local', force_reload=False)
            else:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv5 model. repo={repo}, weights={weights}, err={e}")
        try:
            model.conf = self.conf_thres
        except Exception:
            pass
        return model

    def _preprocess(self, bgr):
        try:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge([l2, a, b])
            out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

            gamma = float(os.getenv("WATER_GAMMA", "1.1"))
            if abs(gamma - 1.0) > 1e-3:
                table = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255.0).astype(np.uint8)
                out = cv2.LUT(out, table)

            if os.getenv("WATER_SHARPEN", "1").strip() not in ("0", "false", "False"):
                blur = cv2.GaussianBlur(out, (0, 0), 1.0)
                out = cv2.addWeighted(out, 1.25, blur, -0.25, 0)
            return out
        except Exception:
            return bgr

    def _water_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        low = np.array(self.water_hsv_low, dtype=np.uint8)
        high = np.array(self.water_hsv_high, dtype=np.uint8)
        mask = cv2.inRange(hsv, low, high)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        return mask

    @staticmethod
    def _clamp_bbox(x1, y1, x2, y2, w, h):
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h - 1))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return x1, y1, x2, y2

    def _bbox_water_ratio(self, water_mask, x1, y1, x2, y2):
        try:
            h, w = water_mask.shape[:2]
            x1, y1, x2, y2 = self._clamp_bbox(x1, y1, x2, y2, w, h)
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            y_mid = int(y1 + 0.45 * bh)
            roi_lower = water_mask[y_mid:y2, x1:x2]
            s_lower = float(np.mean(roi_lower) / 255.0) if roi_lower.size else 0.0

            y_bot = int(y1 + 0.80 * bh)
            roi_bottom = water_mask[y_bot:y2, x1:x2]
            s_bottom = float(np.mean(roi_bottom) / 255.0) if roi_bottom.size else 0.0

            pad = int(0.15 * max(bw, bh))
            x1e = max(0, x1 - pad)
            x2e = min(w - 1, x2 + pad)
            y1e = max(0, y1 - pad)
            y2e = min(h - 1, y2 + pad)
            roi_expand = water_mask[y_mid:y2e, x1e:x2e]
            s_expand = float(np.mean(roi_expand) / 255.0) if roi_expand.size else 0.0

            cx = int((x1 + x2) * 0.5)
            cy = min(h - 1, y2 - 1)
            s_hit = 1.0 if water_mask[cy, cx] > 0 else 0.0

            return float(max(s_lower, s_bottom * 1.1, s_expand * 0.9, s_hit))
        except Exception:
            return 0.0

    def _map_raw_to_label(self, raw_name, conf, water_ratio):
        n = (raw_name or "").strip().lower()
        if "drown" in n:
            return "drowning"
        if "swim" in n:
            return "swimming"
        if n in ("person_in_water", "person-in-water", "personwater", "in_water", "in-water"):
            return "person_in_water"
        if n == "person" and self.enable_water_heuristic:
            if water_ratio >= self.water_ratio_thres and conf >= self.conf_thres_water:
                return "person_in_water"
        return None

    def _collect_best_by_label(self, detections_np, water_mask=None):
        if detections_np is None or len(detections_np) == 0:
            return {}
        best = {}
        for det in detections_np:
            x1, y1, x2, y2, conf, cls = det
            conf = float(conf)
            if conf < min(self.conf_thres, self.conf_thres_water):
                continue

            cls_i = int(cls)
            try:
                raw_name = self.yolo.names[cls_i]
            except Exception:
                raw_name = str(cls_i)

            water_ratio = 0.0
            rn = str(raw_name).strip().lower()
            if water_mask is not None and rn in ("person", "person_in_water", "person-in-water", "in_water", "in-water"):
                water_ratio = self._bbox_water_ratio(water_mask, x1, y1, x2, y2)

            label = self._map_raw_to_label(raw_name, conf, water_ratio)
            if label is None:
                if rn in ("drowning", "swimming", "person_in_water"):
                    label = rn
            if label is None:
                continue
            if self.target_labels and (label not in self.target_labels):
                continue

            prev = best.get(label)
            if prev is None or conf > prev[4]:
                best[label] = (int(x1), int(y1), int(x2), int(y2), conf, str(raw_name), float(water_ratio))
        return best

    def _update_track(self, label, bbox, conf, now):
        st = self._tracks.get(label)
        if st is None:
            st = {"count": 0, "last_ts": 0.0, "last_bbox": None, "ema_conf": 0.0}
            self._tracks[label] = st
        if st["last_ts"] and (now - st["last_ts"] > self.track_timeout_s):
            st["count"] = 0
            st["last_bbox"] = None
            st["ema_conf"] = 0.0
        st["last_ts"] = now
        st["last_bbox"] = bbox
        st["ema_conf"] = 0.7 * st["ema_conf"] + 0.3 * float(conf)
        st["count"] += 1
        return st

    def _should_publish(self, label, track_state):
        if label == "drowning":
            return track_state["count"] >= self.confirm_frames_drowning
        if label == "person_in_water":
            return track_state["count"] >= self.confirm_frames_person_in_water
        if label == "swimming":
            return True
        return True

    @staticmethod
    def _center_of_bbox(x1, y1, x2, y2):
        return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)

    @staticmethod
    def _draw_cross(img, cx, cy, color=(0, 255, 0), size=10, thickness=2):
        cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
        cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)

    def msg_receiver(self, message):
        global time_last
        if time.time() - time_last < time_to_wait:
            return
        time_last = time.time()

        try:
            cv_image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            h, w = cv_image.shape[:2]
            frame_cx, frame_cy = (w // 2, h // 2)

            # draw camera center
            self._draw_cross(cv_image, frame_cx, frame_cy, color=(255, 0, 0), size=15, thickness=2)
            cv2.putText(cv_image, f"CAM_C=({frame_cx},{frame_cy})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            img_infer = self._preprocess(cv_image)

            water_mask = None
            if self.enable_water_heuristic:
                try:
                    water_mask = self._water_mask(img_infer)
                except Exception:
                    water_mask = None

            results = self.yolo(img_infer)
            det = None
            try:
                det = results.xyxy[0].detach().cpu().numpy()
            except Exception:
                if hasattr(results, "pred") and len(results.pred) > 0:
                    det = results.pred[0].detach().cpu().numpy()

            best_by_label = self._collect_best_by_label(det, water_mask=water_mask)

            now = time.time()
            colors = {
                "drowning": (0, 0, 255),
                "person_in_water": (0, 0, 255),
                "swimming": (0, 165, 255),
            }

            # ===== if no person_in_water -> stop tracking safely =====
            if "person_in_water" not in best_by_label:
                self.lost_count += 1
                if self.lost_count >= self.lost_reset_frames:
                    self.controller.stop_tracking("lost_target")
                    PID_VX.reset()
                    PID_VY.reset()
                    self._stable_cnt = 0
                # publish annotated frame
                new_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                self.newimg_pub.publish(new_msg)
                return
            else:
                self.lost_count = 0

            publish_labels = []

            # process detections
            for label, info in best_by_label.items():
                x1, y1, x2, y2, conf, raw_name, water_ratio = info

                col = colors.get(label, (255, 255, 0))
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), col, 2)

                txt = f"{label} {conf:.2f}"
                if label == "person_in_water" and self.enable_water_heuristic and raw_name.strip().lower() == "person":
                    txt += f" water={water_ratio:.2f}"
                cv2.putText(cv_image, txt, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

                tr = self._update_track(label, (x1, y1, x2, y2), conf, now)
                if self._should_publish(label, tr):
                    publish_labels.append((label, conf))

                # ===== Centering logic for person_in_water =====
                if label == "person_in_water" and self._should_publish(label, tr):
                    bbox_cx, bbox_cy = self._center_of_bbox(x1, y1, x2, y2)
                    self._draw_cross(cv_image, bbox_cx, bbox_cy, color=(0, 0, 255), size=10, thickness=2)
                    cv2.line(cv_image, (frame_cx, frame_cy), (bbox_cx, bbox_cy), (255, 255, 255), 2)

                    error_x, error_y, dx, dy = compute_center_errors(bbox_cx, bbox_cy, frame_cx, frame_cy, w, h)
                    vx, vy = pid_centering_velocity(error_x, error_y)

                    # Start/Update tracking state (so goto() can suspend/resume correctly)
                    self.controller.start_tracking("person_in_water")
                    self.controller.update_tracking_cmd(vx, vy, 0.0, error_x, error_y)

                    # Apply velocity immediately (this is what actually centers)
                    self.controller.send_local_ned_velocity(vx, vy, 0.0)

                    # stop condition
                    if abs(error_x) < self.center_err_thres and abs(error_y) < self.center_err_thres:
                        self._stable_cnt += 1
                    else:
                        self._stable_cnt = 0

                    if self._stable_cnt >= self.center_stable_frames:
                        self.controller.stop_tracking("centered")
                        PID_VX.reset()
                        PID_VY.reset()
                        self._stable_cnt = 0

                    cv2.putText(cv_image,
                                f"ex={error_x:.2f} ey={error_y:.2f} vx={vx:.2f} vy={vy:.2f}",
                                (bbox_cx + 10, bbox_cy + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(cv_image,
                                f"dx={dx:.0f}px dy={dy:.0f}px",
                                (bbox_cx + 10, bbox_cy + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # ===== Send zones to server (still works; no waiting/hold) =====
            if publish_labels and (now - self.last_send_time >= self.send_interval):
                lat = self.vehicle.location.global_relative_frame.lat
                lon = self.vehicle.location.global_relative_frame.lon
                if lat is not None and lon is not None:
                    radius_m = self.zone_radius_m
                    if self.scale_zone_with_alt:
                        try:
                            alt = self.vehicle.location.global_relative_frame.alt
                            if alt is not None:
                                radius_m = max(self.zone_radius_m, min(60.0, float(alt) * 5.0))
                        except Exception:
                            pass

                    for (label, conf) in publish_labels:
                        self.detected_marker[label] = {
                            "lat": float(lat),
                            "lon": float(lon),
                            "radius_m": float(radius_m),
                            "conf": float(conf),
                            "ts": float(now),
                        }
                    try:
                        self.controller.send_aruco_marker_to_server(self.detected_marker)
                    except Exception as e:
                        print("Error sending zones:", e)

                self.last_send_time = now

            new_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.newimg_pub.publish(new_msg)

        except Exception as e:
            print("YOLOv5 processing error:", e)


# =========================
# Utility singleton
# =========================
_controller = None

def get_controller(connection_str='tcp:127.0.0.1:5763', takeoff_height=3):
    global _controller
    if _controller is None:
        _controller = DroneController(connection_str=connection_str, takeoff_height=takeoff_height)
    return _controller
