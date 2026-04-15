

import os
import time
import threading
import warnings
import logging
import serial
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import csv
import math
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from dronekit import connect, VehicleMode
from pymavlink import mavutil
from drop import BallDropper
from PID_control import PIDController, LowPassFilter
from Kalma_filter import KalmanCenter2D
from Halio_detect import HailoHEFDetector
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
MODEL_PATH = "/home/pi/UAV/models/04042026/best.hef"
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

#====================
# Aim point for 5 ball 
#====================
BALL_AIM_POINTS = {
    1: (350, 210),
    2: (330, 230),
    3: (310, 250),
    4: (240, 220),
    5: (318, 160),
}

TARGET_TO_BALL_ID = {
    "RED1": 1,
    "RED2": 2,
    "YELLOW_LEFT": 3,
    "YELLOW_RIGHT": 4,
    "BLUE": 5,
}

CLASS_BLUE = 0
CLASS_H_MARKER = 1
CLASS_RED = 2
CLASS_YELLOW = 3

DROP_UART_PORT = "/dev/ttyUSB0"
DROP_UART_BAUDRATE = 9600
DROP_TRIGGER_PERCENT = 90.0

# =========================
# Drop condition by line L
# Sau khi dat DROP_TRIGGER_PERCENT, chi drop khi do dai duong L
# noi tu diem ngam aim -> tam circle nho hon nguong cho phep.
# Luu y: can calibrate lai DROP_LINE_PX_PER_CM theo camera thuc te.
# Vi du hien tai dat 10 px ~ 1 cm de de tune tren field.
# =========================
DROP_LINE_MAX_CM = 5.0
DROP_LINE_PX_PER_CM = 10.0
DROP_LINE_MAX_PX = DROP_LINE_MAX_CM * DROP_LINE_PX_PER_CM
DRAW_DROP_LINE_L = True

log.info(f"[MODEL] HEF = {MODEL_PATH}")

# =========================
# PID controllers (tach rieng theo tung mau)
# =========================
# Giu nguyen flow code cu, nhung moi mau se co cap PID_X / PID_Y rieng:
#   - RED    : dung cho RED, RED1, RED2
#   - YELLOW : dung cho YELLOW, YELLOW_LEFT, YELLOW_RIGHT
#   - BLUE   : dung cho BLUE
#   - H_MARKER: giu rieng de khong anh huong visual landing


# RED: B1-B4 theo yêu cầu
# - Ki = 0
# - Kp giảm nhẹ để vào tâm mượt hơn
# - Kd_X rất nhỏ, Kd_Y nhỏ nhưng mạnh hơn X để dập dội trên trục ey -> vx
# =========================
# PID controllers (tach rieng theo tung target)
# 5 target: RED1, RED2, YELLOW_LEFT, YELLOW_RIGHT, BLUE
# + 1 target H_MARKER
# =========================

# RED1: vao tam mem hon, uu tien giam overshoot luc ball 1
PID_RED1_X = PIDController(0.0035, 0.0,  0.00000015, max_output=0.08, integral_limit=300, derivative_filter_tau=0.06, derivative_limit=1800)
PID_RED1_Y = PIDController(0.004, 0.0, 0.00000015, max_output=0.1, integral_limit=300, derivative_filter_tau=0.06, derivative_limit=1800)

PID_RED2_X = PIDController(0.0035, 0.0, 0.0000001, max_output=0.08, integral_limit=300, derivative_filter_tau=0.06, derivative_limit=1800)
PID_RED2_Y = PIDController(0.0035, 0.0,  0.0000002, max_output=0.1, integral_limit=300, derivative_filter_tau=0.06, derivative_limit=1800)

PID_YELLOW_LEFT_X = PIDController(0.004, 0.0,  0.0000001, max_output=0.06, integral_limit=220, derivative_filter_tau=0.06, derivative_limit=2200)
PID_YELLOW_LEFT_Y = PIDController(0.003, 0.0, 0.0000001, max_output=0.08, integral_limit=220, derivative_filter_tau=0.06, derivative_limit=2300)

PID_YELLOW_RIGHT_X = PIDController(0.003, 0.0,  0.0000001, max_output=0.05, integral_limit=220, derivative_filter_tau=0.06, derivative_limit=2300)
PID_YELLOW_RIGHT_Y = PIDController(0.003, 0.0, 0.0000001, max_output=0.05, integral_limit=220, derivative_filter_tau=0.06, derivative_limit=2300)

PID_BLUE_X   = PIDController(0.0041, 0.0,  0.0000001, max_output=0.05, integral_limit=300, derivative_filter_tau=0.05, derivative_limit=2450)
PID_BLUE_Y   = PIDController(0.0038,0.0, 0.00000015, max_output=0.05, integral_limit=300, derivative_filter_tau=0.05, derivative_limit=2500)

# =========================
# GA tuner bindings / config
# Chỉ thêm hậu xử lý sau mission: đọc PID log -> nhận dạng mô hình lỗi -> GA tìm gain tốt hơn
# Không làm đổi flow mission hiện tại
# =========================
PID_RUNTIME_BINDINGS = {
    "PID_RED1_X": PID_RED1_X,
    "PID_RED1_Y": PID_RED1_Y,
    "PID_RED2_X": PID_RED2_X,
    "PID_RED2_Y": PID_RED2_Y,
    "PID_YELLOW_LEFT_X": PID_YELLOW_LEFT_X,
    "PID_YELLOW_LEFT_Y": PID_YELLOW_LEFT_Y,
    "PID_YELLOW_RIGHT_X": PID_YELLOW_RIGHT_X,
    "PID_YELLOW_RIGHT_Y": PID_YELLOW_RIGHT_Y,
    "PID_BLUE_X": PID_BLUE_X,
    "PID_BLUE_Y": PID_BLUE_Y,
}

PID_GA_TARGET_AXIS = {
    "PID_RED1_X": {"target": "RED1", "axis": "X"},
    "PID_RED1_Y": {"target": "RED1", "axis": "Y"},
    "PID_RED2_X": {"target": "RED2", "axis": "X"},
    "PID_RED2_Y": {"target": "RED2", "axis": "Y"},
    "PID_YELLOW_LEFT_X": {"target": "YELLOW_LEFT", "axis": "X"},
    "PID_YELLOW_LEFT_Y": {"target": "YELLOW_LEFT", "axis": "Y"},
    "PID_YELLOW_RIGHT_X": {"target": "YELLOW_RIGHT", "axis": "X"},
    "PID_YELLOW_RIGHT_Y": {"target": "YELLOW_RIGHT", "axis": "Y"},
    "PID_BLUE_X": {"target": "BLUE", "axis": "X"},
    "PID_BLUE_Y": {"target": "BLUE", "axis": "Y"},
}

GA_PID_PRINT_ORDER = [
    "PID_RED1_X",
    "PID_RED1_Y",
    "PID_RED2_X",
    "PID_RED2_Y",
    "PID_YELLOW_LEFT_X",
    "PID_YELLOW_LEFT_Y",
    "PID_YELLOW_RIGHT_X",
    "PID_YELLOW_RIGHT_Y",
    "PID_BLUE_X",
    "PID_BLUE_Y",
]

GA_ENABLE_POST_TUNING = True
GA_MIN_SAMPLES_PER_AXIS = 12
GA_POP_SIZE = 28
GA_GENERATIONS = 32
GA_ELITE = 6
GA_MUTATION_RATE = 0.28
GA_CROSSOVER_RATE = 0.82
GA_RANDOM_SEED = 42


# H marker: giu rieng nhu logic land hien tai
PID_H_X = PIDController(0.01, 0.0, 0.0, max_output=0.1, integral_limit=300, derivative_filter_tau=0.05, derivative_limit=1800)
PID_H_Y = PIDController(0.01, 0.0, 0.0, max_output=0.1, integral_limit=300, derivative_filter_tau=0.05, derivative_limit=1800)
# =========================
# Low-pass filters for pixel error (ex, ey)
# Bản mới dùng tau + dt để khi FPS thay đổi thì độ lọc vẫn ổn định.
LPF_TAU_X = 0.12
LPF_TAU_Y = 0.12

# soft brake + settle zone + hysteresis
BRAKE_ZONE_PX = 40.0
CENTER_TOL_ENTER_PX = 10.0
CENTER_TOL_EXIT_PX = 20.0
CENTER_STOP_PX = 3.0
CENTER_SETTLE_FRAMES = 4
CENTER_FINE_SCALE = 0.45
CENTER_MIN_CMD = 0.012

# schedule theo err_norm
SLOW_ZONE_1_PX = 80.0
SLOW_ZONE_2_PX = 30.0
SLOW_ZONE_1_KP_SCALE = 0.78
SLOW_ZONE_1_OUT_SCALE = 0.58
SLOW_ZONE_2_KP_SCALE = 0.55
SLOW_ZONE_2_OUT_SCALE = 0.30

# measurement stabilization
NEAR_CENTER_ERR_PX = 90.0
NEAR_CENTER_JUMP_REJECT_PX = 15.0
CENTER_EMA_ALPHA_FAR = 0.35
CENTER_EMA_ALPHA_NEAR = 0.18

# kalman filter for bbox center
KF_PROCESS_ACCEL_VAR = 120.0 
# du doan nếu bbox center nhảy mạnh vì rung drone: để vừa phải hoặc hơi thấp
#nếu target thật di chuyển nhanh trên ảnh: tăng lên chút

KF_MEASUREMENT_VAR = 49.0 # correct
#
KF_INIT_POS_VAR = 25.0
KF_INIT_VEL_VAR = 400.0
KF_NEAR_CENTER_R_SCALE = 4.0
KF_LOST_RESET_SEC = 0.60


# =========================
# GA hậu xử lý PID từ log mission
# =========================
class PIDLogGeneticTuner:
    def __init__(self, output_dir, run_ts):
        self.output_dir = output_dir
        self.run_ts = str(run_ts)
        self.rng = random.Random(GA_RANDOM_SEED)

    def _safe_float(self, v, default=0.0):
        try:
            return float(v)
        except Exception:
            return float(default)

    def _runtime_pid_spec(self, pid_name):
        pid_obj = PID_RUNTIME_BINDINGS[pid_name]
        kp, ki, kd, max_output = pid_obj.get_gains()
        return {
            "kp": float(kp),
            "ki": float(ki),
            "kd": float(kd),
            "max_output": float(getattr(pid_obj, "max_output", max_output)),
            "integral_limit": float(getattr(pid_obj, "integral_limit", 300.0)),
            "derivative_filter_tau": float(getattr(pid_obj, "derivative_filter_tau", 0.06)),
            "derivative_limit": float(getattr(pid_obj, "derivative_limit", 2000.0)),
        }

    def _load_rows(self, csv_path):
        rows = []
        if not csv_path or not os.path.exists(csv_path):
            return rows
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append({
                        "t": self._safe_float(r.get("t_rel_s", 0.0)),
                        "phase": str(r.get("phase", "")),
                        "target": str(r.get("target", "")).upper(),
                        "ex": self._safe_float(r.get("ex", 0.0)),
                        "ey": self._safe_float(r.get("ey", 0.0)),
                        "vx": self._safe_float(r.get("vx", 0.0)),
                        "vy": self._safe_float(r.get("vy", 0.0)),
                        "err_norm": self._safe_float(r.get("err_norm", 0.0)),
                        "note": str(r.get("note", "")),
                    })
        except Exception as e:
            log.error(f"[GA_PID] load csv failed: {e}")
        return rows

    def _extract_axis_series(self, rows, target_name, axis_name):
        target_rows = [r for r in rows if r.get("phase") == "seek" and str(r.get("target", "")).upper() == str(target_name).upper()]
        target_rows.sort(key=lambda r: float(r.get("t", 0.0)))
        if len(target_rows) < max(3, int(GA_MIN_SAMPLES_PER_AXIS)):
            return None

        series = []
        last_t = None
        for r in target_rows:
            t = float(r.get("t", 0.0))
            if last_t is None:
                dt = 1.0 / 20.0
            else:
                dt = max(1e-3, min(0.25, t - last_t))
            last_t = t

            if axis_name == "X":
                error = float(r.get("ex", 0.0))
                control = float(r.get("vy", 0.0))
            else:
                error = float(-r.get("ey", 0.0))
                control = float(r.get("vx", 0.0))

            series.append({
                "t": t,
                "dt": dt,
                "e": error,
                "u": control,
            })

        if len(series) < max(3, int(GA_MIN_SAMPLES_PER_AXIS)):
            return None
        return series

    def _fit_axis_model(self, series):
        if series is None or len(series) < 3:
            return None

        x_rows = []
        y_vals = []
        for i in range(len(series) - 1):
            e_k = float(series[i]["e"])
            u_k = float(series[i]["u"])
            dt_k = max(1e-3, float(series[i]["dt"]))
            e_n = float(series[i + 1]["e"])
            x_rows.append([e_k, u_k * dt_k, 1.0])
            y_vals.append(e_n)

        if len(x_rows) < 3:
            return None

        try:
            X = np.asarray(x_rows, dtype=np.float64)
            y = np.asarray(y_vals, dtype=np.float64)
            theta, *_ = np.linalg.lstsq(X, y, rcond=None)
            A, B, C = [float(v) for v in theta]
        except Exception as e:
            log.error(f"[GA_PID] fit model failed: {e}")
            return None

        if not np.isfinite(A):
            A = 1.0
        if not np.isfinite(B):
            B = 0.0
        if not np.isfinite(C):
            C = 0.0

        A = max(-1.25, min(1.25, A))
        B = max(-25.0, min(25.0, B))
        C = max(-40.0, min(40.0, C))
        return {"A": A, "B": B, "C": C}

    def _make_bounds(self, base_spec):
        kp0 = max(1e-7, float(base_spec["kp"]))
        kd0 = max(0.0, float(base_spec["kd"]))

        kp_lo = max(1e-5, kp0 * 0.35)
        kp_hi = max(kp_lo * 1.25, kp0 * 3.20)

        kd_floor = 0.0 if kd0 <= 1e-10 else kd0 * 0.10
        kd_lo = max(0.0, kd_floor)
        kd_hi = max(5e-8, kd0 * 8.0, kd_lo + 8e-8)

        ki0 = max(0.0, float(base_spec["ki"]))
        if ki0 <= 1e-12:
            ki_lo, ki_hi = 0.0, 0.0
        else:
            ki_lo = max(0.0, ki0 * 0.25)
            ki_hi = max(ki_lo * 1.25, ki0 * 3.0)

        return {
            "kp": (kp_lo, kp_hi),
            "ki": (ki_lo, ki_hi),
            "kd": (kd_lo, kd_hi),
        }

    def _rand_uniform(self, lo, hi):
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            return lo
        return self.rng.uniform(lo, hi)

    def _random_candidate(self, bounds, base_spec):
        return {
            "kp": self._rand_uniform(*bounds["kp"]),
            "ki": float(base_spec["ki"]) if bounds["ki"][0] == bounds["ki"][1] else self._rand_uniform(*bounds["ki"]),
            "kd": self._rand_uniform(*bounds["kd"]),
        }

    def _mutate_candidate(self, cand, bounds, base_spec):
        child = dict(cand)
        for key in ("kp", "kd"):
            if self.rng.random() < float(GA_MUTATION_RATE):
                lo, hi = bounds[key]
                center = float(child[key])
                span = max(1e-12, hi - lo)
                sigma = 0.18 * span
                mutated = center + self.rng.gauss(0.0, sigma)
                child[key] = max(lo, min(hi, mutated))

        if bounds["ki"][0] == bounds["ki"][1]:
            child["ki"] = float(base_spec["ki"])
        elif self.rng.random() < float(GA_MUTATION_RATE):
            lo, hi = bounds["ki"]
            span = max(1e-12, hi - lo)
            sigma = 0.18 * span
            mutated = float(child.get("ki", base_spec["ki"])) + self.rng.gauss(0.0, sigma)
            child["ki"] = max(lo, min(hi, mutated))

        return child

    def _crossover(self, a, b, bounds, base_spec):
        if self.rng.random() > float(GA_CROSSOVER_RATE):
            return dict(a)
        child = {}
        for key in ("kp", "ki", "kd"):
            av = float(a.get(key, base_spec[key]))
            bv = float(b.get(key, base_spec[key]))
            alpha = self.rng.random()
            v = alpha * av + (1.0 - alpha) * bv
            lo, hi = bounds[key]
            child[key] = max(lo, min(hi, v)) if hi > lo else float(base_spec[key])
        return child

    def _simulate_candidate(self, candidate, axis_spec, model, series):
        if model is None or series is None or len(series) < 3:
            return float("inf")

        A = float(model["A"]); B = float(model["B"]); C = float(model["C"])
        kp = float(candidate["kp"])
        ki = float(candidate.get("ki", axis_spec["ki"]))
        kd = float(candidate["kd"])
        max_output = float(axis_spec["max_output"])
        derivative_filter_tau = max(1e-6, float(axis_spec["derivative_filter_tau"]))
        derivative_limit = float(axis_spec["derivative_limit"])
        integral_limit = max(1e-6, float(axis_spec["integral_limit"]))

        e = float(series[0]["e"])
        prev_e = e
        integ = 0.0
        d_f = 0.0
        total = 0.0

        for i in range(len(series) - 1):
            dt = max(1e-3, float(series[i]["dt"]))
            dedt = (e - prev_e) / dt
            alpha = dt / (derivative_filter_tau + dt)
            d_f = (1.0 - alpha) * d_f + alpha * dedt
            d_f = max(-derivative_limit, min(derivative_limit, d_f))

            integ += e * dt
            integ = max(-integral_limit, min(integral_limit, integ))

            u = kp * e + ki * integ + kd * d_f
            u = max(-max_output, min(max_output, u))

            e_next = A * e + B * (u * dt) + C
            if not np.isfinite(e_next):
                return float("inf")

            total += (
                1.25 * abs(e_next)
                + 0.006 * (e_next * e_next)
                + 4.0 * abs(u)
                + 0.08 * abs(e_next - e)
            )

            prev_e = e
            e = float(e_next)

        final_err = abs(e)
        total += 1.5 * final_err
        return float(total)

    def _score_logged_baseline(self, series):
        if series is None or len(series) < 3:
            return float("inf")
        total = 0.0
        for i in range(len(series) - 1):
            e = float(series[i + 1]["e"])
            u = float(series[i]["u"])
            e_prev = float(series[i]["e"])
            total += 1.25 * abs(e) + 0.006 * (e * e) + 4.0 * abs(u) + 0.08 * abs(e - e_prev)
        total += 1.5 * abs(float(series[-1]["e"]))
        return float(total)

    def _format_pid_line(self, pid_name, spec):
        return (
            f"{pid_name} = PIDController("
            f"{spec['kp']:.9f}, {spec['ki']:.9f}, {spec['kd']:.9f}, "
            f"max_output={spec['max_output']:.3f}, integral_limit={int(round(spec['integral_limit']))}, "
            f"derivative_filter_tau={spec['derivative_filter_tau']:.3f}, derivative_limit={int(round(spec['derivative_limit']))})"
        )

    def tune_from_csv(self, csv_path):
        rows = self._load_rows(csv_path)
        if not rows:
            log.warning("[GA_PID] no csv rows -> skip")
            return None

        summary = {
            "csv": csv_path,
            "txt": None,
            "json": None,
            "result_lines": [],
            "per_axis": {},
        }

        for pid_name in GA_PID_PRINT_ORDER:
            axis_info = PID_GA_TARGET_AXIS[pid_name]
            target_name = axis_info["target"]
            axis_name = axis_info["axis"]
            axis_spec = self._runtime_pid_spec(pid_name)
            series = self._extract_axis_series(rows, target_name, axis_name)

            if series is None:
                summary["per_axis"][pid_name] = {
                    "status": "skip_not_enough_samples",
                    "samples": 0,
                    "base": axis_spec,
                    "best": axis_spec,
                }
                summary["result_lines"].append(self._format_pid_line(pid_name, axis_spec))
                continue

            model = self._fit_axis_model(series)
            if model is None:
                summary["per_axis"][pid_name] = {
                    "status": "skip_fit_failed",
                    "samples": len(series),
                    "base": axis_spec,
                    "best": axis_spec,
                }
                summary["result_lines"].append(self._format_pid_line(pid_name, axis_spec))
                continue

            bounds = self._make_bounds(axis_spec)
            base_candidate = {
                "kp": float(axis_spec["kp"]),
                "ki": float(axis_spec["ki"]),
                "kd": float(axis_spec["kd"]),
            }

            population = [dict(base_candidate)]
            while len(population) < int(GA_POP_SIZE):
                population.append(self._random_candidate(bounds, axis_spec))

            scored = []
            for _ in range(int(GA_GENERATIONS)):
                scored = []
                for cand in population:
                    score = self._simulate_candidate(cand, axis_spec, model, series)
                    scored.append((score, dict(cand)))
                scored.sort(key=lambda x: x[0])

                next_pop = [dict(c) for _, c in scored[:int(GA_ELITE)]]
                while len(next_pop) < int(GA_POP_SIZE):
                    parent_a = self.rng.choice(scored[:max(int(GA_ELITE), 2)])[1]
                    parent_b = self.rng.choice(scored[:max(int(GA_ELITE), 2)])[1]
                    child = self._crossover(parent_a, parent_b, bounds, axis_spec)
                    child = self._mutate_candidate(child, bounds, axis_spec)
                    next_pop.append(child)
                population = next_pop

            if not scored:
                best_candidate = dict(base_candidate)
                best_score = float("inf")
            else:
                best_score, best_candidate = scored[0]

            base_score = self._simulate_candidate(base_candidate, axis_spec, model, series)
            logged_score = self._score_logged_baseline(series)

            if not np.isfinite(best_score) or best_score >= base_score:
                chosen = dict(base_candidate)
                chosen_score = base_score
                improved = False
            else:
                chosen = dict(best_candidate)
                chosen_score = best_score
                improved = True

            final_spec = dict(axis_spec)
            final_spec.update({
                "kp": float(chosen["kp"]),
                "ki": float(chosen.get("ki", axis_spec["ki"])),
                "kd": float(chosen["kd"]),
            })

            summary["per_axis"][pid_name] = {
                "status": "ok" if improved else "keep_current",
                "samples": len(series),
                "model": model,
                "base": axis_spec,
                "base_score": float(base_score),
                "logged_score": float(logged_score),
                "best": final_spec,
                "best_score": float(chosen_score),
                "improved": bool(improved),
            }
            summary["result_lines"].append(self._format_pid_line(pid_name, final_spec))

        txt_path = os.path.join(self.output_dir, f"ga_pid_recommendation_{self.run_ts}.txt")
        json_path = os.path.join(self.output_dir, f"ga_pid_recommendation_{self.run_ts}.json")

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("# GA PID recommendation generated from PID log\n\n")
                for idx, pid_name in enumerate(GA_PID_PRINT_ORDER):
                    if idx in (2, 4, 6, 8):
                        f.write("\n")
                    axis_res = summary["per_axis"].get(pid_name, {})
                    status = axis_res.get("status", "unknown")
                    samples = axis_res.get("samples", 0)
                    f.write(f"# {pid_name} | status={status} | samples={samples}\n")
                    f.write(self._format_pid_line(pid_name, axis_res.get("best", self._runtime_pid_spec(pid_name))))
                    f.write("\n")
            summary["txt"] = txt_path
            log.info(f"[GA_PID] saved txt -> {txt_path}")
        except Exception as e:
            log.error(f"[GA_PID] write txt failed: {e}")

        try:
            json_payload = {
                "csv": summary["csv"],
                "run_ts": self.run_ts,
                "per_axis": summary["per_axis"],
                "result_lines": summary["result_lines"],
            }
            with open(json_path, "w", encoding="utf-8") as f:
                import json
                json.dump(json_payload, f, indent=2)
            summary["json"] = json_path
            log.info(f"[GA_PID] saved json -> {json_path}")
        except Exception as e:
            log.error(f"[GA_PID] write json failed: {e}")

        log.info("[GA_PID] ===== RECOMMENDED PID SET =====")
        for line in summary["result_lines"]:
            log.info(line)

        return summary


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
        connection_str="/dev/ttyAMA0",
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
        self._ui_line_target = None
        self._ui_line_ok = False
        self._ui_line_len_px = 0.0
        self._ui_line_len_cm = 0.0
        self._ui_line_max_cm = float(DROP_LINE_MAX_CM)
        self._ui_line_p1 = None
        self._ui_line_p2 = None

        self.drop_port = DROP_UART_PORT
        self.drop_baudrate = int(DROP_UART_BAUDRATE)
        self.drop_trigger_percent = float(DROP_TRIGGER_PERCENT)
        self.dropper = None

        try:
            self.dropper = BallDropper(port=self.drop_port, baudrate=self.drop_baudrate)
        except Exception as e:
            self.dropper = None
            log.error(f"[DROP] init failed: {e}")

        log.info(f"[Connecting to vehicle on]: {self.connection_str}")
        self.vehicle = connect(self.connection_str, baud=115200, wait_ready=True, timeout=60)

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
            max_hold_sec=0.9,
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

        # Low-pass filter state for pixel errors before PID
        self.lpf_ex = LowPassFilter(tau=LPF_TAU_X)
        self.lpf_ey = LowPassFilter(tau=LPF_TAU_Y)
        self._h_center_state = {}
        self._fallback_center_kf = self._create_center_kalman()

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


    def _reset_error_filters(self, ex0=0.0, ey0=0.0):
        try:
            self.lpf_ex.reset(ex0)
            self.lpf_ey.reset(ey0)
        except Exception:
            pass

    def _filter_error(self, ex, ey, dt=None):
        try:
            ex_f = float(self.lpf_ex.update(ex, dt=dt))
        except Exception:
            ex_f = float(ex)
        try:
            ey_f = float(self.lpf_ey.update(ey, dt=dt))
        except Exception:
            ey_f = float(ey)
        return ex_f, ey_f

    def _stabilize_target_center(self, raw_center, prev_center=None, err_norm=None):
        if raw_center is None:
            return prev_center

        cx, cy = float(raw_center[0]), float(raw_center[1])
        if prev_center is None:
            return (cx, cy)

        px, py = float(prev_center[0]), float(prev_center[1])
        jump = float(np.hypot(cx - px, cy - py))
        near_center = float(err_norm if err_norm is not None else 1e9) <= float(NEAR_CENTER_ERR_PX)

        if near_center and jump > float(NEAR_CENTER_JUMP_REJECT_PX):
            cx, cy = px, py

        alpha = float(CENTER_EMA_ALPHA_NEAR if near_center else CENTER_EMA_ALPHA_FAR)
        sx = alpha * cx + (1.0 - alpha) * px
        sy = alpha * cy + (1.0 - alpha) * py
        return (sx, sy)

    def _create_center_kalman(self):
        return KalmanCenter2D(
            process_accel_var=KF_PROCESS_ACCEL_VAR,
            measurement_var=KF_MEASUREMENT_VAR,
            init_pos_var=KF_INIT_POS_VAR,
            init_vel_var=KF_INIT_VEL_VAR,
        )

    def _kalman_center_update(self, kf, raw_center, dt, raw_err=None, jump_gate_px=NEAR_CENTER_JUMP_REJECT_PX):
        if raw_center is None:
            return None

        cx = float(raw_center[0])
        cy = float(raw_center[1])
        dt = max(1e-3, float(dt))

        r_scale = 1.0
        pred = kf.predict(dt)
        if pred is not None:
            px, py = pred
            jump = float(np.hypot(cx - px, cy - py))
            near_center = float(raw_err if raw_err is not None else 1e9) <= float(NEAR_CENTER_ERR_PX)
            if near_center and jump > float(jump_gate_px):
                r_scale = float(KF_NEAR_CENTER_R_SCALE)

        return kf.correct(cx, cy, r_scale=r_scale)

    def _normalize_pid_target_key(self, target_tag: str):
        tag = str(target_tag or "").upper()

        if tag in ("RED1", "RED2", "YELLOW_LEFT", "YELLOW_RIGHT", "BLUE", "H_MARKER"):
            return tag

        if tag == "RED":
            return "RED1"   # fallback
        if tag == "YELLOW":
            return "YELLOW_LEFT"  # fallback
        if tag in ("H",):
            return "H_MARKER"

        return "RED1"

    def _get_pid_pair(self, target_tag: str):
        key = self._normalize_pid_target_key(target_tag)
        pid_map = {
            "RED1": (PID_RED1_X, PID_RED1_Y),
            "RED2": (PID_RED2_X, PID_RED2_Y),
            "YELLOW_LEFT": (PID_YELLOW_LEFT_X, PID_YELLOW_LEFT_Y),
            "YELLOW_RIGHT": (PID_YELLOW_RIGHT_X, PID_YELLOW_RIGHT_Y),
            "BLUE": (PID_BLUE_X, PID_BLUE_Y),
            "H_MARKER": (PID_H_X, PID_H_Y),
        }
        return pid_map.get(key, (PID_RED1_X, PID_RED1_Y))

    def _get_control_region_scale(self, target_tag: str, err_norm: float):
        key = self._normalize_pid_target_key(target_tag)
        err_norm = float(err_norm)

        kp1 = float(SLOW_ZONE_1_KP_SCALE)
        out1 = float(SLOW_ZONE_1_OUT_SCALE)
        kp2 = float(SLOW_ZONE_2_KP_SCALE)
        out2 = float(SLOW_ZONE_2_OUT_SCALE)

        # Nhom target can vao tam mem hon
        if key in ("YELLOW_LEFT", "YELLOW_RIGHT", "BLUE"):
            kp1 = min(0.85, kp1 + 0.05)
            out1 = min(0.65, out1 + 0.05)
            kp2 = min(0.70, kp2 + 0.10)
            out2 = min(0.40, out2 + 0.05)

        if err_norm < float(SLOW_ZONE_2_PX):
            return kp2, out2
        if err_norm < float(SLOW_ZONE_1_PX):
            return kp1, out1
        return 1.0, 1.0

    def _soft_brake_scale(self, error_abs: float, brake_px: float = BRAKE_ZONE_PX):
        frac = min(1.0, max(0.0, float(error_abs) / max(1e-6, float(brake_px))))
        return 0.12 + 0.88 * frac

    def _update_center_hysteresis(self, ex: float, ey: float, state: dict, tol_enter: float = CENTER_TOL_ENTER_PX, tol_exit: float = CENTER_TOL_EXIT_PX, settle_frames: int = CENTER_SETTLE_FRAMES):
        inside_prev = bool(state.get("inside", False))
        if inside_prev:
            inside = abs(float(ex)) <= float(tol_exit) and abs(float(ey)) <= float(tol_exit)
        else:
            inside = abs(float(ex)) <= float(tol_enter) and abs(float(ey)) <= float(tol_enter)

        if inside:
            state["settle_count"] = int(state.get("settle_count", 0)) + 1
        else:
            state["settle_count"] = 0

        state["inside"] = inside
        settled = inside and int(state.get("settle_count", 0)) >= int(settle_frames)
        return inside, settled

    def _project_velocity_to_target_line(self, ex: float, ey: float, vx: float, vy: float, enabled: bool = True):
        if not enabled:
            return float(vx), float(vy), 0.0

        n = float(np.hypot(float(ex), float(ey)))
        if n <= 1e-6:
            return 0.0, 0.0, 0.0

        ux = -float(ey) / n
        uy = float(ex) / n
        v_parallel = float(vx) * ux + float(vy) * uy
        return float(v_parallel * ux), float(v_parallel * uy), float(v_parallel)

    def _get_aim_point_for_target(self, target_tag=None, mission_step=None, frame_shape=None):
        """
        Tra ve diem ngam (aim point) trong frame cho tung target / mission_step.
        """
        if frame_shape is not None:
            h, w = frame_shape[:2]
            default_pt = (w // 2, h // 2)
        else:
            default_pt = (320, 320)

        tag = str(target_tag or "").upper()

        # Uu tien map theo target_tag
        ball_id = TARGET_TO_BALL_ID.get(tag, None)

        # Neu chua co thi fallback theo mission_step
        if ball_id is None:
            step = self.mission_step if mission_step is None else int(mission_step)
            mission_to_ball = {
                1: 1,  # RED1
                2: 2,  # RED2
                3: 3,  # YELLOW_LEFT
                4: 4,  # YELLOW_RIGHT
                5: 5,  # BLUE
            }
            ball_id = mission_to_ball.get(step, None)

        return BALL_AIM_POINTS.get(ball_id, default_pt)

    def _compute_error_to_aim_point(self, target_center, target_tag=None, mission_step=None, frame_shape=None):
        """
        Tinh sai so ex, ey tu tam target -> diem ngam aim.
        """
        aim_cx, aim_cy = self._get_aim_point_for_target(
            target_tag=target_tag,
            mission_step=mission_step,
            frame_shape=frame_shape,
        )

        tx, ty = float(target_center[0]), float(target_center[1])
        ex = tx - float(aim_cx)
        ey = ty - float(aim_cy)
        return ex, ey, aim_cx, aim_cy

    # Giu alias de tranh vo neu code cu goi nham ten ham cu
    def _computr_error_to_aim_point(self, target_center, target_tag=None, mission_step=None, frame_shape=None):
        return self._compute_error_to_aim_point(
            target_center=target_center,
            target_tag=target_tag,
            mission_step=mission_step,
            frame_shape=frame_shape,
        )

    def _get_drop_line_limit_px(self):
        return float(DROP_LINE_MAX_PX)

    def _compute_drop_line_metrics(self, target_center, target_tag=None, mission_step=None, frame_shape=None):
        ex, ey, aim_cx, aim_cy = self._compute_error_to_aim_point(
            target_center=target_center,
            target_tag=target_tag,
            mission_step=mission_step,
            frame_shape=frame_shape,
        )

        line_len_px = float(np.hypot(ex, ey))
        px_per_cm = max(1e-6, float(DROP_LINE_PX_PER_CM))
        line_len_cm = line_len_px / px_per_cm
        max_len_px = self._get_drop_line_limit_px()
        max_len_cm = max_len_px / px_per_cm
        ok = bool(line_len_px <= max_len_px)

        return {
            "ex": float(ex),
            "ey": float(ey),
            "aim_cx": int(round(aim_cx)),
            "aim_cy": int(round(aim_cy)),
            "target_cx": int(round(float(target_center[0]))),
            "target_cy": int(round(float(target_center[1]))),
            "line_len_px": float(line_len_px),
            "line_len_cm": float(line_len_cm),
            "max_len_px": float(max_len_px),
            "max_len_cm": float(max_len_cm),
            "ok": ok,
        }

    def _compute_drop_line_metrics_from_detection(self, target_det, target_tag=None, mission_step=None, frame_shape=None):
        """
        Tinh line L truc tiep tu detection raw de hien thi/gate drop realtime hon,
        tach khoi nhanh Kalman + LPF.
        """
        if target_det is None:
            return None

        target_center = target_det.get("center", None)
        if target_center is None:
            return None

        return self._compute_drop_line_metrics(
            target_center=target_center,
            target_tag=target_tag,
            mission_step=mission_step,
            frame_shape=frame_shape,
        )

    def _draw_drop_line_L(self, frame_bgr, target_center, target_tag=None, mission_step=None):
        if frame_bgr is None or target_center is None or not bool(DRAW_DROP_LINE_L):
            return None

        info = self._compute_drop_line_metrics(
            target_center=target_center,
            target_tag=target_tag,
            mission_step=mission_step,
            frame_shape=frame_bgr.shape,
        )

        p1 = (int(info["aim_cx"]), int(info["aim_cy"]))
        p2 = (int(info["target_cx"]), int(info["target_cy"]))
        color = (0, 255, 0) if info["ok"] else (0, 0, 255)

        cv2.line(frame_bgr, p1, p2, color, 2)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(
            frame_bgr,
            f"L={info['line_len_cm']:.1f}cm ({info['line_len_px']:.1f}px)",
            (mid[0] + 6, mid[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        return info

    def _get_active_mission_target_tag(self):
        mapping = {
            1: "RED1",
            2: "RED2",
            3: "YELLOW_LEFT",
            4: "YELLOW_RIGHT",
            5: "BLUE",
            7: "H_MARKER",
        }
        return mapping.get(int(self.mission_step), None)

    def _select_target_det_for_tag(self, colored, target_tag, img_cx=None):
        tagU = str(target_tag or "").upper()
        if not colored:
            return None

        if img_cx is None:
            img_cx = 10 ** 9

        if tagU in ("RED1", "RED2", "RED"):
            red_dets = [d for d in colored if d.get("color_name") == "RED"]
            if not red_dets:
                return None
            if tagU == "RED":
                red_dets.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
                return red_dets[0]
            r1, r2 = self._assign_red_ids_no_promote(red_dets)
            return r1 if tagU == "RED1" else r2

        if tagU in ("YELLOW", "YELLOW_LEFT", "YELLOW_RIGHT"):
            ys = [d for d in colored if d.get("color_name") == "YELLOW"]
            if not ys:
                return None
            if tagU == "YELLOW":
                ys.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
                return ys[0]
            if tagU == "YELLOW_LEFT":
                return min(ys, key=lambda d: d.get("center", (10**9, 0))[0])
            return max(ys, key=lambda d: d.get("center", (-10**9, 0))[0])

        if tagU == "BLUE":
            bs = [d for d in colored if d.get("color_name") == "BLUE"]
            if not bs:
                return None
            bs.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
            return bs[0]

        return None

    def _compute_centering_command(self, target_tag: str, ex_px: float, ey_px: float, max_vxy: float, dt=None, control_state=None, use_projection: bool = True):
        ex_f, ey_f = self._filter_error(ex_px, ey_px, dt=dt)
        err_norm = float(np.hypot(ex_f, ey_f))
        kp_scale, out_scale = self._get_control_region_scale(target_tag, err_norm)

        pid_x, pid_y = self._get_pid_pair(target_tag)
        kp_x, _, kd_x, base_out_x = pid_x.get_gains()
        kp_y, _, kd_y, base_out_y = pid_y.get_gains()

        vx_raw = float(pid_y.update(
            -ey_f,
            dt=dt,
            Kp=kp_y * kp_scale,
            Ki=0.0,
            Kd=kd_y,
            max_output=min(float(max_vxy), float(base_out_y) * out_scale),
        ))
        vy_raw = float(pid_x.update(
            ex_f,
            dt=dt,
            Kp=kp_x * kp_scale,
            Ki=0.0,
            Kd=kd_x,
            max_output=min(float(max_vxy), float(base_out_x) * out_scale),
        ))

        vx, vy, v_parallel = self._project_velocity_to_target_line(ex_f, ey_f, vx_raw, vy_raw, enabled=use_projection)

        brake_x = self._soft_brake_scale(abs(ex_f))
        brake_y = self._soft_brake_scale(abs(ey_f))
        vx *= brake_y
        vy *= brake_x

        if control_state is None:
            control_state = {}
        center_inside, center_settled = self._update_center_hysteresis(ex_f, ey_f, control_state)
        if center_inside:
            vx *= float(CENTER_FINE_SCALE)
            vy *= float(CENTER_FINE_SCALE)

        if center_settled and err_norm <= float(CENTER_STOP_PX) and abs(vx) <= float(CENTER_MIN_CMD) and abs(vy) <= float(CENTER_MIN_CMD):
            vx, vy = 0.0, 0.0

        note = (
            f"err={err_norm:.1f}|kps={kp_scale:.2f}|outs={out_scale:.2f}|"
            f"center={1 if center_inside else 0}|settled={1 if center_settled else 0}|"
            f"vpar={v_parallel:.3f}"
        )
        return ex_f, ey_f, float(vx), float(vy), note, center_inside, center_settled

    def _reset_pid_pair(self, target_tag: str, ex0=0.0, ey0=0.0):
        pid_x, pid_y = self._get_pid_pair(target_tag)
        try:
            pid_x.reset()
        except Exception:
            pass
        try:
            pid_y.reset()
        except Exception:
            pass
        self._reset_error_filters(ex0, ey0)

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

    def _ui_set_drop_line(self, target, info):
        with self._ui_lock:
            self._ui_line_target = target
            self._ui_line_ok = bool(info.get("ok", False)) if info else False
            self._ui_line_len_px = float(info.get("line_len_px", 0.0)) if info else 0.0
            self._ui_line_len_cm = float(info.get("line_len_cm", 0.0)) if info else 0.0
            self._ui_line_max_cm = float(info.get("max_len_cm", DROP_LINE_MAX_CM)) if info else float(DROP_LINE_MAX_CM)
            if info:
                self._ui_line_p1 = (int(info.get("aim_cx", 0)), int(info.get("aim_cy", 0)))
                self._ui_line_p2 = (int(info.get("target_cx", 0)), int(info.get("target_cy", 0)))
            else:
                self._ui_line_p1 = None
                self._ui_line_p2 = None

    def _ui_clear_drop_line(self):
        with self._ui_lock:
            self._ui_line_target = None
            self._ui_line_ok = False
            self._ui_line_len_px = 0.0
            self._ui_line_len_cm = 0.0
            self._ui_line_max_cm = float(DROP_LINE_MAX_CM)
            self._ui_line_p1 = None
            self._ui_line_p2 = None

    def _drop_ball_once(self, target_tag=None, progress_pct=None):
        if self.dropper is None:
            log.warning("[DROP] skipped: dropper not ready")
            return False

        try:
            ok = self.dropper.mo_tung_cai()
            pct_txt = "" if progress_pct is None else f" progress={float(progress_pct):.1f}%"
            if ok:
                log.info(f"[DROP] fired target={target_tag}{pct_txt}")
            else:
                log.warning(f"[DROP] command failed target={target_tag}{pct_txt}")
            return bool(ok)
        except Exception as e:
            log.error(f"[DROP] exception target={target_tag}: {e}")
            return False

    
    def _run_post_mission_ga_from_csv(self, csv_path):
        if not bool(GA_ENABLE_POST_TUNING):
            return None
        try:
            tuner = PIDLogGeneticTuner(output_dir=self._pid_dir, run_ts=self._pid_run_ts)
            return tuner.tune_from_csv(csv_path)
        except Exception as e:
            log.error(f"[GA_PID] post-mission tuning failed: {e}")
            return None

    def stop(self):
        self._stop_flag = True

    def _get_last_h_dets_copy(self):
        with self._det_lock:
            return list(self._last_h) if self._last_h else []

    # ---------- Detection loop ----------
    def _det_loop(self):
        det_hz = 30.0
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

    # ---------- Visualization ----------
    def run_viewer_loop(self):
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
            aim_cx, aim_cy = self._get_aim_point_for_target(
                mission_step= self.mission_step,
                frame_shape = vis.shape,
            )

            cv2.circle(vis, (int(aim_cx), int(aim_cy)), 6, (0, 100, 255), -1)
            cv2.putText(
                vis,
                f"AIM ({int(aim_cx)},{int(aim_cy)})",
                (int(aim_cx) + 8, int(aim_cy) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                3,
            )
            icx, icy = w_img // 2, h_img // 2

            with self._det_lock:
                circle_dets = list(self._last_circle) if self._last_circle else []
                h_dets = list(self._last_h) if self._last_h else []

            colored = self._colorize_dets(vis, circle_dets)
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

                with self._ui_lock:
                    tgt = self._ui_hold_target
                    inz = self._ui_hold_in_zone
                    elp = self._ui_hold_elapsed
                    req = self._ui_hold_required

                if tgt is not None and req >= 0:
                    pct = 100.0 if req <= 1e-6 else min(100.0, (elp / req) * 100.0)
                    state = "HOLD" if inz else "PAUSE"
                    txt = f"{state} {tgt}: {pct:.0f}% ({elp:.1f}/{req:.1f}s)"
                    color = (0, 255, 0) if inz else (0, 255, 255)
                    cv2.putText(vis, txt, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                active_target_tag = self._get_active_mission_target_tag()
                realtime_line_info = None
                if active_target_tag in ("RED1", "RED2", "YELLOW_LEFT", "YELLOW_RIGHT", "BLUE"):
                    active_target_det = self._select_target_det_for_tag(
                        colored,
                        active_target_tag,
                        img_cx=icx,
                    )
                    realtime_line_info = self._compute_drop_line_metrics_from_detection(
                        active_target_det,
                        target_tag=active_target_tag,
                        mission_step=self.mission_step,
                        frame_shape=vis.shape,
                    )

                if realtime_line_info is not None:
                    line_tgt = active_target_tag
                    line_ok = bool(realtime_line_info["ok"])
                    line_cm = float(realtime_line_info["line_len_cm"])
                    line_max_cm = float(realtime_line_info["max_len_cm"])
                    line_p1 = (int(realtime_line_info["aim_cx"]), int(realtime_line_info["aim_cy"]))
                    line_p2 = (int(realtime_line_info["target_cx"]), int(realtime_line_info["target_cy"]))
                else:
                    line_tgt = self._ui_line_target
                    line_ok = self._ui_line_ok
                    line_cm = self._ui_line_len_cm
                    line_max_cm = self._ui_line_max_cm
                    line_p1 = self._ui_line_p1
                    line_p2 = self._ui_line_p2

                if line_tgt is not None and line_p1 is not None and line_p2 is not None:
                    line_color = (0, 255, 0) if line_ok else (0, 0, 255)
                    cv2.line(vis, line_p1, line_p2, line_color, 2)
                    mid = ((line_p1[0] + line_p2[0]) // 2, (line_p1[1] + line_p2[1]) // 2)
                    cv2.putText(
                        vis,
                        f"L={line_cm:.1f}cm <= {line_max_cm:.1f}cm ? {'OK' if line_ok else 'NO'}",
                        (mid[0] + 8, mid[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        line_color,
                        2,
                    )
                    cv2.putText(
                        vis,
                        f"DROP_LINE {line_tgt}: {line_cm:.1f}/{line_max_cm:.1f}cm",
                        (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        line_color,
                        2,
                    )

                if self._recording:
                    cv2.putText(vis, "REC", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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

    def _append_pid_sample(self, phase, target, ex, ey, vx, vy, note=""):
        try:
            err_norm = float(np.hypot(float(ex), float(ey)))
            sample = {
                "t": time.time(),
                "phase": str(phase),
                "target": str(target),
                "ex": float(ex),
                "ey": float(ey),
                "vx": float(vx),
                "vy": float(vy),
                "err_norm": err_norm,
                "note": str(note or ""),
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
            return {}

        t0 = rows[0]["t"]
        ts = self._pid_run_ts
        csv_path = os.path.join(self._pid_dir, f"pid_trace_{ts}.csv")
        overview_png = os.path.join(self._pid_dir, f"pid_overview_{ts}.png")

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t_rel_s", "phase", "target", "ex", "ey", "vx", "vy", "err_norm", "note"])
                for r in rows:
                    writer.writerow([
                        f"{r['t'] - t0:.3f}",
                        r["phase"],
                        r["target"],
                        f"{r['ex']:.6f}",
                        f"{r['ey']:.6f}",
                        f"{r['vx']:.6f}",
                        f"{r['vy']:.6f}",
                        f"{r.get('err_norm', 0.0):.6f}",
                        r.get("note", ""),
                    ])
        except Exception as e:
            log.error(f"[PID_PLOT] CSV export failed: {e}")
            csv_path = None

        exports = {"csv": csv_path, "overview_png": None, "target_pngs": []}

        try:
            t = [r["t"] - t0 for r in rows]
            ex = [r["ex"] for r in rows]
            ey = [r["ey"] for r in rows]
            vx = [r["vx"] for r in rows]
            vy = [r["vy"] for r in rows]
            err = [r.get("err_norm", float(np.hypot(r["ex"], r["ey"]))) for r in rows]

            fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
            axes[0].plot(t, ex, label="ex (px)")
            axes[0].plot(t, ey, label="ey (px)")
            axes[0].axhline(0.0, linewidth=1.0, linestyle="--")
            axes[0].set_ylabel("Pixel error")
            axes[0].set_title("PID overview - error")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(t, err, label="sqrt(ex^2 + ey^2)")
            axes[1].axhline(0.0, linewidth=1.0, linestyle="--")
            axes[1].set_ylabel("Error norm")
            axes[1].set_title("PID overview - total tracking error")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            axes[2].plot(t, vx, label="vx (m/s)")
            axes[2].plot(t, vy, label="vy (m/s)")
            axes[2].axhline(0.0, linewidth=1.0, linestyle="--")
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("Velocity")
            axes[2].set_title("PID overview - controller output")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

            prev_key = None
            for r in rows:
                key = (r["phase"], r["target"])
                if key != prev_key:
                    x = r["t"] - t0
                    for ax in axes:
                        ax.axvline(x, linewidth=0.8, linestyle=":", alpha=0.5)
                    prev_key = key

            fig.tight_layout()
            fig.savefig(overview_png, dpi=170, bbox_inches="tight")
            plt.close(fig)
            exports["overview_png"] = overview_png
            log.info(f"[PID_PLOT] saved overview -> {overview_png}")
        except Exception as e:
            log.error(f"[PID_PLOT] overview PNG export failed: {e}")

        try:
            groups = defaultdict(list)
            for r in rows:
                groups[str(r.get("target", "UNK"))].append(r)

            for target_name, g in groups.items():
                if len(g) < 2:
                    continue
                t_local0 = g[0]["t"]
                tg = [r["t"] - t_local0 for r in g]
                exg = [r["ex"] for r in g]
                eyg = [r["ey"] for r in g]
                vxg = [r["vx"] for r in g]
                vyg = [r["vy"] for r in g]
                erg = [r.get("err_norm", float(np.hypot(r["ex"], r["ey"]))) for r in g]

                fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
                axes[0].plot(tg, exg, label="ex (px)")
                axes[0].plot(tg, eyg, label="ey (px)")
                axes[0].axhline(0.0, linewidth=1.0, linestyle="--")
                axes[0].set_ylabel("Pixel error")
                axes[0].set_title(f"Target {target_name} - error")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()

                axes[1].plot(tg, erg, label="error norm")
                axes[1].axhline(0.0, linewidth=1.0, linestyle="--")
                axes[1].set_ylabel("Error norm")
                axes[1].set_title(f"Target {target_name} - total error")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()

                axes[2].plot(tg, vxg, label="vx (m/s)")
                axes[2].plot(tg, vyg, label="vy (m/s)")
                axes[2].axhline(0.0, linewidth=1.0, linestyle="--")
                axes[2].set_xlabel("Time (s)")
                axes[2].set_ylabel("Velocity")
                axes[2].set_title(f"Target {target_name} - controller output")
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()

                prev_phase = None
                for r in g:
                    phase = str(r.get("phase", ""))
                    if phase != prev_phase:
                        x = r["t"] - t_local0
                        for ax in axes:
                            ax.axvline(x, linewidth=0.8, linestyle=":", alpha=0.5)
                        prev_phase = phase

                fig.tight_layout()
                safe_target = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in target_name)
                target_png = os.path.join(self._pid_dir, f"pid_target_{safe_target}_{ts}.png")
                fig.savefig(target_png, dpi=170, bbox_inches="tight")
                plt.close(fig)
                exports["target_pngs"].append(target_png)
                log.info(f"[PID_PLOT] saved target plot -> {target_png}")
        except Exception as e:
            log.error(f"[PID_PLOT] target PNG export failed: {e}")

        return exports

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

    def pid_drive_to_center_H(
        self,
        ex_px: float,
        ey_px: float,
        tol_px: int = 5,
        max_vxy: float = 0.15,
        send_now: bool = False,
        vz: float = 0.0,
    ):
        now = time.time()
        dt = max(1e-3, now - float(getattr(self, "_last_h_pid_t", now - 1.0 / 30.0)))
        self._last_h_pid_t = now

        ex_f, ey_f, vx, vy, note, _, _ = self._compute_centering_command(
            "H_MARKER",
            ex_px,
            ey_px,
            max_vxy=max_vxy,
            dt=dt,
            control_state=self._h_center_state,
            use_projection=False,
        )

        log.info(
            f"[PID_CONTROL] target=H_MARKER "
            f"ex_raw={float(ex_px):.1f} ey_raw={float(ey_px):.1f} "
            f"ex_lpf={ex_f:.1f} ey_lpf={ey_f:.1f} "
            f"vx={vx:.3f} vy={vy:.3f} {note}"
        )

        self._append_pid_sample("land_h", "H_MARKER", ex_f, ey_f, vx, vy, note=note)

        if send_now:
            self.send_local_ned_velocity(vx, vy, float(vz))

        return ex_f, ey_f, vx, vy

    def seek_target_and_center(
        self,
        target_tag: str,
        direction: str,
        search_speed: float = 0.7,
        tol_px: int = 10,
        hold_s: float = 5.0,
        timeout_s: float = 30.0,
        pre_delay_s: float = 0.0,
        loop_hz: float = 20.0,
        hold_bbox_pad: int = 0,
        duration_s: float = None,
    ):
        try:
            self._reset_pid_pair(target_tag)
        except Exception:
            pass

        if pre_delay_s and float(pre_delay_s) > 0.0:
            vx0, vy0 = self._direction_to_vxvy(direction, search_speed)
            t_pre = time.time()
            while not self._stop_flag and (time.time() - t_pre) < float(pre_delay_s):
                self.send_local_ned_velocity(vx0, vy0, 0.0)
                time.sleep(0.1)
            try:
                self._reset_pid_pair(target_tag)
            except Exception:
                pass

        start_t = time.time()
        hold_required = max(0.0, float(hold_s))
        hold_accum = 0.0
        hold_last_in_zone_t = None

        def hold_done():
            return hold_required <= 1e-6 or hold_accum >= hold_required

        drop_trigger_pct = float(max(0.0, min(100.0, self.drop_trigger_percent)))
        drop_trigger_time = hold_required * drop_trigger_pct / 100.0
        drop_fired = False
        self._ui_clear_drop_line()

        dt_sleep = 1.0 / float(max(1e-6, loop_hz))
        max_v_seek = 0.30
        max_v_hold = 0.18

        yellow_lock = None
        lock_dist = int(getattr(self, "YELLOW_LOCK_DIST", 250))
        lock_gate2 = lock_dist * lock_dist
        margin = int(getattr(self, "YELLOW_SIDE_MARGIN", 40))

        duration_logged = False
        stable_center = None
        bbox_kf = self._create_center_kalman()
        last_kf_t = time.time()
        last_seen_target_t = None
        ctrl_state = {"inside": False, "settle_count": 0}
        last_pid_t = time.time()

        def pid_drive_to_center(ex_px: float, ey_px: float, max_vxy: float):
            nonlocal last_pid_t
            now_pid = time.time()
            dt_pid = max(1e-3, now_pid - last_pid_t)
            last_pid_t = now_pid

            ex_f, ey_f, vx, vy, note, center_inside, center_settled = self._compute_centering_command(
                tagU,
                ex_px,
                ey_px,
                max_vxy=max_vxy,
                dt=dt_pid,
                control_state=ctrl_state,
                use_projection=(self._normalize_pid_target_key(tagU) != "H_MARKER"),
            )

            log.info(
                f"[PID_CONTROL] target={tagU} "
                f"ex_raw={float(ex_px):.1f} ey_raw={float(ey_px):.1f} "
                f"ex_lpf={ex_f:.1f} ey_lpf={ey_f:.1f} "
                f"vx={vx:.3f} vy={vy:.3f} {note}"
            )

            self._append_pid_sample("seek", tagU, ex_f, ey_f, vx, vy, note=note)
            self.send_local_ned_velocity(vx, vy, 0.0)
            return ex_f, ey_f, vx, vy, center_inside, center_settled

        while not self._stop_flag:
            now = time.time()
            if timeout_s is not None and (now - start_t) > float(timeout_s):
                log.error(f"[SEEK] Timeout target={target_tag} dir={direction}")
                self.send_local_ned_velocity(0.0, 0.0, 0.0)
                self._ui_clear_hold()
                return False

            frame_bgr = self.camera.get_frame()
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            with self._det_lock:
                dets = list(self._last_circle) if self._last_circle else []

            h, w = frame_bgr.shape[:2]
            img_cx, img_cy = w //2, h //2

            aim_cx, aim_cy = self._get_aim_point_for_target(
                target_tag   = target_tag,
                mission_step = self.mission_step,
                frame_shape  = frame_bgr.shape, 
            )

            colored = self._colorize_dets(frame_bgr, dets)
            tagU = (target_tag or "").upper()
            target_det = None

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
                                if cx < img_cx - margin:
                                    target_det = cand
                                    yellow_lock = target_det.get("center", None)
                            else:
                                cand = max(ys, key=lambda d: d.get("center", (-10**9, 0))[0])
                                cx = cand.get("center", (0, 0))[0]
                                if cx > img_cx + margin:
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

            duration_expired = duration_s is not None and (now - start_t) >= float(duration_s)

            if target_det is None:
                hold_last_in_zone_t = None
                stable_center = None
                ctrl_state["inside"] = False
                ctrl_state["settle_count"] = 0
                self._ui_set_hold(target_tag, False, min(hold_accum, hold_required), hold_required)
                self._ui_clear_drop_line()

                now_kf = time.time()
                dt_kf = max(1e-3, now_kf - last_kf_t)
                last_kf_t = now_kf
                if bbox_kf.initialized:
                    bbox_kf.predict(dt_kf)
                if last_seen_target_t is not None and (now_kf - last_seen_target_t) > float(KF_LOST_RESET_SEC):
                    bbox_kf.reset()

                if duration_expired:
                    if not duration_logged:
                        log.info(f"[SEEK] duration_s={duration_s}s expired(target={target_tag}, dir={direction})")
                        duration_logged = True
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
                else:
                    vx, vy = self._direction_to_vxvy(direction, search_speed)
                    self.send_local_ned_velocity(vx, vy, 0.0)

            else:
                raw_tx, raw_ty = target_det.get("center", (img_cx, img_cy))
                raw_ex = float(raw_tx - aim_cx)
                raw_ey = float(raw_ty - aim_cy)
                raw_err = float(np.hypot(raw_ex, raw_ey))

                now_kf = time.time()
                dt_kf = max(1e-3, now_kf - last_kf_t)
                last_kf_t = now_kf
                last_seen_target_t = now_kf

                kf_center = self._kalman_center_update(
                    bbox_kf,
                    (raw_tx, raw_ty),
                    dt=dt_kf,
                    raw_err=raw_err,
                )
                if kf_center is None:
                    stable_center = self._stabilize_target_center((raw_tx, raw_ty), stable_center, err_norm=raw_err)
                    tx, ty = stable_center if stable_center is not None else (raw_tx, raw_ty)
                else:
                    stable_center = kf_center
                    tx, ty = stable_center

                ex = float(tx - aim_cx)
                ey = float(ty - aim_cy)

                # Line L tinh truc tiep tu detection raw -> giam delay hien thi/drop gate,
                # khong di qua nhanh Kalman + LPF.
                line_info = self._compute_drop_line_metrics_from_detection(
                    target_det=target_det,
                    target_tag=target_tag,
                    mission_step=self.mission_step,
                    frame_shape=frame_bgr.shape,
                )
                self._ui_set_drop_line(target_tag, line_info)

                x1, y1, x2, y2 = target_det.get("bbox", (0, 0, 0, 0))
                pad = int(hold_bbox_pad)
                if pad > 0:
                    x1 -= pad
                    y1 -= pad
                    x2 += pad
                    y2 += pad

                in_zone = (x1 <= aim_cx <= x2) and (y1 <= aim_cy <= y2)

                if in_zone:
                    if hold_last_in_zone_t is None:
                        hold_last_in_zone_t = now
                        if hold_accum <= 1e-6:
                            try:
                                self._reset_pid_pair(tagU, ex, ey)
                            except Exception:
                                pass
                            ctrl_state["inside"] = False
                            ctrl_state["settle_count"] = 0
                            last_pid_t = time.time()
                            log.info(f"[HOLD-ZONE] start target={target_tag}")
                        else:
                            pct_resume = 100.0 * hold_accum / max(hold_required, 1e-6)
                            log.info(f"[HOLD-ZONE] resume target={target_tag} progress={pct_resume:.1f}%")
                    else:
                        hold_accum += max(0.0, now - hold_last_in_zone_t)
                        hold_last_in_zone_t = now

                    hold_view = min(hold_accum, hold_required)
                    self._ui_set_hold(target_tag, True, hold_view, hold_required)
                    ex_f, ey_f, vx, vy, center_inside, center_settled = pid_drive_to_center(ex, ey, max_v_hold)

                    drop_gate_ready = bool(line_info.get("ok", False))
                    drop_gate_msg = (
                        f"L={line_info.get('line_len_cm', 0.0):.2f}cm/"
                        f"{line_info.get('max_len_cm', 0.0):.2f}cm"
                    )

                    if (not drop_fired) and hold_required > 0.0 and hold_accum >= drop_trigger_time:
                        progress_pct = 100.0 * min(hold_accum, hold_required) / max(hold_required, 1e-6)
                        if drop_gate_ready:
                            # Dung yen truoc khi drop de tang do chinh xac
                            self.send_local_ned_velocity(0.0, 0.0, 0.0)
                            time.sleep(0.12)
                            self._drop_ball_once(target_tag, progress_pct)
                            drop_fired = True
                            log.info(
                                f"[DROP_GATE] target={target_tag} pass | "
                                f"progress={progress_pct:.1f}% | {drop_gate_msg}"
                            )
                        else:
                            log.info(
                                f"[DROP_GATE] target={target_tag} wait-line | "
                                f"progress={progress_pct:.1f}% | {drop_gate_msg}"
                            )

                    # Chi hoan thanh step khi da du hold_s va da drop thanh cong.
                    if hold_done() and drop_fired:
                        log.info(
                            f"[HOLD-ZONE] done target={target_tag} "
                            f"hold={min(hold_accum, hold_required):.1f}/{hold_required:.1f}s "
                            f"center_inside={int(center_inside)} settled={int(center_settled)} | {drop_gate_msg}"
                        )
                        self.send_local_ned_velocity(0.0, 0.0, 0.0)
                        self._ui_clear_hold()
                        self._ui_clear_drop_line()
                        return True

                else:
                    if hold_done() and drop_fired:
                        log.info(
                            f"[HOLD-ZONE] done target={target_tag} "
                            f"hold={min(hold_accum, hold_required):.1f}/{hold_required:.1f}s"
                        )
                        self.send_local_ned_velocity(0.0, 0.0, 0.0)
                        self._ui_clear_hold()
                        self._ui_clear_drop_line()
                        return True

                    if hold_last_in_zone_t is not None:
                        pct_pause = 100.0 * min(hold_accum, hold_required) / max(hold_required, 1e-6)
                        log.info(f"[HOLD-ZONE] pause target={target_tag} progress={pct_pause:.1f}%")

                    hold_last_in_zone_t = None
                    hold_view = min(hold_accum, hold_required)
                    self._ui_set_hold(target_tag, False, hold_view, hold_required)
                    pid_drive_to_center(ex, ey, max_v_seek)

            time.sleep(dt_sleep)

        self.send_local_ned_velocity(0.0, 0.0, 0.0)
        self._ui_clear_hold()
        self._ui_clear_drop_line()
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
        switch_to_land_alt_m: float = 1.5, # đúng yêu cầu của bạn
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
            self._reset_pid_pair("H_MARKER")
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

        hold_start = None
        h_lock = None
        h_center_kf = self._create_center_kalman()
        last_h_kf_t = time.time()
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

                now_kf = time.time()
                dt_kf = max(1e-3, now_kf - last_h_kf_t)
                last_h_kf_t = now_kf
                if h_center_kf.initialized:
                    h_center_kf.predict(dt_kf)

                if (now - last_seen_h) > float(lost_h_timeout_s):
                    h_center_kf.reset()
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
            raw_tx, raw_ty = best.get("center", (icx, icy))
            raw_ex = float(raw_tx - icx)
            raw_ey = float(raw_ty - icy)
            raw_err = float(np.hypot(raw_ex, raw_ey))

            now_kf = time.time()
            dt_kf = max(1e-3, now_kf - last_h_kf_t)
            last_h_kf_t = now_kf

            h_kf_center = self._kalman_center_update(
                h_center_kf,
                (raw_tx, raw_ty),
                dt=dt_kf,
                raw_err=raw_err,
            )
            if h_kf_center is None:
                tx, ty = raw_tx, raw_ty
            else:
                tx, ty = h_kf_center

            ex = float(tx - icx)
            ey = float(ty - icy)

            # PID ngang để kéo về tâm marker H
            ex_f, ey_f, vx, vy = self.pid_drive_to_center_H(
                ex_px=ex,
                ey_px=ey,
                tol_px=tol_px,
                max_vxy=max_vxy if (alt is None or alt > 1.2) else min(max_vxy, 0.18),
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
                    ok = self.move_with_timer("forward", 2.0, 1.0)

                    ok = self.seek_target_and_center("RED1", "forward", 0.4, hold_s=6.5, timeout_s=80.0, pre_delay_s=4, duration_s= 10)
                    if not ok:
                        break
                    self.mission_step = 7

                elif self.mission_step == 2:
                    log.info("[Step 2] FORWARD -> RED2 enter bbox then hold 5s")
                    ok = self.seek_target_and_center("RED2", "forward", 0.4, hold_s=4, timeout_s=60.0, pre_delay_s=2, duration_s=5)
                    if not ok:
                        break
                    self.mission_step = 3

                elif self.mission_step == 3:
                    log.info("[Step 3] LEFT -> YELLOW_LEFT enter bbox then hold 5s")
                    ok = self.seek_target_and_center("YELLOW_LEFT", "left", 0.7, hold_s=4.5, timeout_s=60.0, pre_delay_s=1.5,duration_s=2.5)
                    if not ok:
                        break
                    self.mission_step = 4

                elif self.mission_step == 4:
                    log.info("[Step 4] RIGHT -> YELLOW_RIGHT enter bbox then hold 5s")
                    ok = self.seek_target_and_center("YELLOW_RIGHT", "right", 0.5, hold_s=6.5, timeout_s=60.0, pre_delay_s=3.5, duration_s=5)
                    if not ok:
                        break
                    self.mission_step = 5

                elif self.mission_step == 5:
                    log.info("[Step 5] Move: left 2.5s then forward 2.5s -> BLUE enter bbox hold 5s")
                    ok = self.move_with_timer("forward", 2.0, 0.5)
                    time.sleep(1)
                    if not ok:
                        break
                    ok = self.seek_target_and_center("BLUE","left", 0.5, hold_s=3.5, timeout_s=80.0,  pre_delay_s=1.5, duration_s=2)
                    self.mission_step = 6

                elif self.mission_step == 6:
                    log.info("[Step 6] Backward 2.5s")
                    ok = self.move_with_timer("backward", 8.5, 0.7)
                    time.sleep(1)
                    if not ok:
                        break
                    self.mission_step = 7

                elif self.mission_step == 7:
                    log.info("[Step 7] LAND on H (visual-servo)")
                    # self.vehicle.mode = VehicleMode("LAND")
                    res = self.land_to_h(timeout_s=90)
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
            if alt >= targetHeight * 0.8:
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
        # Giu nguyen signature cu de tranh vo cac noi goi ham cu.
        now = time.time()
        dt = max(1e-3, now - float(getattr(self, "_last_control_pid_t", now - 1.0 / 30.0)))
        self._last_control_pid_t = now
        state = getattr(self, "_fallback_center_state", None)
        if state is None:
            state = {"inside": False, "settle_count": 0}
            self._fallback_center_state = state

        try:
            frame_bgr = self.camera.get_frame()
            if frame_bgr is not None:
                h, w = frame_bgr.shape[:2]
                icx, icy = w // 2, h // 2
                raw_center = (float(icx + ex), float(icy + ey))
                kf_center = self._kalman_center_update(
                    self._fallback_center_kf,
                    raw_center,
                    dt=dt,
                    raw_err=float(np.hypot(ex, ey)),
                )
                if kf_center is not None:
                    ex = float(kf_center[0] - icx)
                    ey = float(kf_center[1] - icy)
        except Exception:
            pass

        ex_f, ey_f, vx, vy, note, _, _ = self._compute_centering_command(
            "RED",
            ex,
            ey,
            max_vxy=0.09,
            dt=dt,
            control_state=state,
            use_projection=True,
        )
        log.info(
            f"[PID_CONTROL] target=RED ex_raw={float(ex):.1f} ey_raw={float(ey):.1f} "
            f"ex_lpf={ex_f:.1f} ey_lpf={ey_f:.1f} vx={vx:.3f} vy={vy:.3f} {note}"
        )
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
            plot_exports = self._export_pid_plots()
            if plot_exports:
                log.info(f"[PID_PLOT] export summary: {plot_exports}")
                ga_exports = self._run_post_mission_ga_from_csv(plot_exports.get("csv"))
                if ga_exports:
                    log.info(f"[GA_PID] export summary: {ga_exports}")
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
            if self.dropper is not None:
                self.dropper.close()
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
    # connection_str = "/dev/ttyAMA0"
    connection_str = "/dev/ttyACM0"

    takeoff_height = 4.5
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