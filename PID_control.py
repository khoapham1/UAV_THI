import time
import numpy as np


class LowPassFilter:
    def __init__(self, alpha=0.25, init_value=0.0):
        """
        alpha in (0,1]:
            - alpha nhỏ  -> lọc mạnh hơn, mượt hơn, nhưng trễ hơn
            - alpha lớn  -> bám nhanh hơn, nhưng lọc ít hơn
        """
        self.alpha = float(np.clip(alpha, 1e-6, 1.0))
        self.init_value = float(init_value)
        self.reset()

    def reset(self, value=None):
        self.initialized = False
        self.last_value = self.init_value if value is None else float(value)

    def update(self, value):
        value = float(value)
        if not self.initialized:
            self.last_value = value
            self.initialized = True
            return self.last_value

        self.last_value = self.alpha * value + (1.0 - self.alpha) * self.last_value
        return self.last_value


class PIDController:
    def __init__(self, Kp, Ki, Kd, max_output, integral_limit=None):
        """
        Kp, Ki, Kd      : hệ số PID
        max_output      : giới hạn đầu ra (vd: tốc độ m/s, góc servo, RC value)
        integral_limit  : giới hạn tích phân (chống windup), None = không giới hạn
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error):
        """
        error: sai lệch hiện tại (vd: error_x hoặc error_y)
        return: output PID đã clamp
        """
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0.0:
            return 0.0

        # ===== Integral =====
        self.integral += error * dt

        # Anti-windup
        if self.integral_limit is not None:
            self.integral = np.clip(
                self.integral,
                -self.integral_limit,
                self.integral_limit
            )

        # ===== Derivative =====
        derivative = (error - self.last_error) / dt

        output = (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * derivative
        )

        # Clamp output
        output = np.clip(output, -self.max_output, self.max_output)

        # Update state
        self.last_error = error
        self.last_time = current_time

        return float(output)
