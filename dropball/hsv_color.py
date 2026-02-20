import cv2
import numpy as np

class HSVColorClassifier:
    def __init__(self, color_threshold=0.05, debug=False):
        # M? r?ng range HSV
        self.color_ranges = {
            "RED": [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([160, 50, 50]), np.array([180, 255, 255]))
            ],
            "BLUE": [
                (np.array([90, 50, 50]), np.array([140, 255, 255]))
            ],
            "YELLOW": [
                (np.array([15, 40, 40]), np.array([45, 255, 255]))  # M? r?ng range v�ng
            ]
        }

        self.draw_colors = {
            "RED": (0, 0, 255),
            "BLUE": (255, 0, 0),
            "YELLOW": (0, 255, 255)
        }

        self.kernel = np.ones((5, 5), np.uint8)
        self.color_threshold = color_threshold
        self.debug = debug

    def classify(self, roi):
        if roi is None or roi.size == 0:
            return None

        if roi.shape[0] < 20 or roi.shape[1] < 20:
            roi = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_LINEAR)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_area = roi.shape[0] * roi.shape[1]

        best_color = None
        best_ratio = 0
        color_ratios = {}

        for color, ranges in self.color_ranges.items():
            mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                mask_total = cv2.bitwise_or(mask_total, mask)

            # Morphology d? ?n d?nh
            mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, self.kernel)
            mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, self.kernel)

            pixels = cv2.countNonZero(mask_total)
            ratio = pixels / roi_area
            color_ratios[color] = ratio

            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color

        if self.debug:
            print(f"[HSV DEBUG] Ratios: {color_ratios}, Best: {best_color} ({best_ratio:.3f})")

        if best_ratio > self.color_threshold:
            return best_color, self.draw_colors[best_color]

        return None