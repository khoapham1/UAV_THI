import cv2
import math

class CenterTracker:
    def __init__(self, offset=(0, 0), circle_detect=300):
        self.offset_x = offset[0]
        self.offset_y = offset[1]
        self.circle_detect = circle_detect

    def draw_center(self, frame):
        h, w, _ = frame.shape
        cx = w // 2 + self.offset_x
        cy = h // 2 + self.offset_y

        cv2.circle(frame, (cx, cy), 5, (0, 100, 255), 60)
        cv2.circle(frame, (cx, cy), self.circle_detect, (255, 0, 0), 2)

        cv2.arrowedLine(frame, (cx, cy), (cx + 50, cy), (255, 0, 0), 1)
        cv2.arrowedLine(frame, (cx, cy), (cx, cy - 50), (255, 0, 0), 1)

    def compute_error(self, frame, bbox):
        """
        bbox: (x1, y1, x2, y2)
        return: error_x, error_y ho?c None
        """
        if bbox is None:
            return None

        h, w, _ = frame.shape
        center_x = w // 2 + self.offset_x
        center_y = h // 2 + self.offset_y

        x1, y1, x2, y2 = bbox
        obj_cx = (x1 + x2) // 2
        obj_cy = (y1 + y2) // 2

        dist = math.hypot(obj_cx - center_x, obj_cy - center_y)

        if dist > self.circle_detect:
            return None

        error_x = float(obj_cx - center_x)
        error_y = float(obj_cy - center_y)

        return error_x, error_y
