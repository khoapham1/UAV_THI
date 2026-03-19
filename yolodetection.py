import onnxruntime as ort
import numpy as np
import cv2

class YOLODetector:
    def __init__(self, model_path, input_size=320, conf_thres=0.8):
        self.input_size = input_size
        self.conf_thres = conf_thres

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, 0)

    def detect_best(self, frame):
        h, w, _ = frame.shape
        inp = self.preprocess(frame)

        outs = self.session.run(None, {self.input_name: inp})
        out = outs[0]

        # Robust squeeze: (1,N,85) or (N,85) or (1,85,N)...
        out = np.array(out)
        out = np.squeeze(out)

        # If got (85, N) -> transpose to (N, 85)
        if out.ndim == 2 and out.shape[0] < out.shape[1] and out.shape[0] in (6, 7, 85):
            out = out.T

        if out.ndim != 2:
            return None

        best = None
        best_conf = 0.0

        for det in out:
            # YOLOv5: [x,y,w,h, obj, cls...]
            if det.shape[0] < 6:
                continue

            obj = float(det[4])
            # nếu model chỉ 1 class circle thì score = obj; nếu nhiều class thì nhân cls_conf
            cls_conf = float(np.max(det[5:])) if det.shape[0] > 5 else 1.0
            conf = obj * cls_conf

            if conf > best_conf and conf > self.conf_thres:
                best = det
                best_conf = conf

        if best is None:
            return None

        cx, cy, bw, bh = best[:4]

        scale_x = w / self.input_size
        scale_y = h / self.input_size

        cx = int(cx * scale_x)
        cy = int(cy * scale_y)
        bw = int(bw * scale_x)
        bh = int(bh * scale_y)

        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w - 1, cx + bw // 2)
        y2 = min(h - 1, cy + bh // 2)

        if x2 <= x1 or y2 <= y1:
            return None

        return {"bbox": (x1, y1, x2, y2), "center": (cx, cy), "conf": float(best_conf)}
