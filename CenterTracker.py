import cv2
import math
import time


class CircleMemoryTracker:
    """
    Giu detections circle trong mot khoang ngan khi model hut frame.
    - Khi detect lai: match theo tam gan nhat -> cap nhat track.
    - Khi mat detect: tra ve bbox/tam cu trong mot khoang thoi gian ngan.
    - Co danh dau is_memory/source de phan biet detect that vs du lieu nho.
    """

    def __init__(
        self,
        max_missing_frames: int = 30,
        max_hold_sec: float = 0.8,
        match_dist_px: float = 140.0,
        max_tracks: int = 8,
        conf_decay: float = 0.92,
        predict_motion: bool = True,
        offset=(0, 0),
        circle_detect=None,
        **kwargs,
    ):
        # backward compatibility with old call sites:
        #   CircleMemoryTracker(offset=(0,0), circle_detect=250)
        if circle_detect is not None:
            match_dist_px = float(circle_detect)
        self.offset = tuple(offset) if offset is not None else (0, 0)
        self.max_missing_frames = int(max_missing_frames)
        self.max_hold_sec = float(max_hold_sec)
        self.match_dist_px = float(match_dist_px)
        self.max_tracks = int(max_tracks)
        self.conf_decay = float(conf_decay)
        self.predict_motion = bool(predict_motion)
        self.tracks = []

    @staticmethod
    def _clip_box(box, w, h):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return (x1, y1, x2, y2)

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return (int((x1 + x2) * 0.5), int((y1 + y2) * 0.5))

    @staticmethod
    def _dist2(p, q):
        dx = float(p[0]) - float(q[0])
        dy = float(p[1]) - float(q[1])
        return dx * dx + dy * dy

    @staticmethod
    def _make_det(track):
        return {
            "bbox": tuple(track["bbox"]),
            "center": tuple(track["center"]),
            "conf": float(track["conf"]),
            "cls": int(track.get("cls", 0)),
            "is_memory": bool(track.get("is_memory", False)),
            "source": track.get("source", "detected"),
            "miss_count": int(track.get("miss_count", 0)),
        }

    def _new_track(self, det, now, w, h):
        bbox = self._clip_box(det.get("bbox", (0, 0, 1, 1)), w, h)
        center = det.get("center") or self._center(bbox)
        return {
            "bbox": bbox,
            "center": (int(center[0]), int(center[1])),
            "conf": float(det.get("conf", 0.0)),
            "cls": int(det.get("cls", 0)),
            "last_seen": float(now),
            "last_update": float(now),
            "miss_count": 0,
            "vx": 0.0,
            "vy": 0.0,
            "is_memory": False,
            "source": "detected",
        }

    def _predict_track(self, track, now, w, h):
        dt = max(0.0, float(now) - float(track.get("last_update", now)))
        x1, y1, x2, y2 = track["bbox"]
        if self.predict_motion and dt > 0.0:
            dx = int(round(float(track.get("vx", 0.0)) * min(dt, 0.20)))
            dy = int(round(float(track.get("vy", 0.0)) * min(dt, 0.20)))
        else:
            dx = dy = 0

        pred_box = self._clip_box((x1 + dx, y1 + dy, x2 + dx, y2 + dy), w, h)
        track["bbox"] = pred_box
        track["center"] = self._center(pred_box)
        track["conf"] = max(0.05, float(track.get("conf", 0.0)) * self.conf_decay)
        track["miss_count"] = int(track.get("miss_count", 0)) + 1
        track["last_update"] = float(now)
        track["is_memory"] = True
        track["source"] = "memory"
        return track

    def update(self, detections, frame_shape):
        now = time.time()
        h, w = frame_shape[:2]
        detections = list(detections or [])
        tracks = list(self.tracks)

        detections.sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)

        used_track_ids = set()
        used_det_ids = set()
        gate2 = self.match_dist_px * self.match_dist_px

        for di, det in enumerate(detections):
            det_bbox = self._clip_box(det.get("bbox", (0, 0, 1, 1)), w, h)
            det_center = det.get("center") or self._center(det_bbox)

            best_ti = None
            best_d2 = None

            for ti, tr in enumerate(tracks):
                if ti in used_track_ids:
                    continue
                if int(det.get("cls", -999)) != int(tr.get("cls", -999)):
                    continue
                d2 = self._dist2(det_center, tr.get("center", det_center))
                if d2 <= gate2 and (best_d2 is None or d2 < best_d2):
                    best_d2 = d2
                    best_ti = ti

            if best_ti is None:
                continue

            tr = tracks[best_ti]
            dt = max(1e-6, now - float(tr.get("last_seen", now)))
            prev_center = tr.get("center", det_center)

            tr["bbox"] = det_bbox
            tr["center"] = (int(det_center[0]), int(det_center[1]))
            tr["conf"] = float(det.get("conf", tr.get("conf", 0.0)))
            tr["cls"] = int(det.get("cls", tr.get("cls", 0)))
            tr["vx"] = (float(tr["center"][0]) - float(prev_center[0])) / dt
            tr["vy"] = (float(tr["center"][1]) - float(prev_center[1])) / dt
            tr["miss_count"] = 0
            tr["last_seen"] = float(now)
            tr["last_update"] = float(now)
            tr["is_memory"] = False
            tr["source"] = "detected"

            used_track_ids.add(best_ti)
            used_det_ids.add(di)

        for di, det in enumerate(detections):
            if di in used_det_ids:
                continue
            tracks.append(self._new_track(det, now, w, h))

        kept = []
        for ti, tr in enumerate(tracks):
            if ti in used_track_ids or tr.get("last_seen", 0.0) == now:
                kept.append(tr)
                continue

            age = float(now) - float(tr.get("last_seen", now))
            miss_count = int(tr.get("miss_count", 0))
            if miss_count < self.max_missing_frames and age <= self.max_hold_sec:
                kept.append(self._predict_track(tr, now, w, h))

        kept.sort(
            key=lambda tr: (
                0 if tr.get("is_memory", False) else 1,
                float(tr.get("conf", 0.0)),
                -float(tr.get("miss_count", 0)),
            ),
            reverse=True,
        )
        self.tracks = kept[: self.max_tracks]
        return [self._make_det(tr) for tr in self.tracks]


