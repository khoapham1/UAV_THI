"""Common utilities for Hailo-8L inference on YOLO26 (NMS-free dual-head model)"""

import numpy as np
import cv2
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List
from hailo_platform import VDevice, HEF, ConfigureParams, InputVStreamParams, OutputVStreamParams, InferVStreams, HailoStreamInterface, FormatType


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Vectorized sigmoid"""
    # Tránh cảnh báo tràn số khi np.exp nhận số âm lớn
    return 1.0 / (1.0 + np.exp(np.clip(-x, -88.72, 88.72)))


# ============================================================================
# Common Image Operations
# ============================================================================

def letterbox_image(img, target_size=640, color=(114, 114, 114)):
    """Resize image with aspect ratio preservation (letterbox)"""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Dùng nội suy nhanh hơn cho tác vụ realtime
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    padded = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return padded, scale, pad_w, pad_h

def load_and_preprocess_image(img, target_size: int = 640, normalize: bool = False) -> Tuple[np.ndarray, Tuple[int, int], float, int, int]:
    # YOLO models expect RGB, but cv2.imread loads as BGR, so convert
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    orig_h, orig_w = img.shape[:2]
    
    padded, scale, pad_w, pad_h = letterbox_image(img, target_size)
    
    if normalize:
        padded = padded.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(padded, axis=0)
    else:
        input_tensor = np.expand_dims(padded, axis=0).astype(np.uint8)
    
    return input_tensor, (orig_h, orig_w), scale, pad_w, pad_h

def scale_detections_to_original(detections: List[dict], orig_h: int, orig_w: int, scale: float, pad_w: int, pad_h: int) -> List[dict]:
    """Scale detection coordinates from inference space (640x640) to original image space
    
    Args:
        detections: List of detection dicts with x1, y1, x2, y2 coordinates
        orig_h, orig_w: Original image dimensions
        scale: Scale factor used in preprocessing
        pad_w, pad_h: Padding used in preprocessing
    """
    for det in detections:
        # 1. Reverse padding, 2. Reverse scaling
        det['x1'] = max(0, min((det['x1'] - pad_w) / scale, orig_w))
        det['y1'] = max(0, min((det['y1'] - pad_h) / scale, orig_h))
        det['x2'] = max(0, min((det['x2'] - pad_w) / scale, orig_w))
        det['y2'] = max(0, min((det['y2'] - pad_h) / scale, orig_h))
        
    return detections

@dataclass
class InferenceStats:
    """Runtime statistics for inference"""
    preprocess_time: float
    hailo_inference_time: float
    postprocess_time: float
    total_time: float
    hailo_output_shape: str
    final_output_shape: str


class HailoPythonInferenceEngine:
    """Encapsulates Hailo-8L backbone + Python head inference"""
    
    def __init__(self, hef_path: str):
        """Initialize the hybrid inference engine"""
        self.hef_path = hef_path
        
        # Initialize Hailo
        self.target = VDevice()
        self.hef = HEF(hef_path)
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        
        # Setup vstreams
        self.input_vstream_params = InputVStreamParams.make(self.network_group)
        self.output_vstream_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        
        # Cache trước bảng tên để tránh việc khai báo dictionary liên tục trong vòng lặp (Tối ưu memory)
        self.name_map = {
            'yolo26n/conv64': 'cls_80',
            'yolo26n/conv61': 'reg_80',
            'yolo26n/conv80': 'cls_40',
            'yolo26n/conv77': 'reg_40',
            'yolo26n/conv94': 'cls_20',
            'yolo26n/conv91': 'reg_20',
        }

        print(f"✓ Hailo engine initialized: {hef_path}")
        print(f"✓ Using Python head for post-processing.")
    
    @staticmethod
    def preprocess(img, width: int = 640, height: int = 640, normalize: bool = False) -> Tuple[np.ndarray, Tuple[int, int], float, int, int]:
        """Preprocess image to tensor and return original dimensions + stats"""
        return load_and_preprocess_image(img, width, normalize)
    
    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        """Load and read original image"""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return img

    #FOR 4 classes
    def _run_python_head(self, dequantized_results: Dict, conf_threshold: float) -> List[dict]:
        """Run python head logic on dequantized Hailo outputs (vectorized)."""
        
        tensors = {}
        for name, data in dequantized_results.items():
            if name in self.name_map:
                tensors[self.name_map[name]] = data
        
        required_tensors = ['cls_80', 'cls_40', 'cls_20', 'reg_80', 'reg_40', 'reg_20']
        if len(tensors) < 6:
            missing = [t for t in required_tensors if t not in tensors]
            print(f"Error: Missing tensors: {missing}")
            return []

        STRIDES = [8, 16, 32]
        GRID_SIZES = [80, 40, 20]
        
        # 2. LỚP BẢO VỆ
        is_probability = False
        test_cls = tensors['cls_80'][0]
        if test_cls.min() >= 0.0 and test_cls.max() <= 1.0:
            is_probability = True
            
        logit_threshold = -np.log(1.0 / conf_threshold - 1.0)
        
        # Khởi tạo danh sách kết quả thô
        raw_x = []
        raw_y = []
        raw_w = []
        raw_h = []
        raw_scores = []
        raw_cls = []
        
        for scale_idx in range(len(STRIDES)):
            stride = STRIDES[scale_idx]
            grid_dim = GRID_SIZES[scale_idx]
            
            cls_data = tensors[f'cls_{grid_dim}'][0]  
            reg_data = tensors[f'reg_{grid_dim}'][0]  
            
            cls_flat = cls_data.reshape(-1, 4)
            reg_flat = reg_data.reshape(-1, 4)
            
            max_logits = cls_flat.max(axis=1)       
            class_ids = cls_flat.argmax(axis=1)      
            
            # Tính toán mặt nạ (Mask)
            if is_probability:
                mask = max_logits > conf_threshold
            else:
                mask = max_logits > logit_threshold
                
            if not mask.any():
                continue
            
            indices = np.where(mask)[0]
            
            # Tính điểm số tự tin
            if is_probability:
                scores = max_logits[indices]
            else:
                scores = sigmoid(max_logits[indices])
                
            cls = class_ids[indices]
            
            # Giải mã tọa độ theo dạng mảng Numpy (Nhanh hơn gấp 3 lần so với vòng for)
            rows = indices // grid_dim
            cols = indices % grid_dim
            
            l = reg_flat[indices, 0]
            t = reg_flat[indices, 1]
            r = reg_flat[indices, 2]
            b = reg_flat[indices, 3]
            
            x1 = (cols + 0.5 - l) * stride
            y1 = (rows + 0.5 - t) * stride
            x2 = (cols + 0.5 + r) * stride
            y2 = (rows + 0.5 + b) * stride
            
            # Lưu trữ tọa độ dưới dạng [x, y, w, h] chuẩn bị cho NMS
            raw_x.extend(x1.tolist())
            raw_y.extend(y1.tolist())
            raw_w.extend((x2 - x1).tolist())
            raw_h.extend((y2 - y1).tolist())
            raw_scores.extend(scores.tolist())
            raw_cls.extend(cls.tolist())
        
        # 3. HẬU XỬ LÝ: NMS Tối ưu hóa mảng
        final_results = []
        if len(raw_x) > 0:
            # Tạo list of lists trong 1 dòng duy nhất bằng list comprehension
            boxes_for_nms = [[float(raw_x[i]), float(raw_y[i]), float(raw_w[i]), float(raw_h[i])] for i in range(len(raw_x))]
            scores_for_nms = [float(s) for s in raw_scores]
            
            nms_indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, conf_threshold, 0.45)
            
            coco_classes = DetectionPostProcessor._load_coco_classes()
            if len(nms_indices) > 0:
                for i in nms_indices.flatten():
                    final_results.append({
                        'x1': boxes_for_nms[i][0],
                        'y1': boxes_for_nms[i][1],
                        'x2': boxes_for_nms[i][0] + boxes_for_nms[i][2],
                        'y2': boxes_for_nms[i][1] + boxes_for_nms[i][3],
                        'conf': scores_for_nms[i],
                        'cls_id': raw_cls[i],
                        'cls_name': coco_classes.get(raw_cls[i], 'N/A')
                    })
                    
        return final_results
    
    def infer(self, input_data: np.ndarray, verbose: bool = False, save_output: bool = False, conf_threshold: float = 0.5) -> Tuple[List[dict], InferenceStats]:
        """Run hybrid inference pipeline with Python head"""
        stats = InferenceStats(
            preprocess_time=0, hailo_inference_time=0, 
            postprocess_time=0, total_time=0,
            hailo_output_shape="", final_output_shape=""
        )
        
        t_start = time.perf_counter()
        
        if verbose:
            print(f"[INFERENCE] Input shape: {input_data.shape}, dtype: {input_data.dtype}")
        
        with self.network_group.activate() as active_group:
            with InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params) as infer_pipeline:
                
                # A. Hailo Backbone Inference
                if verbose: print(f"[STAGE 1] Running Hailo backbone...")
                t_hailo = time.perf_counter()
                hailo_results = infer_pipeline.infer(input_data)
                stats.hailo_inference_time = time.perf_counter() - t_hailo
                stats.hailo_output_shape = str({k: v.shape for k, v in hailo_results.items()})
                if verbose: print(f"  ✓ Hailo inference: {stats.hailo_inference_time*1000:.2f}ms")

                # B. Python Head
                if verbose: print(f"[STAGE 2] Running Python Head...")
                t_post = time.perf_counter()
                
                detections = self._run_python_head(hailo_results, conf_threshold)
                stats.postprocess_time = time.perf_counter() - t_post
                stats.final_output_shape = f"{len(detections)} detections"
                if verbose: print(f"  ✓ Python head: {stats.postprocess_time*1000:.2f}ms")

        stats.total_time = time.perf_counter() - t_start
        
        # ====================================================================
        # ĐOẠN CODE THÊM MỚI: IN RESULT VÀ TÍNH TRUNG BÌNH DETECTIONS 0 -> 5
        # ====================================================================
        if verbose:
            print(f"\n[SUMMARY] Pipeline timing:")
            print(f"  Hailo:   {stats.hailo_inference_time*1000:7.2f}ms")
            print(f"  PyHead:  {stats.postprocess_time*1000:7.2f}ms")
            print(f"  Total:   {stats.total_time*1000:7.2f}ms")
            
            print(f"\n[INSPECT RESULTS] Total Detections: {len(detections)}")
            if len(detections) > 0:
                print("  -> Cấu trúc của 1 result (Detection 0):")
                print(f"     {detections[0]}")
                
                # Lấy ra tối đa 6 detections đầu tiên (0, 1, 2, 3, 4, 5)
                limit = min(6, len(detections))
                subset = detections[:limit]
                
                # Tính tổng và trung bình
                avg_x1 = sum(d['x1'] for d in subset) / limit
                avg_y1 = sum(d['y1'] for d in subset) / limit
                avg_x2 = sum(d['x2'] for d in subset) / limit
                avg_y2 = sum(d['y2'] for d in subset) / limit
                avg_conf = sum(d['conf'] for d in subset) / limit
                
                print(f"\n  -> Trung bình các chỉ số của {limit} detections đầu tiên:")
                print(f"     Average x1   : {avg_x1:.2f}")
                print(f"     Average y1   : {avg_y1:.2f}")
                print(f"     Average x2   : {avg_x2:.2f}")
                print(f"     Average y2   : {avg_y2:.2f}")
                print(f"     Average conf : {avg_conf:.4f}")
            else:
                print("  -> Không có detection nào vượt qua conf_threshold!")
        # ====================================================================
        
        return detections, stats

# ONNX / Hybrid engines removed to reduce dependencies

class DetectionPostProcessor:
    """Postprocess detections and draw bboxes using YOLO COCO classes"""
    
    COCO_CLASSES = None
    
    @classmethod
    def _load_coco_classes(cls):
        """Return COCO class names (hardcoded to avoid dependencies)"""
        if cls.COCO_CLASSES is not None:
            return cls.COCO_CLASSES
        
        cls.COCO_CLASSES = {
            0: 'blue', 1: 'h_marker', 2: 'red', 3: 'yellow'
        }
        
        return cls.COCO_CLASSES
    
    @staticmethod
    def postprocess(detections: np.ndarray, conf_threshold: float = 0.5) -> List[dict]:
        """Parse detections and filter by confidence
        
        Args:
            detections: Shape (num_detections, 6) - [x1, y1, x2, y2, conf, cls]
            conf_threshold: Confidence threshold
            
        Returns:
            List of dicts with keys: [x1, y1, x2, y2, conf, cls_id, cls_name]
        """
        classes = DetectionPostProcessor._load_coco_classes()
        results = []
        
        for det in detections:
            print(det)
            conf = det[4]
            if conf >= conf_threshold:
                cls_id = int(det[5])
                cls_name = classes.get(cls_id) if isinstance(classes, dict) else classes[cls_id]
                results.append({
                    'x1': float(det[0]),
                    'y1': float(det[1]),
                    'x2': float(det[2]),
                    'y2': float(det[3]),
                    'conf': float(conf),
                    'cls_id': cls_id,
                    'cls_name': cls_name
                })
        return results
    
    @staticmethod
    def draw_bboxes(image: np.ndarray, detections: list, thickness: int = 2) -> np.ndarray:
        """Draw bounding boxes on image"""
        img = image # Bỏ hàm .copy() để giảm thiểu việc tạo bản sao ảnh, ghi đè trực tiếp lên RAM
        h, w = img.shape[:2]
        
        colors = [(255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
        
        for det in detections:
            x1 = int(max(0, det['x1']))
            y1 = int(max(0, det['y1']))
            x2 = int(min(w, det['x2']))
            y2 = int(min(h, det['y2']))
            
            color = colors[det['cls_id']]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{det['cls_name']} {det['conf']:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img