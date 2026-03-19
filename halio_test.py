import time
import cv2
from pathlib import Path
from threading import Thread
from common import (
    HailoPythonInferenceEngine, DetectionPostProcessor,
    scale_detections_to_original
)

MODEL_PATH = "models/model.hef"
VIDEO_SOURCE = "/home/pi/Documents/test/video/record_20260226_165922.mp4" 
INPUT_SIZE = (640, 640)

#WAIT TIME
TARGET_FPS = 30.0
FRAME_TIME = 1.0 / TARGET_FPS  # ~0.033s (33ms)

#CONFIG FOR MODEL
VERBOSE = False
NORMALIZE = False
CONFIDENCE_THRESHOLD = 0.5 # Nguong tin cay de loc cac vat the (50%)

# --- CLASS CAMERA ĐA LUỒNG (Giữ nguyên) ---
class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream = cv2.VideoCapture(stream_id)
        # Cấu hình Camera
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30) # Cài đặt phần cứng camera là 30
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    """Run detection on single image using hybrid Hailo + Python head pipeline"""
    #LOAD MODEL
    engine = HailoPythonInferenceEngine(MODEL_PATH)
    
    #RUN CAMERA
    webcam = WebcamStream(stream_id=0).start()
    time.sleep(1.0)
    try:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        while True:
            t_start = time.perf_counter()
            # Load original image for visualization
            # orig_image = HailoPythonInferenceEngine.load_image(args.image)
            # orig_image = webcam.read()
            _, orig_image = cap.read()

            if orig_image is None:
                print("\tNo Frame!")
                continue
            
            orig_h, orig_w = orig_image.shape[:2]
            
            # Preprocess for inference
            input_data, orig_size, scale, pad_w, pad_h = HailoPythonInferenceEngine.preprocess(orig_image, normalize=NORMALIZE)
            
            # Run inference
            results, stats = engine.infer(input_data, verbose=VERBOSE, save_output=True, conf_threshold=CONFIDENCE_THRESHOLD)
            
            # Calculate time-consuming
            total_time = time.perf_counter() - t_start
            wait_time = FRAME_TIME - total_time

            if wait_time > 0:
                # Nếu xử lý nhanh quá, thì ngủ cho đủ thời gian còn lại
                time.sleep(wait_time)
                actual_fps = TARGET_FPS
            else:
                # Nếu xử lý chậm hơn 33ms (Drop frame)
                actual_fps = 1.0 / total_time
                print(f"Cảnh báo: Drop FPS! Xử lý mất {total_time*1000:.1f}ms")

            # Scale detections to original image space
            results = scale_detections_to_original(results, orig_h, orig_w, scale, pad_w, pad_h)
            
            # Draw bboxes on original image
            output_image = DetectionPostProcessor.draw_bboxes(orig_image, results, thickness=2)
            # Hiển thị FPS thực tế
            cv2.putText(output_image, f"FPS: {int(actual_fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Hiển thị tải CPU (Load)
            cpu_load = (total_time / FRAME_TIME) * 100
            cv2.putText(output_image, f"Load: {int(cpu_load)}%", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("UAV CONTEST", output_image)
            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        webcam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()