import cv2
import time
from threading import Thread
from ultralytics import YOLO

# --- CẤU HÌNH ---
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS  # ~0.033s (33ms)

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

# --- MAIN LOOP ---
def main():
    # Load Model
    model_path = "models/model4_openvino_model" 
    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        print(f"Lỗi: {e}")
        return

    # Khởi động Camera
    webcam = WebcamStream(stream_id=0).start()
    time.sleep(1.0) 

    # frame = cv2.imread("datasets/datasets_yolo26/train/images/img719_jpg.rf.11864da677076996341962620d219394.jpg")

    try:
        while True:
            # 1. Bấm giờ bắt đầu
            start_time = time.time()

            # 2. Lấy frame & Xử lý
            frame = webcam.read()
            if frame is None: continue

            # Chạy YOLO
            results = model.predict(
                source=frame,
                imgsz=320,
                conf=0.5,
                max_det=1,
                agnostic_nms=True, 
                verbose=False,
                stream=False
            )

            # Vẽ kết quả
            annotated_frame = results[0].plot()

            # 3. Tính toán thời gian đã tiêu tốn
            process_time = time.time() - start_time
            
            # 4. Logic Hãm FPS (Frame Pacing)
            wait_time = FRAME_TIME - process_time
            
            if wait_time > 0:
                # Nếu xử lý nhanh quá, thì ngủ cho đủ thời gian còn lại
                time.sleep(wait_time)
                actual_fps = TARGET_FPS
            else:
                # Nếu xử lý chậm hơn 33ms (Drop frame)
                actual_fps = 1.0 / process_time
                print(f"Cảnh báo: Drop FPS! Xử lý mất {process_time*1000:.1f}ms")

            # Hiển thị FPS thực tế
            cv2.putText(annotated_frame, f"FPS: {int(actual_fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Hiển thị tải CPU (Load)
            cpu_load = (process_time / FRAME_TIME) * 100
            cv2.putText(annotated_frame, f"Load: {int(cpu_load)}%", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('UAV Tracking', annotated_frame)

            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        webcam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()