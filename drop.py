import serial
import time
import os
import time
import threading
import warnings
import logging
import serial
from dataclasses import dataclass
from pathlib import Path


warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("dropball")


class BallDropper:
    def __init__(self, port, baudrate=9600, connect_on_init=True):
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.ser = None
        if connect_on_init:
            self.connect()

    def connect(self):
        try:
            if self.ser is not None and getattr(self.ser, "is_open", False):
                return True
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2.0)  # Arduino Nano/Uno thuong reset khi mo cong serial
            log.info(f"[DROP] Serial connected: {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            self.ser = None
            log.error(f"[DROP] Serial connect failed ({self.port}): {e}")
            return False

    def _write_and_readline(self, payload: bytes):
        if self.ser is None or not getattr(self.ser, "is_open", False):
            if not self.connect():
                return False, "serial_not_ready"
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        try:
            self.ser.write(payload)
            self.ser.flush()
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                line = ""
            return True, line
        except Exception as e:
            log.error(f"[DROP] Serial write failed: {e}")
            return False, str(e)

    def mo_tung_cai(self):
        ok, line = self._write_and_readline(b'1')
        if ok:
            log.info(f"[DROP] command=1 sent ack='{line}'")
        return ok

    def dong_tat_ca(self):
        ok, line = self._write_and_readline(b'0')
        if ok:
            log.info(f"[DROP] command=0 sent ack='{line}'")
        return ok

    def close(self):
        try:
            if self.ser is not None:
                self.ser.close()
                log.info("[DROP] Serial closed")
        except Exception as e:
            log.error(f"[DROP] Serial close failed: {e}")
        finally:
            self.ser = None



# from ball_dropper import BallDropper

# # Khởi tạo đối tượng (Thay COM3 bằng cổng thực tế của bạn)
# dropper = BallDropper(port='COMxx')
# dropper.mo_tung_cai()