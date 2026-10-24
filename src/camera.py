"""
Camera module for the PCB AOI desktop GUI — smooth live detection.

Same 3-stage pipeline as live_detection.py:
  Stage 1  CameraReader     — always exposes the latest raw frame
  Stage 2  InferenceWorker  — stores raw detection coords (NOT an annotated image)
  Stage 3  display loop     — draws coords on each fresh raw frame → smooth 30 FPS video

Reuses detector.model so no second YOLO instance is loaded into VRAM.
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from detector import model as yolo_model   # reuse — saves GPU VRAM

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT       = Path(__file__).resolve().parent.parent
CONFIDENCE         = 0.25
INFERENCE_DEVICE   = 0 if torch.cuda.is_available() else "cpu"
USE_HALF_PRECISION = torch.cuda.is_available()
INFER_IMGSZ        = 640

CLASS_COLORS = {
    "open":             (0,   0,   255),
    "short":            (255, 0,   0  ),
    "mouse_bite":       (0,   165, 255),
    "spur":             (0,   200, 0  ),
    "spurious_copper":  (255, 0,   255),
    "missing_hole":     (0,   255, 255),
}
DEFAULT_COLOR = (200, 200, 200)


# ─── Stage 1: Camera reader ───────────────────────────────────────────────────
class CameraReader(threading.Thread):
    def __init__(self, idx: int = 0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {idx}.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame  = None
        self._lock   = threading.Lock()
        self.stopped = False
        self.start()

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            with self._lock:
                self._frame = frame

    @property
    def frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()


# ─── Stage 2: Inference worker ────────────────────────────────────────────────
class InferenceWorker(threading.Thread):
    """
    Stores raw detection data (box coords + labels), NOT an annotated image.
    The display loop applies these coords to each fresh camera frame so
    the video is always live even between inference cycles.
    """

    def __init__(self, camera: CameraReader):
        super().__init__(daemon=True)
        self.camera   = camera
        self._dets    = []
        self._lock    = threading.Lock()
        self.stopped  = False
        self.fps      = 0.0
        self.infer_ms = 0.0
        self._times: list[float] = []
        self.start()

    def run(self):
        prev = time.perf_counter()
        while not self.stopped:
            frame = self.camera.frame
            if frame is None:
                time.sleep(0.005)
                continue

            t0 = time.perf_counter()
            results = yolo_model.predict(
                source=frame,
                conf=CONFIDENCE,
                verbose=False,
                device=INFERENCE_DEVICE,
                half=USE_HALF_PRECISION,
                imgsz=INFER_IMGSZ,
            )
            t1 = time.perf_counter()

            dets = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                name = yolo_model.names[int(box.cls[0])]
                dets.append((x1, y1, x2, y2, conf, name))

            ms = (t1 - t0) * 1000
            self._times.append(ms)
            if len(self._times) > 20:
                self._times.pop(0)
            self.infer_ms = sum(self._times) / len(self._times)

            now      = time.perf_counter()
            self.fps = 1.0 / max(now - prev, 1e-6)
            prev     = now

            with self._lock:
                self._dets = dets

    @property
    def detections(self) -> list:
        with self._lock:
            return list(self._dets)

    def stop(self):
        self.stopped = True


# ─── Box drawing helper ───────────────────────────────────────────────────────
def draw_detections(frame: np.ndarray, dets: list) -> np.ndarray:
    for (x1, y1, x2, y2, conf, name) in dets:
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


# ─── Stage 3: Display loop ────────────────────────────────────────────────────
def start_live_camera():
    """
    Blocking display loop — run from a daemon thread in the GUI.
    Press q to quit, s to save the current frame.
    """
    try:
        camera = CameraReader(0)
    except RuntimeError as exc:
        print(f"Unable to open camera: {exc}")
        return

    inference = InferenceWorker(camera)

    while camera.frame is None:
        if camera.stopped:
            return
        time.sleep(0.01)

    save_folder = PROJECT_ROOT / "output"
    save_folder.mkdir(exist_ok=True)

    disp_prev  = time.perf_counter()
    disp_times: list[float] = []

    while True:
        frame = camera.frame          # freshest raw frame — never stale
        if frame is None or camera.stopped:
            break

        dets  = inference.detections  # latest box coords from background thread
        frame = draw_detections(frame, dets)

        total        = len(dets)
        status       = "FAIL" if total > 0 else "PASS"
        status_color = (0, 0, 255) if total > 0 else (0, 255, 0)

        now = time.perf_counter()
        disp_times.append(1.0 / max(now - disp_prev, 1e-6))
        disp_prev = now
        if len(disp_times) > 30:
            disp_times.pop(0)
        disp_fps = sum(disp_times) / len(disp_times)

        cv2.rectangle(frame, (0, 0), (450, 170), (12, 12, 12), cv2.FILLED)
        cv2.putText(frame, "PCB LIVE INSPECTION",
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Total Defects : {total}",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Status        : {status}",
                    (10, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Disp FPS : {disp_fps:5.1f}",
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Det  FPS : {inference.fps:5.1f}  ({inference.infer_ms:.0f} ms)",
                    (10, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("PCB Live Inspection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            filename = save_folder / f"capture_{int(time.time())}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"Saved : {filename}")

    inference.stop()
    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_live_camera()