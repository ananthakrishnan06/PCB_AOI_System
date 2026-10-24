"""
PCB Live Detection — Smooth 3-stage pipeline.

The key insight
---------------
  Previous approach stored an 'annotated frame' from YOLO's result.plot().
  The display loop re-showed the same static annotated image until the next
  inference finished → video appeared frozen / laggy between detections.

  This version:
    Stage 1  CameraReader  thread → always stores the LATEST raw frame
    Stage 2  InferenceWorker thread → runs YOLO, stores raw DETECTION DATA
                                      (box coords + labels) — NOT an image
    Stage 3  Display loop (main)  → grabs freshest RAW frame from Stage 1,
                                    draws latest boxes from Stage 2 onto it,
                                    shows at full 30 FPS — always live video

  Result: video is always real-time.  Boxes refresh at inference rate.

Controls  q = quit,  s = save frame
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "runs" / "detect" / "train-7" / "weights" / "best.pt"
OUTPUT_DIR   = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CAMERA_INDEX       = 0
CONFIDENCE         = 0.25
INFERENCE_DEVICE   = 0 if torch.cuda.is_available() else "cpu"
USE_HALF_PRECISION = torch.cuda.is_available()
INFER_IMGSZ        = 640   # lower = faster inference (try 416 or 320 for speed)

# Per-class box colours (BGR)
CLASS_COLORS = {
    "open":             (0,   0,   255),
    "short":            (255, 0,   0  ),
    "mouse_bite":       (0,   165, 255),
    "spur":             (0,   200, 0  ),
    "spurious_copper":  (255, 0,   255),
    "missing_hole":     (0,   255, 255),
}
DEFAULT_COLOR = (200, 200, 200)

# ─── Load model ───────────────────────────────────────────────────────────────
print("Loading YOLO model...")
model = YOLO(str(MODEL_PATH))
if USE_HALF_PRECISION:
    model.to("cuda:0")

# Warmup — eliminates the first-frame latency spike on CUDA
_w = np.zeros((INFER_IMGSZ, INFER_IMGSZ, 3), dtype=np.uint8)
model.predict(source=_w, conf=0.99, verbose=False,
              device=INFERENCE_DEVICE, half=USE_HALF_PRECISION, imgsz=INFER_IMGSZ)
del _w
print("Model ready!\n")


# ─── Stage 1: Camera reader ───────────────────────────────────────────────────
class CameraReader(threading.Thread):
    """
    Daemon thread that reads frames as fast as the camera allows.
    Always overwrites self._frame with the latest one so consumers
    never process stale data from the OS buffer.
    """

    def __init__(self, idx: int = 0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {idx}.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep OS buffer minimal
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
                self._frame = frame             # overwrite — latest wins

    @property
    def frame(self):
        """Return a copy of the most recent raw frame."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()


# ─── Stage 2: Inference worker ────────────────────────────────────────────────
class InferenceWorker(threading.Thread):
    """
    Daemon thread that runs YOLO on the latest raw frame and stores
    only the RAW DETECTION DATA — never an annotated image.

    Storing coordinates instead of a rendered image means the display
    thread can apply the boxes to a brand-new raw frame every render
    cycle, keeping the video perfectly live.
    """

    def __init__(self, camera: CameraReader):
        super().__init__(daemon=True)
        self.camera     = camera
        # Each detection: (x1, y1, x2, y2, confidence_float, class_name_str)
        self._dets      = []
        self._lock      = threading.Lock()
        self.stopped    = False
        self.fps        = 0.0
        self.infer_ms   = 0.0
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
            results = model.predict(
                source=frame,
                conf=CONFIDENCE,
                verbose=False,
                device=INFERENCE_DEVICE,
                half=USE_HALF_PRECISION,
                imgsz=INFER_IMGSZ,
            )
            t1 = time.perf_counter()

            # Extract raw numbers only — no image rendering here
            dets = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                name = model.names[int(box.cls[0])]
                dets.append((x1, y1, x2, y2, conf, name))

            # Rolling inference stats
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
            return list(self._dets)     # shallow copy, safe to use

    def stop(self):
        self.stopped = True


# ─── Box drawing helper ───────────────────────────────────────────────────────
def draw_detections(frame: np.ndarray, dets: list) -> np.ndarray:
    """
    Draw bounding boxes directly onto a raw frame using fast OpenCV
    primitives.  Much faster than YOLO's result.plot() because it
    avoids PIL / matplotlib rendering.
    """
    for (x1, y1, x2, y2, conf, name) in dets:
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)
        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Label background + text
        label = f"{name}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


# ─── Stage 3: Display loop ────────────────────────────────────────────────────
def run():
    camera    = CameraReader(CAMERA_INDEX)
    inference = InferenceWorker(camera)

    # Wait for the first raw frame
    print("Waiting for camera...")
    while camera.frame is None:
        if camera.stopped:
            print("Camera failed to open.")
            return
        time.sleep(0.01)
    print("Live feed active. Press 'q' to quit, 's' to save.\n")

    disp_prev  = time.perf_counter()
    disp_times: list[float] = []

    while True:
        # ── Always grab the freshest raw frame ──────────────────────────────
        frame = camera.frame
        if frame is None or camera.stopped:
            break

        # ── Overlay the most recent detection boxes (fast OpenCV draw) ──────
        dets  = inference.detections
        frame = draw_detections(frame, dets)

        total        = len(dets)
        status       = "FAIL" if total > 0 else "PASS"
        status_color = (0, 0, 255) if total > 0 else (0, 255, 0)

        # ── Display-rate FPS (independent of inference) ─────────────────────
        now = time.perf_counter()
        disp_times.append(1.0 / max(now - disp_prev, 1e-6))
        disp_prev = now
        if len(disp_times) > 30:
            disp_times.pop(0)
        disp_fps = sum(disp_times) / len(disp_times)

        # ── HUD overlay ──────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (450, 170), (12, 12, 12), cv2.FILLED)

        cv2.putText(frame, "PCB LIVE INSPECTION",
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Defects  : {total}",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Status   : {status}",
                    (10, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"Disp FPS : {disp_fps:5.1f}",
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Det  FPS : {inference.fps:5.1f}  ({inference.infer_ms:.0f} ms)",
                    (10, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)

        # ── Show & key handling ──────────────────────────────────────────────
        cv2.imshow("PCB Live Inspection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            ts       = int(time.time())
            filename = OUTPUT_DIR / f"capture_{ts}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"Saved: {filename}")

    inference.stop()
    camera.stop()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    run()
