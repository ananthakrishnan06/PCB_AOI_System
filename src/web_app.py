"""Local browser interface for the PCB AOI system."""

import base64
from collections import deque
from datetime import datetime
import queue
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from detector import detect_image


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
SCAN_HISTORY = deque(maxlen=50)


class DedicatedInferenceWorker:
    """
    Runs YOLO inference in a single persistent background thread.
    This avoids PyTorch CUDA stream/context re-initialization penalty
    (1.5 - 3.7 seconds) that happens on every new HTTP worker thread.
    """
    def __init__(self):
        self.work_queue = queue.Queue(maxsize=3)
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()

    def _run(self):
        # Warmup inside this dedicated worker thread to initialize CUDA
        dummy = np.zeros((540, 960, 3), dtype=np.uint8)
        try:
            detect_image(dummy, confidence=0.99, return_annotated=False)
        except Exception:
            pass

        while True:
            image, confidence, return_annotated, reply_queue = self.work_queue.get()
            t0 = time.perf_counter()
            try:
                res = detect_image(image, confidence=confidence, return_annotated=return_annotated)
                elapsed = time.perf_counter() - t0
                reply_queue.put((res, elapsed, None))
            except Exception as err:
                elapsed = time.perf_counter() - t0
                reply_queue.put((None, elapsed, str(err)))

    def inspect(self, image, confidence, return_annotated=True, timeout=5.0):
        # Drop stale frames if backlog builds up from live camera
        while self.work_queue.qsize() >= 2:
            try:
                dropped = self.work_queue.get_nowait()
                dropped[3].put((None, 0.0, "Skipped stale frame"))
            except queue.Empty:
                break

        reply_queue = queue.Queue(maxsize=1)
        self.work_queue.put((image, confidence, return_annotated, reply_queue))
        res, elapsed, err = reply_queue.get(timeout=timeout)
        if err:
            raise RuntimeError(err)
        return res, elapsed


INFERENCE_WORKER = DedicatedInferenceWorker()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_data_url(image) -> str:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not success:
        raise ValueError("Could not prepare the annotated image.")
    encoded_text = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_text}"


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/history")
def history():
    return jsonify(scans=list(SCAN_HISTORY))


@app.post("/api/inspect")
def inspect():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="Select a PCB image before starting an inspection."), 400
    if not allowed_file(upload.filename):
        return jsonify(error="Use a JPG, JPEG, PNG, BMP, or WEBP image."), 400

    try:
        confidence = float(request.form.get("confidence", 25)) / 100
        confidence = min(max(confidence, 0.01), 0.99)
    except ValueError:
        return jsonify(error="Confidence must be a number between 1 and 99."), 400

    is_live = request.form.get("live", "").lower() in ("true", "1")

    file_name = secure_filename(upload.filename)
    encoded_image = np.frombuffer(upload.read(), dtype=np.uint8)
    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(error="The uploaded file could not be decoded as an image."), 400

    try:
        # For live inspection, skip server-side plot and JPEG encoding for ultra-fast response
        result, elapsed = INFERENCE_WORKER.inspect(
            image, confidence=confidence, return_annotated=not is_live
        )

        response = {
            "status": result["status"],
            "total": result["total"],
            "summary": result["summary"],
            "defects": result["defects"],
            "processing_seconds": round(elapsed, 3),
            "image_name": file_name,
            "image_width": result["image_width"],
            "image_height": result["image_height"],
        }

        if not is_live and result["annotated"] is not None:
            response["annotated_image"] = image_data_url(result["annotated"])

            SCAN_HISTORY.appendleft({
                "timestamp": datetime.now().strftime("%d %b %Y, %H:%M:%S"),
                "image_name": file_name,
                "status": result["status"],
                "total": result["total"],
                "processing_seconds": round(elapsed, 3),
                "confidence": round(confidence * 100),
                "summary": result["summary"],
            })

        return jsonify(response)
    except Exception as error:
        return jsonify(error=f"Inspection could not be completed: {error}"), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="Image is too large. The maximum upload size is 16 MB."), 413


if __name__ == "__main__":
    print("Dedicated YOLO worker active; starting AOI web server on port 5000.")
    app.run(host="127.0.0.1", port=5000, debug=False)
