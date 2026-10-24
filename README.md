# PCB Automatic Optical Inspection (AOI) System

AI-powered Automated Optical Inspection (AOI) system for detecting and classifying defects on Printed Circuit Boards (PCBs) in real time. Built with **YOLOv11**, **PyTorch (CUDA FP16)**, **OpenCV**, **Flask**, and **CustomTkinter**.

---

## Key Features

- **Real-Time 30 FPS Live Inspection**: Decoupled asynchronous architecture prevents camera lag and displays live video with high-framerate bounding box tracking.
- **Three Interfaces**:
  - **Modern Web Dashboard**: Real-time browser console featuring dual-viewport camera view, confidence controls, taxonomy breakdown, and CSV export.
  - **Desktop GUI**: Clean, dark-themed CustomTkinter dashboard for standalone operator workstations.
  - **CLI Diagnostic Tools**: Standalone scripts for single-image batch inference and ground-truth label visualization.
- **Dedicated CUDA Inference Worker**: Persistent background worker thread eliminates PyTorch CUDA re-initialization overhead (~40ms inference per frame).
- **Comprehensive Incident Logging**: Detailed defect table listing class, confidence percentage, center coordinates $(X, Y)$, and bounding box geometry.

---

## Model Performance

Trained on the **TDDPCB** benchmark dataset (~10,000 images, 6 defect classes) for 50 epochs using YOLOv11 nano:

| Metric | Score |
|---|---|
| **mAP@0.5** | **92.4%** |
| **Precision** | **91.8%** |
| **Recall** | **86.6%** |
| **mAP@0.5:0.95** | **57.1%** |

### Defect Classes Detected
1. `open` — Open Circuit
2. `short` — Short Circuit
3. `mouse_bite` — Edge Mouse Bite
4. `spur` — Trace Spur
5. `spurious_copper` — Excess / Spurious Copper
6. `missing_hole` — Missing Drill Hole

Model weights are saved at: `runs/detect/train-7/weights/best.pt`

---

## Architecture

```
                    +--------------------+
                    | Camera / PCB Image |
                    +---------+----------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
    +-------------------+           +-------------------+
    |   Web Dashboard   |           |    Desktop GUI    |
    |  (Flask + Canvas) |           |  (CustomTkinter)  |
    +---------+---------+           +---------+---------+
              |                               |
              +---------------+---------------+
                              |
                              v
                +----------------------------+
                |  Dedicated Inference Queue |
                +-------------+--------------+
                              |
                              v
                +----------------------------+
                |   YOLOv11n (CUDA FP16)     |
                |   runs/.../train-7/best.pt |
                +----------------------------+
```

---

## Installation

### Prerequisites
- Python 3.10 – 3.12
- NVIDIA GPU with CUDA drivers (recommended for real-time 30 FPS inference)

### Setup
```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/PCB_AOI_System.git
cd PCB_AOI_System

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

### 1. Web Inspection Dashboard (Recommended)
```bash
cd src
python web_app.py
```
Open **`http://127.0.0.1:5000`** in your browser.
- Click **Live Feed** to start real-time optical inspection.
- Or click **Browse Image** → **Trigger Scan** to inspect a photo.

### 2. Desktop GUI
```bash
cd src
python gui.py
```

### 3. Standalone Live Camera (OpenCV Window)
```bash
cd src
python live_detection.py
```

---

## Project Structure

```
PCB_AOI_System/
├── dataset/
│   └── data.yaml             # Dataset class and split configuration
├── runs/
│   └── detect/train-7/       # Training run metrics, confusion matrices
│       └── weights/best.pt   # Trained YOLOv11 model weights
├── src/
│   ├── web_app.py            # Flask server with dedicated CUDA worker
│   ├── gui.py                # CustomTkinter operator dashboard
│   ├── detector.py           # Core YOLO detection engine (FP16)
│   ├── camera.py             # Threaded camera capture pipeline
│   ├── live_detection.py     # Standalone OpenCV live inspector
│   ├── report.py             # Plain-text inspection report generator
│   ├── detect_image.py       # Single-image CLI tester
│   ├── pcb_defect_map.py     # Defect map generator with callout lines
│   ├── visualize_labels.py   # Ground-truth label viewer
│   ├── templates/index.html  # Modern Tailwind CSS dashboard UI
│   └── static/
│       ├── app.js            # Real-time 30 FPS canvas rendering & controls
│       └── app.css           # Styling and dark theme variables
├── requirements.txt
├── .gitignore
└── README.md
```

---

## License

MIT License. Open-source for academic and industrial research.
