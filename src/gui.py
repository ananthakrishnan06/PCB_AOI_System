from pathlib import Path
from tkinter import filedialog
import time
import threading

import customtkinter as ctk
import cv2
from PIL import Image

from camera import start_live_camera
from detector import detect_image



COLORS = {
    "canvas": "#10141A", "panel": "#181C22", "panel_active": "#1C2026",
    "well": "#0A0E14", "border": "#30363D", "text": "#DFE2EB",
    "muted": "#9AA0AE", "blue": "#3B82F6", "blue_hover": "#2563EB",
    "cyan": "#4CD7F6", "green": "#4EDEA3", "red": "#FF8A80",
    "red_bg": "#4A2024", "amber": "#FBBF24",
}

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class AOIDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.selected_image_path = None
        self.title("PCB Automatic Optical Inspection System")
        self.geometry("1440x900")
        self.minsize(1180, 760)
        self.configure(fg_color=COLORS["canvas"])
        self._build_header()
        self._build_toolbar()
        self._build_workspace()
        self._build_footer()

    def _label(self, parent, text, size=12, weight="normal", color=None, **kwargs):
        return ctk.CTkLabel(parent, text=text, font=("Inter", size, weight), text_color=color or COLORS["text"], **kwargs)

    def _panel(self, parent, **kwargs):
        return ctk.CTkFrame(parent, fg_color=COLORS["panel"], border_color=COLORS["border"], border_width=1, corner_radius=12, **kwargs)

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=COLORS["well"], corner_radius=0, height=76)
        header.pack(fill="x")
        header.pack_propagate(False)
        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.pack(side="left", padx=24, pady=14)
        ctk.CTkLabel(brand, text="AOI", width=42, height=42, corner_radius=8, fg_color=COLORS["blue"], font=("JetBrains Mono", 14, "bold"), text_color="white").pack(side="left", padx=(0, 12))
        titles = ctk.CTkFrame(brand, fg_color="transparent")
        titles.pack(side="left")
        self._label(titles, "PCB AUTOMATIC OPTICAL INSPECTION", 18, "bold").pack(anchor="w")
        self._label(titles, "AI-POWERED DEFECT DETECTION  /  QUALITY CONTROL", 10, "bold", COLORS["muted"]).pack(anchor="w", pady=(2, 0))
        right = ctk.CTkFrame(header, fg_color="transparent")
        right.pack(side="right", padx=24)
        self._badge(right, "MODEL  best.pt", COLORS["cyan"]).pack(side="left", padx=6)
        self.system_badge = self._badge(right, "●  SYSTEM READY", COLORS["green"])
        self.system_badge.pack(side="left", padx=6)

    def _badge(self, parent, text, color):
        return ctk.CTkLabel(parent, text=text, font=("JetBrains Mono", 10, "bold"), text_color=color, fg_color=COLORS["panel_active"], corner_radius=5, padx=10, pady=7)

    def _button(self, parent, text, command, primary):
        return ctk.CTkButton(parent, text=text, command=command, height=36, corner_radius=6, font=("JetBrains Mono", 11, "bold"), fg_color=COLORS["blue"] if primary else COLORS["panel"], hover_color=COLORS["blue_hover"] if primary else COLORS["panel_active"], border_width=0 if primary else 1, border_color=COLORS["border"], text_color="white")

    def _build_toolbar(self):
        toolbar = ctk.CTkFrame(self, fg_color=COLORS["canvas"], height=68)
        toolbar.pack(fill="x", padx=24, pady=(14, 0))
        toolbar.pack_propagate(False)
        self._button(toolbar, "BROWSE IMAGE", self.browse_image, False).pack(side="left", padx=(0, 8), pady=12)
        self.inspect_button = self._button(toolbar, "RUN INSPECTION", self.detect, True)
        self.inspect_button.pack(side="left", padx=8, pady=12)
        self.live_button = self._button(toolbar, "LIVE CAMERA", self._start_live_camera, False)
        self.live_button.pack(side="left", padx=8, pady=12)
        self._button(toolbar, "RESET", self.reset, False).pack(side="left", padx=8, pady=12)
        threshold = ctk.CTkFrame(toolbar, fg_color=COLORS["panel"], corner_radius=6)
        threshold.pack(side="right", pady=12)
        self._label(threshold, "CONFIDENCE THRESHOLD", 10, "bold", COLORS["muted"]).pack(side="left", padx=(12, 8))
        slider = ctk.CTkSlider(threshold, from_=0, to=100, number_of_steps=100, width=120, progress_color=COLORS["cyan"], button_color=COLORS["cyan"], button_hover_color=COLORS["cyan"], command=self._update_threshold)
        slider.set(25)
        slider.pack(side="left", padx=(0, 8))
        self.threshold_value = self._label(threshold, "25%", 11, "bold", COLORS["cyan"], width=38)
        self.threshold_value.pack(side="left", padx=(0, 10))

    def _build_workspace(self):
        workspace = ctk.CTkFrame(self, fg_color="transparent")
        workspace.pack(fill="both", expand=True, padx=24, pady=(0, 14))
        workspace.grid_columnconfigure((0, 1), weight=1, uniform="preview")
        workspace.grid_rowconfigure(0, weight=3)
        workspace.grid_rowconfigure(1, weight=2)
        self._build_image_panel(workspace, 0, "ORIGINAL PCB IMAGE", "Awaiting image input", "original")
        self._build_image_panel(workspace, 1, "INSPECTION RESULT", "Run inspection to view annotations", "result")
        self._build_summary(workspace)
        self._build_defect_table(workspace)

    def _build_image_panel(self, parent, column, heading, placeholder, kind):
        panel = self._panel(parent)
        panel.grid(row=0, column=column, sticky="nsew", padx=(0, 7) if column == 0 else (7, 0), pady=(0, 7))
        header = ctk.CTkFrame(panel, fg_color="transparent", height=38)
        header.pack(fill="x", padx=14, pady=(8, 0))
        self._label(header, heading, 11, "bold", COLORS["cyan"] if kind == "result" else COLORS["muted"]).pack(side="left")
        self._label(header, "CAMERA FEED" if kind == "original" else "YOLO INFERENCE", 10, "bold", COLORS["muted"]).pack(side="right")
        image = ctk.CTkLabel(panel, text=placeholder, font=("Inter", 14), text_color=COLORS["muted"], fg_color=COLORS["well"], corner_radius=8)
        image.pack(fill="both", expand=True, padx=14, pady=(4, 10))
        meta = self._label(panel, "NO FILE SELECTED" if kind == "original" else "STATUS: STANDBY", 10, "bold", COLORS["muted"])
        meta.pack(anchor="w", padx=14, pady=(0, 10))
        if kind == "original":
            self.left_image, self.image_meta = image, meta
        else:
            self.right_image, self.result_meta = image, meta

    def _build_summary(self, parent):
        panel = self._panel(parent)
        panel.grid(row=1, column=0, sticky="nsew", padx=(0, 7), pady=(7, 0))
        self._label(panel, "INSPECTION SUMMARY", 11, "bold", COLORS["cyan"]).pack(anchor="w", padx=14, pady=(12, 8))
        self.status_card = ctk.CTkFrame(panel, fg_color="#123126", corner_radius=8)
        self.status_card.pack(fill="x", padx=14, pady=(0, 10))
        self.status_value = self._label(self.status_card, "READY", 24, "bold", COLORS["green"])
        self.status_value.pack(side="left", padx=14, pady=11)
        self.status_detail = self._label(self.status_card, "SELECT A PCB IMAGE TO BEGIN", 10, "bold", COLORS["muted"])
        self.status_detail.pack(side="right", padx=14)
        metrics = ctk.CTkFrame(panel, fg_color="transparent")
        metrics.pack(fill="x", padx=14)
        for title, attr in (("TOTAL DEFECTS", "total_metric"), ("PROCESS TIME", "time_metric"), ("THRESHOLD", "threshold_metric")):
            metric = ctk.CTkFrame(metrics, fg_color=COLORS["well"], corner_radius=6)
            metric.pack(side="left", fill="x", expand=True, padx=(0, 6))
            self._label(metric, title, 9, "bold", COLORS["muted"]).pack(anchor="w", padx=10, pady=(8, 0))
            value = self._label(metric, "--" if attr != "threshold_metric" else "25%", 18, "bold", COLORS["text"])
            value.pack(anchor="w", padx=10, pady=(1, 8))
            setattr(self, attr, value)
        self.summary_text = self._label(panel, "Defect categories will appear after inspection.", 11, "normal", COLORS["muted"], wraplength=480, justify="left")
        self.summary_text.pack(anchor="w", padx=14, pady=10)

    def _build_defect_table(self, parent):
        panel = self._panel(parent)
        panel.grid(row=1, column=1, sticky="nsew", padx=(7, 0), pady=(7, 0))
        self._label(panel, "DEFECT DETAILS", 11, "bold", COLORS["cyan"]).pack(anchor="w", padx=14, pady=(12, 8))
        header = ctk.CTkFrame(panel, fg_color=COLORS["well"], corner_radius=4)
        header.pack(fill="x", padx=14)
        for text, width in (("ID", 35), ("DEFECT TYPE", 145), ("CONFIDENCE", 90), ("CENTER POSITION", 130)):
            self._label(header, text, 9, "bold", COLORS["muted"], width=width, anchor="w").pack(side="left", padx=6, pady=7)
        self.defect_box = ctk.CTkTextbox(panel, fg_color=COLORS["well"], text_color=COLORS["muted"], font=("JetBrains Mono", 11), corner_radius=6, border_width=0)
        self.defect_box.pack(fill="both", expand=True, padx=14, pady=(5, 12))
        self.defect_box.insert("0.0", "No inspection data available.")
        self.defect_box.configure(state="disabled")

    def _update_threshold(self, value):
        percentage = int(value)
        self.threshold_value.configure(text=f"{percentage}%")
        self.threshold_metric.configure(text=f"{percentage}%")

    def browse_image(self):
        path = filedialog.askopenfilename(title="Select PCB Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        self.selected_image_path = path
        self._set_preview(self.left_image, Image.open(path))
        self.image_meta.configure(text=f"FILE: {Path(path).name.upper()}", text_color=COLORS["cyan"])
        self._clear_results()
        self._set_ready_state("IMAGE LOADED — READY FOR INSPECTION")

    def detect(self):
        if not self.selected_image_path:
            self._set_fail_state("NO IMAGE SELECTED", "SELECT A PCB IMAGE BEFORE RUNNING INSPECTION")
            return
        self.inspect_button.configure(state="disabled", text="ANALYZING...")
        self.system_badge.configure(text="●  ANALYZING", text_color=COLORS["amber"])
        self.update_idletasks()

        def _run():
            started = time.perf_counter()
            try:
                result = detect_image(self.selected_image_path)
                elapsed = time.perf_counter() - started
                # Schedule UI update back on the main thread
                self.after(0, lambda: self._on_detect_complete(result, elapsed))
            except Exception as error:
                self.after(0, lambda: self._set_fail_state("INSPECTION ERROR", str(error)))
            finally:
                self.after(0, lambda: self.inspect_button.configure(state="normal", text="RUN INSPECTION"))

        threading.Thread(target=_run, daemon=True).start()

    def _on_detect_complete(self, result, elapsed):
        self._set_preview(self.right_image, Image.fromarray(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB)))
        self._show_result(result, elapsed)
        self._show_defects(result)

    def _start_live_camera(self):
        """Launch start_live_camera() in a daemon thread so the GUI stays responsive."""
        if getattr(self, "_live_thread", None) and self._live_thread.is_alive():
            return  # Already running — ignore duplicate clicks
        self._live_thread = threading.Thread(target=start_live_camera, daemon=True)
        self._live_thread.start()

    def _set_preview(self, widget, image):
        image.thumbnail((570, 360))
        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        widget.configure(image=ctk_image, text="")
        widget.image = ctk_image

    def _show_result(self, result, elapsed):
        passed = result["status"] == "PASS"
        color = COLORS["green"] if passed else COLORS["red"]
        self.status_card.configure(fg_color="#123126" if passed else COLORS["red_bg"])
        self.status_value.configure(text=result["status"], text_color=color)
        self.status_detail.configure(text="NO DEFECTS DETECTED" if passed else "DEFECTS REQUIRE REVIEW", text_color=color)
        self.total_metric.configure(text=str(result["total"]), text_color=color)
        self.time_metric.configure(text=f"{elapsed:.2f}s")
        self.result_meta.configure(text=f"STATUS: {result['status']}  |  {result['total']} DEFECT(S)", text_color=color)
        self.system_badge.configure(text="●  INSPECTION COMPLETE", text_color=COLORS["green"])
        summary = "  •  ".join(f"{name.upper()}: {count}" for name, count in result["summary"].items()) or "No defects detected. PCB passed the configured visual inspection threshold."
        self.summary_text.configure(text=summary, text_color=color if passed else COLORS["amber"])

    def _show_defects(self, result):
        lines = [f"{index:<5}{defect['class']:<18}{defect['confidence']:>6.2f}%     {str(defect['center']):<18}" for index, defect in enumerate(result["defects"], start=1)]
        self.defect_box.configure(state="normal")
        self.defect_box.delete("0.0", "end")
        self.defect_box.insert("0.0", "\n".join(lines) if lines else "No defects detected — inspection passed.")
        self.defect_box.configure(state="disabled")

    def _set_ready_state(self, message):
        self.status_card.configure(fg_color="#123126")
        self.status_value.configure(text="READY", text_color=COLORS["green"])
        self.status_detail.configure(text=message, text_color=COLORS["muted"])
        self.system_badge.configure(text="●  SYSTEM READY", text_color=COLORS["green"])

    def _set_fail_state(self, status, detail):
        self.status_card.configure(fg_color=COLORS["red_bg"])
        self.status_value.configure(text=status, text_color=COLORS["red"])
        self.status_detail.configure(text=detail.upper(), text_color=COLORS["red"])
        self.system_badge.configure(text="●  ACTION REQUIRED", text_color=COLORS["red"])

    def _clear_results(self):
        self.right_image.configure(image=None, text="Run inspection to view annotations")
        self.right_image.image = None
        self.result_meta.configure(text="STATUS: STANDBY", text_color=COLORS["muted"])
        self.total_metric.configure(text="--", text_color=COLORS["text"])
        self.time_metric.configure(text="--")
        self.summary_text.configure(text="Defect categories will appear after inspection.", text_color=COLORS["muted"])
        self.defect_box.configure(state="normal")
        self.defect_box.delete("0.0", "end")
        self.defect_box.insert("0.0", "No inspection data available.")
        self.defect_box.configure(state="disabled")

    def reset(self):
        self.selected_image_path = None
        self.left_image.configure(image=None, text="Awaiting image input")
        self.left_image.image = None
        self.image_meta.configure(text="NO FILE SELECTED", text_color=COLORS["muted"])
        self._clear_results()
        self._set_ready_state("SELECT A PCB IMAGE TO BEGIN")

    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color=COLORS["well"], height=30, corner_radius=0)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        self._label(footer, "CAMERA: NOT CONNECTED", 9, "bold", COLORS["muted"]).pack(side="left", padx=24)
        self._label(footer, "MODEL: RUNS/DETECT/TRAIN-7/WEIGHTS/BEST.PT", 9, "bold", COLORS["muted"]).pack(side="right", padx=24)


if __name__ == "__main__":
    AOIDashboard().mainloop()
