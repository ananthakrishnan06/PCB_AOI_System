const el = (id) => document.getElementById(id);
const imageInput = el("imageInput");
const inspectButton = el("inspectButton");
const confidenceInput = el("confidenceInput");

let selectedFile = null;
let currentResult = null;
let cameraStream = null;
let liveInspectionTimer = null;
let inspectionInProgress = false;
let cameraFrameRate = null;
let liveDetectionTimes = [];
let animFrameId = null;
let currentLiveDetections = [];
let captureCanvas = null;

function toast(message, isError = false) {
  let node = el("toast");
  if (!node) {
    node = document.createElement("div");
    node.id = "toast";
    node.className = "fixed bottom-10 left-1/2 -translate-x-1/2 z-50 border rounded px-4 py-2 font-label-code text-label-code shadow-xl";
    document.body.append(node);
  }
  node.className = `fixed bottom-10 left-1/2 -translate-x-1/2 z-50 border rounded px-4 py-2 font-label-code text-label-code shadow-xl ${
    isError ? "bg-error-container border-error text-error" : "bg-surface-container-high border-secondary text-on-surface"
  }`;
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 3000);
}

function setBusy(busy) {
  if (!inspectButton) return;
  inspectButton.disabled = busy;
  const label = inspectButton.querySelector("span:last-child");
  if (label) label.textContent = busy ? "ANALYZING..." : "TRIGGER SCAN";
}

function severity(defect) {
  return ["short", "missing_hole", "open"].includes(defect.class)
    ? "text-red-400 bg-red-950/60 border-red-500"
    : "text-amber-300 bg-amber-950/60 border-amber-500";
}

function buildRows(defects) {
  if (!defects || !defects.length) {
    return '<tr class="bg-surface-container-low"><td class="py-6 px-3 text-on-surface-variant text-center" colspan="6">No defects detected — inspection passed.</td></tr>';
  }
  return defects.map((defect, index) => `
    <tr data-defect-index="${index}" class="bg-surface-container-low hover:bg-surface-container-high transition-colors cursor-pointer">
      <td class="py-2 px-3 font-bold text-primary">#${String(index + 1).padStart(2, "0")}</td>
      <td class="py-2 px-3">
        <span class="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold border ${severity(defect)}">
          ${defect.class.replaceAll("_", " ")}
        </span>
      </td>
      <td class="py-2 px-3 font-bold text-on-surface">${defect.confidence.toFixed(2)}%</td>
      <td class="py-2 px-3 text-on-surface">(${defect.center.join(", ")})</td>
      <td class="py-2 px-3 text-on-surface-variant">${defect.bbox.join(", ")}</td>
      <td class="py-2 px-3 text-right">
        <button data-review="${index}" class="px-2 py-1 bg-surface-container-high border border-outline-variant rounded text-on-surface text-[10px] uppercase font-bold hover:text-secondary">
          Review Crop
        </button>
      </td>
    </tr>
  `).join("");
}

function refreshTaxonomy(summary) {
  if (!summary) return;
  document.querySelectorAll(".grid.grid-cols-3.gap-1\\.5 > div").forEach((chip) => {
    const a = chip.querySelector("span:first-child");
    const b = chip.querySelector("span:last-child");
    if (!a || !b) return;
    const key = a.textContent.trim().toLowerCase().replaceAll(" ", "_")
      .replace("spurious_cu", "spurious_copper")
      .replace("open_circuit", "open")
      .replace("short_circuit", "short");
    b.textContent = summary[key] || 0;
  });
}

function setLiveFeedLabel(active) {
  const btn = el("liveFeedButton");
  if (!btn) return;
  const label = btn.querySelector("span:nth-child(2)");
  if (label) label.textContent = active ? "Stop Feed" : "Live Feed";
}

function renderLiveFeedOverlay() {
  if (!cameraStream) return;
  const video = el("liveVideo");
  const canvas = el("liveResultCanvas");

  if (video && canvas && video.videoWidth && video.videoHeight) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext("2d");
    // Draw current video frame to canvas at full 30 FPS display rate
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Draw the latest detected bounding boxes smoothly
    for (let i = 0; i < currentLiveDetections.length; i++) {
      const d = currentLiveDetections[i];
      const [x1, y1, x2, y2] = d.bbox;
      const isCritical = ["short", "missing_hole", "open"].includes(d.class);
      const strokeColor = isCritical ? "#ef4444" : "#f59e0b";
      const bgColor = isCritical ? "rgba(239, 68, 68, 0.85)" : "rgba(245, 158, 11, 0.85)";

      // Bounding box
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label badge
      const label = `#${i + 1} ${d.class.replace(/_/g, " ").toUpperCase()} [${d.confidence.toFixed(1)}%]`;
      ctx.font = "bold 11px 'JetBrains Mono', monospace";
      const textWidth = ctx.measureText(label).width;
      const labelY = Math.max(16, y1 - 4);

      ctx.fillStyle = bgColor;
      ctx.fillRect(x1, labelY - 14, textWidth + 8, 16);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + 4, labelY - 2);
    }
  }

  animFrameId = requestAnimationFrame(renderLiveFeedOverlay);
}

function closeCamera(notify = true) {
  if (liveInspectionTimer) {
    clearTimeout(liveInspectionTimer);
    liveInspectionTimer = null;
  }
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
  }
  cameraStream = null;
  cameraFrameRate = null;
  currentLiveDetections = [];

  const video = el("liveVideo");
  if (video) {
    video.srcObject = null;
    video.remove();
  }

  const canvas = el("liveResultCanvas");
  if (canvas) {
    canvas.remove();
  }

  const resultImg = el("resultImage");
  if (resultImg) {
    resultImg.style.display = "block";
  }

  setLiveFeedLabel(false);
  const inference = el("liveInference");
  if (inference) inference.textContent = "INFERENCE: STANDBY";
  if (notify) toast("Live camera stopped.");
}

async function startCamera() {
  if (cameraStream) {
    closeCamera();
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    toast("This browser does not support camera access. Use Chrome or Edge on localhost.", true);
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width: { ideal: 960, max: 1280 },
        height: { ideal: 540, max: 720 },
        frameRate: { ideal: 30, max: 30 }
      },
      audio: false
    });

    // 1. Setup Live Video in the Left Card
    const video = document.createElement("video");
    video.id = "liveVideo";
    video.autoplay = true;
    video.playsInline = true;
    video.muted = true;
    video.className = "absolute inset-0 w-full h-full object-contain bg-black z-10";
    video.srcObject = stream;
    el("originalImage").parentElement.append(video);

    // 2. Setup Live Real-Time Canvas in the Right Card
    const resultParent = el("resultImage").parentElement;
    const existingCanvas = el("liveResultCanvas");
    if (existingCanvas) existingCanvas.remove();

    const resultCanvas = document.createElement("canvas");
    resultCanvas.id = "liveResultCanvas";
    resultCanvas.className = "absolute inset-0 w-full h-full object-contain z-10 bg-black";
    resultParent.append(resultCanvas);
    el("resultImage").style.display = "none";

    await new Promise((resolve) => { video.onloadedmetadata = resolve; });
    await video.play();

    cameraStream = stream;
    cameraFrameRate = stream.getVideoTracks()[0]?.getSettings?.().frameRate || 30;
    liveDetectionTimes = [];
    currentLiveDetections = [];

    const inference = el("liveInference");
    if (inference) inference.textContent = `LIVE: ${Math.round(cameraFrameRate)} FPS | INFERENCE: STARTING`;
    setLiveFeedLabel(true);

    // Start 30 FPS rendering loop
    animFrameId = requestAnimationFrame(renderLiveFeedOverlay);

    // Start background inference loop
    liveInspectionTimer = setTimeout(inspectLiveFrame, 50);
    toast("Live camera active: real-time inspection running.");
  } catch (error) {
    const reason = error.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access in your browser and try again."
      : error.name === "NotFoundError"
      ? "No camera was found on this device."
      : error.message;
    toast(`Camera unavailable: ${reason}`, true);
  }
}

async function captureLiveFrame() {
  const video = el("liveVideo");
  if (!video || !video.videoWidth || !video.videoHeight) {
    throw new Error("The camera frame is not ready yet. Wait a moment and try again.");
  }

  if (!captureCanvas) {
    captureCanvas = document.createElement("canvas");
  }

  // Scale to 640 for high performance
  const targetWidth = Math.min(640, video.videoWidth);
  const scale = targetWidth / video.videoWidth;
  captureCanvas.width = targetWidth;
  captureCanvas.height = Math.round(video.videoHeight * scale);

  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

  const blob = await new Promise((resolve) => captureCanvas.toBlob(resolve, "image/jpeg", 0.75));
  if (!blob) throw new Error("Could not capture a frame from the live camera.");

  return {
    file: new File([blob], `live_${Date.now()}.jpg`, { type: "image/jpeg" }),
    capturedWidth: captureCanvas.width,
    capturedHeight: captureCanvas.height
  };
}

async function performInspection(fileData, fromLiveFeed = false) {
  if (inspectionInProgress) return;
  inspectionInProgress = true;
  if (!fromLiveFeed) setBusy(true);

  const file = fileData.file || fileData;
  const body = new FormData();
  body.append("image", file);
  body.append("confidence", confidenceInput.value);
  if (fromLiveFeed) {
    body.append("live", "true");
  }

  try {
    const response = await fetch("/api/inspect", { method: "POST", body });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Inspection failed.");

    currentResult = result;

    if (fromLiveFeed) {
      // Scale detection coordinates to native video resolution
      const video = el("liveVideo");
      if (video && fileData.capturedWidth && fileData.capturedHeight) {
        const scaleX = video.videoWidth / fileData.capturedWidth;
        const scaleY = video.videoHeight / fileData.capturedHeight;

        currentLiveDetections = (result.defects || []).map((d) => ({
          ...d,
          bbox: [
            Math.round(d.bbox[0] * scaleX),
            Math.round(d.bbox[1] * scaleY),
            Math.round(d.bbox[2] * scaleX),
            Math.round(d.bbox[3] * scaleY)
          ],
          center: [
            Math.round(d.center[0] * scaleX),
            Math.round(d.center[1] * scaleY)
          ]
        }));
      } else {
        currentLiveDetections = result.defects || [];
      }

      // Live FPS and latency tracking
      const completedAt = performance.now();
      liveDetectionTimes.push(completedAt);
      liveDetectionTimes = liveDetectionTimes.filter((stamp) => completedAt - stamp <= 2000);
      const measuredFps = liveDetectionTimes.length > 1
        ? ((liveDetectionTimes.length - 1) * 1000 / (liveDetectionTimes.at(-1) - liveDetectionTimes[0])).toFixed(1)
        : "30";

      const inference = el("liveInference");
      if (inference) {
        inference.textContent = `LIVE: ${Math.round(cameraFrameRate || 30)} FPS | DETECTION: ${measuredFps} FPS | ${Math.round(result.processing_seconds * 1000)}ms MODEL`;
      }
    } else {
      // Non-live static inspection result
      el("resultImage").src = result.annotated_image;
      el("resultImage").style.display = "block";
      toast(`Inspection complete: ${result.status}.`);
    }

    // Update status banner
    const statusEl = el("resultStatus");
    const subLabel = el("statusSubLabel");
    if (statusEl) {
      statusEl.textContent = result.status;
      statusEl.className = `font-data-metric-xl text-2xl font-bold tracking-tight ${
        result.status === "PASS" ? "text-tertiary" : "text-error"
      }`;
    }
    if (subLabel) {
      subLabel.textContent = result.status === "PASS" ? "ALL CHECKS PASSED — NO DEFECTS" : "DEFECTS DETECTED — REVIEW REQUIRED";
      subLabel.className = `font-label-caps text-[10px] font-bold tracking-wider ${result.status === "PASS" ? "text-tertiary" : "text-error"}`;
    }

    const totalEl = el("totalMetric");
    if (totalEl) totalEl.textContent = String(result.total).padStart(2, "0");

    const timeEl = el("timeMetric");
    if (timeEl) timeEl.textContent = `${result.processing_seconds}s`;

    el("defectRows").innerHTML = buildRows(result.defects);
    refreshTaxonomy(result.summary);
  } catch (error) {
    if (!fromLiveFeed) toast(error.message, true);
  } finally {
    inspectionInProgress = false;
    if (!fromLiveFeed) setBusy(false);
  }
}

async function inspectLiveFrame() {
  if (!cameraStream) return;
  if (!inspectionInProgress) {
    try {
      const frameData = await captureLiveFrame();
      await performInspection(frameData, true);
    } catch (error) {
      // non-fatal capture error
    }
  }

  if (cameraStream) {
    liveInspectionTimer = setTimeout(inspectLiveFrame, 15);
  }
}

function modal(title, body) {
  const node = document.createElement("div");
  node.className = "fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4";
  node.innerHTML = `
    <section class="max-w-lg w-full bg-surface-container border border-outline-variant rounded-lg p-4 shadow-2xl">
      <div class="flex justify-between items-center border-b border-outline-variant pb-2">
        <h2 class="font-headline-sm text-on-surface font-bold">${title}</h2>
        <button class="text-error text-lg" aria-label="Close">×</button>
      </div>
      <div class="py-4 text-on-surface-variant font-body-md">${body}</div>
    </section>
  `;
  node.addEventListener("click", (event) => {
    if (event.target === node || event.target.tagName === "BUTTON") node.remove();
  });
  document.body.append(node);
}

function exportCsv() {
  if (!currentResult?.defects?.length) {
    toast("Run an inspection before exporting results.", true);
    return;
  }
  const header = ["id", "defect_type", "confidence_percent", "center_x", "center_y", "x1", "y1", "x2", "y2"];
  const records = currentResult.defects.map((d, i) => [i + 1, d.class, d.confidence, ...d.center, ...d.bbox]);
  const csv = [header, ...records].map((row) => row.join(",")).join("\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  link.download = "pcb-aoi-inspection.csv";
  link.click();
  URL.revokeObjectURL(link.href);
  toast("Inspection CSV exported.");
}

el("browseButton")?.addEventListener("click", () => imageInput.click());

imageInput?.addEventListener("change", () => {
  selectedFile = imageInput.files[0];
  if (!selectedFile) return;
  closeCamera(false);
  const source = URL.createObjectURL(selectedFile);
  el("originalImage").src = source;
  el("resultImage").src = source;
  el("resultImage").style.display = "block";
  toast(`Loaded ${selectedFile.name}.`);
});

confidenceInput?.addEventListener("input", () => {
  const value = `${confidenceInput.value}%`;
  el("confidenceValue").textContent = value;
  el("thresholdMetric").textContent = value;
});

el("resetButton")?.addEventListener("click", () => {
  if (imageInput) imageInput.value = "";
  selectedFile = currentResult = null;
  currentLiveDetections = [];
  closeCamera(false);
  el("originalImage").removeAttribute("src");
  el("resultImage").removeAttribute("src");
  el("resultImage").style.display = "block";
  el("resultStatus").textContent = "READY";
  el("resultStatus").className = "font-data-metric-xl text-2xl font-bold tracking-tight text-tertiary";
  const subLabel = el("statusSubLabel");
  if (subLabel) {
    subLabel.textContent = "SELECT IMAGE OR LIVE FEED";
    subLabel.className = "font-label-caps text-[10px] text-on-surface-variant font-bold tracking-wider";
  }
  el("totalMetric").textContent = "--";
  el("timeMetric").textContent = "--";
  el("defectRows").innerHTML = '<tr class="bg-surface-container-low"><td class="py-6 px-3 text-on-surface-variant text-center" colspan="6">No inspection data available.</td></tr>';
  refreshTaxonomy({ open: 0, short: 0, mouse_bite: 0, spur: 0, spurious_copper: 0, missing_hole: 0 });
  toast("Workspace reset.");
});

inspectButton?.addEventListener("click", async () => {
  try {
    const inspectionFile = cameraStream ? (await captureLiveFrame()).file : selectedFile;
    if (!inspectionFile) throw new Error("Select a PCB image or start Live Feed before triggering a scan.");
    await performInspection(inspectionFile, false);
  } catch (error) {
    toast(error.message, true);
  }
});

el("liveFeedButton")?.addEventListener("click", startCamera);
el("cycleStopButton")?.addEventListener("click", () => {
  closeCamera();
  setBusy(false);
  toast("Cycle stopped.");
});

el("fullScreenButton")?.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch (error) {
    toast("Fullscreen is not available in this browser.", true);
  }
});

el("exportButton")?.addEventListener("click", exportCsv);

el("defectRows")?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-review]");
  if (!button || !currentResult) return;
  const d = currentResult.defects[Number(button.dataset.review)];
  modal(
    "Defect Crop Review",
    `<p><strong>${d.class.replaceAll("_", " ")}</strong> — ${d.confidence.toFixed(2)}% confidence</p>
     <p class="mt-2 font-label-code">Center: (${d.center.join(", ")})<br>Bounding box: ${d.bbox.join(", ")}</p>
     ${currentResult.annotated_image ? `<img class="mt-4 max-h-72 mx-auto border border-outline-variant" src="${currentResult.annotated_image}" alt="Annotated inspection result">` : ""}`
  );
});

async function showHistory(title) {
  try {
    const response = await fetch("/api/history");
    const scans = (await response.json()).scans;
    const content = scans.length
      ? `<div class="overflow-x-auto"><table class="w-full text-left font-label-code text-label-code"><thead><tr class="text-on-surface-variant"><th class="p-2">TIME</th><th class="p-2">IMAGE</th><th class="p-2">STATUS</th><th class="p-2">DEFECTS</th><th class="p-2">DURATION</th></tr></thead><tbody>${scans.map((scan) => `<tr class="border-t border-outline-variant"><td class="p-2">${scan.timestamp}</td><td class="p-2">${scan.image_name}</td><td class="p-2 font-bold ${scan.status === "PASS" ? "text-tertiary" : "text-error"}">${scan.status}</td><td class="p-2">${scan.total}</td><td class="p-2">${scan.processing_seconds}s @ ${scan.confidence}%</td></tr>`).join("")}</tbody></table></div>`
      : "<p>No scans have been completed in this server session. Upload a PCB image and run Trigger Scan to add the first record.</p>";
    modal(title, content);
  } catch (error) {
    toast("Scan history is unavailable.", true);
  }
}

function setTheme(dark) {
  document.documentElement.classList.toggle("light-ui", !dark);
  localStorage.setItem("aoi-theme", dark ? "dark" : "light");
}

function showSettings() {
  const dark = !document.documentElement.classList.contains("light-ui");
  modal("System Settings", `<div class="space-y-4"><div class="flex items-center justify-between border border-outline-variant rounded p-3"><div><strong class="text-on-surface">Appearance</strong><p class="text-sm mt-1">Use the industrial low-glare dark interface for inspection work.</p></div><label class="flex items-center gap-2 cursor-pointer"><input id="darkModeToggle" type="checkbox" ${dark ? "checked" : ""}><span>Dark mode</span></label></div><div class="border border-outline-variant rounded p-3"><strong class="text-on-surface">Inference</strong><p class="text-sm mt-1">The confidence cutoff in the toolbar controls the minimum YOLO detection confidence for each scan.</p></div></div>`);
  setTimeout(() => {
    const toggle = el("darkModeToggle");
    if (toggle) toggle.addEventListener("change", () => {
      setTheme(toggle.checked);
      toast(`Dark mode ${toggle.checked ? "enabled" : "disabled"}.`);
    });
  }, 0);
}

function showDocs() {
  modal("Operator Docs — How to operate AOI-CORE", `<ol class="list-decimal pl-5 space-y-3 leading-relaxed"><li><strong class="text-on-surface">Prepare the board image.</strong> Use a clear, top-down PCB photograph in JPG, PNG, BMP, or WEBP format. Keep the board in focus and well lit.</li><li><strong class="text-on-surface">Load the image.</strong> Select <em>Browse Image</em>, choose the PCB file, and confirm it appears in the Original PCB Image panel.</li><li><strong class="text-on-surface">Set confidence.</strong> Adjust Confidence Cutoff. 25% is the project default. Higher values reduce low-confidence detections; lower values show more possible defects.</li><li><strong class="text-on-surface">Run the scan.</strong> Select <em>Trigger Scan</em>. The page uploads the image to the local YOLO model and shows the annotated result on completion.</li><li><strong class="text-on-surface">Review the outcome.</strong> PASS means no detections reached the chosen threshold. FAIL means one or more defects were found. Review each incident in the table and use Review Crop for its coordinates.</li><li><strong class="text-on-surface">Export and track.</strong> Export CSV saves the current defect ledger. Scan History lists every completed scan while the local server remains running.</li><li><strong class="text-on-surface">Live Feed.</strong> Select Live Feed and approve camera access in the browser to preview a connected camera. Cycle Stop ends the live preview safely.</li></ol>`);
}

setTheme(localStorage.getItem("aoi-theme") !== "light");

document.querySelectorAll("nav a").forEach((link) => link.addEventListener("click", (event) => {
  event.preventDefault();
  const target = link.getAttribute("href");
  if (target === "#history") showHistory("Scan History");
  else if (target === "#settings") showSettings();
  else if (target === "#docs") showDocs();
  else if (target === "#inspect") document.querySelector("main").scrollTo({ top: 0, behavior: "smooth" });
}));
