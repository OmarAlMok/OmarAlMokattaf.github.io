"""
Edge Tracer — Canny edge detection, simplified.
Shows only: Edge Map + Overlay.
Thresholds are auto-calibrated (no sliders).
"""

import sys
import os
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea, QFrame, QSizePolicy,
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QPalette
from PyQt6.QtCore import Qt, QThread, pyqtSignal

try:
    from PIL import Image, ImageDraw
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow",
                           "--break-system-packages"])
    from PIL import Image, ImageDraw


# ─────────────────────────────────────────────────────────────────────────────
#  Pure-NumPy Canny Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
    shape = (img.shape[0], img.shape[1], kh, kw)
    strides = (padded.strides[0], padded.strides[1],
               padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.einsum('ijkl,kl->ij', windows, kernel)


def sobel_gradients(gray: np.ndarray):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = convolve2d(gray, Kx)
    gy = convolve2d(gray, Ky)
    mag = np.hypot(gx, gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180
    return gx, gy, mag, angle


def _nms_vectorised(mag: np.ndarray, angle: np.ndarray) -> np.ndarray:
    ang = angle.copy()
    d0   = (ang <  22.5) | (ang >= 157.5)
    d45  = (ang >= 22.5) & (ang <  67.5)
    d90  = (ang >= 67.5) & (ang < 112.5)
    d135 = (ang >= 112.5) & (ang < 157.5)

    m = mag
    local = np.zeros_like(m, dtype=bool)

    q = np.roll(m, -1, axis=1); r = np.roll(m, 1, axis=1)
    local |= d0 & (m >= q) & (m >= r)

    q = np.roll(m, -1, axis=0); r = np.roll(m, 1, axis=0)
    local |= d90 & (m >= q) & (m >= r)

    q = np.roll(np.roll(m, -1, axis=0), 1, axis=1)
    r = np.roll(np.roll(m, 1, axis=0), -1, axis=1)
    local |= d45 & (m >= q) & (m >= r)

    q = np.roll(np.roll(m, -1, axis=0), -1, axis=1)
    r = np.roll(np.roll(m, 1, axis=0),  1, axis=1)
    local |= d135 & (m >= q) & (m >= r)

    out = np.zeros_like(m)
    out[local] = m[local]
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = 0
    return out


def auto_thresholds(nms: np.ndarray):
    """
    Auto-calibrate high/low thresholds using Otsu's method on the NMS
    magnitude histogram. High = Otsu threshold; low = 0.4 × high.
    """
    flat = nms[nms > 0]
    if flat.size == 0:
        return 0.15, 0.05  # fallback ratios
    # Normalise to 0-1 for Otsu computation
    norm = flat / (flat.max() + 1e-9)
    # Otsu on 256 bins
    hist, edges = np.histogram(norm, bins=256, range=(0, 1))
    total = hist.sum()
    best_var, best_t = 0, 0.5
    w0 = 0; mu0 = 0
    mu_total = np.sum(np.arange(256) * hist) / (total + 1e-9)
    for t in range(1, 256):
        w0 += hist[t - 1]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        mu0 = np.sum(np.arange(t) * hist[:t]) / (w0 + 1e-9)
        mu1 = (mu_total * total - mu0 * w0) / (w1 + 1e-9)
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_t = t / 256.0

    # Convert ratio back to absolute threshold
    mx = nms.max()
    high = best_t * mx
    low  = 0.4 * high
    return high, low


def double_threshold(nms: np.ndarray, low: float, high: float):
    strong = nms >= high
    weak   = (nms >= low) & ~strong
    return strong, weak


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    out = strong.astype(np.uint8) * 255
    changed = True
    while changed:
        s = out > 0
        dilated = (
            np.roll(s,  1, 0) | np.roll(s, -1, 0) |
            np.roll(s,  1, 1) | np.roll(s, -1, 1) |
            np.roll(np.roll(s,  1, 0),  1, 1) |
            np.roll(np.roll(s,  1, 0), -1, 1) |
            np.roll(np.roll(s, -1, 0),  1, 1) |
            np.roll(np.roll(s, -1, 0), -1, 1)
        )
        promoted = weak & dilated & ~s
        if not promoted.any():
            changed = False
        else:
            out[promoted] = 255
    return out


def _gray_to_rgb(g: np.ndarray) -> np.ndarray:
    return np.stack([g, g, g], axis=2)


def _edge_overlay(pil_img: Image.Image, edges: np.ndarray) -> np.ndarray:
    base = np.array(pil_img.convert("RGB"), dtype=np.uint8).copy()
    base[edges > 0] = [50, 255, 120]
    return base


def canny_pipeline(pil_img: Image.Image):
    gray = np.array(pil_img.convert("L"), dtype=np.float64)

    kernel  = gaussian_kernel(5, 1.4)
    blurred = convolve2d(gray, kernel)

    gx, gy, mag, angle = sobel_gradients(blurred)
    nms = _nms_vectorised(mag, angle)

    high, low = auto_thresholds(nms)
    strong, weak = double_threshold(nms, low, high)

    edges   = hysteresis(strong, weak)
    edge_bw = _gray_to_rgb(edges)
    overlay = _edge_overlay(pil_img, edges)

    return {
        "edges":   edge_bw,
        "overlay": overlay,
        "hi":      high,
        "lo":      low,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Qt helpers
# ─────────────────────────────────────────────────────────────────────────────


def array_to_qimage(arr: np.ndarray) -> QImage:
    h, w, c = arr.shape
    arr = np.ascontiguousarray(arr)
    return QImage(arr.data, w, h, w * c, QImage.Format.Format_RGB888)


# ─────────────────────────────────────────────────────────────────────────────
#  Worker thread
# ─────────────────────────────────────────────────────────────────────────────


class EdgeWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, pil_img):
        super().__init__()
        self.pil_img = pil_img

    def run(self):
        try:
            self.finished.emit(canny_pipeline(self.pil_img))
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Panel widget
# ─────────────────────────────────────────────────────────────────────────────


class PanelCard(QFrame):
    def __init__(self, title: str, subtitle: str, accent: str, parent=None):
        super().__init__(parent)
        self._full_pixmap = None
        self.setObjectName("panelCard")
        self.setStyleSheet("""
            QFrame#panelCard {
                background-color: #0f0f18;
                border: 1px solid #22223a;
                border-radius: 14px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        # Header
        hdr = QHBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"""
            color: {accent};
            font-family: 'Courier New', monospace;
            font-size: 12px;
            font-weight: bold;
            letter-spacing: 3px;
        """)
        sub_lbl = QLabel(subtitle)
        sub_lbl.setStyleSheet("""
            color: #333358;
            font-family: 'Courier New', monospace;
            font-size: 9px;
        """)
        hdr.addWidget(title_lbl)
        hdr.addStretch()
        hdr.addWidget(sub_lbl)
        layout.addLayout(hdr)

        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background-color: {accent}33;")
        layout.addWidget(div)

        self.img_lbl = QLabel("—")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setMinimumHeight(300)
        self.img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                    QSizePolicy.Policy.Expanding)
        self.img_lbl.setStyleSheet("""
            color: #2a2a40;
            font-size: 24px;
            background-color: #08080e;
            border-radius: 8px;
        """)
        layout.addWidget(self.img_lbl)

    def set_array(self, arr: np.ndarray):
        qi = array_to_qimage(arr)
        self._full_pixmap = QPixmap.fromImage(qi)
        self._refresh()

    def _refresh(self):
        if self._full_pixmap is None:
            return
        scaled = self._full_pixmap.scaled(
            self.img_lbl.width() or 500,
            self.img_lbl.height() or 400,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.img_lbl.setPixmap(scaled)
        self.img_lbl.setStyleSheet("background-color: #08080e; border-radius: 8px;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def clear(self):
        self._full_pixmap = None
        self.img_lbl.setPixmap(QPixmap())
        self.img_lbl.setText("—")
        self.img_lbl.setStyleSheet("""
            color: #2a2a40; font-size: 24px;
            background-color: #08080e; border-radius: 8px;
        """)


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────


class EdgeDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.source_image = None
        self.results      = {}
        self.worker       = None
        self.setWindowTitle("EDGE TRACER")
        self.setMinimumSize(1100, 720)
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar
        topbar = QWidget()
        topbar.setFixedHeight(58)
        topbar.setObjectName("topbar")
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(28, 0, 28, 0)

        brand = QLabel("EDGE TRACER")
        brand.setObjectName("brand")
        tb.addWidget(brand)

        pipe_lbl = QLabel("AUTO-CALIBRATED  ·  CANNY GRADIENT DETECTION")
        pipe_lbl.setObjectName("pipelineLbl")
        tb.addWidget(pipe_lbl)

        tb.addStretch()

        self.status_lbl = QLabel("NO IMAGE LOADED")
        self.status_lbl.setObjectName("statusLbl")
        tb.addWidget(self.status_lbl)

        root.addWidget(topbar)

        # Body
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(28, 20, 28, 28)
        body_layout.setSpacing(18)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.setSpacing(12)

        self.load_btn = QPushButton("⊕  LOAD IMAGE")
        self.load_btn.setObjectName("primaryBtn")
        self.load_btn.setFixedHeight(42)
        self.load_btn.clicked.connect(self._load_image)

        self.detect_btn = QPushButton("◈  DETECT EDGES")
        self.detect_btn.setObjectName("accentBtn")
        self.detect_btn.setFixedHeight(42)
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self._run_detection)

        self.export_btn = QPushButton("↓  EXPORT")
        self.export_btn.setObjectName("downloadBtn")
        self.export_btn.setFixedHeight(42)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export)

        # Source info label
        self.src_info_lbl = QLabel("")
        self.src_info_lbl.setObjectName("srcInfoLbl")

        ctrl.addWidget(self.load_btn)
        ctrl.addWidget(self.detect_btn)
        ctrl.addWidget(self.src_info_lbl)
        ctrl.addStretch()
        ctrl.addWidget(self.export_btn)
        body_layout.addLayout(ctrl)

        # Panels row: source preview + two output cards
        panels_row = QHBoxLayout()
        panels_row.setSpacing(16)

        # ── Narrow source preview card ────────────────────────────
        src_card = QFrame()
        src_card.setObjectName("srcCard")
        src_card.setFixedWidth(180)
        src_card.setStyleSheet("""
            QFrame#srcCard {
                background-color: #0f0f18;
                border: 1px solid #22223a;
                border-radius: 14px;
            }
        """)
        src_card_layout = QVBoxLayout(src_card)
        src_card_layout.setContentsMargins(12, 12, 12, 12)
        src_card_layout.setSpacing(8)

        src_hdr = QHBoxLayout()
        src_title = QLabel("SOURCE")
        src_title.setStyleSheet("""
            color: #6060a0;
            font-family: 'Courier New', monospace;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 3px;
        """)
        src_hdr.addWidget(src_title)
        src_hdr.addStretch()
        src_card_layout.addLayout(src_hdr)

        src_div = QFrame()
        src_div.setFixedHeight(1)
        src_div.setStyleSheet("background-color: #6060a033;")
        src_card_layout.addWidget(src_div)

        self.src_preview_lbl = QLabel("—")
        self.src_preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.src_preview_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                            QSizePolicy.Policy.Expanding)
        self.src_preview_lbl.setStyleSheet("""
            color: #2a2a40;
            font-size: 20px;
            background-color: #08080e;
            border-radius: 8px;
        """)
        src_card_layout.addWidget(self.src_preview_lbl, 1)

        self.src_fname_lbl = QLabel("awaiting input…")
        self.src_fname_lbl.setWordWrap(True)
        self.src_fname_lbl.setStyleSheet("""
            color: #3a3a5a;
            font-family: 'Courier New', monospace;
            font-size: 8px;
        """)
        src_card_layout.addWidget(self.src_fname_lbl)

        panels_row.addWidget(src_card)

        # ── Output cards ──────────────────────────────────────────
        self.card_edges   = PanelCard("EDGE MAP",  "BINARY  ·  HYSTERESIS OUTPUT", "#50fa7b")
        self.card_overlay = PanelCard("OVERLAY",   "EDGES ON ORIGINAL IMAGE",       "#bd93f9")

        panels_row.addWidget(self.card_edges, 1)
        panels_row.addWidget(self.card_overlay, 1)
        body_layout.addLayout(panels_row, 1)

        root.addWidget(body, 1)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d0d14;
                color: #c8c8d8;
            }
            QWidget#topbar {
                background-color: #09090f;
                border-bottom: 1px solid #1a1a28;
            }
            QLabel#brand {
                font-family: 'Courier New', monospace;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 6px;
                color: #e0e0f0;
                margin-right: 24px;
            }
            QLabel#pipelineLbl {
                font-family: 'Courier New', monospace;
                font-size: 9px;
                letter-spacing: 2px;
                color: #252545;
            }
            QLabel#statusLbl {
                font-family: 'Courier New', monospace;
                font-size: 10px;
                letter-spacing: 2px;
                color: #404060;
            }
            QLabel#srcInfoLbl {
                font-family: 'Courier New', monospace;
                font-size: 10px;
                letter-spacing: 1px;
                color: #4a4a6a;
            }
            QPushButton#primaryBtn {
                background-color: #1a1a28;
                color: #9090b0;
                border: 1px solid #2a2a40;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 2px;
                padding: 0 20px;
            }
            QPushButton#primaryBtn:hover {
                background-color: #22223a;
                color: #c0c0e0;
                border-color: #4a4a70;
            }
            QPushButton#accentBtn {
                background-color: #1a1428;
                color: #a070ff;
                border: 1px solid #4a3a80;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 2px;
                padding: 0 20px;
            }
            QPushButton#accentBtn:hover {
                background-color: #22183a;
                color: #c090ff;
                border-color: #7060b0;
            }
            QPushButton#accentBtn:disabled {
                color: #2a2a40;
                border-color: #1a1a28;
            }
            QPushButton#downloadBtn {
                background-color: #101a16;
                color: #40c080;
                border: 1px solid #1e5040;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 2px;
                padding: 0 20px;
            }
            QPushButton#downloadBtn:hover {
                background-color: #162a20;
                color: #60e0a0;
                border-color: #40a070;
            }
            QPushButton#downloadBtn:disabled {
                color: #2a2a40;
                border-color: #1a1a28;
            }
            QScrollBar:vertical {
                background: #0d0d14;
                width: 6px;
            }
            QScrollBar::handle:vertical {
                background: #2a2a40;
                border-radius: 3px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if not path:
            return
        try:
            self.source_image = Image.open(path).convert("RGB")
            w, h = self.source_image.size
            fname = os.path.basename(path)

            # Populate source preview card
            arr = np.array(self.source_image)
            qi  = array_to_qimage(arr)
            pix = QPixmap.fromImage(qi).scaled(
                156, 9999,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.src_preview_lbl.setPixmap(pix)
            self.src_preview_lbl.setStyleSheet(
                "background-color: #08080e; border-radius: 8px;"
            )
            self.src_fname_lbl.setText(f"{fname}\n{w} × {h} px")

            self.src_info_lbl.setText(f"{fname}  ·  {w} × {h} px")
            self.status_lbl.setText(f"LOADED  ·  {fname}")
            self.detect_btn.setEnabled(True)
            self.export_btn.setEnabled(False)
            self.results = {}
            self.card_edges.clear()
            self.card_overlay.clear()
        except Exception as e:
            self.status_lbl.setText(f"ERROR: {str(e)[:70]}")

    def _run_detection(self):
        if not self.source_image:
            return
        self.detect_btn.setEnabled(False)
        self.status_lbl.setText("PROCESSING…")
        QApplication.processEvents()
        self.worker = EdgeWorker(self.source_image)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_done(self, results: dict):
        self.results = results
        self.card_edges.set_array(results["edges"])
        self.card_overlay.set_array(results["overlay"])
        w, h = self.source_image.size
        self.status_lbl.setText(
            f"DONE  ·  {w}×{h}  ·  hi={results['hi']:.1f}  lo={results['lo']:.1f}"
        )
        self.detect_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self.status_lbl.setText(f"ERROR: {msg[:70]}")
        self.detect_btn.setEnabled(True)

    def _export(self):
        if not self.results:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "edge_detection_results.png",
            "PNG Image (*.png)"
        )
        if not save_path:
            return
        try:
            panels = [self.results["edges"], self.results["overlay"]]
            titles = ["EDGE MAP", "OVERLAY"]
            colors = [(80, 250, 123), (189, 147, 249)]

            H, W  = panels[0].shape[:2]
            gap   = 12
            lbl_h = 28
            total_w = 2 * W + gap
            total_h = H + lbl_h

            canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            canvas[lbl_h:, :W]     = panels[0]
            canvas[lbl_h:, W+gap:] = panels[1]

            pil_canvas = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil_canvas)

            for idx, (title, color) in enumerate(zip(titles, colors)):
                x0 = idx * (W + gap)
                tw = len(title) * 6
                draw.text((x0 + W // 2 - tw // 2, 8), title, fill=color)

            pil_canvas.save(save_path)
            self.status_lbl.setText(f"EXPORTED  ·  {os.path.basename(save_path)}")
        except Exception as e:
            self.status_lbl.setText(f"EXPORT ERROR: {str(e)[:60]}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(13, 13, 20))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(200, 200, 216))
    pal.setColor(QPalette.ColorRole.Base,            QColor(16, 16, 24))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(20, 20, 32))
    pal.setColor(QPalette.ColorRole.Button,          QColor(26, 26, 40))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(200, 200, 216))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(80, 60, 160))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(230, 230, 250))
    app.setPalette(pal)

    win = EdgeDetectorApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
