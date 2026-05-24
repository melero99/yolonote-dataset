"""
╔══════════════════════════════════════════════════════════════╗
║          YOLO Annotator Pro — Gestor completo de datasets    ║
╚══════════════════════════════════════════════════════════════╝

Instalación:
    pip install PyQt5 opencv-python

Ejecución:
    python yolo_annotator.py

Flujo de trabajo:
    1. File → New Project  (crea un proyecto nuevo)
    2. File → Open Project (abre un .yannotator existente)
    3. Añade clases en el panel izquierdo
    4. Selecciona una clase y dibuja bounding boxes
    5. Asigna cada imagen a TRAIN / VALID / TEST
    6. Project → Export Dataset  genera la estructura final

Atajos:
    ← / →          Imagen anterior / siguiente
    1-9            Seleccionar clase por número
    Ctrl+S         Guardar proyecto
    Ctrl+Z         Deshacer última caja
    Doble clic     Eliminar caja bajo el cursor
    Rueda ratón    Zoom in / out
"""

import sys, os, json, shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QInputDialog,
    QColorDialog, QMessageBox, QSizePolicy, QStatusBar,
    QAction, QMenuBar, QShortcut, QDialog, QDialogButtonBox,
    QLineEdit, QFormLayout, QListWidget, QListWidgetItem,
    QProgressDialog, QSpinBox, QDoubleSpinBox, QGroupBox,
    QRadioButton, QButtonGroup, QScrollArea, QSplitter,
    QComboBox, QCheckBox, QTabWidget, QTextEdit
)
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage, QCursor,
    QFont, QKeySequence, QIcon, QPalette, QTransform, QWheelEvent
)
import cv2


# ══════════════════════════════════════════════════════════════
#  FUNCIÓN HELPER PARA RECURSOS
# ══════════════════════════════════════════════════════════════

def resource_path(relative_path):
    """Obtiene la ruta absoluta del recurso, funciona para dev y PyInstaller"""
    try:
        # PyInstaller crea una carpeta temporal y guarda la ruta en _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ══════════════════════════════════════════════════════════════
#  ESTILOS GLOBALES
# ══════════════════════════════════════════════════════════════

DARK_BG      = "#0d0d1f"
PANEL_BG     = "#12122a"
CARD_BG      = "#1a1a3a"
BORDER_COLOR = "#2a2a5a"
TEXT_PRIMARY = "#e0e0f0"
TEXT_MUTED   = "#7777aa"
ACCENT       = "#4a4aaa"

SPLIT_COLORS = {
    "train": QColor("#1a5c2a"),
    "valid": QColor("#1a3a6a"),
    "test":  QColor("#5c2a1a"),
    None:    QColor("#333355"),
}
SPLIT_TEXT = {
    "train": "TRAIN",
    "valid": "VALID",
    "test":  "TEST",
    None:    "—",
}

def btn_style(bg="#2a2a5a", hover="#3a3a7a", border=BORDER_COLOR):
    return f"""
        QPushButton {{
            background-color: {bg};
            color: {TEXT_PRIMARY};
            border: 1px solid {border};
            border-radius: 6px;
            padding: 5px 12px;
            font-size: 12px;
        }}
        QPushButton:hover {{ background-color: {hover}; }}
        QPushButton:pressed {{ background-color: #1a1a4a; }}
        QPushButton:disabled {{ color: #444466; border-color: #222244; }}
    """


# ══════════════════════════════════════════════════════════════
#  MODELOS DE DATOS
# ══════════════════════════════════════════════════════════════

@dataclass
class YoloClass:
    id: int
    name: str
    color_hex: str = "#FF4444"

    @property
    def color(self) -> QColor:
        return QColor(self.color_hex)

    def to_dict(self):
        return {"id": self.id, "name": self.name, "color": self.color_hex}

    @staticmethod
    def from_dict(d):
        return YoloClass(id=d["id"], name=d["name"], color_hex=d.get("color", "#FF4444"))


@dataclass
class BoundingBox:
    class_id: int
    x1: int; y1: int; x2: int; y2: int

    def normalized(self, w, h):
        cx = ((self.x1 + self.x2) / 2) / w
        cy = ((self.y1 + self.y2) / 2) / h
        bw = abs(self.x2 - self.x1) / w
        bh = abs(self.y2 - self.y1) / h
        return cx, cy, bw, bh

    def to_yolo_line(self, w, h):
        cx, cy, bw, bh = self.normalized(w, h)
        return f"{self.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    def is_valid(self, min_px=5):
        return abs(self.x2-self.x1) >= min_px and abs(self.y2-self.y1) >= min_px

    def pixel_info(self):
        return dict(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2,
                    cx=(self.x1+self.x2)//2, cy=(self.y1+self.y2)//2,
                    w=abs(self.x2-self.x1), h=abs(self.y2-self.y1))

    def to_dict(self):
        return {"class_id": self.class_id,
                "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @staticmethod
    def from_dict(d):
        return BoundingBox(class_id=d["class_id"],
                           x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


@dataclass
class ImageRecord:
    path: str
    split: Optional[str] = None
    verified: bool = False
    boxes: List[BoundingBox] = field(default_factory=list)

    def to_dict(self):
        return {
            "path": self.path,
            "split": self.split,
            "verified": self.verified,
            "boxes": [b.to_dict() for b in self.boxes],
        }

    @staticmethod
    def from_dict(d):
        return ImageRecord(
            path=d["path"],
            split=d.get("split"),
            verified=d.get("verified", False),
            boxes=[BoundingBox.from_dict(b) for b in d.get("boxes", [])],
        )


@dataclass
class Project:
    name: str = "Sin título"
    images_folder: str = ""
    current_index: int = 0
    classes: List[YoloClass] = field(default_factory=list)
    images: List[ImageRecord] = field(default_factory=list)
    project_file: Optional[str] = None

    def to_dict(self):
        return {
            "name": self.name,
            "images_folder": self.images_folder,
            "current_index": self.current_index,
            "classes": [c.to_dict() for c in self.classes],
            "images": [i.to_dict() for i in self.images],
        }

    @staticmethod
    def from_dict(d):
        return Project(
            name=d.get("name", "Sin título"),
            images_folder=d.get("images_folder", ""),
            current_index=d.get("current_index", 0),
            classes=[YoloClass.from_dict(c) for c in d.get("classes", [])],
            images=[ImageRecord.from_dict(i) for i in d.get("images", [])],
        )

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        self.project_file = path

    @staticmethod
    def load(path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        p = Project.from_dict(d)
        p.project_file = path
        return p

    def get_class(self, class_id: int) -> Optional[YoloClass]:
        return next((c for c in self.classes if c.id == class_id), None)

    def next_class_id(self) -> int:
        return max((c.id for c in self.classes), default=-1) + 1

    def stats(self):
        total = len(self.images)
        splits = {s: sum(1 for i in self.images if i.split == s)
                  for s in ("train", "valid", "test")}
        unassigned = sum(1 for i in self.images if i.split is None)
        annotated  = sum(1 for i in self.images if i.boxes)
        return dict(total=total, annotated=annotated, unassigned=unassigned, **splits)


# ══════════════════════════════════════════════════════════════
#  CANVAS DE ANOTACIÓN (con zoom)
# ══════════════════════════════════════════════════════════════

class AnnotationCanvas(QWidget):
    box_added     = pyqtSignal(BoundingBox)
    box_removed   = pyqtSignal()
    status_update = pyqtSignal(str)

    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background-color: {DARK_BG};")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

        self._pixmap: Optional[QPixmap] = None
        self._img_w = self._img_h = 0

        self._boxes: List[BoundingBox] = []
        self._classes: List[YoloClass] = []
        self._selected_class_id: Optional[int] = None

        self._drawing = False
        self._draw_start: Optional[QPointF] = None
        self._draw_end:   Optional[QPointF] = None

        self._zoom   = 1.0
        self._offset = QPointF(0, 0)
        self._pan_active = False
        self._pan_last:  Optional[QPoint] = None

    # ── API pública ────────────────────────────────────────────

    def load_image(self, path: str) -> bool:
        img = cv2.imread(path)
        if img is None:
            return False
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._img_h, self._img_w = h, w
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._boxes  = []
        self._fit_to_window()
        self.update()
        return True

    def set_boxes(self, boxes: List[BoundingBox]):
        self._boxes = list(boxes)
        self.update()

    def get_boxes(self) -> List[BoundingBox]:
        return list(self._boxes)

    def set_classes(self, classes: List[YoloClass]):
        self._classes = classes

    def set_selected_class(self, class_id: Optional[int]):
        self._selected_class_id = class_id
        self.setCursor(Qt.CrossCursor if class_id is not None else Qt.ArrowCursor)

    def remove_last_box(self):
        if self._boxes:
            self._boxes.pop()
            self.update()

    def clear_boxes(self):
        self._boxes.clear()
        self.update()

    # ── Zoom & layout ──────────────────────────────────────────

    def _fit_to_window(self):
        if self._pixmap is None:
            return
        cw, ch = self.width() or 800, self.height() or 600
        self._zoom = min(cw / self._img_w, ch / self._img_h) * 0.95
        self._center_image()

    def _center_image(self):
        if self._pixmap is None:
            return
        cw, ch = self.width(), self.height()
        iw = self._img_w * self._zoom
        ih = self._img_h * self._zoom
        self._offset = QPointF((cw - iw) / 2, (ch - ih) / 2)

    def _canvas_to_img(self, pt: QPointF) -> QPointF:
        x = (pt.x() - self._offset.x()) / self._zoom
        y = (pt.y() - self._offset.y()) / self._zoom
        x = max(0.0, min(x, float(self._img_w)))
        y = max(0.0, min(y, float(self._img_h)))
        return QPointF(x, y)

    def _img_to_canvas(self, x: float, y: float) -> QPointF:
        return QPointF(x * self._zoom + self._offset.x(),
                       y * self._zoom + self._offset.y())

    # ── Eventos ────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        mouse_pos = QPointF(event.pos())
        self._offset = QPointF(
            mouse_pos.x() - (mouse_pos.x() - self._offset.x()) * factor,
            mouse_pos.y() - (mouse_pos.y() - self._offset.y()) * factor,
        )
        self._zoom = max(0.1, min(self._zoom * factor, 20.0))
        self.update()
        self.status_update.emit(f"Zoom: {self._zoom*100:.0f}%")

    def mousePressEvent(self, event):
        if self._pixmap is None:
            return
        if event.button() == Qt.MiddleButton:
            self._pan_active = True
            self._pan_last = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            if self._selected_class_id is not None:
                self._drawing    = True
                self._draw_start = QPointF(event.pos())
                self._draw_end   = QPointF(event.pos())

    def mouseMoveEvent(self, event):
        if self._pan_active and self._pan_last:
            dx = event.pos().x() - self._pan_last.x()
            dy = event.pos().y() - self._pan_last.y()
            self._offset += QPointF(dx, dy)
            self._pan_last = event.pos()
            self.update()
        elif self._drawing:
            self._draw_end = QPointF(event.pos())
            self.update()

        if self._pixmap:
            ip = self._canvas_to_img(QPointF(event.pos()))
            self.status_update.emit(
                f"x:{int(ip.x())}  y:{int(ip.y())}  |  imagen:{self._img_w}×{self._img_h}  |  zoom:{self._zoom*100:.0f}%"
            )

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_active = False
            cursor = Qt.CrossCursor if self._selected_class_id is not None else Qt.ArrowCursor
            self.setCursor(cursor)
        elif event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            if self._draw_start and self._draw_end and self._selected_class_id is not None:
                p1 = self._canvas_to_img(self._draw_start)
                p2 = self._canvas_to_img(self._draw_end)
                box = BoundingBox(
                    class_id=self._selected_class_id,
                    x1=int(min(p1.x(), p2.x())),
                    y1=int(min(p1.y(), p2.y())),
                    x2=int(max(p1.x(), p2.x())),
                    y2=int(max(p1.y(), p2.y())),
                )
                if box.is_valid():
                    self._boxes.append(box)
                    self.box_added.emit(box)
            self._draw_start = self._draw_end = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap:
            ip = self._canvas_to_img(QPointF(event.pos()))
            for i in range(len(self._boxes)-1, -1, -1):
                b = self._boxes[i]
                if b.x1 <= ip.x() <= b.x2 and b.y1 <= ip.y() <= b.y2:
                    self._boxes.pop(i)
                    self.box_removed.emit()
                    self.update()
                    break

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom_step(1.2)
        elif event.key() == Qt.Key_Minus:
            self._zoom_step(1/1.2)
        elif event.key() == Qt.Key_0:
            self._fit_to_window()
            self.update()

    def _zoom_step(self, factor):
        cx, cy = self.width()/2, self.height()/2
        self._offset = QPointF(
            cx - (cx - self._offset.x()) * factor,
            cy - (cy - self._offset.y()) * factor,
        )
        self._zoom = max(0.1, min(self._zoom * factor, 20.0))
        self.update()

    # ── Pintado ────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(DARK_BG))

        if self._pixmap is None:
            p.setPen(QColor("#555577"))
            p.setFont(QFont("Arial", 15))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "📂  Abre o crea un proyecto para empezar")
            p.end()
            return

        iw = int(self._img_w * self._zoom)
        ih = int(self._img_h * self._zoom)
        ox, oy = int(self._offset.x()), int(self._offset.y())
        scaled = self._pixmap.scaled(iw, ih, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        p.drawPixmap(ox, oy, scaled)

        for box in self._boxes:
            cls = self._get_class(box.class_id)
            self._draw_box(p, box, cls)

        if self._drawing and self._draw_start and self._draw_end and self._selected_class_id is not None:
            cls = self._get_class(self._selected_class_id)
            color = cls.color if cls else QColor("#ffff00")
            x1 = min(self._draw_start.x(), self._draw_end.x())
            y1 = min(self._draw_start.y(), self._draw_end.y())
            x2 = max(self._draw_start.x(), self._draw_end.x())
            y2 = max(self._draw_start.y(), self._draw_end.y())
            pen = QPen(color, 2, Qt.DashLine)
            p.setPen(pen)
            fill = QColor(color); fill.setAlpha(35)
            p.setBrush(QBrush(fill))
            p.drawRect(QRectF(x1, y1, x2-x1, y2-y1))

        p.end()

    def _draw_box(self, p: QPainter, box: BoundingBox, cls: Optional[YoloClass]):
        color = cls.color if cls else QColor("#ff0000")
        name  = cls.name  if cls else f"#{box.class_id}"

        c1 = self._img_to_canvas(box.x1, box.y1)
        c2 = self._img_to_canvas(box.x2, box.y2)
        rect = QRectF(c1, c2)

        fill = QColor(color); fill.setAlpha(40)
        p.setBrush(QBrush(fill))
        p.setPen(QPen(color, 2))
        p.drawRect(rect)

        fm   = p.fontMetrics()
        tw   = fm.horizontalAdvance(name) + 10
        th   = fm.height() + 5
        tag  = QRectF(c1.x(), c1.y() - th, tw, th)
        if tag.top() < 0:
            tag.moveTop(c1.y())
        p.fillRect(tag, color)
        p.setPen(Qt.white)
        p.setFont(QFont("Consolas", 9, QFont.Bold))
        p.drawText(tag, Qt.AlignCenter, name)

    def _get_class(self, cid: int) -> Optional[YoloClass]:
        return next((c for c in self._classes if c.id == cid), None)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap and self._zoom < 0.11:
            self._fit_to_window()
        self.update()


# ══════════════════════════════════════════════════════════════
#  PANEL DE CLASES
# ══════════════════════════════════════════════════════════════

class ClassPanel(QWidget):
    class_selected      = pyqtSignal(int)
    class_deselected    = pyqtSignal()
    class_add_requested    = pyqtSignal()
    class_delete_requested = pyqtSignal(int)
    split_assigned      = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(230)
        self.setStyleSheet(f"""
            QWidget {{ background-color: {PANEL_BG}; color: {TEXT_PRIMARY}; }}
            QListWidget {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                color: {TEXT_PRIMARY};
                font-size: 12px;
            }}
            QListWidget::item {{ padding: 5px; border-radius: 3px; }}
            QListWidget::item:selected {{ background-color: {ACCENT}; }}
            QLabel {{ color: {TEXT_MUTED}; font-size: 11px; }}
            QGroupBox {{
                color: {TEXT_MUTED};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                margin-top: 8px;
                font-size: 11px;
                font-weight: bold;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; padding: 0 6px; }}
        """)
        self._undo_cb = None
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # ── Clases ──
        grp_cls = QGroupBox("CLASES")
        cls_lay = QVBoxLayout(grp_cls)
        cls_lay.setSpacing(4)

        self.class_list = QListWidget()
        self.class_list.itemClicked.connect(self._on_class_clicked)
        cls_lay.addWidget(self.class_list)

        row = QHBoxLayout()
        btn_add = QPushButton("＋ Añadir")
        btn_add.setStyleSheet(btn_style("#2a4a2a", "#3a6a3a"))
        btn_add.clicked.connect(lambda: self.class_add_requested.emit())

        btn_del = QPushButton("✕ Quitar")
        btn_del.setStyleSheet(btn_style("#4a2a2a", "#6a3a3a"))
        btn_del.clicked.connect(self._on_delete_clicked)

        row.addWidget(btn_add)
        row.addWidget(btn_del)
        cls_lay.addLayout(row)

        btn_desel = QPushButton("⊘ Deseleccionar")
        btn_desel.setStyleSheet(btn_style())
        btn_desel.clicked.connect(self._deselect)
        cls_lay.addWidget(btn_desel)
        lay.addWidget(grp_cls)

        hint = QLabel("  Teclas 1–9 seleccionan clase")
        hint.setStyleSheet(f"color: #555588; font-size: 10px; font-style: italic;")
        lay.addWidget(hint)

        # ── Anotaciones ──
        grp_ann = QGroupBox("ANOTACIONES (imagen actual)")
        ann_lay = QVBoxLayout(grp_ann)
        self.ann_list = QListWidget()
        self.ann_list.setStyleSheet(f"QListWidget {{ background: {DARK_BG}; border-radius: 4px; }}")
        ann_lay.addWidget(self.ann_list)
        btn_undo = QPushButton("↩ Deshacer última")
        btn_undo.setStyleSheet(btn_style())
        btn_undo.clicked.connect(lambda: self._undo_cb and self._undo_cb())
        ann_lay.addWidget(btn_undo)
        lay.addWidget(grp_ann)

        # ── Split asignación ──
        grp_split = QGroupBox("ASIGNAR IMAGEN A SPLIT")
        split_lay = QVBoxLayout(grp_split)
        self.split_label = QLabel("Split actual: —")
        self.split_label.setAlignment(Qt.AlignCenter)
        self.split_label.setStyleSheet("font-size: 13px; font-weight: bold; color: white;")
        split_lay.addWidget(self.split_label)
        row2 = QHBoxLayout()
        for split, color in [("train","#1a5c2a"), ("valid","#1a3a6a"), ("test","#5c2a1a")]:
            b = QPushButton(split.upper())
            b.setStyleSheet(btn_style(color, color))
            b.clicked.connect(lambda _, s=split: self.split_assigned.emit(s))
            row2.addWidget(b)
        split_lay.addLayout(row2)
        lay.addWidget(grp_split)

        lay.addStretch()

        self.stats_label = QLabel("Sin proyecto")
        self.stats_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        self.stats_label.setWordWrap(True)
        lay.addWidget(self.stats_label)

    # ── API ────────────────────────────────────────────────────

    def set_undo_callback(self, fn):
        self._undo_cb = fn

    def load_classes(self, classes: List[YoloClass]):
        self.class_list.clear()
        for cls in classes:
            item = QListWidgetItem(f"  {cls.id+1}.  {cls.name}")
            item.setForeground(cls.color)
            pix = QPixmap(12, 12); pix.fill(cls.color)
            item.setIcon(QIcon(pix))
            self.class_list.addItem(item)

    def update_annotations(self, boxes: List[BoundingBox], classes: List[YoloClass]):
        self.ann_list.clear()
        cls_map = {c.id: c for c in classes}
        for i, b in enumerate(boxes):
            cls = cls_map.get(b.class_id)
            name = cls.name if cls else f"class_{b.class_id}"
            info = b.pixel_info()
            item = QListWidgetItem(f"  #{i+1} {name}  [{info['w']}×{info['h']}]")
            if cls:
                item.setForeground(cls.color)
            self.ann_list.addItem(item)

    def update_split_label(self, split: Optional[str]):
        text  = SPLIT_TEXT.get(split, "—")
        color = SPLIT_COLORS.get(split, QColor("#333355"))
        self.split_label.setText(f"Split actual: {text}")
        self.split_label.setStyleSheet(
            f"font-size: 13px; font-weight: bold; "
            f"color: white; background-color: {color.name()}; "
            f"border-radius: 4px; padding: 4px;"
        )

    def update_stats(self, stats: dict):
        self.stats_label.setText(
            f"Total: {stats['total']} | Anotadas: {stats['annotated']}\n"
            f"Train: {stats['train']}  Valid: {stats['valid']}  Test: {stats['test']}\n"
            f"Sin asignar: {stats['unassigned']}"
        )

    # ── Handlers internos ──────────────────────────────────────

    def _on_delete_clicked(self):
        row = self.class_list.currentRow()
        if row >= 0:
            self.class_delete_requested.emit(row)
        else:
            QMessageBox.information(self, "Sin selección",
                                    "Selecciona una clase de la lista para eliminarla.")

    def _on_class_clicked(self, item):
        row = self.class_list.row(item)
        self.class_selected.emit(row)

    def _deselect(self):
        self.class_list.clearSelection()
        self.class_deselected.emit()

    # ── Diálogo de nueva clase ──

    def open_add_class_dialog(self) -> tuple:
        dialog = QDialog(self)
        dialog.setWindowTitle("Nueva clase")
        dialog.setStyleSheet(f"background:{CARD_BG}; color:{TEXT_PRIMARY};")
        lay = QFormLayout(dialog)

        name_in = QLineEdit()
        name_in.setPlaceholderText("ej. jugador, balón, portero…")
        name_in.setStyleSheet(
            f"background:{DARK_BG}; color:white; "
            f"border:1px solid {BORDER_COLOR}; border-radius:4px; padding:4px;"
        )
        lay.addRow("Nombre:", name_in)

        chosen = [QColor("#FF4444")]
        cbtn = QPushButton("  Elegir color")
        cbtn.setStyleSheet(
            f"background:{chosen[0].name()}; color:white; border-radius:4px; padding:4px;"
        )
        def pick():
            c = QColorDialog.getColor(chosen[0], dialog)
            if c.isValid():
                chosen[0] = c
                cbtn.setStyleSheet(
                    f"background:{c.name()}; color:white; border-radius:4px; padding:4px;"
                )
        cbtn.clicked.connect(pick)
        lay.addRow("Color:", cbtn)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.setStyleSheet("color:white;")
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        lay.addRow(btns)

        if dialog.exec_() == QDialog.Accepted:
            name = name_in.text().strip()
            if name:
                return name, chosen[0]
        return None, None


# ══════════════════════════════════════════════════════════════
#  DIÁLOGO DE EXPORTACIÓN
# ══════════════════════════════════════════════════════════════

class ExportDialog(QDialog):
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Exportar Dataset")
        self.setMinimumWidth(480)
        self.setStyleSheet(f"background:{CARD_BG}; color:{TEXT_PRIMARY};")
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(12)

        stats = self.project.stats()
        info = QLabel(
            f"📊  Train: {stats['train']}  |  Valid: {stats['valid']}  |  "
            f"Test: {stats['test']}  |  Sin asignar: {stats['unassigned']}"
        )
        info.setStyleSheet(
            f"color:{TEXT_MUTED}; font-size:11px; padding:6px; "
            f"background:{DARK_BG}; border-radius:4px;"
        )
        lay.addWidget(info)

        lay.addWidget(QLabel("Carpeta de destino del dataset:"))
        row = QHBoxLayout()
        self.dest_edit = QLineEdit()
        self.dest_edit.setPlaceholderText("Selecciona carpeta…")
        self.dest_edit.setStyleSheet(
            f"background:{DARK_BG}; color:white; "
            f"border:1px solid {BORDER_COLOR}; border-radius:4px; padding:4px;"
        )
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(32)
        btn_browse.setStyleSheet(btn_style())
        btn_browse.clicked.connect(self._browse)
        row.addWidget(self.dest_edit)
        row.addWidget(btn_browse)
        lay.addLayout(row)

        lay.addWidget(QLabel("Nombre del dataset:"))
        self.name_edit = QLineEdit(self.project.name.replace(" ", "_"))
        self.name_edit.setStyleSheet(
            f"background:{DARK_BG}; color:white; "
            f"border:1px solid {BORDER_COLOR}; border-radius:4px; padding:4px;"
        )
        lay.addWidget(self.name_edit)

        self.copy_images = QCheckBox("Copiar imágenes (si no marcado, solo genera labels)")
        self.copy_images.setChecked(True)
        self.copy_images.setStyleSheet(f"color:{TEXT_PRIMARY};")
        lay.addWidget(self.copy_images)

        self.gen_yaml = QCheckBox("Generar data.yaml para YOLO")
        self.gen_yaml.setChecked(True)
        self.gen_yaml.setStyleSheet(f"color:{TEXT_PRIMARY};")
        lay.addWidget(self.gen_yaml)

        grp = QGroupBox("Auto-asignación de imágenes sin split")
        grp.setStyleSheet(
            f"QGroupBox {{ color:{TEXT_MUTED}; border:1px solid {BORDER_COLOR}; "
            f"border-radius:4px; margin-top:6px; }}"
        )
        grp_lay = QHBoxLayout(grp)
        self.auto_split = QCheckBox("Activar auto-split")
        self.auto_split.setStyleSheet(f"color:{TEXT_PRIMARY};")
        grp_lay.addWidget(self.auto_split)
        grp_lay.addWidget(QLabel("Train %:"))
        self.pct_train = QSpinBox(); self.pct_train.setRange(1,98); self.pct_train.setValue(80)
        grp_lay.addWidget(self.pct_train)
        grp_lay.addWidget(QLabel("Valid %:"))
        self.pct_valid = QSpinBox(); self.pct_valid.setRange(1,98); self.pct_valid.setValue(10)
        grp_lay.addWidget(self.pct_valid)
        lay.addWidget(grp)

        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setFixedHeight(130)
        preview.setStyleSheet(
            f"background:{DARK_BG}; color:{TEXT_MUTED}; "
            f"font-family:Consolas; font-size:11px; border-radius:4px;"
        )
        preview.setPlainText(
            "dataset_name/\n"
            "├── train/\n│   ├── images/\n│   └── labels/\n"
            "├── valid/\n│   ├── images/\n│   └── labels/\n"
            "├── test/\n│   ├── images/\n│   └── labels/\n"
            "└── data.yaml"
        )
        lay.addWidget(preview)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.setStyleSheet("color:white;")
        btns.button(QDialogButtonBox.Ok).setText("🚀  Exportar")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de destino")
        if folder:
            self.dest_edit.setText(folder)

    def get_config(self):
        return {
            "dest": self.dest_edit.text().strip(),
            "name": self.name_edit.text().strip() or "dataset",
            "copy_images": self.copy_images.isChecked(),
            "gen_yaml": self.gen_yaml.isChecked(),
            "auto_split": self.auto_split.isChecked(),
            "pct_train": self.pct_train.value(),
            "pct_valid": self.pct_valid.value(),
        }


# ══════════════════════════════════════════════════════════════
#  VENTANA PRINCIPAL
# ══════════════════════════════════════════════════════════════

class YoloAnnotatorPro(QMainWindow):

    SUPPORTED = {".jpg",".jpeg",".png",".bmp",".tiff",".tif",".webp"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Annotator Pro")
        
        # Configurar icono de la ventana
        icon_path = resource_path("logoapp.png")
        if os.path.exists(icon_path):
            app_icon = QIcon(icon_path)
            self.setWindowIcon(app_icon)
            # También establecer el icono de la aplicación
            QApplication.setWindowIcon(app_icon)
        
        self.setMinimumSize(1150, 720)
        self._apply_theme()

        self.project: Optional[Project] = None
        self._dirty = False

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self.statusBar().showMessage("Bienvenido. Crea un proyecto nuevo o abre uno existente.")

    # ── UI ─────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        # Panel lateral
        self.panel = ClassPanel()
        self.panel.class_selected.connect(self._on_class_selected)
        self.panel.class_deselected.connect(self._on_class_deselected)
        self.panel.class_add_requested.connect(self._handle_add_class)
        self.panel.class_delete_requested.connect(self._handle_delete_class)
        self.panel.split_assigned.connect(self._assign_split)
        self.panel.set_undo_callback(self._undo_last)
        root.addWidget(self.panel)

        # Centro
        center = QWidget()
        center.setStyleSheet(f"background:{DARK_BG};")
        clayout = QVBoxLayout(center)
        clayout.setContentsMargins(0,0,0,0)
        clayout.setSpacing(0)

        # Info bar
        self.info_bar = QLabel("Sin proyecto")
        self.info_bar.setAlignment(Qt.AlignCenter)
        self.info_bar.setStyleSheet(
            f"background:{PANEL_BG}; color:{TEXT_MUTED}; padding:6px; "
            f"font-size:12px; font-family:Consolas,monospace; "
            f"border-bottom:1px solid {BORDER_COLOR};"
        )
        clayout.addWidget(self.info_bar)

        # Nav + canvas
        nav = QHBoxLayout()
        nav.setContentsMargins(0,0,0,0)
        nav.setSpacing(0)

        self.btn_prev = self._nav_btn("◀", self._go_prev)
        nav.addWidget(self.btn_prev)

        self.canvas = AnnotationCanvas()
        self.canvas.box_added.connect(self._on_box_added)
        self.canvas.box_removed.connect(self._on_box_removed)
        self.canvas.status_update.connect(self.statusBar().showMessage)
        nav.addWidget(self.canvas, 1)

        self.btn_next = self._nav_btn("▶", self._go_next)
        nav.addWidget(self.btn_next)
        clayout.addLayout(nav, 1)

        # Barra inferior
        bot = QWidget()
        bot.setStyleSheet(f"background:{PANEL_BG}; border-top:1px solid {BORDER_COLOR};")
        bot_lay = QHBoxLayout(bot)
        bot_lay.setContentsMargins(10,5,10,5)

        self.verified_chk = QCheckBox("✔ Imagen verificada")
        self.verified_chk.setStyleSheet(f"color:{TEXT_PRIMARY};")
        self.verified_chk.stateChanged.connect(self._toggle_verified)
        bot_lay.addWidget(self.verified_chk)

        self.output_lbl = QLabel("💾  Proyecto sin guardar")
        self.output_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px;")
        bot_lay.addWidget(self.output_lbl)
        bot_lay.addStretch()

        for label, cb, style in [
            ("Guardar proyecto  Ctrl+S", self._save_project, btn_style("#2a3a6a","#4a6aaa")),
            ("📤 Exportar dataset",       self._export_dataset, btn_style("#3a2a6a","#6a4aaa")),
        ]:
            b = QPushButton(label)
            b.setStyleSheet(style)
            b.clicked.connect(cb)
            bot_lay.addWidget(b)

        clayout.addWidget(bot)
        root.addWidget(center, 1)

    def _nav_btn(self, text, cb):
        b = QPushButton(text)
        b.setFixedWidth(48)
        b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        b.setStyleSheet(f"""
            QPushButton {{ background:{PANEL_BG}; color:#8888cc; border:none; font-size:22px; }}
            QPushButton:hover {{ background:{CARD_BG}; color:white; }}
            QPushButton:disabled {{ color:#333355; }}
        """)
        b.clicked.connect(cb)
        return b

    def _build_menu(self):
        mb = self.menuBar()
        mb.setStyleSheet(f"background:{PANEL_BG}; color:{TEXT_PRIMARY};")

        fm = mb.addMenu("Archivo")
        self._add_action(fm, "🆕  Nuevo proyecto",            "Ctrl+N", self._new_project)
        self._add_action(fm, "📂  Abrir proyecto",             "Ctrl+O", self._open_project)
        self._add_action(fm, "💾  Guardar proyecto",           "Ctrl+S", self._save_project)
        self._add_action(fm, "💾  Guardar como…",              "",       self._save_project_as)
        fm.addSeparator()
        self._add_action(fm, "📁  Añadir carpeta de imágenes", "",       self._add_images_folder)
        fm.addSeparator()
        self._add_action(fm, "📤  Exportar dataset",           "",       self._export_dataset)
        self._add_action(fm, "📋  Exportar classes.txt",       "",       self._export_classes_txt)

        pm = mb.addMenu("Proyecto")
        self._add_action(pm, "🔀  Auto-asignar splits",  "",  self._auto_split_dialog)
        self._add_action(pm, "📊  Estadísticas",         "",  self._show_stats)
        self._add_action(pm, "🗑  Limpiar sin split",     "",  self._clear_unassigned)

        vm = mb.addMenu("Vista")
        self._add_action(vm, "🔍  Ajustar a ventana  (0)", "",
                         lambda: (self.canvas._fit_to_window(), self.canvas.update()))
        self._add_action(vm, "🔍+  Zoom in  (+)",  "", lambda: self.canvas._zoom_step(1.3))
        self._add_action(vm, "🔍−  Zoom out  (−)", "", lambda: self.canvas._zoom_step(1/1.3))

        tm = mb.addMenu("Herramientas")
        self._add_action(tm, "🎬  Extractor de Frames…", "Ctrl+E", self._open_frame_extractor)

        hm = mb.addMenu("Ayuda")
        self._add_action(hm, "ℹ️  Atajos de teclado", "", self._show_help)

    def _add_action(self, menu, label, shortcut, callback):
        a = QAction(label, self)
        if shortcut:
            a.setShortcut(shortcut)
        a.triggered.connect(callback)
        menu.addAction(a)
        return a

    def _build_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left),  self, self._go_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._go_next)
        QShortcut(QKeySequence("Ctrl+Z"),     self, self._undo_last)
        QShortcut(QKeySequence("Ctrl+E"),     self, self._open_frame_extractor)
        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self,
                      lambda idx=i-1: self._select_class_by_index(idx))

    # ── Proyecto: nuevo / abrir / guardar ──────────────────────

    def _new_project(self):
        if not self._confirm_unsaved():
            return
        name, ok = QInputDialog.getText(self, "Nuevo proyecto", "Nombre del proyecto:")
        if not ok or not name.strip():
            return
        self.project = Project(name=name.strip())
        self._dirty = True
        self._refresh_all()
        self.statusBar().showMessage(
            f"Proyecto '{name}' creado. Añade imágenes con Archivo → Añadir carpeta."
        )

    def _open_project(self):
        if not self._confirm_unsaved():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir proyecto",
            filter="Proyectos YOLO (*.yannotator);;Todos (*)"
        )
        if not path:
            return
        try:
            self.project = Project.load(path)
            self._dirty = False
            self._refresh_all()
            self._go_to(self.project.current_index)
            self.statusBar().showMessage(
                f"Proyecto '{self.project.name}' cargado. {len(self.project.images)} imágenes."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo abrir el proyecto:\n{e}")

    def _save_project(self):
        if self.project is None:
            return
        self._store_current()
        if not self.project.project_file:
            self._save_project_as()
            return
        try:
            self.project.save(self.project.project_file)
            self._dirty = False
            self.output_lbl.setText(f"💾  {Path(self.project.project_file).name}")
            self.statusBar().showMessage("✅ Proyecto guardado.")
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))

    def _save_project_as(self):
        if self.project is None:
            return
        self._store_current()
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar proyecto como",
            self.project.name + ".yannotator",
            "Proyectos YOLO (*.yannotator)"
        )
        if path:
            try:
                self.project.save(path)
                self._dirty = False
                self.output_lbl.setText(f"💾  {Path(path).name}")
                self.statusBar().showMessage(f"✅ Guardado en {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _add_images_folder(self):
        if self.project is None:
            QMessageBox.information(self, "Sin proyecto", "Crea o abre un proyecto primero.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de imágenes")
        if not folder:
            return
        folder_path = Path(folder)
        existing = {r.path for r in self.project.images}
        added = 0
        for p in sorted(folder_path.iterdir()):
            if p.suffix.lower() in self.SUPPORTED and str(p) not in existing:
                self.project.images.append(ImageRecord(path=str(p)))
                added += 1
        self._dirty = True
        self._refresh_all()
        if self._current_idx() < 0 and self.project.images:
            self._go_to(0)
        self.statusBar().showMessage(
            f"✅ {added} imágenes añadidas de '{folder_path.name}'."
        )

    # ── Navegación ─────────────────────────────────────────────

    def _current_idx(self) -> int:
        return self.project.current_index if self.project else -1

    def _go_to(self, idx: int):
        if not self.project or not self.project.images:
            return
        self._store_current()
        idx = max(0, min(idx, len(self.project.images)-1))
        self.project.current_index = idx
        rec = self.project.images[idx]

        if not self.canvas.load_image(rec.path):
            self.statusBar().showMessage(f"❌ No se puede cargar: {rec.path}")
            return

        self.canvas.set_boxes(rec.boxes)
        self.canvas.set_classes(self.project.classes)

        total     = len(self.project.images)
        name      = Path(rec.path).name
        split_txt = SPLIT_TEXT.get(rec.split, "—")
        verified  = "✔" if rec.verified else ""
        self.info_bar.setText(
            f"[{idx+1} / {total}]   {name}   |   Split: {split_txt}   "
            f"{verified}   |   {len(rec.boxes)} anotaciones"
        )
        self.verified_chk.blockSignals(True)
        self.verified_chk.setChecked(rec.verified)
        self.verified_chk.blockSignals(False)

        self.panel.update_annotations(rec.boxes, self.project.classes)
        self.panel.update_split_label(rec.split)
        self.panel.update_stats(self.project.stats())

        self.btn_prev.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < total-1)

    def _go_prev(self):
        if self.project:
            self._go_to(self._current_idx() - 1)

    def _go_next(self):
        if self.project:
            self._go_to(self._current_idx() + 1)

    def _store_current(self):
        if not self.project or not self.project.images:
            return
        idx = self._current_idx()
        if 0 <= idx < len(self.project.images):
            self.project.images[idx].boxes = self.canvas.get_boxes()

    # ── Clases ─────────────────────────────────────────────────

    def _on_class_selected(self, row: int):
        if not self.project or row >= len(self.project.classes):
            return
        cls = self.project.classes[row]
        self.canvas.set_selected_class(cls.id)
        self.statusBar().showMessage(
            f"Clase: {cls.name}  (id={cls.id})  |  tecla {row+1}"
        )

    def _on_class_deselected(self):
        self.canvas.set_selected_class(None)

    def _select_class_by_index(self, idx: int):
        if not self.project or idx >= len(self.project.classes):
            return
        self.panel.class_list.setCurrentRow(idx)
        self._on_class_selected(idx)

    def _handle_add_class(self):
        if not self.project:
            QMessageBox.information(self, "Sin proyecto",
                                    "Crea o abre un proyecto primero.")
            return
        name, color = self.panel.open_add_class_dialog()
        if name:
            cls = YoloClass(
                id=self.project.next_class_id(),
                name=name,
                color_hex=color.name(),
            )
            self.project.classes.append(cls)
            self._dirty = True
            self._refresh_classes()
            self.statusBar().showMessage(
                f"Clase '{name}' añadida (id={cls.id})."
            )

    def _handle_delete_class(self, row: int):
        if not self.project or row >= len(self.project.classes):
            return
        cls = self.project.classes[row]
        reply = QMessageBox.question(
            self, "Eliminar clase",
            f"¿Eliminar la clase '{cls.name}'?\n"
            f"Las cajas de esta clase en las imágenes no se borrarán del proyecto.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.project.classes.pop(row)
        # Reasignar IDs correlativos
        for i, c in enumerate(self.project.classes):
            c.id = i
        self._dirty = True
        self._refresh_classes()
        self.canvas.set_selected_class(None)
        self.statusBar().showMessage(f"Clase '{cls.name}' eliminada.")

    def _refresh_classes(self):
        if not self.project:
            return
        self.panel.load_classes(self.project.classes)
        self.canvas.set_classes(self.project.classes)

    def _refresh_all(self):
        if not self.project:
            return
        self.setWindowTitle(f"YOLO Annotator Pro — {self.project.name}")
        self._refresh_classes()
        self.panel.update_stats(self.project.stats())

    # ── Anotaciones ────────────────────────────────────────────

    def _on_box_added(self, box: BoundingBox):
        self._store_current()
        if self.project and self.project.images:
            rec = self.project.images[self._current_idx()]
            self.panel.update_annotations(rec.boxes, self.project.classes)
        self._dirty = True

    def _on_box_removed(self):
        self._store_current()
        if self.project and self.project.images:
            rec = self.project.images[self._current_idx()]
            self.panel.update_annotations(rec.boxes, self.project.classes)
        self._dirty = True

    def _undo_last(self):
        self.canvas.remove_last_box()
        self._store_current()
        if self.project and self.project.images:
            rec = self.project.images[self._current_idx()]
            self.panel.update_annotations(rec.boxes, self.project.classes)
        self._dirty = True

    # ── Split ──────────────────────────────────────────────────

    def _assign_split(self, split: str):
        if not self.project or not self.project.images:
            return
        rec = self.project.images[self._current_idx()]
        rec.split = split
        self.panel.update_split_label(split)
        self.panel.update_stats(self.project.stats())
        split_txt = SPLIT_TEXT[split]
        parts = self.info_bar.text().split("|")
        if len(parts) >= 2:
            parts[1] = f"   Split: {split_txt}   "
            self.info_bar.setText("|".join(parts))
        self._dirty = True

    def _toggle_verified(self, state):
        if not self.project or not self.project.images:
            return
        self.project.images[self._current_idx()].verified = bool(state)
        self._dirty = True

    def _auto_split_dialog(self):
        if not self.project:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Auto-asignar splits")
        dialog.setStyleSheet(f"background:{CARD_BG}; color:{TEXT_PRIMARY};")
        lay = QFormLayout(dialog)

        pct_train = QSpinBox(); pct_train.setRange(1,98); pct_train.setValue(80)
        pct_valid = QSpinBox(); pct_valid.setRange(1,98); pct_valid.setValue(10)
        only_unassigned = QCheckBox("Solo imágenes sin split")
        only_unassigned.setChecked(True)
        only_unassigned.setStyleSheet(f"color:{TEXT_PRIMARY};")

        lay.addRow("Train %:", pct_train)
        lay.addRow("Valid %:", pct_valid)
        lay.addRow("Test %: resto", QLabel("(calculado automáticamente)"))
        lay.addRow(only_unassigned)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.setStyleSheet("color:white;")
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        lay.addRow(btns)

        if dialog.exec_() != QDialog.Accepted:
            return

        import random
        pt, pv = pct_train.value(), pct_valid.value()
        images = [r for r in self.project.images
                  if not only_unassigned.isChecked() or r.split is None]
        random.shuffle(images)
        n = len(images)
        n_train = int(n * pt / 100)
        n_valid = int(n * pv / 100)
        for i, rec in enumerate(images):
            if i < n_train:              rec.split = "train"
            elif i < n_train + n_valid:  rec.split = "valid"
            else:                        rec.split = "test"
        self.panel.update_stats(self.project.stats())
        self._dirty = True
        self.statusBar().showMessage(f"Auto-split aplicado a {n} imágenes.")

    def _clear_unassigned(self):
        if not self.project:
            return
        n = sum(1 for r in self.project.images if r.split is None)
        if n == 0:
            QMessageBox.information(self, "Info", "No hay imágenes sin asignar.")
            return
        reply = QMessageBox.question(
            self, "Confirmar",
            f"¿Eliminar {n} imágenes sin split del proyecto?\n(No borra archivos del disco)",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.project.images = [r for r in self.project.images if r.split is not None]
            self._dirty = True
            self.panel.update_stats(self.project.stats())

    # ── Exportación ────────────────────────────────────────────

    def _export_dataset(self):
        if not self.project:
            QMessageBox.information(self, "Sin proyecto", "Crea o abre un proyecto primero.")
            return
        self._store_current()

        dlg = ExportDialog(self.project, self)
        if dlg.exec_() != QDialog.Accepted:
            return

        cfg = dlg.get_config()
        if not cfg["dest"]:
            QMessageBox.warning(self, "Sin destino", "Selecciona una carpeta de destino.")
            return

        if cfg["auto_split"]:
            import random
            unassigned = [r for r in self.project.images if r.split is None]
            random.shuffle(unassigned)
            n = len(unassigned)
            nt = int(n * cfg["pct_train"] / 100)
            nv = int(n * cfg["pct_valid"] / 100)
            for i, rec in enumerate(unassigned):
                rec.split = "train" if i < nt else ("valid" if i < nt+nv else "test")

        root = Path(cfg["dest"]) / cfg["name"]

        for split in ("train", "valid", "test"):
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)

        prog = QProgressDialog(
            "Exportando dataset…", "Cancelar", 0, len(self.project.images), self
        )
        prog.setWindowModality(Qt.WindowModal)
        prog.setStyleSheet(f"background:{CARD_BG}; color:{TEXT_PRIMARY};")

        counts = {"train":0, "valid":0, "test":0, "skipped":0}
        for i, rec in enumerate(self.project.images):
            prog.setValue(i)
            if prog.wasCanceled():
                break
            if not rec.split:
                counts["skipped"] += 1
                continue

            img_path  = Path(rec.path)
            split_dir = root / rec.split

            if cfg["copy_images"]:
                dest_img = split_dir / "images" / img_path.name
                shutil.copy2(str(img_path), str(dest_img))

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            label_path = split_dir / "labels" / (img_path.stem + ".txt")
            with open(label_path, "w") as f:
                for box in rec.boxes:
                    f.write(box.to_yolo_line(iw, ih) + "\n")
            counts[rec.split] += 1

        prog.setValue(len(self.project.images))

        if cfg["gen_yaml"]:
            yaml_path = root / "data.yaml"
            with open(yaml_path, "w") as f:
                f.write(f"path: {root}\n")
                f.write(f"train: train/images\n")
                f.write(f"val:   valid/images\n")
                f.write(f"test:  test/images\n\n")
                f.write(f"nc: {len(self.project.classes)}\n")
                names = [c.name for c in sorted(self.project.classes, key=lambda x: x.id)]
                f.write(f"names: {names}\n")

        classes_path = root / "classes.txt"
        with open(classes_path, "w") as f:
            for cls in sorted(self.project.classes, key=lambda x: x.id):
                f.write(cls.name + "\n")

        QMessageBox.information(
            self, "✅ Exportación completada",
            f"Dataset exportado en:\n{root}\n\n"
            f"Train: {counts['train']}  |  Valid: {counts['valid']}  |  Test: {counts['test']}\n"
            f"Omitidas (sin split): {counts['skipped']}"
        )
        self.statusBar().showMessage(f"✅ Dataset exportado en {root}")

    def _export_classes_txt(self):
        if not self.project or not self.project.classes:
            QMessageBox.information(self, "Sin clases", "No hay clases definidas.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar classes.txt", "classes.txt")
        if path:
            with open(path, "w") as f:
                for cls in sorted(self.project.classes, key=lambda x: x.id):
                    f.write(cls.name + "\n")
            self.statusBar().showMessage(f"✅ {path}")

    # ── Stats ──────────────────────────────────────────────────

    def _show_stats(self):
        if not self.project:
            return
        s = self.project.stats()
        total_boxes = sum(len(r.boxes) for r in self.project.images)
        cls_counts  = {}
        for rec in self.project.images:
            for b in rec.boxes:
                cls_counts[b.class_id] = cls_counts.get(b.class_id, 0) + 1
        cls_lines = "\n".join(
            f"  {self.project.get_class(cid).name if self.project.get_class(cid) else cid}: {n}"
            for cid, n in sorted(cls_counts.items())
        )
        QMessageBox.information(self, "Estadísticas del proyecto", f"""
Proyecto: {self.project.name}

Imágenes:
  Total:       {s['total']}
  Anotadas:    {s['annotated']}
  Verificadas: {sum(1 for r in self.project.images if r.verified)}
  Sin asignar: {s['unassigned']}

Splits:
  Train: {s['train']}
  Valid: {s['valid']}
  Test:  {s['test']}

Cajas totales: {total_boxes}

Por clase:
{cls_lines if cls_lines else '  (sin anotaciones)'}
        """)

    # ── Cierre ─────────────────────────────────────────────────

    def _confirm_unsaved(self) -> bool:
        if not self._dirty or not self.project:
            return True
        reply = QMessageBox.question(
            self, "Cambios sin guardar",
            "Hay cambios sin guardar. ¿Continuar de todas formas?",
            QMessageBox.Yes | QMessageBox.No
        )
        return reply == QMessageBox.Yes

    def closeEvent(self, event):
        self._store_current()
        if self._dirty and self.project:
            reply = QMessageBox.question(
                self, "Guardar al salir",
                "¿Guardar el proyecto antes de salir?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Yes:
                self._save_project()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        event.accept()

    # ── Extractor de frames ────────────────────────────────────

    def _open_frame_extractor(self):
        dlg = FrameExtractorDialog(project=self.project, parent=self)
        dlg.exec_()
        if self.project:
            self._refresh_all()
            self.panel.update_stats(self.project.stats())

    # ── Ayuda ──────────────────────────────────────────────────

    def _show_help(self):
        QMessageBox.information(self, "Atajos de teclado", """
Herramientas:
  Ctrl+E             Abrir Extractor de Frames

Navegación:
  ←  /  →           Imagen anterior / siguiente
  Rueda ratón       Zoom in / out (centrado en cursor)
  Botón central     Pan (arrastrar vista)
  0                 Ajustar zoom a ventana
  +  /  −           Zoom in / out (paso fijo)

Anotación:
  Clic + arrastrar  Dibujar bounding box
  Doble clic        Eliminar caja bajo el cursor
  Ctrl+Z            Deshacer última caja
  1 – 9             Seleccionar clase por número

Proyecto:
  Ctrl+N            Nuevo proyecto
  Ctrl+O            Abrir proyecto
  Ctrl+S            Guardar proyecto

Splits:
  Botones TRAIN / VALID / TEST en el panel izquierdo
  asignan la imagen actual a ese split.
        """)

    # ── Tema ───────────────────────────────────────────────────

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow   {{ background:{DARK_BG}; }}
            QMenuBar      {{ background:{PANEL_BG}; color:{TEXT_PRIMARY};
                            border-bottom:1px solid {BORDER_COLOR}; }}
            QMenuBar::item:selected {{ background:{CARD_BG}; }}
            QMenu         {{ background:{CARD_BG}; color:{TEXT_PRIMARY};
                            border:1px solid {BORDER_COLOR}; }}
            QMenu::item:selected {{ background:{ACCENT}; }}
            QStatusBar    {{ background:{PANEL_BG}; color:{TEXT_MUTED};
                            font-family:Consolas,monospace; font-size:11px;
                            border-top:1px solid {BORDER_COLOR}; }}
            QDialog       {{ background:{CARD_BG}; }}
            QMessageBox   {{ background:{CARD_BG}; color:{TEXT_PRIMARY}; }}
            QInputDialog  {{ background:{CARD_BG}; color:{TEXT_PRIMARY}; }}
            QScrollBar:vertical {{ background:{DARK_BG}; width:8px; }}
            QScrollBar::handle:vertical {{ background:{BORDER_COLOR}; border-radius:4px; }}
        """)


# ══════════════════════════════════════════════════════════════
#  EXTRACTOR DE FRAMES (integrado)
# ══════════════════════════════════════════════════════════════

class FrameExtractorDialog(QDialog):
    def __init__(self, project: Optional["Project"], parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Extractor de Frames")
        self.setMinimumSize(580, 520)
        self.setStyleSheet(f"background:{CARD_BG}; color:{TEXT_PRIMARY};")
        self._last_output_folder: Optional[Path] = None
        self._running = False
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(16, 16, 16, 16)

        # ── Fuente ──
        grp_src = QGroupBox("Fuente del vídeo")
        grp_src.setStyleSheet(self._grp_style())
        src_lay = QVBoxLayout(grp_src)

        mode_row = QHBoxLayout()
        self.rb_local   = QRadioButton("Archivo local")
        self.rb_youtube = QRadioButton("URL de YouTube")
        self.rb_local.setChecked(True)
        for rb in (self.rb_local, self.rb_youtube):
            rb.setStyleSheet(f"color:{TEXT_PRIMARY};")
            mode_row.addWidget(rb)
        mode_row.addStretch()
        src_lay.addLayout(mode_row)

        self.local_widget = QWidget()
        local_row = QHBoxLayout(self.local_widget)
        local_row.setContentsMargins(0,0,0,0)
        self.local_edit = QLineEdit()
        self.local_edit.setPlaceholderText("Ruta al archivo de vídeo…")
        self.local_edit.setStyleSheet(self._input_style())
        btn_browse_vid = QPushButton("…")
        btn_browse_vid.setFixedWidth(34)
        btn_browse_vid.setStyleSheet(btn_style())
        btn_browse_vid.clicked.connect(self._browse_video)
        local_row.addWidget(self.local_edit)
        local_row.addWidget(btn_browse_vid)
        src_lay.addWidget(self.local_widget)

        self.yt_widget = QWidget()
        yt_lay = QVBoxLayout(self.yt_widget)
        yt_lay.setContentsMargins(0,0,0,0)
        self.yt_edit = QLineEdit()
        self.yt_edit.setPlaceholderText("https://www.youtube.com/watch?v=...")
        self.yt_edit.setStyleSheet(self._input_style())
        yt_lay.addWidget(self.yt_edit)
        yt_note = QLabel("⚠️  Requiere:  pip install yt-dlp")
        yt_note.setStyleSheet(f"color:#aa8800; font-size:10px;")
        yt_lay.addWidget(yt_note)
        self.yt_widget.hide()
        src_lay.addWidget(self.yt_widget)

        self.rb_local.toggled.connect(self._toggle_source)
        lay.addWidget(grp_src)

        # ── Opciones ──
        grp_opt = QGroupBox("Opciones de extracción")
        grp_opt.setStyleSheet(self._grp_style())
        opt_form = QFormLayout(grp_opt)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 30.0)
        self.fps_spin.setValue(1.0)
        self.fps_spin.setSingleStep(0.5)
        self.fps_spin.setSuffix("  frame/s")
        self.fps_spin.setStyleSheet(self._input_style())
        opt_form.addRow("Cadencia de extracción:", self.fps_spin)

        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["jpg", "png"])
        self.fmt_combo.setStyleSheet(self._input_style())
        opt_form.addRow("Formato de imagen:", self.fmt_combo)

        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)
        self.quality_spin.setSuffix("  %")
        self.quality_spin.setStyleSheet(self._input_style())
        opt_form.addRow("Calidad JPEG:", self.quality_spin)

        self.minw_spin = QSpinBox()
        self.minw_spin.setRange(0, 7680)
        self.minw_spin.setValue(0)
        self.minw_spin.setSuffix("  px  (0 = sin límite)")
        self.minw_spin.setStyleSheet(self._input_style())
        opt_form.addRow("Ancho mínimo:", self.minw_spin)

        lay.addWidget(grp_opt)

        # ── Salida ──
        grp_out = QGroupBox("Carpeta de salida")
        grp_out.setStyleSheet(self._grp_style())
        out_lay = QHBoxLayout(grp_out)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Selecciona carpeta base…")
        self.out_edit.setStyleSheet(self._input_style())
        btn_browse_out = QPushButton("…")
        btn_browse_out.setFixedWidth(34)
        btn_browse_out.setStyleSheet(btn_style())
        btn_browse_out.clicked.connect(self._browse_output)
        out_lay.addWidget(self.out_edit)
        out_lay.addWidget(btn_browse_out)
        lay.addWidget(grp_out)

        # ── Log ──
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(110)
        self.log.setStyleSheet(
            f"background:{DARK_BG}; color:#aaffaa; "
            f"font-family:Consolas,monospace; font-size:11px; border-radius:4px;"
        )
        lay.addWidget(self.log)

        # ── Progreso ──
        from PyQt5.QtWidgets import QProgressBar
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setStyleSheet(f"""
            QProgressBar {{ background:{DARK_BG}; border-radius:4px;
                           color:white; text-align:center; }}
            QProgressBar::chunk {{ background:{ACCENT}; border-radius:4px; }}
        """)
        lay.addWidget(self.progress)

        # ── Botones ──
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  Extraer frames")
        self.btn_start.setStyleSheet(btn_style("#1a5c2a", "#2a8a3a"))
        self.btn_start.clicked.connect(self._start_extraction)

        self.btn_add = QPushButton("➕  Añadir al proyecto")
        self.btn_add.setStyleSheet(btn_style("#2a3a6a", "#4a6aaa"))
        self.btn_add.setEnabled(False)
        self.btn_add.clicked.connect(self._add_to_project)

        btn_close = QPushButton("Cerrar")
        btn_close.setStyleSheet(btn_style())
        btn_close.clicked.connect(self.close)

        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_add)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

    def _toggle_source(self, local: bool):
        self.local_widget.setVisible(local)
        self.yt_widget.setVisible(not local)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecciona vídeo",
            filter="Vídeos (*.mp4 *.mkv *.avi *.mov *.webm *.ts);;Todos (*)"
        )
        if path:
            self.local_edit.setText(path)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Carpeta base de salida")
        if folder:
            self.out_edit.setText(folder)

    def _log(self, msg: str):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        QApplication.processEvents()

    def _start_extraction(self):
        if self._running:
            return

        out_folder = self.out_edit.text().strip()
        if not out_folder:
            QMessageBox.warning(self, "Sin carpeta", "Selecciona una carpeta de salida.")
            return

        is_local = self.rb_local.isChecked()

        if is_local:
            video_path_str = self.local_edit.text().strip()
            if not video_path_str:
                QMessageBox.warning(self, "Sin vídeo", "Selecciona un archivo de vídeo.")
                return
            if not Path(video_path_str).exists():
                QMessageBox.warning(self, "No encontrado",
                                    f"No existe el archivo:\n{video_path_str}")
                return
        else:
            if not self.yt_edit.text().strip():
                QMessageBox.warning(self, "Sin URL", "Introduce una URL de YouTube.")
                return

        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.progress.setValue(0)
        self.log.clear()

        base_out = Path(out_folder)

        try:
            if is_local:
                video_path = Path(self.local_edit.text().strip())
                self._extract(video_path, video_path.stem, base_out)
            else:
                self._download_and_extract(self.yt_edit.text().strip(), base_out)
        except Exception as e:
            self._log(f"❌ Error: {e}")
        finally:
            self._running = False
            self.btn_start.setEnabled(True)

    def _limpiar_nombre(self, nombre: str) -> str:
        import re
        nombre = re.sub(r'[\\/*?:"<>|]', "", nombre)
        nombre = nombre.replace(" ", "_")
        return nombre[:80]

    def _crear_carpeta(self, base: Path, nombre: str) -> Path:
        nombre  = self._limpiar_nombre(nombre)
        carpeta = base / nombre
        if carpeta.exists():
            i = 2
            while (base / f"{nombre}_{i}").exists():
                i += 1
            carpeta = base / f"{nombre}_{i}"
        carpeta.mkdir(parents=True, exist_ok=True)
        return carpeta

    def _download_and_extract(self, url: str, base_out: Path):
        try:
            import yt_dlp
        except ImportError:
            self._log("❌  yt-dlp no instalado. Ejecuta:  pip install yt-dlp")
            return

        temp_dir = base_out / "_temp_yt"
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = None

        try:
            self._log("⬇️  Conectando con YouTube…")
            QApplication.processEvents()

            opciones = {
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "outtmpl": str(temp_dir / "%(title)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
            }

            with yt_dlp.YoutubeDL(opciones) as ydl:
                info = ydl.extract_info(url, download=True)
                titulo = info.get("title", "video_youtube")

            candidates = (list(temp_dir.glob("*.mp4")) +
                          list(temp_dir.glob("*.mkv")) +
                          list(temp_dir.glob("*.webm")))
            if not candidates:
                self._log("❌  No se encontró el archivo descargado.")
                return

            video_path = candidates[0]
            self._log(f"✅  Descargado: {video_path.stem}")
            self._extract(video_path, video_path.stem, base_out)

        finally:
            if video_path and video_path.exists():
                video_path.unlink()
            if temp_dir.exists():
                try:
                    temp_dir.rmdir()
                except OSError:
                    pass

    def _extract(self, video_path: Path, nombre: str, base_out: Path):
        carpeta  = self._crear_carpeta(base_out, nombre)
        fps_obj  = self.fps_spin.value()
        fmt      = self.fmt_combo.currentText()
        quality  = self.quality_spin.value()
        min_w    = self.minw_spin.value()
        params   = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self._log(f"❌  No se puede abrir: {video_path}")
            return

        fps_video    = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duracion_s   = total_frames / fps_video
        intervalo    = max(1, round(fps_video / fps_obj))
        ancho        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._log(f"📹  {video_path.name}")
        self._log(
            f"    Resolución: {ancho}×{alto}  |  {fps_video:.1f} fps  |  {duracion_s/60:.1f} min"
        )
        self._log(
            f"    Extrayendo: {fps_obj} frame/s  →  "
            f"~{int(duracion_s * fps_obj)} frames estimados"
        )
        self._log(f"    Destino:    {carpeta}")

        if min_w and ancho < min_w:
            self._log(
                f"⚠️  Vídeo más estrecho que el mínimo ({ancho} < {min_w}). "
                f"Se extrae igualmente."
            )

        guardados = 0
        frame_idx = 0
        self.progress.setMaximum(max(total_frames, 1))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % intervalo == 0:
                nombre_frame = f"frame_{frame_idx:07d}.{fmt}"
                cv2.imwrite(str(carpeta / nombre_frame), frame, params)
                guardados += 1
                if guardados % 50 == 0:
                    self.progress.setValue(frame_idx)
                    self._log(
                        f"    [{frame_idx/max(total_frames,1)*100:5.1f}%]  "
                        f"{guardados} frames guardados…"
                    )
            frame_idx += 1

        cap.release()
        self.progress.setValue(total_frames)
        self._log(f"\n✅  Completado: {guardados} frames en '{carpeta.name}/'")
        self._last_output_folder = carpeta

        if self.project is not None:
            self.btn_add.setEnabled(True)

    def _add_to_project(self):
        if not self._last_output_folder or not self.project:
            return
        folder    = self._last_output_folder
        SUPPORTED = {".jpg",".jpeg",".png",".bmp",".tiff",".tif",".webp"}
        existing  = {r.path for r in self.project.images}
        added = 0
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in SUPPORTED and str(p) not in existing:
                self.project.images.append(ImageRecord(path=str(p)))
                added += 1
        self._log(f"➕  {added} imágenes añadidas al proyecto '{self.project.name}'.")
        if self.parent() and hasattr(self.parent(), '_refresh_all'):
            self.parent()._refresh_all()
            if added > 0 and len(self.project.images) == added:
                self.parent()._go_to(0)
        QMessageBox.information(
            self, "Añadido",
            f"{added} imágenes añadidas al proyecto.\n"
            "Puedes cerrar esta ventana y empezar a anotar."
        )

    def _grp_style(self):
        return (
            f"QGroupBox {{ color:{TEXT_MUTED}; border:1px solid {BORDER_COLOR}; "
            f"border-radius:6px; margin-top:8px; font-size:11px; font-weight:bold; }}"
            f"QGroupBox::title {{ subcontrol-origin:margin; padding:0 6px; }}"
        )

    def _input_style(self):
        return (
            f"background:{DARK_BG}; color:{TEXT_PRIMARY}; "
            f"border:1px solid {BORDER_COLOR}; border-radius:4px; padding:4px;"
        )


# ══════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Annotator Pro")
    app.setStyle("Fusion")

    # Configurar icono global de la aplicación
    icon_path = resource_path("logoapp.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(DARK_BG))
    pal.setColor(QPalette.WindowText,      QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.Base,            QColor(PANEL_BG))
    pal.setColor(QPalette.AlternateBase,   QColor(CARD_BG))
    pal.setColor(QPalette.Text,            QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.Button,          QColor(CARD_BG))
    pal.setColor(QPalette.ButtonText,      QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.Highlight,       QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)

    win = YoloAnnotatorPro()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
