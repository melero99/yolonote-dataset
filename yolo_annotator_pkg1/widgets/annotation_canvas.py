"""
╔══════════════════════════════════════════════════════════════╗
║  widgets/annotation_canvas.py  —  Canvas central de dibujo  ║
╚══════════════════════════════════════════════════════════════╝

Widget central que muestra la imagen activa y permite:
  - Dibujar bounding boxes arrastrando el ratón.
  - Eliminar boxes con doble clic.
  - Zoom con la rueda del ratón.
  - Pan con el botón central del ratón.

Las funciones de extracción de vídeo y utilidades de fichero
fueron movidas a yolo_annotator_pkg/functions/.

Signals:
    box_added     (BoundingBox): se emite al terminar de dibujar una caja.
    box_removed   ():            se emite al eliminar una caja (doble clic o undo).
    status_update (str):         mensaje para la barra de estado.
"""
import cv2
from pathlib import Path
from typing import List, Optional

from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage,
    QFont, QWheelEvent,
)

from yolo_annotator_pkg1.models import YoloClass, BoundingBox
from yolo_annotator_pkg1.resources import DARK_BG, TEXT_MUTED, BORDER_COLOR
from yolo_annotator_pkg1.functions.draw_box import draw_box as draw_box_fn


class AnnotationCanvas(QWidget):
    """
    Canvas central de anotación con zoom y pan.

    Responsabilidad única: renderizar la imagen y gestionar la
    interacción de dibujo/eliminación de bounding boxes.
    NO contiene lógica de clases ni de balance — esas están
    en ClassPanel y en los controllers.

    Signals:
        box_added     (BoundingBox): nueva caja creada por el usuario.
        box_removed   ():            caja eliminada (doble clic o undo externo).
        status_update (str):         texto para QMainWindow.statusBar().

    Args:
        parent: widget padre Qt.
    """

    box_added     = pyqtSignal(BoundingBox)
    box_removed   = pyqtSignal()
    status_update = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        self.setStyleSheet(f"background-color: {DARK_BG};")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # ── Estado de la imagen ────────────────────────────────
        self._pixmap:   Optional[QPixmap]      = None
        self._img_w:    int                    = 0
        self._img_h:    int                    = 0

        # ── Estado de las anotaciones ──────────────────────────
        self._boxes:         List[BoundingBox] = []
        self._classes:       List[YoloClass]   = []
        self._selected_class_id: Optional[int] = None

        # ── Estado de dibujado ────────────────────────────────
        self._drawing:   bool          = False
        self._start_pt:  Optional[QPointF] = None
        self._current_pt: Optional[QPointF] = None

        # ── Estado de zoom / pan ───────────────────────────────
        self._scale:    float  = 1.0
        self._offset:   QPointF = QPointF(0, 0)
        self._panning:  bool   = False
        self._pan_start: QPointF = QPointF(0, 0)

        # ── Callback de undo ──────────────────────────────────
        self._undo_cb: Optional[callable] = None

    # ══════════════════════════════════════════════════════════
    #  API PÚBLICA
    # ══════════════════════════════════════════════════════════

    def set_undo_callback(self, fn):
        """Registra la función a llamar al solicitar deshacer desde el canvas."""
        self._undo_cb = fn

    def load_image(self, path: str) -> bool:
        """
        Carga una imagen desde disco y la muestra ajustada a la ventana.

        Args:
            path: Ruta absoluta al archivo de imagen.

        Returns:
            True si la imagen se cargó correctamente, False en caso contrario.
        """
        img = cv2.imread(path)
        if img is None:
            return False

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._img_w  = w
        self._img_h  = h
        self._fit_to_window()
        self.update()
        return True

    def set_boxes(self, boxes: List[BoundingBox]):
        """Establece las anotaciones a mostrar y repinta."""
        self._boxes = list(boxes)
        self.update()

    def get_boxes(self) -> List[BoundingBox]:
        """Devuelve la lista de bounding boxes actuales."""
        return list(self._boxes)

    def set_classes(self, classes: List[YoloClass]):
        """Actualiza la lista de clases usada para colorear los boxes."""
        self._classes = classes
        self.update()

    def set_selected_class(self, class_id: Optional[int]):
        """
        Establece la clase activa para dibujar nuevas cajas.

        Args:
            class_id: id de la clase, o None para deseleccionar.
        """
        self._selected_class_id = class_id

    def remove_last_box(self):
        """Elimina la última bounding box añadida y emite box_removed."""
        if self._boxes:
            self._boxes.pop()
            self.box_removed.emit()
            self.update()

    # ══════════════════════════════════════════════════════════
    #  NAVEGACIÓN DE ZOOM / PAN
    # ══════════════════════════════════════════════════════════

    def _fit_to_window(self):
        """Ajusta escala y offset para que la imagen quepa en el canvas."""
        if not self._pixmap:
            return
        cw, ch = self.width(), self.height()
        iw, ih = self._img_w, self._img_h
        if cw == 0 or ch == 0 or iw == 0 or ih == 0:
            return
        self._scale = min(cw / iw, ch / ih)
        self._offset = QPointF(
            (cw - iw * self._scale) / 2,
            (ch - ih * self._scale) / 2,
        )

    def _zoom_step(self, factor: float):
        """
        Aplica un paso de zoom centrado en el centro del canvas.

        Args:
            factor: >1 acerca, <1 aleja.
        """
        cx, cy = self.width() / 2, self.height() / 2
        self._scale *= factor
        self._offset = QPointF(
            cx - (cx - self._offset.x()) * factor,
            cy - (cy - self._offset.y()) * factor,
        )
        self.update()

    def _widget_to_image(self, pt: QPointF) -> QPointF:
        """Convierte coordenadas de widget a coordenadas de imagen (píxeles)."""
        return QPointF(
            (pt.x() - self._offset.x()) / self._scale,
            (pt.y() - self._offset.y()) / self._scale,
        )

    def _image_to_widget(self, pt: QPointF) -> QPointF:
        """Convierte coordenadas de imagen a coordenadas de widget."""
        return QPointF(
            pt.x() * self._scale + self._offset.x(),
            pt.y() * self._scale + self._offset.y(),
        )

    # ══════════════════════════════════════════════════════════
    #  EVENTOS DE RATÓN
    # ══════════════════════════════════════════════════════════

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            # Iniciar pan
            self._panning   = True
            self._pan_start = QPointF(event.pos())
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            if self._selected_class_id is None:
                self.status_update.emit("⚠ Selecciona una clase antes de dibujar.")
                return
            self._drawing   = True
            self._start_pt  = self._widget_to_image(QPointF(event.pos()))
            self._current_pt = self._start_pt

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = QPointF(event.pos()) - self._pan_start
            self._offset += delta
            self._pan_start = QPointF(event.pos())
            self.update()
            return

        if self._drawing:
            self._current_pt = self._widget_to_image(QPointF(event.pos()))
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return

        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            if self._start_pt and self._current_pt:
                self._finalize_box()

    def mouseDoubleClickEvent(self, event):
        """Doble clic izquierdo: elimina la caja más cercana al cursor."""
        if event.button() == Qt.LeftButton:
            img_pt = self._widget_to_image(QPointF(event.pos()))
            self._delete_box_at(img_pt)

    def wheelEvent(self, event: QWheelEvent):
        """Rueda del ratón: zoom centrado en el cursor."""
        delta  = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1 / 1.15)
        cx, cy = event.pos().x(), event.pos().y()
        self._scale *= factor
        self._offset = QPointF(
            cx - (cx - self._offset.x()) * factor,
            cy - (cy - self._offset.y()) * factor,
        )
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_0:
            self._fit_to_window()
            self.update()
        elif event.key() == Qt.Key_Plus:
            self._zoom_step(1.3)
        elif event.key() == Qt.Key_Minus:
            self._zoom_step(1 / 1.3)

    def resizeEvent(self, event):
        if self._pixmap:
            self._fit_to_window()
        super().resizeEvent(event)

    # ══════════════════════════════════════════════════════════
    #  LÓGICA DE DIBUJO / BORRADO
    # ══════════════════════════════════════════════════════════

    def _finalize_box(self):
        """
        Convierte las coordenadas de arrastre en una BoundingBox normalizada
        y la emite vía box_added.
        """
        x1 = min(self._start_pt.x(), self._current_pt.x())
        y1 = min(self._start_pt.y(), self._current_pt.y())
        x2 = max(self._start_pt.x(), self._current_pt.x())
        y2 = max(self._start_pt.y(), self._current_pt.y())

        # Clampear a los límites de la imagen
        x1 = max(0.0, min(x1, self._img_w))
        y1 = max(0.0, min(y1, self._img_h))
        x2 = max(0.0, min(x2, self._img_w))
        y2 = max(0.0, min(y2, self._img_h))

        if x2 - x1 < 4 or y2 - y1 < 4:
            # Caja demasiado pequeña → ignorar
            return

        # Normalizar a coordenadas YOLO (0..1)
        cx = (x1 + x2) / 2 / self._img_w
        cy = (y1 + y2) / 2 / self._img_h
        bw = (x2 - x1) / self._img_w
        bh = (y2 - y1) / self._img_h

        box = BoundingBox(
            class_id=self._selected_class_id,
            cx=cx, cy=cy, w=bw, h=bh,
            img_w=self._img_w, img_h=self._img_h,
        )
        self._boxes.append(box)
        self.box_added.emit(box)
        self.update()

    def _delete_box_at(self, img_pt: QPointF):
        """
        Elimina la caja cuyo centro está más cerca de img_pt.

        Args:
            img_pt: Coordenadas en espacio de imagen donde se hizo doble clic.
        """
        best_idx  = -1
        best_dist = float("inf")

        for i, b in enumerate(self._boxes):
            info = b.pixel_info()
            bx   = info["x"] + info["w"] / 2
            by   = info["y"] + info["h"] / 2
            dist = ((img_pt.x() - bx) ** 2 + (img_pt.y() - by) ** 2) ** 0.5
            # Solo considerar si el clic cae dentro de la caja
            if (info["x"] <= img_pt.x() <= info["x"] + info["w"] and
                    info["y"] <= img_pt.y() <= info["y"] + info["h"]):
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i

        if best_idx >= 0:
            self._boxes.pop(best_idx)
            self.box_removed.emit()
            self.update()

    # ══════════════════════════════════════════════════════════
    #  PINTADO
    # ══════════════════════════════════════════════════════════

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # ── Fondo ─────────────────────────────────────────────
        p.fillRect(self.rect(), QColor(DARK_BG))

        if not self._pixmap:
            p.setPen(QColor(TEXT_MUTED))
            p.setFont(QFont("Consolas", 13))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Sin imagen\n\nAbre un proyecto y selecciona una imagen")
            # No llamar p.end() explícitamente — Qt lo hace al salir del método
            return

        # ── Imagen transformada ────────────────────────────────
        p.save()
        p.translate(self._offset)
        p.scale(self._scale, self._scale)
        p.drawPixmap(0, 0, self._pixmap)

        # ── Bounding boxes existentes ──────────────────────────
        cls_map = {c.id: c for c in self._classes}
        for box in self._boxes:
            draw_box_fn(p, box, cls_map, self._img_w, self._img_h)

        # ── Caja en construcción ───────────────────────────────
        if self._drawing and self._start_pt and self._current_pt:
            x1 = min(self._start_pt.x(), self._current_pt.x())
            y1 = min(self._start_pt.y(), self._current_pt.y())
            x2 = max(self._start_pt.x(), self._current_pt.x())
            y2 = max(self._start_pt.y(), self._current_pt.y())
            pen = QPen(QColor("#ffffff"), 1.5 / self._scale, Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

        p.restore()
        # No llamar p.end() explícitamente — Qt lo hace al destruir el objeto