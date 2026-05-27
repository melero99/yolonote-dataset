"""
╔══════════════════════════════════════════════════════════════╗
║  functions/draw_box.py  —  Renderizado de bounding boxes     ║
╚══════════════════════════════════════════════════════════════╝

Función pura para dibujar una BoundingBox sobre un QPainter.
Sin dependencias de Qt más allá de QtGui/QtCore.
"""
from typing import Dict, Optional

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont

from yolo_annotator_pkg1.models import BoundingBox, YoloClass


def draw_box(
    painter: QPainter,
    box: BoundingBox,
    cls_map: Dict[int, YoloClass],
    img_w: int,
    img_h: int,
) -> None:
    """
    Dibuja una BoundingBox sobre el painter activo.

    Dibuja el rectángulo con el color de la clase, y una etiqueta con
    el nombre en la esquina superior izquierda.  El painter debe estar
    en el espacio de coordenadas de la imagen (escalado ya aplicado).

    Args:
        painter: QPainter activo con la transformación de imagen aplicada.
        box:     BoundingBox a dibujar.
        cls_map: Mapa {class_id: YoloClass} para resolver colores y nombres.
        img_w:   Ancho de la imagen en píxeles.
        img_h:   Alto de la imagen en píxeles.
    """
    cls: Optional[YoloClass] = cls_map.get(box.class_id)
    color = QColor(cls.color_hex) if cls else QColor("#ff4444")

    # Coordenadas absolutas de la caja
    x = (box.cx - box.w / 2) * img_w
    y = (box.cy - box.h / 2) * img_h
    w = box.w * img_w
    h = box.h * img_h

    # ── Rectángulo ─────────────────────────────────────────────
    pen = QPen(color, 1.5)
    pen.setCosmetic(True)  # grosor constante independiente del zoom
    painter.setPen(pen)
    painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
    painter.drawRect(QRectF(x, y, w, h))

    # ── Etiqueta ───────────────────────────────────────────────
    label = cls.name if cls else f"cls_{box.class_id}"
    font  = QFont("Consolas", 8)
    font.setBold(True)
    painter.setFont(font)

    fm          = painter.fontMetrics()
    label_w     = fm.horizontalAdvance(label) + 6
    label_h     = fm.height() + 2
    label_rect  = QRectF(x, y - label_h, label_w, label_h)

    # Fondo de la etiqueta
    painter.setBrush(QBrush(color))
    painter.setPen(Qt.NoPen)
    painter.drawRect(label_rect)

    # Texto de la etiqueta
    painter.setPen(QPen(QColor("#ffffff")))
    painter.drawText(label_rect, Qt.AlignCenter, label)
