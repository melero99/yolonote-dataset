"""
Paquete de widgets Qt reutilizables de YOLO Annotator Pro.

Exporta:
    AnnotationCanvas   — canvas central de dibujo con zoom/pan.
    ClassBalanceBar    — barra de balance por clase.
    ClassPanel         — panel lateral de clases y splits.
"""
from yolo_annotator_pkg1.widgets.annotation_canvas import AnnotationCanvas
from yolo_annotator_pkg1.widgets.class_balance_bar import ClassBalanceBar
from yolo_annotator_pkg1.widgets.class_panel import ClassPanel

__all__ = ["AnnotationCanvas", "ClassBalanceBar", "ClassPanel"]
