"""
Paquete de diálogos Qt de YOLO Annotator Pro.

Exporta:
    AutoSplitDialog        — auto-asignación de splits.
    BalanceDialog          — balance inteligente de clases.
    ExportDialog           — exportación del dataset.
    FrameExtractorDialog   — extracción de frames de vídeo.
"""
from yolo_annotator_pkg1.dialogs.auto_split_dialog import AutoSplitDialog
from yolo_annotator_pkg1.dialogs.balance_dialog import BalanceDialog
from yolo_annotator_pkg1.dialogs.export_dialog import ExportDialog
from yolo_annotator_pkg1.dialogs.frame_extractor_dialog import FrameExtractorDialog

__all__ = [
    "AutoSplitDialog",
    "BalanceDialog",
    "ExportDialog",
    "FrameExtractorDialog",
]