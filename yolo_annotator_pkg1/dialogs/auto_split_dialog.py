"""
╔══════════════════════════════════════════════════════════════╗
║  dialogs/auto_split_dialog.py  —  Auto-asignación de splits  ║
╚══════════════════════════════════════════════════════════════╝

Pequeño diálogo modal para configurar la auto-asignación aleatoria
de imágenes a los splits TRAIN / VALID / TEST.

Antes estaba incrustado directamente en YoloAnnotatorPro._auto_split_dialog;
ahora es un componente reutilizable y testeable.
"""
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QSpinBox, QLabel,
    QCheckBox, QDialogButtonBox,
)

from yolo_annotator_pkg1.resources import CARD_BG, TEXT_PRIMARY


class AutoSplitDialog(QDialog):
    """
    Diálogo para configurar el auto-split aleatorio de imágenes.

    Args:
        parent: Widget padre Qt.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-asignar splits")
        self.setStyleSheet(f"background:{CARD_BG}; color:{TEXT_PRIMARY};")
        self._build_ui()

    def _build_ui(self):
        lay = QFormLayout(self)

        # ── Porcentajes ────────────────────────────────────────
        self.pct_train = QSpinBox()
        self.pct_train.setRange(1, 98)
        self.pct_train.setValue(80)

        self.pct_valid = QSpinBox()
        self.pct_valid.setRange(1, 98)
        self.pct_valid.setValue(10)

        lay.addRow("Train %:", self.pct_train)
        lay.addRow("Val %:", self.pct_valid)
        lay.addRow("Test %: resto", QLabel("(calculado automáticamente)"))

        # ── Opción: solo imágenes sin split ───────────────────
        self.only_unassigned = QCheckBox("Solo imágenes sin split")
        self.only_unassigned.setChecked(True)
        self.only_unassigned.setStyleSheet(f"color:{TEXT_PRIMARY};")
        lay.addRow(self.only_unassigned)

        # ── Botones ────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.setStyleSheet("color:white;")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addRow(btns)

    # ══════════════════════════════════════════════════════════
    #  API PÚBLICA
    # ══════════════════════════════════════════════════════════

    def get_config(self) -> dict:
        """
        Devuelve la configuración introducida por el usuario.

        Returns:
            Dict con pct_train (int), pct_val (int),
            only_unassigned (bool).
        """
        return {
            "pct_train":       self.pct_train.value(),
            "pct_val":         self.pct_valid.value(),
            "only_unassigned": self.only_unassigned.isChecked(),
        }