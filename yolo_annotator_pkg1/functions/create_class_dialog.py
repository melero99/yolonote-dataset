"""
╔══════════════════════════════════════════════════════════════╗
║  functions/create_class_dialog.py  —  Diálogo nueva clase    ║
╚══════════════════════════════════════════════════════════════╝

Función pura que abre un diálogo modal para crear una nueva
clase YOLO: nombre + color.  Devuelve (nombre, QColor) o
(None, None) si el usuario cancela.
"""
from typing import Optional, Tuple

from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QLineEdit,
    QDialogButtonBox, QColorDialog, QPushButton, QHBoxLayout,
)
from PyQt5.QtGui import QColor

from yolo_annotator_pkg1.resources import CARD_BG, TEXT_PRIMARY, btn_style


def open_add_class_dialog(parent=None) -> Tuple[Optional[str], Optional[QColor]]:
    """
    Abre el diálogo de creación de una nueva clase.

    Args:
        parent: Widget padre Qt (puede ser None).

    Returns:
        Tupla (nombre, color) si el usuario aceptó,
        (None, None) si canceló o el nombre está vacío.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Añadir clase")
    dlg.setStyleSheet(f"background:{CARD_BG}; color:{TEXT_PRIMARY};")
    dlg.setMinimumWidth(320)

    lay   = QFormLayout(dlg)
    name_edit = QLineEdit()
    name_edit.setPlaceholderText("Nombre de la clase…")
    name_edit.setStyleSheet(
        f"background:#0d0d1f; color:{TEXT_PRIMARY}; "
        f"border:1px solid #2a2a5a; border-radius:4px; padding:4px;"
    )
    lay.addRow("Nombre:", name_edit)

    # ── Selector de color ──────────────────────────────────────
    chosen_color = [QColor("#4a9aff")]   # lista mutable para closure

    color_row = QHBoxLayout()
    color_btn = QPushButton("  Elegir color  ")
    color_btn.setStyleSheet(btn_style())
    color_preview = QPushButton()
    color_preview.setFixedSize(28, 28)
    color_preview.setStyleSheet(
        f"background:{chosen_color[0].name()}; border-radius:4px; border:1px solid #2a2a5a;"
    )
    color_preview.setEnabled(False)
    color_row.addWidget(color_btn)
    color_row.addWidget(color_preview)
    color_row.addStretch()
    lay.addRow("Color:", color_row)

    def pick_color():
        c = QColorDialog.getColor(chosen_color[0], parent, "Elige color de clase")
        if c.isValid():
            chosen_color[0] = c
            color_preview.setStyleSheet(
                f"background:{c.name()}; border-radius:4px; border:1px solid #2a2a5a;"
            )

    color_btn.clicked.connect(pick_color)

    btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    btns.setStyleSheet("color:white;")
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)
    lay.addRow(btns)

    if dlg.exec_() == QDialog.Accepted:
        name = name_edit.text().strip()
        if name:
            return name, chosen_color[0]
    return None, None
