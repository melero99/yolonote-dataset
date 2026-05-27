"""
╔══════════════════════════════════════════════════════════════╗
║  dialogs/export_dialog.py  —  Diálogo de exportación         ║
╚══════════════════════════════════════════════════════════════╝

Diálogo modal para configurar la exportación del dataset al
formato YOLO estándar (images/ + labels/ + data.yaml).

Opciones:
  - Carpeta de destino y nombre del dataset.
  - Copiar imágenes o solo generar etiquetas.
  - Generar data.yaml para YOLO v5/v8.
  - Auto-asignar split a imágenes sin asignar.
  - Exportar solo frames con anotaciones (omite imágenes vacías).
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QLineEdit, QSpinBox, QCheckBox,
    QDialogButtonBox, QFileDialog, QTextEdit,
)
from PyQt5.QtCore import Qt

from yolo_annotator_pkg1.models import Project
from yolo_annotator_pkg1.resources import (
    DARK_BG, CARD_BG, BORDER_COLOR, TEXT_PRIMARY, TEXT_MUTED,
    btn_style,
)


class ExportDialog(QDialog):
    """
    Diálogo de exportación del dataset al formato YOLO.

    Muestra las estadísticas actuales del proyecto y permite
    al usuario configurar todos los parámetros de exportación.

    Args:
        project: Proyecto a exportar (solo lectura en el diálogo).
        parent:  Widget padre Qt.
    """

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Exportar Dataset")
        self.setMinimumWidth(480)
        self.setStyleSheet(f"background:{CARD_BG}; color:{TEXT_PRIMARY};")
        self._build_ui()

    # ══════════════════════════════════════════════════════════
    #  CONSTRUCCIÓN DE UI
    # ══════════════════════════════════════════════════════════

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(12)

        # ── Resumen de splits ──────────────────────────────────
        stats = self.project.stats()
        info  = QLabel(
            f"📊  Train: {stats['train']}  |  Val: {stats['val']}  |  "
            f"Test: {stats['test']}  |  Sin asignar: {stats['unassigned']}"
        )
        info.setStyleSheet(
            f"color:{TEXT_MUTED}; font-size:11px; padding:6px; "
            f"background:{DARK_BG}; border-radius:4px;"
        )
        lay.addWidget(info)

        # ── Carpeta de destino ─────────────────────────────────
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

        # ── Nombre del dataset ─────────────────────────────────
        lay.addWidget(QLabel("Nombre del dataset:"))
        self.name_edit = QLineEdit(self.project.name.replace(" ", "_"))
        self.name_edit.setStyleSheet(
            f"background:{DARK_BG}; color:white; "
            f"border:1px solid {BORDER_COLOR}; border-radius:4px; padding:4px;"
        )
        lay.addWidget(self.name_edit)

        # ── Opciones ───────────────────────────────────────────
        self.copy_images = QCheckBox("Copiar imágenes (si no marcado, solo genera labels)")
        self.copy_images.setChecked(True)
        self.copy_images.setStyleSheet(f"color:{TEXT_PRIMARY};")
        lay.addWidget(self.copy_images)

        self.gen_yaml = QCheckBox("Generar data.yaml para YOLO")
        self.gen_yaml.setChecked(True)
        self.gen_yaml.setStyleSheet(f"color:{TEXT_PRIMARY};")
        lay.addWidget(self.gen_yaml)

        self.only_annotated = QCheckBox(
            "Exportar solo frames con anotaciones (omite im\u00e1genes vac\u00edas)"
        )
        self.only_annotated.setChecked(False)
        self.only_annotated.setStyleSheet(f"color:{TEXT_PRIMARY};")
        lay.addWidget(self.only_annotated)

        # ── Auto-split de imágenes sin asignar ────────────────
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
        self.pct_train = QSpinBox()
        self.pct_train.setRange(1, 98)
        self.pct_train.setValue(80)
        grp_lay.addWidget(self.pct_train)
        grp_lay.addWidget(QLabel("Val %:"))
        self.pct_valid = QSpinBox()
        self.pct_valid.setRange(1, 98)
        self.pct_valid.setValue(10)
        grp_lay.addWidget(self.pct_valid)
        lay.addWidget(grp)

        # ── Vista previa de la estructura ─────────────────────
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
            "├── val/\n│   ├── images/\n│   └── labels/\n"
            "├── test/\n│   ├── images/\n│   └── labels/\n"
            "└── data.yaml"
        )
        lay.addWidget(preview)

        # ── Botones ────────────────────────────────────────────
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

    # ══════════════════════════════════════════════════════════
    #  API PÚBLICA
    # ══════════════════════════════════════════════════════════

    def get_config(self) -> dict:
        """
        Devuelve la configuración de exportación introducida por el usuario.

        Returns:
            Dict con dest, name, copy_images, gen_yaml, auto_split,
            pct_train, pct_val, only_annotated.
        """
        return {
            "dest":        self.dest_edit.text().strip(),
            "name":        self.name_edit.text().strip() or "dataset",
            "copy_images": self.copy_images.isChecked(),
            "gen_yaml":    self.gen_yaml.isChecked(),
            "auto_split":  self.auto_split.isChecked(),
            "pct_train":   self.pct_train.value(),
            "pct_val":    self.pct_valid.value(),
            "only_annotated": self.only_annotated.isChecked(),
        }