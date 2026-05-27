"""
╔══════════════════════════════════════════════════════════════╗
║  widgets/class_panel.py  —  Panel lateral de clases          ║
╚══════════════════════════════════════════════════════════════╝

Panel izquierdo de la ventana principal.  Contiene:
  - Lista de clases con contadores en tiempo real
  - Barras de balance por clase (ClassBalanceBar)
  - Indicador global de balance
  - Lista de anotaciones de la imagen actual
  - Botones de asignación de split (TRAIN / VALID / TEST)
  - Estadísticas compactas del proyecto

Responsabilidad única: mostrar estado y emitir señales.
Toda la lógica de negocio reside en los controllers.

Signals:
    class_selected         (int):  fila de la clase seleccionada.
    class_deselected       ():     usuario deseleccionó la clase.
    class_add_requested    ():     usuario pulsó "Añadir clase".
    class_delete_requested (int):  usuario pulsó "Quitar clase" con fila seleccionada.
    split_assigned         (str):  "train" | "val" | "test".
"""
from typing import List, Dict

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QGroupBox, QDialog, QFormLayout, QLineEdit,
    QDialogButtonBox, QColorDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QColor

from yolo_annotator_pkg1.resources import (
    DARK_BG, PANEL_BG, CARD_BG, BORDER_COLOR,
    TEXT_PRIMARY, TEXT_MUTED, ACCENT,
    SPLIT_COLORS, SPLIT_TEXT,
    IMBALANCE_WARN, IMBALANCE_CRIT,
    btn_style,
)
from yolo_annotator_pkg1.models import YoloClass, BoundingBox
from yolo_annotator_pkg1.widgets.class_balance_bar import ClassBalanceBar
from yolo_annotator_pkg1.functions.create_class_dialog import (
    open_add_class_dialog as open_add_class_dialog_fn,
)


class ClassPanel(QWidget):
    """
    Panel lateral izquierdo de la ventana principal.

    Solo emite señales; no modifica el modelo directamente.
    La lógica de negocio se delega a los controllers a través
    de los slots conectados en YoloAnnotatorPro.

    Args:
        parent: widget padre Qt.
    """

    class_selected         = pyqtSignal(int)
    class_deselected       = pyqtSignal()
    class_add_requested    = pyqtSignal()
    class_delete_requested = pyqtSignal(int)
    split_assigned         = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(250)
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

        # ── Estado interno de balance ──────────────────────────
        self._undo_cb: callable      = None
        self._class_counts: Dict[int, int] = {}
        self._total_boxes: int       = 0
        self._bar_widgets: List[ClassBalanceBar] = []

        self._build_ui()

    # ══════════════════════════════════════════════════════════
    #  CONSTRUCCIÓN DE UI
    # ══════════════════════════════════════════════════════════

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # ── Grupo CLASES / ANOTACIONES ─────────────────────────
        grp_cls = QGroupBox("CLASES  /  ANOTACIONES")
        cls_lay = QVBoxLayout(grp_cls)
        cls_lay.setSpacing(4)

        self.class_list = QListWidget()
        self.class_list.setMinimumHeight(90)
        self.class_list.itemClicked.connect(self._on_class_clicked)
        cls_lay.addWidget(self.class_list)

        # Área de barras de balance (una por clase)
        self.balance_area   = QWidget()
        self.balance_layout = QVBoxLayout(self.balance_area)
        self.balance_layout.setContentsMargins(0, 0, 0, 0)
        self.balance_layout.setSpacing(2)
        cls_lay.addWidget(self.balance_area)

        # Indicador global
        self.balance_indicator = QLabel("Balance: —")
        self.balance_indicator.setAlignment(Qt.AlignCenter)
        self.balance_indicator.setStyleSheet(
            "font-size: 11px; font-weight: bold; border-radius: 4px; padding: 3px;"
        )
        cls_lay.addWidget(self.balance_indicator)

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
        hint.setStyleSheet("color: #555588; font-size: 10px; font-style: italic;")
        lay.addWidget(hint)

        # ── Grupo ANOTACIONES imagen actual ───────────────────
        grp_ann = QGroupBox("ANOTACIONES (imagen actual)")
        ann_lay = QVBoxLayout(grp_ann)

        self.ann_list = QListWidget()
        self.ann_list.setMaximumHeight(100)
        self.ann_list.setStyleSheet(
            f"QListWidget {{ background: {DARK_BG}; border-radius: 4px; }}"
        )
        ann_lay.addWidget(self.ann_list)

        btn_undo = QPushButton("↩ Deshacer última")
        btn_undo.setStyleSheet(btn_style())
        btn_undo.clicked.connect(lambda: self._undo_cb and self._undo_cb())
        ann_lay.addWidget(btn_undo)
        lay.addWidget(grp_ann)

        # ── Grupo SPLIT ───────────────────────────────────────
        grp_split = QGroupBox("ASIGNAR IMAGEN A SPLIT")
        split_lay = QVBoxLayout(grp_split)

        self.split_label = QLabel("Split actual: —")
        self.split_label.setAlignment(Qt.AlignCenter)
        self.split_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: white;"
        )
        split_lay.addWidget(self.split_label)

        row2 = QHBoxLayout()
        for split, color in [("train", "#1a5c2a"), ("val", "#1a3a6a"), ("test", "#5c2a1a")]:
            b = QPushButton(split.upper())
            b.setStyleSheet(btn_style(color, color))
            b.clicked.connect(lambda _, s=split: self.split_assigned.emit(s))
            row2.addWidget(b)
        split_lay.addLayout(row2)
        lay.addWidget(grp_split)

        lay.addStretch()

        # ── Estadísticas compactas ─────────────────────────────
        self.stats_label = QLabel("Sin proyecto")
        self.stats_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px;"
        )
        self.stats_label.setWordWrap(True)
        lay.addWidget(self.stats_label)

    # ══════════════════════════════════════════════════════════
    #  API PÚBLICA
    # ══════════════════════════════════════════════════════════

    def set_undo_callback(self, fn):
        """Registra la función a llamar cuando se pulse 'Deshacer última'."""
        self._undo_cb = fn

    def load_classes(self, classes: List[YoloClass]):
        """
        Rellena la lista de clases con nombre, color e icono.

        Preserva los contadores actuales (_class_counts) si ya existen.

        Args:
            classes: Lista de YoloClass ordenadas por id.
        """
        self.class_list.clear()
        for cls in classes:
            cnt  = self._class_counts.get(cls.id, 0)
            item = QListWidgetItem(f"  {cls.id+1}.  {cls.name}   [{cnt}]")
            item.setForeground(cls.color)
            pix = QPixmap(12, 12)
            pix.fill(cls.color)
            item.setIcon(QIcon(pix))
            self.class_list.addItem(item)

    def update_class_counts(self, classes: List[YoloClass],
                            counts: Dict[int, int], total: int):
        """
        Refresca los contadores en la lista y las barras de balance.

        Debe llamarse cada vez que cambia cualquier anotación o se
        navega a una nueva imagen.

        Args:
            classes: Lista completa de clases del proyecto.
            counts:  Mapa {class_id: nº anotaciones totales en el proyecto}.
            total:   Suma de todos los valores de counts.
        """
        self._class_counts = counts
        self._total_boxes  = total

        # ── Actualizar textos de la lista ──────────────────────
        for row, cls in enumerate(classes):
            item = self.class_list.item(row)
            if item:
                cnt = counts.get(cls.id, 0)
                item.setText(f"  {cls.id+1}.  {cls.name}   [{cnt}]")

        # ── Reconstruir barras de balance ──────────────────────
        # Limpiar las anteriores para reconstruir desde cero
        while self.balance_layout.count():
            child = self.balance_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._bar_widgets.clear()

        if not classes or total == 0:
            self.balance_indicator.setText("Balance: sin datos")
            self.balance_indicator.setStyleSheet(
                f"font-size:11px; font-weight:bold; "
                f"color:{TEXT_MUTED}; border-radius:4px; padding:3px;"
            )
            return

        target = 1.0 / len(classes)   # distribución uniforme ideal

        for cls in classes:
            cnt   = counts.get(cls.id, 0)
            ratio = cnt / total if total > 0 else 0.0

            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(4)

            lbl = QLabel(f"{cls.name[:10]}")
            lbl.setFixedWidth(72)
            lbl.setStyleSheet(f"color:{cls.color_hex}; font-size:10px;")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            bar = ClassBalanceBar()
            bar.set_data(ratio, target, cls.color)

            pct_lbl = QLabel(f"{ratio*100:.0f}%")
            pct_lbl.setFixedWidth(30)
            pct_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")

            row_l.addWidget(lbl)
            row_l.addWidget(bar, 1)
            row_l.addWidget(pct_lbl)
            self.balance_layout.addWidget(row_w)
            self._bar_widgets.append(bar)

        # ── Indicador global de balance ────────────────────────
        vals         = [counts.get(c.id, 0) for c in classes]
        ratio_global = (max(vals) - min(vals)) / max(vals) if max(vals) > 0 else 0.0
        if ratio_global > IMBALANCE_CRIT:
            color = "#cc2222"
            text  = f"⚠ DESBALANCE CRÍTICO ({ratio_global*100:.0f}%)"
        elif ratio_global > IMBALANCE_WARN:
            color = "#cc8800"
            text  = f"⚡ Desbalance moderado ({ratio_global*100:.0f}%)"
        else:
            color = "#228b22"
            text  = f"✔ Dataset equilibrado ({ratio_global*100:.0f}%)"

        self.balance_indicator.setText(text)
        self.balance_indicator.setStyleSheet(
            f"font-size:11px; font-weight:bold; color:white; "
            f"background:{color}; border-radius:4px; padding:3px;"
        )

    def update_annotations(self, boxes: List[BoundingBox], classes: List[YoloClass]):
        """
        Rellena la lista de anotaciones de la imagen actual.

        Args:
            boxes:   Bounding boxes de la imagen activa.
            classes: Lista de clases para resolver nombres.
        """
        self.ann_list.clear()
        cls_map = {c.id: c for c in classes}
        for i, b in enumerate(boxes):
            cls  = cls_map.get(b.class_id)
            name = cls.name if cls else f"class_{b.class_id}"
            info = b.pixel_info()
            item = QListWidgetItem(f"  #{i+1} {name}  [{info['w']}×{info['h']}]")
            if cls:
                item.setForeground(cls.color)
            self.ann_list.addItem(item)

    def update_split_label(self, split):
        """
        Actualiza la etiqueta de split con color de fondo.

        Args:
            split: "train" | "val" | "test" | None.
        """
        text  = SPLIT_TEXT.get(split, "—")
        color = SPLIT_COLORS.get(split, QColor("#333355"))
        self.split_label.setText(f"Split actual: {text}")
        self.split_label.setStyleSheet(
            f"font-size:13px; font-weight:bold; color:white; "
            f"background-color:{color.name()}; border-radius:4px; padding:4px;"
        )

    def update_stats(self, stats: dict):
        """
        Actualiza la etiqueta de estadísticas compactas del proyecto.

        Args:
            stats: Resultado de Project.stats().
        """
        self.stats_label.setText(
            f"Total: {stats['total']} | Anotadas: {stats['annotated']}\n"
            f"Train: {stats['train']}  Val: {stats['val']}  Test: {stats['test']}\n"
            f"Sin asignar: {stats['unassigned']}"
        )

    def open_add_class_dialog(self) -> tuple:
        """
        Abre el diálogo de nueva clase.

        Returns:
            (nombre: str, color: QColor) si se aceptó, (None, None) si se canceló.
        """
        return open_add_class_dialog_fn(self)

    # ══════════════════════════════════════════════════════════
    #  HANDLERS INTERNOS
    # ══════════════════════════════════════════════════════════

    def _on_delete_clicked(self):
        """Emite class_delete_requested con la fila seleccionada, o avisa si no hay ninguna."""
        row = self.class_list.currentRow()
        if row >= 0:
            self.class_delete_requested.emit(row)
        else:
            QMessageBox.information(
                self, "Sin selección",
                "Selecciona una clase de la lista para eliminarla."
            )

    def _on_class_clicked(self, item):
        """Emite class_selected con la fila del item clicado."""
        self.class_selected.emit(self.class_list.row(item))

    def _deselect(self):
        """Limpia la selección y emite class_deselected."""
        self.class_list.clearSelection()
        self.class_deselected.emit()