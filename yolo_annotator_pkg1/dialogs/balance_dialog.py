"""
╔══════════════════════════════════════════════════════════════╗
║  dialogs/balance_dialog.py  —  Balance inteligente           ║
╚══════════════════════════════════════════════════════════════╝

Diálogo modal para configurar y previsualizar el balance
automático de clases en el dataset.

Estrategia 0 — Eliminar anotaciones excedentes (recomendado):
  Para cada clase que supera su límite porcentual, busca la imagen
  que más contribuye al exceso y elimina anotaciones de esa clase
  en dicha imagen.

Estrategia 1 — Excluir imágenes del split (conservador):
  Marca como split=None las imágenes que más aportan al desequilibrio,
  sin borrar ninguna anotación.
"""
from typing import Dict, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QScrollArea, QWidget, QSpinBox, QComboBox,
    QTextEdit,
)
from PyQt5.QtCore import Qt

from yolo_annotator_pkg1.models import Project
from yolo_annotator_pkg1.resources import (
    DARK_BG, CARD_BG, BORDER_COLOR, TEXT_PRIMARY, TEXT_MUTED,
    IMBALANCE_WARN, IMBALANCE_CRIT,
    btn_style, grp_style, input_style,
)


class BalanceDialog(QDialog):
    """
    Diálogo de balance inteligente del dataset.

    Permite al usuario configurar un porcentaje máximo por clase y
    calcular una vista previa del impacto antes de aplicar los cambios.

    Args:
        project: Proyecto activo (solo lectura en el diálogo).
        parent:  Widget padre Qt.
    """

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("⚖ Balance Inteligente de Dataset")
        self.setMinimumSize(640, 580)
        self.setStyleSheet(f"background:{CARD_BG}; color:{TEXT_PRIMARY};")

        self._class_spins:    Dict[int, QSpinBox] = {}
        self._preview_result: Optional[dict]       = None

        self._build_ui()
        self._refresh_current_state()

    # ══════════════════════════════════════════════════════════
    #  CONSTRUCCIÓN DE UI
    # ══════════════════════════════════════════════════════════

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        # ── Estado actual ──────────────────────────────────────
        grp_state = QGroupBox("Estado actual del dataset")
        grp_state.setStyleSheet(grp_style())
        state_lay = QVBoxLayout(grp_state)

        self.state_table = QTextEdit()
        self.state_table.setReadOnly(True)
        self.state_table.setFixedHeight(110)
        self.state_table.setStyleSheet(
            f"background:{DARK_BG}; color:{TEXT_PRIMARY}; "
            f"font-family:Consolas; font-size:11px; border-radius:4px;"
        )
        state_lay.addWidget(self.state_table)
        lay.addWidget(grp_state)

        # ── Configuración de límites por clase ─────────────────
        grp_cfg = QGroupBox(
            "Configurar límites por clase  (% máximo sobre el total de anotaciones)"
        )
        grp_cfg.setStyleSheet(grp_style())
        cfg_lay = QVBoxLayout(grp_cfg)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(160)
        scroll.setStyleSheet(f"background:{DARK_BG}; border:none;")
        inner     = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setSpacing(6)

        counts     = self.project.class_counts()
        total      = sum(counts.values()) or 1
        n_cls      = len(self.project.classes)
        default_pct = max(10, int(100 / n_cls * 1.5)) if n_cls else 50

        from PyQt5.QtGui import QPixmap
        for cls in self.project.classes:
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(4, 2, 4, 2)

            pix = QPixmap(12, 12)
            pix.fill(cls.color)
            ico = QLabel()
            ico.setPixmap(pix)
            row_l.addWidget(ico)

            lbl = QLabel(f"{cls.name}")
            lbl.setFixedWidth(120)
            lbl.setStyleSheet(f"color:{cls.color_hex}; font-size:12px;")
            row_l.addWidget(lbl)

            cnt     = counts.get(cls.id, 0)
            cur_pct = cnt * 100 // total
            cur_lbl = QLabel(f"Actual: {cnt}  ({cur_pct}%)")
            cur_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px;")
            cur_lbl.setFixedWidth(130)
            row_l.addWidget(cur_lbl)

            spin = QSpinBox()
            spin.setRange(1, 100)
            spin.setValue(default_pct)
            spin.setSuffix(" % máx")
            spin.setFixedWidth(90)
            spin.setStyleSheet(input_style())
            row_l.addWidget(spin)
            row_l.addStretch()

            self._class_spins[cls.id] = spin
            inner_lay.addWidget(row_w)

        inner_lay.addStretch()
        scroll.setWidget(inner)
        cfg_lay.addWidget(scroll)

        strat_row = QHBoxLayout()
        strat_row.addWidget(QLabel("Estrategia:"))
        self.strat_combo = QComboBox()
        self.strat_combo.addItems([
            "Eliminar anotaciones excedentes (recomendado)",
            "Excluir imágenes del split (conservador)",
        ])
        self.strat_combo.setStyleSheet(input_style())
        try:
            self.strat_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        except Exception:
            pass
        self.strat_combo.setMinimumWidth(220)
        strat_row.addWidget(self.strat_combo, 1)
        cfg_lay.addLayout(strat_row)
        lay.addWidget(grp_cfg)

        # ── Vista previa ───────────────────────────────────────
        grp_prev = QGroupBox("Vista previa del resultado")
        grp_prev.setStyleSheet(grp_style())
        prev_lay = QVBoxLayout(grp_prev)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFixedHeight(110)
        self.preview_text.setStyleSheet(
            f"background:{DARK_BG}; color:#aaffaa; "
            f"font-family:Consolas; font-size:11px; border-radius:4px;"
        )
        self.preview_text.setPlainText("Pulsa 'Calcular vista previa' para ver el impacto.")
        prev_lay.addWidget(self.preview_text)
        lay.addWidget(grp_prev)

        # ── Botones ────────────────────────────────────────────
        btn_row = QHBoxLayout()

        btn_preview = QPushButton("🔍  Calcular vista previa")
        btn_preview.setStyleSheet(btn_style("#2a3a5a", "#4a6aaa"))
        btn_preview.clicked.connect(self._compute_preview)

        self.btn_apply = QPushButton("⚖  Realizar balanceo")
        self.btn_apply.setStyleSheet(btn_style("#2a5a2a", "#3a8a3a"))
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self.accept)

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet(btn_style())
        btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(btn_preview)
        btn_row.addWidget(self.btn_apply)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        lay.addLayout(btn_row)

    # ══════════════════════════════════════════════════════════
    #  LÓGICA DE PREVIEW Y CONFIG
    # ══════════════════════════════════════════════════════════

    def _refresh_current_state(self):
        """Rellena el área de estado con la distribución actual de clases."""
        counts = self.project.class_counts()
        total  = sum(counts.values())
        lines  = [f"  {'Clase':<20} {'Anotaciones':>12} {'%':>6}",
                  "  " + "─" * 42]
        for cls in self.project.classes:
            cnt = counts.get(cls.id, 0)
            pct = cnt * 100 / total if total else 0
            bar = "█" * int(pct / 5)
            lines.append(f"  {cls.name:<20} {cnt:>12}  {pct:>5.1f}%  {bar}")
        lines.append(f"\n  Total anotaciones: {total}")
        ratio = self.project.imbalance_ratio()
        level = "CRÍTICO" if ratio > IMBALANCE_CRIT else ("MODERADO" if ratio > IMBALANCE_WARN else "OK")
        lines.append(f"  Índice de desbalance: {ratio*100:.1f}%  [{level}]")
        self.state_table.setPlainText("\n".join(lines))

    def _compute_preview(self):
        """Simula el balanceo y muestra el impacto sin tocar el proyecto."""
        result = self._simulate()
        self._preview_result = result
        lines = ["  Impacto estimado del balanceo:\n"]
        for cls in self.project.classes:
            before = result["before"].get(cls.id, 0)
            after  = result["after"].get(cls.id, 0)
            diff   = after - before
            sign   = "+" if diff >= 0 else ""
            lines.append(f"  {cls.name:<20}  {before:>6} → {after:>6}  ({sign}{diff:+d})")
        lines.append(f"\n  Imágenes afectadas: {result['images_affected']}")
        lines.append(f"  Anotaciones eliminadas: {result['boxes_removed']}")
        if self.strat_combo.currentIndex() == 1:
            lines.append(f"  Imágenes excluidas de split: {result['images_excluded']}")
        self.preview_text.setPlainText("\n".join(lines))
        self.btn_apply.setEnabled(True)

    def _simulate(self) -> dict:
        """
        Simula el balanceo iterativo (idéntico a _apply_balance) sin tocar el proyecto.

        La simulación replica exactamente la lógica de YoloAnnotatorPro._apply_balance:
        en cada iteración detecta la clase más desbalanceada y elimina una sola
        anotación / excluye una imagen, repitiendo hasta que no haya offenders.

        Returns:
            Dict con claves before, after, boxes_removed, images_affected,
            images_excluded, limits, strategy.
        """
        import copy

        limits   = {cls.id: self._class_spins[cls.id].value() / 100.0
                    for cls in self.project.classes}
        strategy = self.strat_combo.currentIndex()

        counts_before = self.project.class_counts()

        # Copia profunda de boxes para simular sin modificar el proyecto
        sim_boxes: Dict[int, list] = {
            id(rec): list(rec.boxes)
            for rec in self.project.images
        }
        sim_splits: Dict[int, Optional[str]] = {
            id(rec): rec.split
            for rec in self.project.images
        }

        def sim_counts() -> Dict[int, int]:
            c: Dict[int, int] = {cls.id: 0 for cls in self.project.classes}
            for rec in self.project.images:
                rid = id(rec)
                if strategy == 1 and sim_splits[rid] is None:
                    continue
                for b in sim_boxes[rid]:
                    c[b.class_id] = c.get(b.class_id, 0) + 1
            return c

        boxes_removed   = 0
        images_modified: set = set()
        images_excluded = 0

        MAX_ITERATIONS = 500
        for _ in range(MAX_ITERATIONS):
            counts = sim_counts()
            total  = sum(counts.values())
            if total == 0:
                break

            offenders = {
                cid: counts[cid]
                for cid, lim in limits.items()
                if counts.get(cid, 0) / total > lim + 0.001
            }
            if not offenders:
                break

            worst_cid = max(offenders, key=lambda c: offenders[c] / total - limits[c])

            if strategy == 0:
                # Imagen que más contribuye al exceso
                best_rec = max(
                    (rec for rec in self.project.images
                     if any(b.class_id == worst_cid for b in sim_boxes[id(rec)])),
                    key=lambda rec: sum(1 for b in sim_boxes[id(rec)] if b.class_id == worst_cid),
                    default=None,
                )
                if best_rec is None:
                    break
                rid = id(best_rec)
                for i, b in enumerate(sim_boxes[rid]):
                    if b.class_id == worst_cid:
                        sim_boxes[rid].pop(i)
                        boxes_removed += 1
                        images_modified.add(rid)
                        break
            else:
                best_rec = max(
                    (rec for rec in self.project.images
                     if sim_splits[id(rec)] is not None and
                     any(b.class_id == worst_cid for b in sim_boxes[id(rec)])),
                    key=lambda rec: sum(1 for b in sim_boxes[id(rec)] if b.class_id == worst_cid),
                    default=None,
                )
                if best_rec is None:
                    break
                rid = id(best_rec)
                if sim_splits[rid] is not None:
                    sim_splits[rid] = None
                    images_excluded += 1
                    images_modified.add(rid)

        counts_after = sim_counts()

        return {
            "before":           counts_before,
            "after":            counts_after,
            "boxes_removed":    boxes_removed,
            "images_affected":  len(images_modified),
            "images_excluded":  images_excluded,
            "limits":           limits,
            "strategy":         strategy,
        }

    def get_config(self) -> dict:
        """
        Devuelve la configuración final para que el controller la aplique.

        Returns:
            Dict con 'limits' ({class_id: max_ratio}) y 'strategy' (0 o 1).
        """
        return {
            "limits":   {cls.id: self._class_spins[cls.id].value() / 100.0
                         for cls in self.project.classes},
            "strategy": self.strat_combo.currentIndex(),
        }