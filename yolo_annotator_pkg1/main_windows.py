"""
╔══════════════════════════════════════════════════════════════╗
║  main_window.py  —  Ventana principal (ensamblador)          ║
╚══════════════════════════════════════════════════════════════╝

YoloAnnotatorPro es exclusivamente un ensamblador:
  - Instancia los widgets (ClassPanel, AnnotationCanvas).
  - Conecta señales con los handlers de negocio definidos aquí.
  - Delega la lógica de diálogos a yolo_annotator_pkg/dialogs/.

No contiene lógica de extracción, balance ni exportación inline;
esas operaciones están en los métodos de esta clase pero bien
separadas por región con comentarios de nivel 1.
"""
import os
import sys
import json
import shutil
import random
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QInputDialog,
    QMessageBox, QSizePolicy, QAction, QShortcut,
    QDialog, QProgressDialog, QCheckBox, QApplication,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QKeySequence, QIcon, QPalette
import cv2

from yolo_annotator_pkg1.resources import (
    resource_path,
    DARK_BG, PANEL_BG, CARD_BG, BORDER_COLOR,
    TEXT_PRIMARY, TEXT_MUTED, ACCENT,
    SPLIT_COLORS, SPLIT_TEXT,
    IMBALANCE_WARN, IMBALANCE_CRIT,
    btn_style, SUPPORTED_EXTS,
)
from yolo_annotator_pkg1.models import YoloClass, BoundingBox, ImageRecord, Project
from yolo_annotator_pkg1.widgets.annotation_canvas import AnnotationCanvas
from yolo_annotator_pkg1.widgets.class_panel import ClassPanel
from yolo_annotator_pkg1.dialogs import (
    AutoSplitDialog,
    BalanceDialog,
    ExportDialog,
    FrameExtractorDialog,
)


class YoloAnnotatorPro(QMainWindow):
    """
    Ventana principal de YOLO Annotator Pro.

    Responsabilidad: ensamblar widgets y coordinar los flujos de
    trabajo del usuario.  Toda la lógica de extracción y utilidades
    de bajo nivel reside en yolo_annotator_pkg/functions/.
    Los diálogos modales están en yolo_annotator_pkg/dialogs/.
    Los widgets reutilizables están en yolo_annotator_pkg/widgets/.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Annotator Pro")

        # ── Icono de aplicación ────────────────────────────────
        icon_path = resource_path("logoapp.png")
        if os.path.exists(icon_path):
            app_icon = QIcon(icon_path)
            self.setWindowIcon(app_icon)
            QApplication.setWindowIcon(app_icon)

        self.setMinimumSize(1180, 720)
        self._apply_theme()

        # ── Estado del proyecto ────────────────────────────────
        self.project: Optional[Project] = None
        self._dirty: bool               = False

        # ── Construcción de UI ─────────────────────────────────
        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self.statusBar().showMessage(
            "Bienvenido. Crea un proyecto nuevo o abre uno existente."
        )

        # ── Timer de notificación de desbalance (cada 30 s) ────
        self._balance_timer = QTimer(self)
        self._balance_timer.timeout.connect(self._check_balance_notification)
        self._balance_timer.start(30_000)

    # ══════════════════════════════════════════════════════════
    #  CONSTRUCCIÓN DE UI
    # ══════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Panel lateral ──────────────────────────────────────
        self.panel = ClassPanel()
        self.panel.class_selected.connect(self._on_class_selected)
        self.panel.class_deselected.connect(self._on_class_deselected)
        self.panel.class_add_requested.connect(self._handle_add_class)
        self.panel.class_delete_requested.connect(self._handle_delete_class)
        self.panel.split_assigned.connect(self._assign_split)
        self.panel.set_undo_callback(self._undo_last)
        root.addWidget(self.panel)

        # ── Área central ───────────────────────────────────────
        center  = QWidget()
        center.setStyleSheet(f"background:{DARK_BG};")
        clayout = QVBoxLayout(center)
        clayout.setContentsMargins(0, 0, 0, 0)
        clayout.setSpacing(0)

        # Barra de info de imagen actual
        self.info_bar = QLabel("Sin proyecto")
        self.info_bar.setAlignment(Qt.AlignCenter)
        self.info_bar.setStyleSheet(
            f"background:{PANEL_BG}; color:{TEXT_MUTED}; padding:6px; "
            f"font-size:12px; font-family:Consolas,monospace; "
            f"border-bottom:1px solid {BORDER_COLOR};"
        )
        clayout.addWidget(self.info_bar)

        # Canvas con botones de navegación a los lados
        nav = QHBoxLayout()
        nav.setContentsMargins(0, 0, 0, 0)
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
        bot.setStyleSheet(
            f"background:{PANEL_BG}; border-top:1px solid {BORDER_COLOR};"
        )
        bot_lay = QHBoxLayout(bot)
        bot_lay.setContentsMargins(10, 5, 10, 5)

        self.verified_chk = QCheckBox("✔ Imagen verificada")
        self.verified_chk.setStyleSheet(f"color:{TEXT_PRIMARY};")
        self.verified_chk.stateChanged.connect(self._toggle_verified)
        bot_lay.addWidget(self.verified_chk)

        self.output_lbl = QLabel("💾  Proyecto sin guardar")
        self.output_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px;")
        bot_lay.addWidget(self.output_lbl)
        bot_lay.addStretch()

        for label, cb, style in [
            ("💾 Guardar  Ctrl+S", self._save_project,       btn_style("#2a3a6a", "#4a6aaa")),
            ("⚖ Balance  Ctrl+B", self._open_balance_dialog, btn_style("#3a2a6a", "#6a3a8a")),
            ("📤 Exportar",        self._export_dataset,      btn_style("#3a2a6a", "#6a4aaa")),
        ]:
            b = QPushButton(label)
            b.setStyleSheet(style)
            b.clicked.connect(cb)
            bot_lay.addWidget(b)

        clayout.addWidget(bot)
        root.addWidget(center, 1)

    def _nav_btn(self, text: str, cb) -> QPushButton:
        """Crea un botón de navegación lateral (◀ / ▶)."""
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
        self._add_action(pm, "🔀  Auto-asignar splits",  "",       self._auto_split_dialog)
        self._add_action(pm, "⚖   Balance inteligente", "Ctrl+B", self._open_balance_dialog)
        self._add_action(pm, "📊  Estadísticas",          "",       self._show_stats)
        self._add_action(pm, "🗑  Limpiar sin split",      "",       self._clear_unassigned)

        vm = mb.addMenu("Vista")
        self._add_action(vm, "🔍  Ajustar a ventana  (0)", "",
                         lambda: (self.canvas._fit_to_window(), self.canvas.update()))
        self._add_action(vm, "🔍+  Zoom in  (+)",  "", lambda: self.canvas._zoom_step(1.3))
        self._add_action(vm, "🔍−  Zoom out  (−)", "", lambda: self.canvas._zoom_step(1 / 1.3))

        tm = mb.addMenu("Herramientas")
        self._add_action(tm, "🎬  Extractor de Frames…", "Ctrl+E", self._open_frame_extractor)

        hm = mb.addMenu("Ayuda")
        self._add_action(hm, "ℹ️  Atajos de teclado", "", self._show_help)

    def _add_action(self, menu, label: str, shortcut: str, callback) -> QAction:
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
        QShortcut(QKeySequence("Ctrl+B"),     self, self._open_balance_dialog)
        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self,
                      lambda idx=i - 1: self._select_class_by_index(idx))

    # ══════════════════════════════════════════════════════════
    #  PROYECTO  (nuevo / abrir / guardar)
    # ══════════════════════════════════════════════════════════

    def _new_project(self):
        if not self._confirm_unsaved():
            return
        name, ok = QInputDialog.getText(self, "Nuevo proyecto", "Nombre del proyecto:")
        if not ok or not name.strip():
            return
        self.project = Project(name=name.strip())
        self._dirty  = True
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
            self._dirty  = False
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
        existing    = {r.path for r in self.project.images}
        added       = 0
        for p in sorted(folder_path.iterdir()):
            if p.suffix.lower() in SUPPORTED_EXTS and str(p) not in existing:
                self.project.images.append(ImageRecord(path=str(p)))
                added += 1
        self._dirty = True
        self._refresh_all()
        if self._current_idx() < 0 and self.project.images:
            self._go_to(0)
        self.statusBar().showMessage(f"✅ {added} imágenes añadidas de '{folder_path.name}'.")

    # ══════════════════════════════════════════════════════════
    #  NAVEGACIÓN
    # ══════════════════════════════════════════════════════════

    def _current_idx(self) -> int:
        return self.project.current_index if self.project else -1

    def _go_to(self, idx: int):
        """Navega a la imagen en la posición idx y actualiza toda la UI."""
        if not self.project or not self.project.images:
            return
        self._store_current()
        idx = max(0, min(idx, len(self.project.images) - 1))
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
        self._update_class_counters()

        self.btn_prev.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < total - 1)

    def _go_prev(self):
        if self.project:
            self._go_to(self._current_idx() - 1)

    def _go_next(self):
        if self.project:
            self._go_to(self._current_idx() + 1)

    def _store_current(self):
        """Persiste en el modelo los boxes actuales del canvas."""
        if not self.project or not self.project.images:
            return
        idx = self._current_idx()
        if 0 <= idx < len(self.project.images):
            self.project.images[idx].boxes = self.canvas.get_boxes()

    # ══════════════════════════════════════════════════════════
    #  CLASES
    # ══════════════════════════════════════════════════════════

    def _on_class_selected(self, row: int):
        if not self.project or row >= len(self.project.classes):
            return
        cls = self.project.classes[row]
        self.canvas.set_selected_class(cls.id)
        self.statusBar().showMessage(f"Clase: {cls.name}  (id={cls.id})  |  tecla {row+1}")

    def _on_class_deselected(self):
        self.canvas.set_selected_class(None)

    def _select_class_by_index(self, idx: int):
        if not self.project or idx >= len(self.project.classes):
            return
        self.panel.class_list.setCurrentRow(idx)
        self._on_class_selected(idx)

    def _handle_add_class(self):
        if not self.project:
            QMessageBox.information(self, "Sin proyecto", "Crea o abre un proyecto primero.")
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
            self._update_class_counters()
            self.statusBar().showMessage(f"Clase '{name}' añadida (id={cls.id}).")

    def _handle_delete_class(self, row: int):
        if not self.project or row >= len(self.project.classes):
            return
        cls   = self.project.classes[row]
        reply = QMessageBox.question(
            self, "Eliminar clase",
            f"¿Eliminar la clase '{cls.name}'?\n"
            f"Las cajas de esta clase en las imágenes no se borrarán del proyecto.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.project.classes.pop(row)
        for i, c in enumerate(self.project.classes):
            c.id = i
        self._dirty = True
        self._refresh_classes()
        self.canvas.set_selected_class(None)
        self._update_class_counters()
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
        self._update_class_counters()

    def _update_class_counters(self):
        """Recalcula y actualiza contadores y barras de balance en tiempo real."""
        if not self.project:
            return
        counts = self.project.class_counts()
        total  = sum(counts.values())
        self.panel.update_class_counts(self.project.classes, counts, total)

    # ══════════════════════════════════════════════════════════
    #  ANOTACIONES
    # ══════════════════════════════════════════════════════════

    def _on_box_added(self, box: BoundingBox):
        self._store_current()
        if self.project and self.project.images:
            rec = self.project.images[self._current_idx()]
            self.panel.update_annotations(rec.boxes, self.project.classes)
        self._dirty = True
        self._update_class_counters()

    def _on_box_removed(self):
        self._store_current()
        if self.project and self.project.images:
            rec = self.project.images[self._current_idx()]
            self.panel.update_annotations(rec.boxes, self.project.classes)
        self._dirty = True
        self._update_class_counters()

    def _undo_last(self):
        self.canvas.remove_last_box()
        self._store_current()
        if self.project and self.project.images:
            rec = self.project.images[self._current_idx()]
            self.panel.update_annotations(rec.boxes, self.project.classes)
        self._dirty = True
        self._update_class_counters()

    # ══════════════════════════════════════════════════════════
    #  SPLIT
    # ══════════════════════════════════════════════════════════

    def _assign_split(self, split: str):
        if not self.project or not self.project.images:
            return
        rec       = self.project.images[self._current_idx()]
        rec.split = split
        self.panel.update_split_label(split)
        self.panel.update_stats(self.project.stats())
        # Actualizar la info bar sin recargar la imagen
        split_txt = SPLIT_TEXT[split]
        parts     = self.info_bar.text().split("|")
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
        """Abre AutoSplitDialog y aplica el resultado al proyecto."""
        if not self.project:
            return
        dlg = AutoSplitDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return

        cfg    = dlg.get_config()
        pt, pv = cfg["pct_train"], cfg["pct_val"]
        images = [
            r for r in self.project.images
            if not cfg["only_unassigned"] or r.split is None
        ]
        random.shuffle(images)
        n       = len(images)
        n_train = int(n * pt / 100)
        n_val   = int(n * pv / 100)
        for i, rec in enumerate(images):
            if i < n_train:
                rec.split = "train"
            elif i < n_train + n_val:
                rec.split = "val"
            else:
                rec.split = "test"
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

    # ══════════════════════════════════════════════════════════
    #  BALANCE INTELIGENTE
    # ══════════════════════════════════════════════════════════

    def _open_balance_dialog(self):
        if not self.project:
            QMessageBox.information(self, "Sin proyecto", "Crea o abre un proyecto primero.")
            return
        if len(self.project.classes) < 2:
            QMessageBox.information(self, "Sin datos",
                                    "Necesitas al menos 2 clases para usar el balance.")
            return
        counts = self.project.class_counts()
        if sum(counts.values()) == 0:
            QMessageBox.information(self, "Sin anotaciones",
                                    "No hay anotaciones en el proyecto todavía.")
            return

        self._store_current()
        dlg = BalanceDialog(self.project, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        self._apply_balance(dlg.get_config())

    def _apply_balance(self, cfg: dict):
        """
        Aplica el balanceo al proyecto según la configuración de BalanceDialog.

        Estrategia 0 — Eliminar anotaciones excedentes:
          Elimina anotaciones de la clase más desbalanceada en la imagen
          que más contribuye al exceso, iterando hasta cumplir los límites.

        Estrategia 1 — Excluir imágenes del split:
          Marca como split=None las imágenes que más contribuyen al exceso.
        """
        limits   = cfg["limits"]    # {class_id: max_ratio}
        strategy = cfg["strategy"]  # 0 = borrar annots, 1 = excluir imágenes

        boxes_removed   = 0
        images_affected = 0

        MAX_ITERATIONS = 100
        for _ in range(MAX_ITERATIONS):
            counts = self.project.class_counts()
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
                # Imagen que más aporta al exceso de worst_cid
                best_img = max(
                    (rec for rec in self.project.images
                     if any(b.class_id == worst_cid for b in rec.boxes)),
                    key=lambda rec: sum(1 for b in rec.boxes if b.class_id == worst_cid),
                    default=None,
                )
                if best_img is None:
                    break
                for i, b in enumerate(best_img.boxes):
                    if b.class_id == worst_cid:
                        best_img.boxes.pop(i)
                        boxes_removed   += 1
                        images_affected += 1
                        break
            else:
                best_img = max(
                    (rec for rec in self.project.images
                     if rec.split is not None and
                     any(b.class_id == worst_cid for b in rec.boxes)),
                    key=lambda rec: sum(1 for b in rec.boxes if b.class_id == worst_cid),
                    default=None,
                )
                if best_img is None:
                    break
                best_img.split   = None
                images_affected += 1

        self._dirty = True

        # Recargar la imagen actual por si fue modificada
        if self.project.images:
            rec = self.project.images[self._current_idx()]
            self.canvas.set_boxes(rec.boxes)
            self.panel.update_annotations(rec.boxes, self.project.classes)

        self._update_class_counters()
        self.panel.update_stats(self.project.stats())

        ratio_after = self.project.imbalance_ratio()
        if strategy == 0:
            msg = (
                f"✅ Balanceo completado.\n\n"
                f"Anotaciones eliminadas: {boxes_removed}\n"
                f"Imágenes afectadas: {images_affected}\n"
                f"Índice de desbalance resultante: {ratio_after*100:.1f}%"
            )
        else:
            msg = (
                f"✅ Balanceo completado.\n\n"
                f"Imágenes excluidas del split: {images_affected}\n"
                f"Índice de desbalance resultante: {ratio_after*100:.1f}%"
            )
        QMessageBox.information(self, "Balance completado", msg)
        self.statusBar().showMessage(
            f"Balance aplicado — desbalance resultante: {ratio_after*100:.1f}%"
        )

    def _check_balance_notification(self):
        """Se ejecuta cada 30 s para notificar desbalances críticos."""
        if not self.project or len(self.project.classes) < 2:
            return
        ratio = self.project.imbalance_ratio()
        if ratio > IMBALANCE_CRIT:
            self.statusBar().showMessage(
                f"⚠ DESBALANCE CRÍTICO detectado ({ratio*100:.0f}%) — "
                f"Usa Proyecto → Balance inteligente  (Ctrl+B)"
            )

    # ══════════════════════════════════════════════════════════
    #  EXPORTACIÓN
    # ══════════════════════════════════════════════════════════

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

        # ══════════════════════════════════════════════════════
        #  SANITY CHECKS
        # ══════════════════════════════════════════════════════
        warnings = []

        # 1. Imágenes sin anotaciones
        unannotated = [r for r in self.project.images if not r.boxes]
        if unannotated:
            # Ofrecer tres opciones: exportar todo, solo anotados, o cancelar
            msg_unannotated = (
                f"{len(unannotated)} imagen(es) no tienen ninguna anotación.\n\n"
                "¿Qué deseas hacer?"
            )
            _dlg_ua = QMessageBox(self)
            _dlg_ua.setWindowTitle("⚠️  Imágenes sin anotaciones")
            _dlg_ua.setText(msg_unannotated)
            _dlg_ua.setIcon(QMessageBox.Warning)
            _btn_all    = _dlg_ua.addButton("Exportar todo igualmente", QMessageBox.AcceptRole)
            _btn_only   = _dlg_ua.addButton("Exportar solo frames anotados", QMessageBox.ActionRole)
            _btn_cancel = _dlg_ua.addButton("Cancelar", QMessageBox.RejectRole)
            _dlg_ua.setDefaultButton(_btn_cancel)
            _dlg_ua.exec_()
            clicked = _dlg_ua.clickedButton()
            if clicked == _btn_cancel:
                return
            if clicked == _btn_only:
                cfg["only_annotated"] = True   # forzar filtro aunque no se marcó en el diálogo

        # 2. Clases definidas sin ninguna caja en todo el proyecto
        counts = self.project.class_counts()
        empty_classes = [
            c.name for c in self.project.classes if counts.get(c.id, 0) == 0
        ]
        if empty_classes:
            warnings.append(
                f"• {len(empty_classes)} clase(s) no tienen anotaciones: "
                + ", ".join(empty_classes)
            )

        # 3. Imágenes sin split (y auto-split desactivado)
        unassigned = [r for r in self.project.images if r.split is None]
        if unassigned and not cfg["auto_split"]:
            warnings.append(
                f"• {len(unassigned)} imagen(es) no tienen split asignado y serán omitidas."
            )

        # 4. Imágenes cuyo archivo ya no existe en disco
        missing = [r for r in self.project.images if not Path(r.path).exists()]
        if missing:
            warnings.append(
                f"• {len(missing)} imagen(es) no se encuentran en disco:\n"
                + "\n".join(f"    {Path(r.path).name}" for r in missing[:5])
                + ("  …" if len(missing) > 5 else "")
            )

        # 5. Split muy desbalanceado (ratio > 90 %)
        ratio = self.project.imbalance_ratio()
        if ratio > 0.9 and len(self.project.classes) >= 2:
            warnings.append(
                f"• El dataset está muy desbalanceado ({ratio*100:.0f}%). "
                "Considera usar Balance inteligente (Ctrl+B) antes de exportar."
            )

        if warnings:
            msg = "Se encontraron los siguientes avisos antes de exportar:\n\n"
            msg += "\n\n".join(warnings)
            msg += "\n\n¿Deseas continuar de todas formas?"
            reply = QMessageBox.warning(
                self, "⚠️  Avisos de exportación", msg,
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return

        # Consolidar flag only_annotated (puede venir del diálogo o de la elección anterior)
        only_annotated = cfg.get("only_annotated", False)

        # ── Auto-split de imágenes sin asignar ────────────────
        if cfg["auto_split"]:
            unassigned = [r for r in self.project.images if r.split is None]
            random.shuffle(unassigned)
            n  = len(unassigned)
            nt = int(n * cfg["pct_train"] / 100)
            nv = int(n * cfg["pct_val"] / 100)
            for i, rec in enumerate(unassigned):
                rec.split = "train" if i < nt else ("val" if i < nt + nv else "test")

        root = Path(cfg["dest"]) / cfg["name"]
        for split in ("train", "val", "test"):
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)

        prog = QProgressDialog(
            "Exportando dataset…", "Cancelar",
            0, len(self.project.images), self
        )
        prog.setWindowModality(Qt.WindowModal)
        prog.setStyleSheet(f"background:{CARD_BG}; color:{TEXT_PRIMARY};")

        counts = {"train": 0, "val": 0, "test": 0, "skipped": 0, "no_annot": 0}
        for i, rec in enumerate(self.project.images):
            prog.setValue(i)
            if prog.wasCanceled():
                break
            if not rec.split:
                counts["skipped"] += 1
                continue

            # Filtrar imágenes sin anotaciones si el usuario lo eligió
            if only_annotated and not rec.boxes:
                counts["no_annot"] += 1
                continue

            img_path  = Path(rec.path)
            split_dir = root / rec.split

            if cfg["copy_images"]:
                shutil.copy2(str(img_path), str(split_dir / "images" / img_path.name))

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            # Ordenar las cajas por class_id para evitar errores de índice en YOLO
            sorted_boxes = sorted(rec.boxes, key=lambda b: b.class_id)
            with open(split_dir / "labels" / (img_path.stem + ".txt"), "w") as f:
                for box in sorted_boxes:
                    f.write(box.to_yolo_line(iw, ih) + "\n")
            counts[rec.split] += 1

        prog.setValue(len(self.project.images))

        if cfg["gen_yaml"]:
            with open(root / "data.yaml", "w") as f:
                f.write(
                    f"path: {root}\ntrain: train/images\nval: val/images\n"
                    f"test: test/images\n\n"
                )
                f.write(f"nc: {len(self.project.classes)}\n")
                names = [c.name for c in sorted(self.project.classes, key=lambda x: x.id)]
                f.write(f"names: {names}\n")

        with open(root / "classes.txt", "w") as f:
            for cls in sorted(self.project.classes, key=lambda x: x.id):
                f.write(cls.name + "\n")

        summary_lines = [
            f"Dataset exportado en:\n{root}\n",
            f"Train: {counts['train']}  |  Val: {counts['val']}  |  Test: {counts['test']}",
            f"Omitidas (sin split): {counts['skipped']}",
        ]
        if counts.get("no_annot", 0):
            summary_lines.append(
                f"Omitidas (sin anotaciones): {counts['no_annot']}"
            )
        QMessageBox.information(
            self, "✅ Exportación completada",
            "\n".join(summary_lines),
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

    # ══════════════════════════════════════════════════════════
    #  HERRAMIENTAS
    # ══════════════════════════════════════════════════════════

    def _open_frame_extractor(self):
        dlg = FrameExtractorDialog(project=self.project, parent=self)
        dlg.exec_()
        if self.project:
            self._refresh_all()
            self.panel.update_stats(self.project.stats())

    # ══════════════════════════════════════════════════════════
    #  ESTADÍSTICAS Y AYUDA
    # ══════════════════════════════════════════════════════════

    def _show_stats(self):
        if not self.project:
            return
        s           = self.project.stats()
        total_boxes = sum(len(r.boxes) for r in self.project.images)
        counts      = self.project.class_counts()
        cls_lines   = "\n".join(
            f"  {self.project.get_class(cid).name if self.project.get_class(cid) else cid}: {n}"
            for cid, n in sorted(counts.items())
        )
        ratio = self.project.imbalance_ratio()
        level = "CRÍTICO ⚠" if ratio > IMBALANCE_CRIT else (
            "MODERADO ⚡" if ratio > IMBALANCE_WARN else "OK ✔"
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
  Val: {s['val']}
  Test:  {s['test']}

Cajas totales: {total_boxes}

Por clase:
{cls_lines if cls_lines else '  (sin anotaciones)'}

Índice de desbalance: {ratio*100:.1f}%  [{level}]
        """)

    def _show_help(self):
        QMessageBox.information(self, "Atajos de teclado", """
Herramientas:
  Ctrl+E             Abrir Extractor de Frames
  Ctrl+B             Balance inteligente de dataset

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
  Botones TRAIN / VALID / TEST en el panel izquierdo.
        """)

    # ══════════════════════════════════════════════════════════
    #  CIERRE Y CONFIRMACIÓN
    # ══════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════
    #  TEMA
    # ══════════════════════════════════════════════════════════

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