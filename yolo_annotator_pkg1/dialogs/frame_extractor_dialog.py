"""
╔══════════════════════════════════════════════════════════════╗
║  dialogs/frame_extractor_dialog.py  —  Extractor de frames   ║
╚══════════════════════════════════════════════════════════════╝

Diálogo modal para extraer frames de vídeos locales o de YouTube.

Responsabilidad: solo orquesta la UI y delega la extracción real
a los módulos de yolo_annotator_pkg/functions/:
  - extract_local.py    → vídeos locales
  - extract_youtube.py  → streams de YouTube vía yt-dlp
  - video_utils.py      → helpers de bajo nivel (imwrite, cap_open)

La lógica de extracción ya NO vive en este archivo.
"""
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QWidget, QRadioButton, QLineEdit, QDoubleSpinBox,
    QSpinBox, QComboBox, QFileDialog, QTextEdit, QProgressBar,
    QMessageBox, QApplication,
)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTimeEdit
from PyQt5.QtCore import QTime

from yolo_annotator_pkg1.models import ImageRecord
from yolo_annotator_pkg1.resources import (
    DARK_BG, CARD_BG, TEXT_PRIMARY, TEXT_MUTED, ACCENT,
    btn_style, grp_style, input_style,
)
from yolo_annotator_pkg1.functions.extract_local import extract_local
from yolo_annotator_pkg1.functions.extract_youtube import download_and_extract
from yolo_annotator_pkg1.resources import SUPPORTED_EXTS


class FrameExtractorDialog(QDialog):
    """
    Diálogo de extracción de frames de vídeo.

    Permite al usuario seleccionar una fuente (archivo local o URL de
    YouTube), configurar los parámetros de extracción y lanzar el
    proceso.  Opcionalmente añade los frames resultantes al proyecto.

    Args:
        project: Proyecto activo (puede ser None si no hay proyecto abierto).
        parent:  Widget padre Qt.
    """

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Extractor de Frames")
        self.setMinimumSize(600, 720)
        self.setStyleSheet(f"background:{CARD_BG}; color:{TEXT_PRIMARY};")
        self._last_output_folder = None
        self._running            = False
        self._build_ui()

    # ══════════════════════════════════════════════════════════
    #  CONSTRUCCIÓN DE UI
    # ══════════════════════════════════════════════════════════

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(16, 16, 16, 16)

        # ── Fuente ────────────────────────────────────────────
        grp_src = QGroupBox("Fuente del vídeo")
        grp_src.setStyleSheet(grp_style())
        src_lay = QVBoxLayout(grp_src)

        mode_row = QHBoxLayout()
        self.rb_local   = QRadioButton("Archivo local")
        self.rb_youtube = QRadioButton("URL de YouTube")
        self.rb_local.setChecked(True)
        for rb in (self.rb_local, self.rb_youtube):
            rb.setStyleSheet(f"color:{TEXT_PRIMARY};")
            mode_row.addWidget(rb)
        mode_row.addStretch()
        src_lay.addLayout(mode_row)

        # ── Widget archivo local ───────────────────────────────
        self.local_widget = QWidget()
        local_row = QHBoxLayout(self.local_widget)
        local_row.setContentsMargins(0, 0, 0, 0)
        self.local_edit = QLineEdit()
        self.local_edit.setPlaceholderText("Ruta al archivo de vídeo…")
        self.local_edit.setStyleSheet(input_style())
        btn_bv = QPushButton("…")
        btn_bv.setFixedWidth(34)
        btn_bv.setStyleSheet(btn_style())
        btn_bv.clicked.connect(self._browse_video)
        local_row.addWidget(self.local_edit)
        local_row.addWidget(btn_bv)
        src_lay.addWidget(self.local_widget)

        # ── Widget YouTube ─────────────────────────────────────
        self.yt_widget = QWidget()
        yt_lay = QVBoxLayout(self.yt_widget)
        yt_lay.setContentsMargins(0, 0, 0, 0)
        self.yt_edit = QLineEdit()
        self.yt_edit.setPlaceholderText("https://www.youtube.com/watch?v=...")
        self.yt_edit.setStyleSheet(input_style())
        yt_lay.addWidget(self.yt_edit)

        res_row = QHBoxLayout()
        res_label = QLabel("Calidad:")
        res_label.setStyleSheet(f"color:{TEXT_PRIMARY};")
        self.yt_quality_combo = QComboBox()
        self.yt_quality_combo.addItems([
            "Mejor disponible", "1080p", "720p", "480p", "360p", "240p", "144p",
        ])
        self.yt_quality_combo.setStyleSheet(input_style())
        try:
            self.yt_quality_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        except Exception:
            pass
        self.yt_quality_combo.setMinimumWidth(140)
        res_row.addWidget(res_label)
        res_row.addWidget(self.yt_quality_combo)
        res_row.addStretch()
        yt_lay.addLayout(res_row)

        yt_note = QLabel("⚠️  Se necesita ffmpeg para descargar 1080p/720p (streams separados).")
        yt_note.setStyleSheet("color:#aa8800; font-size:10px;")
        yt_lay.addWidget(yt_note)
        self.yt_widget.hide()
        src_lay.addWidget(self.yt_widget)

        self.rb_local.toggled.connect(self._toggle_source)
        lay.addWidget(grp_src)

        # ── Opciones de extracción ─────────────────────────────
        grp_opt = QGroupBox("Opciones de extracción")
        grp_opt.setStyleSheet(grp_style())
        from PyQt5.QtWidgets import QFormLayout
        opt_form = QFormLayout(grp_opt)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 30.0)
        self.fps_spin.setValue(1.0)
        self.fps_spin.setSingleStep(0.5)
        self.fps_spin.setSuffix("  frame/s")
        self.fps_spin.setStyleSheet(input_style())
        opt_form.addRow("Cadencia:", self.fps_spin)

        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["jpg", "png"])
        self.fmt_combo.setStyleSheet(input_style())
        try:
            self.fmt_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        except Exception:
            pass
        self.fmt_combo.setMinimumWidth(100)
        opt_form.addRow("Formato:", self.fmt_combo)

        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)
        self.quality_spin.setSuffix("  %")
        self.quality_spin.setStyleSheet(input_style())
        opt_form.addRow("Calidad JPEG:", self.quality_spin)

        self.minw_spin = QSpinBox()
        self.minw_spin.setRange(0, 7680)
        self.minw_spin.setValue(0)
        self.minw_spin.setSuffix("  px  (0=sin límite)")
        self.minw_spin.setStyleSheet(input_style())
        opt_form.addRow("Ancho mínimo:", self.minw_spin)

        # ── Rango de tiempo ────────────────────────────────────
        time_row = QHBoxLayout()
        lbl_desde = QLabel("Desde:")
        lbl_desde.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:12px;")
        self.time_start = QTimeEdit()
        self.time_start.setDisplayFormat("mm:ss")
        self.time_start.setTime(QTime(0, 0, 0))
        self.time_start.setStyleSheet(input_style())
        self.time_start.setMinimumWidth(90)
        lbl_hasta = QLabel("Hasta:")
        lbl_hasta.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:12px;")
        self.time_end = QTimeEdit()
        self.time_end.setDisplayFormat("mm:ss")
        self.time_end.setTime(QTime(0, 0, 0))
        self.time_end.setStyleSheet(input_style())
        self.time_end.setMinimumWidth(90)
        lbl_fin = QLabel("(00:00 = hasta el final)")
        lbl_fin.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
        time_row.addWidget(lbl_desde)
        time_row.addWidget(self.time_start)
        time_row.addSpacing(12)
        time_row.addWidget(lbl_hasta)
        time_row.addWidget(self.time_end)
        time_row.addSpacing(6)
        time_row.addWidget(lbl_fin)
        time_row.addStretch()
        opt_form.addRow("Rango:", time_row)
        lay.addWidget(grp_opt)

        # ── Carpeta de salida ──────────────────────────────────
        grp_out = QGroupBox("Carpeta de salida")
        grp_out.setStyleSheet(grp_style())
        out_lay = QHBoxLayout(grp_out)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Selecciona carpeta base…")
        self.out_edit.setStyleSheet(input_style())
        btn_bo = QPushButton("…")
        btn_bo.setFixedWidth(34)
        btn_bo.setStyleSheet(btn_style())
        btn_bo.clicked.connect(self._browse_output)
        out_lay.addWidget(self.out_edit)
        out_lay.addWidget(btn_bo)
        lay.addWidget(grp_out)

        # ── Log ────────────────────────────────────────────────
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(130)
        self.log.setMaximumHeight(220)
        self.log.setStyleSheet(
            f"background:{DARK_BG}; color:#aaffaa; "
            f"font-family:Consolas,monospace; font-size:11px; border-radius:4px;"
        )
        lay.addWidget(self.log)

        # ── Barra de progreso ──────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setStyleSheet(f"""
            QProgressBar {{ background:{DARK_BG}; border-radius:4px;
                           color:white; text-align:center; }}
            QProgressBar::chunk {{ background:{ACCENT}; border-radius:4px; }}
        """)
        lay.addWidget(self.progress)

        # ── Botones de acción ──────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  Extraer frames")
        self.btn_start.setStyleSheet(btn_style("#1a5c2a", "#2a8a3a"))
        self.btn_start.clicked.connect(self._start_extraction)

        self.btn_add = QPushButton("➕  Añadir al proyecto")
        self.btn_add.setStyleSheet(btn_style("#2a3a6a", "#4a6aaa"))
        self.btn_add.setEnabled(False)
        self.btn_add.clicked.connect(self._add_to_project)

        btn_close = QPushButton("Cerrar")
        btn_close.setStyleSheet(btn_style())
        btn_close.clicked.connect(self.close)

        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_add)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

    # ══════════════════════════════════════════════════════════
    #  HELPERS DE UI
    # ══════════════════════════════════════════════════════════

    def _toggle_source(self, local: bool):
        """Muestra / oculta los widgets de fuente según el radio button."""
        self.local_widget.setVisible(local)
        self.yt_widget.setVisible(not local)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecciona vídeo",
            filter="Vídeos (*.mp4 *.mkv *.avi *.mov *.webm *.ts);;Todos (*)"
        )
        if path:
            self.local_edit.setText(path)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Carpeta base de salida")
        if folder:
            self.out_edit.setText(folder)

    def _log(self, msg: str):
        """Añade un mensaje al área de log y desplaza al final."""
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum()
        )
        QApplication.processEvents()

    def _time_to_seconds(self, te: QTimeEdit) -> int:
        """
        Convierte un QTimeEdit en segundos totales.

        Returns:
            Segundos totales; 0 significa "hasta el final".
        """
        t = te.time()
        return t.minute() * 60 + t.second()

    # ══════════════════════════════════════════════════════════
    #  EXTRACCIÓN — ORQUESTACIÓN (delega a functions/)
    # ══════════════════════════════════════════════════════════

    def _start_extraction(self):
        """Valida la entrada y delega la extracción al módulo correspondiente."""
        if self._running:
            return

        out_folder = self.out_edit.text().strip()
        if not out_folder:
            QMessageBox.warning(self, "Sin carpeta", "Selecciona una carpeta de salida.")
            return

        is_local = self.rb_local.isChecked()
        if is_local:
            vps = self.local_edit.text().strip()
            if not vps:
                QMessageBox.warning(self, "Sin vídeo", "Selecciona un archivo de vídeo.")
                return
            if not Path(vps).exists():
                QMessageBox.warning(self, "No encontrado", f"No existe:\n{vps}")
                return
        else:
            if not self.yt_edit.text().strip():
                QMessageBox.warning(self, "Sin URL", "Introduce una URL de YouTube.")
                return

        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.progress.setValue(0)
        self.log.clear()

        base_out   = Path(out_folder)
        t_start_s  = self._time_to_seconds(self.time_start)
        t_end_s    = self._time_to_seconds(self.time_end)
        fps_obj    = self.fps_spin.value()
        fmt        = self.fmt_combo.currentText()
        quality    = self.quality_spin.value()
        min_w      = self.minw_spin.value()

        try:
            carpeta = None
            if is_local:
                vp = Path(self.local_edit.text().strip())
                carpeta = extract_local(
                    vp, vp.stem, base_out,
                    fps_obj=fps_obj, fmt=fmt, quality=quality,
                    min_w=min_w, t_start_s=t_start_s, t_end_s=t_end_s,
                    logger=self._log,
                    progress_set=lambda v: self.progress.setValue(v),
                    progress_set_max=lambda m: self.progress.setMaximum(m),
                )
            else:
                quality_str = self.yt_quality_combo.currentText()
                carpeta = download_and_extract(
                    self.yt_edit.text().strip(), base_out,
                    quality=quality_str,
                    t_start_s=t_start_s, t_end_s=t_end_s,
                    fps_obj=fps_obj, fmt=fmt, quality_jpeg=quality,
                    logger=self._log,
                    progress_set=lambda v: self.progress.setValue(v),
                    progress_set_max=lambda m: self.progress.setMaximum(m),
                )
            if carpeta:
                self._last_output_folder = carpeta
                if self.project is not None:
                    self.btn_add.setEnabled(True)
        except Exception as e:
            self._log(f"❌ Error: {e}")
        finally:
            self._running = False
            self.btn_start.setEnabled(True)

    def _add_to_project(self):
        """Añade los frames extraídos al proyecto activo."""
        if not self._last_output_folder or not self.project:
            return

        existing = {r.path for r in self.project.images}
        added    = 0
        for p in sorted(self._last_output_folder.iterdir()):
            if p.suffix.lower() in SUPPORTED_EXTS and str(p) not in existing:
                self.project.images.append(ImageRecord(path=str(p)))
                added += 1

        self._log(f"➕  {added} imágenes añadidas al proyecto '{self.project.name}'.")

        # Refrescar la ventana principal si está disponible
        if self.parent() and hasattr(self.parent(), "_refresh_all"):
            self.parent()._refresh_all()
            if added > 0 and len(self.project.images) == added:
                self.parent()._go_to(0)

        QMessageBox.information(
            self, "Añadido",
            f"{added} imágenes añadidas.\nPuedes cerrar esta ventana y empezar a anotar."
        )