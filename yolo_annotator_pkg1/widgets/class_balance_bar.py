"""
╔══════════════════════════════════════════════════════════════╗
║  widgets/class_balance_bar.py  —  Barra de balance por clase ║
╚══════════════════════════════════════════════════════════════╝

Widget visual de barra horizontal que muestra el porcentaje real
de una clase frente a su objetivo uniforme.

Colores de la barra:
  - Verde  → desviación dentro de IMBALANCE_WARN
  - Naranja → desviación entre WARN y CRIT
  - Rojo   → desviación por encima de IMBALANCE_CRIT

Una línea blanca discontinua marca el porcentaje objetivo.
"""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import Qt

from yolo_annotator_pkg1.resources import (
    DARK_BG, BORDER_COLOR, IMBALANCE_WARN, IMBALANCE_CRIT,
)


class ClassBalanceBar(QWidget):
    """
    Barra visual coloreada según el nivel de balance de una clase.

    Uso::

        bar = ClassBalanceBar()
        bar.set_data(ratio=0.45, target=0.33, color=QColor("#4a4aaa"))

    Args:
        parent: widget padre Qt.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(18)
        # Estado interno — se actualiza con set_data()
        self._ratio  = 0.0    # porcentaje real de esta clase (0..1)
        self._target = 0.0    # porcentaje objetivo uniforme (0..1)
        self._color  = QColor("#4a4aaa")

    def set_data(self, ratio: float, target: float, color: QColor):
        """
        Actualiza los datos de la barra y la repinta.

        Args:
            ratio:  Porcentaje real de la clase (0.0–1.0).
            target: Porcentaje objetivo ideal (0.0–1.0).
            color:  Color base de la clase.
        """
        self._ratio  = max(0.0, min(1.0, ratio))
        self._target = max(0.0, min(1.0, target))
        self._color  = color
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # ── Fondo ─────────────────────────────────────────────
        p.setBrush(QBrush(QColor(DARK_BG)))
        p.setPen(QPen(QColor(BORDER_COLOR)))
        p.drawRoundedRect(0, 0, w - 1, h - 1, 4, 4)

        if self._ratio > 0:
            bar_w = int((w - 2) * self._ratio)

            # ── Color según desviación respecto al objetivo ────
            dev = abs(self._ratio - self._target)
            if dev > IMBALANCE_CRIT:
                bar_color = QColor("#8b0000")   # rojo crítico
            elif dev > IMBALANCE_WARN:
                bar_color = QColor("#8b6000")   # naranja moderado
            else:
                bar_color = self._color         # color de la clase

            p.setBrush(QBrush(bar_color))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(1, 1, bar_w, h - 2, 3, 3)

        # ── Línea de objetivo (blanca discontinua) ─────────────
        if self._target > 0:
            tx = int((w - 2) * self._target)
            p.setPen(QPen(QColor("#ffffff"), 2, Qt.DashLine))
            p.drawLine(tx, 0, tx, h)
        # No llamar p.end() explícitamente — Qt lo hace automáticamente