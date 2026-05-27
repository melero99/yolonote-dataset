"""
╔══════════════════════════════════════════════════════════════╗
║  resources.py  —  Constantes visuales y utilidades globales  ║
╚══════════════════════════════════════════════════════════════╝

Centraliza toda la paleta de colores, estilos de botón y
constantes de configuración de la aplicación.
"""
import sys
import os
from PyQt5.QtGui import QColor


def resource_path(relative_path):
    """Resuelve la ruta a un recurso, compatible con PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath('.')
    return os.path.join(base_path, relative_path)


# ══════════════════════════════════════════════════════════════
#  PALETA DE COLORES GLOBAL
# ══════════════════════════════════════════════════════════════

DARK_BG      = "#0d0d1f"
PANEL_BG     = "#12122a"
CARD_BG      = "#1a1a3a"
BORDER_COLOR = "#2a2a5a"
TEXT_PRIMARY = "#e0e0f0"
TEXT_MUTED   = "#7777aa"
ACCENT       = "#4a4aaa"

SPLIT_COLORS = {
    "train": QColor("#1a5c2a"),
    "val": QColor("#1a3a6a"),
    "test":  QColor("#5c2a1a"),
    None:    QColor("#333355"),
}
SPLIT_TEXT = {
    "train": "TRAIN",
    "val": "VAL",
    "test":  "TEST",
    None:    "—",
}

# Umbrales de desequilibrio
IMBALANCE_WARN = 0.30
IMBALANCE_CRIT = 0.60

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ══════════════════════════════════════════════════════════════
#  ESTILOS COMUNES  (centralizados para evitar duplicación)
# ══════════════════════════════════════════════════════════════

def btn_style(bg="#2a2a5a", hover="#3a3a7a", border=BORDER_COLOR):
    """Estilo estándar para QPushButton en toda la aplicación."""
    return f"""
        QPushButton {{
            background-color: {bg};
            color: {TEXT_PRIMARY};
            border: 1px solid {border};
            border-radius: 6px;
            padding: 5px 12px;
            font-size: 12px;
        }}
        QPushButton:hover {{ background-color: {hover}; }}
        QPushButton:pressed {{ background-color: #1a1a4a; }}
        QPushButton:disabled {{ color: #444466; border-color: #222244; }}
    """


def grp_style() -> str:
    """Estilo común para QGroupBox en todos los diálogos."""
    return (
        f"QGroupBox {{ color:{TEXT_MUTED}; border:1px solid {BORDER_COLOR}; "
        f"border-radius:6px; margin-top:8px; font-size:11px; font-weight:bold; }}"
        f"QGroupBox::title {{ subcontrol-origin:margin; padding:0 6px; }}"
    )


def input_style() -> str:
    """Estilo común para QLineEdit, QSpinBox, QComboBox en todos los diálogos."""
    return (
        f"background:{DARK_BG}; color:{TEXT_PRIMARY}; "
        f"border:1px solid {BORDER_COLOR}; border-radius:4px; padding:4px;"
    )
