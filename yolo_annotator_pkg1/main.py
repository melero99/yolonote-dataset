"""
main.py — Punto de entrada de YOLO Annotator Pro.

Ejecutar desde la carpeta que contiene yolo_annotator_pkg1/:
    python main.py
"""
import sys
from PyQt5.QtWidgets import QApplication
from yolo_annotator_pkg1.main_windows import YoloAnnotatorPro


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = YoloAnnotatorPro()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
