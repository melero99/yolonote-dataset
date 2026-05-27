"""
╔══════════════════════════════════════════════════════════════╗
║  models.py  —  Modelos de datos del proyecto                 ║
╚══════════════════════════════════════════════════════════════╝

Contiene los dataclasses/clases del dominio:
  - YoloClass    → clase de anotación (id, nombre, color).
  - BoundingBox  → caja de anotación normalizada YOLO.
  - ImageRecord  → registro de imagen con sus boxes y metadatos.
  - Project      → proyecto completo con clases e imágenes.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtGui import QColor


# ══════════════════════════════════════════════════════════════
#  YOLO CLASS
# ══════════════════════════════════════════════════════════════

@dataclass
class YoloClass:
    """
    Clase de anotación YOLO.

    Attributes:
        id:        Índice de la clase (0-based).
        name:      Nombre legible de la clase.
        color_hex: Color en formato #RRGGBB para la UI.
    """
    id:        int
    name:      str
    color_hex: str = "#4a9aff"

    @property
    def color(self) -> QColor:
        """QColor calculado desde color_hex."""
        return QColor(self.color_hex)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "color_hex": self.color_hex}

    @classmethod
    def from_dict(cls, d: dict) -> "YoloClass":
        return cls(id=d["id"], name=d["name"], color_hex=d.get("color_hex", "#4a9aff"))


# ══════════════════════════════════════════════════════════════
#  BOUNDING BOX
# ══════════════════════════════════════════════════════════════

@dataclass
class BoundingBox:
    """
    Caja de anotación en coordenadas YOLO normalizadas (0..1).

    Attributes:
        class_id: Id de la clase YoloClass.
        cx, cy:   Centro de la caja (normalizado).
        w, h:     Ancho y alto (normalizados).
        img_w, img_h: Dimensiones de la imagen de referencia (píxeles).
    """
    class_id: int
    cx:       float
    cy:       float
    w:        float
    h:        float
    img_w:    int = 0
    img_h:    int = 0

    def pixel_info(self) -> Dict[str, int]:
        """
        Coordenadas absolutas en píxeles (esquina superior-izquierda + tamaño).

        Returns:
            Dict con x, y, w, h en píxeles.
        """
        x = int((self.cx - self.w / 2) * self.img_w)
        y = int((self.cy - self.h / 2) * self.img_h)
        w = int(self.w * self.img_w)
        h = int(self.h * self.img_h)
        return {"x": x, "y": y, "w": w, "h": h}

    def to_yolo_line(self, img_w: int = 0, img_h: int = 0) -> str:
        """
        Serializa la caja al formato de línea YOLO (.txt).

        Args:
            img_w, img_h: Se ignoran si ya están almacenados en la caja.

        Returns:
            Cadena "class_id cx cy w h" con 6 decimales.
        """
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "cx": self.cx, "cy": self.cy,
            "w": self.w,  "h": self.h,
            "img_w": self.img_w, "img_h": self.img_h,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoundingBox":
        return cls(
            class_id=d["class_id"],
            cx=d["cx"], cy=d["cy"],
            w=d["w"],   h=d["h"],
            img_w=d.get("img_w", 0),
            img_h=d.get("img_h", 0),
        )


# ══════════════════════════════════════════════════════════════
#  IMAGE RECORD
# ══════════════════════════════════════════════════════════════

@dataclass
class ImageRecord:
    """
    Registro de una imagen dentro del proyecto.

    Attributes:
        path:     Ruta absoluta al archivo de imagen.
        boxes:    Lista de BoundingBox anotadas.
        split:    Partición asignada: "train" | "val" | "test" | None.
        verified: True si el anotador marcó la imagen como revisada.
    """
    path:     str
    boxes:    List[BoundingBox] = field(default_factory=list)
    split:    Optional[str]     = None
    verified: bool              = False

    def to_dict(self) -> dict:
        return {
            "path":     self.path,
            "boxes":    [b.to_dict() for b in self.boxes],
            "split":    self.split,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageRecord":
        return cls(
            path=d["path"],
            boxes=[BoundingBox.from_dict(b) for b in d.get("boxes", [])],
            split=d.get("split"),
            verified=d.get("verified", False),
        )


# ══════════════════════════════════════════════════════════════
#  PROJECT
# ══════════════════════════════════════════════════════════════

class Project:
    """
    Proyecto de anotación YOLO.

    Contiene la lista de clases, la lista de imágenes y el índice
    de la imagen actualmente mostrada en el canvas.

    Args:
        name: Nombre legible del proyecto.
    """

    def __init__(self, name: str = "Nuevo proyecto"):
        self.name:          str                  = name
        self.classes:       List[YoloClass]      = []
        self.images:        List[ImageRecord]    = []
        self.current_index: int                  = 0
        self.project_file:  Optional[str]        = None

    # ── Helpers de clases ──────────────────────────────────────

    def next_class_id(self) -> int:
        """Devuelve el próximo id de clase disponible (max+1 o 0)."""
        if not self.classes:
            return 0
        return max(c.id for c in self.classes) + 1

    def get_class(self, class_id: int) -> Optional[YoloClass]:
        """Busca una clase por id; devuelve None si no existe."""
        for c in self.classes:
            if c.id == class_id:
                return c
        return None

    # ── Estadísticas ───────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """
        Estadísticas básicas del proyecto.

        Returns:
            Dict con total, annotated, unassigned, train, val, test.
        """
        total      = len(self.images)
        annotated  = sum(1 for r in self.images if r.boxes)
        unassigned = sum(1 for r in self.images if r.split is None)
        train      = sum(1 for r in self.images if r.split == "train")
        val        = sum(1 for r in self.images if r.split == "val")
        test       = sum(1 for r in self.images if r.split == "test")
        return {
            "total":      total,
            "annotated":  annotated,
            "unassigned": unassigned,
            "train":      train,
            "val":        val,
            "test":       test,
        }

    def class_counts(self) -> Dict[int, int]:
        """
        Cuenta el total de anotaciones por clase en todo el proyecto.

        Returns:
            Dict {class_id: count}.
        """
        counts: Dict[int, int] = {c.id: 0 for c in self.classes}
        for rec in self.images:
            for box in rec.boxes:
                counts[box.class_id] = counts.get(box.class_id, 0) + 1
        return counts

    def imbalance_ratio(self) -> float:
        """
        Índice de desbalance global: (max - min) / max.

        Returns:
            Valor entre 0.0 (perfectamente equilibrado) y 1.0 (máximo desbalance).
            Devuelve 0.0 si hay menos de 2 clases o no hay anotaciones.
        """
        if len(self.classes) < 2:
            return 0.0
        counts = self.class_counts()
        vals   = [counts.get(c.id, 0) for c in self.classes]
        mx     = max(vals)
        if mx == 0:
            return 0.0
        return (mx - min(vals)) / mx

    # ── Persistencia ───────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Guarda el proyecto en un archivo JSON (.yannotator).

        Args:
            path: Ruta del archivo de destino.
        """
        data = {
            "name":          self.name,
            "current_index": self.current_index,
            "classes":       [c.to_dict() for c in self.classes],
            "images":        [r.to_dict() for r in self.images],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.project_file = path

    @classmethod
    def load(cls, path: str) -> "Project":
        """
        Carga un proyecto desde un archivo JSON (.yannotator).

        Args:
            path: Ruta del archivo a leer.

        Returns:
            Instancia de Project reconstruida.

        Raises:
            ValueError: Si el archivo no es un proyecto válido.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        proj               = cls(name=data.get("name", "Sin nombre"))
        proj.current_index = data.get("current_index", 0)
        proj.classes       = [YoloClass.from_dict(c) for c in data.get("classes", [])]
        proj.images        = [ImageRecord.from_dict(r) for r in data.get("images", [])]
        proj.project_file  = path

        # Clampear el índice por si el proyecto tiene menos imágenes
        if proj.images:
            proj.current_index = max(0, min(proj.current_index, len(proj.images) - 1))

        return proj
