"""
╔══════════════════════════════════════════════════════════════╗
║  controllers/video_utils.py  —  Utilidades de vídeo            ║
╚══════════════════════════════════════════════════════════════╝

Funciones de bajo nivel para manejo de vídeo extraídas de
FrameExtractorDialog.  Son independientes de Qt y testables
en aislamiento.

Antes estaban duplicadas / incrustadas en el diálogo.
"""
import re
import tempfile
import shutil
from pathlib import Path

import cv2


def limpiar_nombre(nombre: str) -> str:
    """
    Elimina caracteres inválidos en nombres de carpeta y trunca a 80 chars.

    Args:
        nombre: Cadena a limpiar (título de vídeo, nombre de fichero…).

    Returns:
        Cadena saneada, segura para usar como nombre de directorio.
    """
    return re.sub(r'[\\/*?:"<>|]', "", nombre).replace(" ", "_")[:80]


def crear_carpeta(base: Path, nombre: str) -> Path:
    """
    Crea una carpeta única bajo base/ con nombre sanitizado.

    Si ya existe <nombre>, prueba <nombre>_2, <nombre>_3, …

    Args:
        base:   Directorio padre donde se creará la subcarpeta.
        nombre: Nombre deseado (se pasa por limpiar_nombre automáticamente).

    Returns:
        Path de la carpeta creada.
    """
    nombre  = limpiar_nombre(nombre)
    carpeta = base / nombre
    if carpeta.exists():
        i = 2
        while (base / f"{nombre}_{i}").exists():
            i += 1
        carpeta = base / f"{nombre}_{i}"
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def imwrite_unicode(filepath: Path, frame, params: list) -> None:
    """
    cv2.imwrite seguro para rutas con caracteres Unicode en Windows.

    cv2.imwrite falla silenciosamente con rutas que contienen
    caracteres no-ASCII en Windows.  Esta función usa imencode
    + escritura binaria de Python para evitarlo.

    Args:
        filepath: Ruta destino (puede contener Unicode).
        frame:    Array numpy BGR a guardar.
        params:   Parámetros de cv2.imencode (p.ej. JPEG quality).
    """
    ext = Path(filepath).suffix.lower()
    ok, buf = cv2.imencode(ext, frame, params)
    if ok:
        Path(filepath).write_bytes(buf.tobytes())


def cap_open_unicode(video_path: Path):
    """
    cv2.VideoCapture seguro para rutas Unicode en Windows.

    cv2.VideoCapture falla con rutas que contienen caracteres no-ASCII
    en Windows.  Si la ruta no es ASCII pura, copia el vídeo a un
    temporal ASCII antes de abrirlo.

    Args:
        video_path: Ruta al fichero de vídeo.

    Returns:
        Tupla (cap: cv2.VideoCapture, tmp_path: str | None).
        tmp_path es la ruta del temporal creado (que el llamador debe
        eliminar con Path(tmp_path).unlink() cuando termine), o None
        si no se creó ningún temporal.
    """
    path_str = str(video_path)
    try:
        path_str.encode("ascii")
        # La ruta es ASCII pura → abrir directamente
        return cv2.VideoCapture(path_str), None
    except UnicodeEncodeError:
        pass

    # Ruta con Unicode → copiar a temporal ASCII
    suffix = Path(video_path).suffix
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    shutil.copy2(video_path, tmp.name)
    return cv2.VideoCapture(tmp.name), tmp.name