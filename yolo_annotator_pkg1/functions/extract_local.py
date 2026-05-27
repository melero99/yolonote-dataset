"""
╔══════════════════════════════════════════════════════════════╗
║  functions/extract_local.py  —  Extracción de vídeo local    ║
╚══════════════════════════════════════════════════════════════╝

Extrae frames de un archivo de vídeo local usando OpenCV.
Es independiente de Qt y testable en aislamiento.
"""
from pathlib import Path
from typing import Callable, Optional

import cv2

from yolo_annotator_pkg1.controllers.video_utils import (
    crear_carpeta,
    imwrite_unicode,
    cap_open_unicode,
)


def extract_local(
    video_path: Path,
    nombre: str,
    base_out: Path,
    fps_obj: float = 1.0,
    fmt: str = "jpg",
    quality: int = 95,
    min_w: int = 0,
    t_start_s: int = 0,
    t_end_s: int = 0,
    logger: Optional[Callable[[str], None]] = None,
    progress_set: Optional[Callable[[int], None]] = None,
    progress_set_max: Optional[Callable[[int], None]] = None,
) -> Optional[Path]:
    """
    Extrae frames de un vídeo local a una carpeta de salida.

    Args:
        video_path:       Ruta al archivo de vídeo.
        nombre:           Nombre base para la subcarpeta de salida.
        base_out:         Carpeta padre de salida.
        fps_obj:          Cadencia de extracción (frames por segundo objetivo).
        fmt:              Formato de salida: "jpg" o "png".
        quality:          Calidad JPEG (1-100).
        min_w:            Ancho mínimo de frame en píxeles (0 = sin límite).
        t_start_s:        Segundo de inicio (0 = desde el principio).
        t_end_s:          Segundo de fin (0 = hasta el final).
        logger:           Función de logging (recibe str).
        progress_set:     Callback para actualizar la barra de progreso (valor).
        progress_set_max: Callback para establecer el máximo de la barra.

    Returns:
        Path de la carpeta con los frames extraídos, o None si falló.
    """
    def log(msg: str):
        if logger:
            logger(msg)

    cap, tmp_path = cap_open_unicode(video_path)
    if not cap.isOpened():
        log(f"❌ No se pudo abrir el vídeo: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_s   = total_frames / source_fps

    # Calcular rango de frames
    frame_start = int(t_start_s * source_fps) if t_start_s > 0 else 0
    frame_end   = int(t_end_s   * source_fps) if t_end_s   > 0 else total_frames
    frame_end   = min(frame_end, total_frames)
    frame_step  = max(1, int(round(source_fps / fps_obj)))

    n_estimated = max(1, (frame_end - frame_start) // frame_step)
    if progress_set_max:
        progress_set_max(n_estimated)

    log(f"📹 {video_path.name}  —  {duration_s:.1f}s  @{source_fps:.1f}fps")
    log(f"   Extrayendo: frames {frame_start}–{frame_end}  (paso {frame_step})")

    carpeta = crear_carpeta(base_out, nombre)
    params  = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else []

    # Ir al frame inicial
    if frame_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    saved   = 0
    skipped = 0
    target_fn = frame_start  # próximo frame objetivo a extraer

    while target_fn < frame_end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_fn)
        ok, frame = cap.read()
        if not ok:
            break

        if min_w > 0 and frame.shape[1] < min_w:
            skipped += 1
        else:
            out_path = carpeta / f"frame_{target_fn:07d}.{fmt}"
            imwrite_unicode(out_path, frame, params)
            saved += 1
            if progress_set:
                progress_set(saved)

        target_fn += frame_step

    cap.release()
    if tmp_path:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass

    log(f"✅ {saved} frames guardados en {carpeta}  ({skipped} omitidos por ancho mínimo)")
    return carpeta
