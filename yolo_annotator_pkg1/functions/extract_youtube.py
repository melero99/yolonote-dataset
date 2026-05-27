"""
╔══════════════════════════════════════════════════════════════╗
║  functions/extract_youtube.py  —  Descarga + extracción YT   ║
╚══════════════════════════════════════════════════════════════╝

Descarga un vídeo de YouTube usando yt-dlp y extrae frames
con OpenCV.  Es independiente de Qt y testable en aislamiento.

Requiere:
    pip install yt-dlp
    (ffmpeg en PATH para calidades 720p+)
"""
import tempfile
from pathlib import Path
from typing import Callable, Optional

from yolo_annotator_pkg1.functions.extract_local import extract_local


# Mapa de texto de calidad UI → formato yt-dlp
_QUALITY_MAP = {
    "Mejor disponible": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
    "1080p":  "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]",
    "720p":   "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
    "480p":   "best[height<=480]/bestvideo[height<=480]+bestaudio",
    "360p":   "best[height<=360]/bestvideo[height<=360]+bestaudio",
    "240p":   "best[height<=240]/bestvideo[height<=240]+bestaudio",
    "144p":   "best[height<=144]/bestvideo[height<=144]+bestaudio",
}


def download_and_extract(
    url: str,
    base_out: Path,
    quality: str = "Mejor disponible",
    t_start_s: int = 0,
    t_end_s: int = 0,
    fps_obj: float = 1.0,
    fmt: str = "jpg",
    quality_jpeg: int = 95,
    logger: Optional[Callable[[str], None]] = None,
    progress_set: Optional[Callable[[int], None]] = None,
    progress_set_max: Optional[Callable[[int], None]] = None,
) -> Optional[Path]:
    """
    Descarga un vídeo de YouTube y extrae frames del mismo.

    Args:
        url:              URL del vídeo de YouTube.
        base_out:         Carpeta base donde se guardarán los frames.
        quality:          Texto de calidad (clave de _QUALITY_MAP).
        t_start_s:        Segundo de inicio de la extracción.
        t_end_s:          Segundo de fin de la extracción.
        fps_obj:          Cadencia de extracción en frames por segundo.
        fmt:              Formato de salida: "jpg" o "png".
        quality_jpeg:     Calidad JPEG (1-100).
        logger:           Función de logging.
        progress_set:     Callback de progreso (valor actual).
        progress_set_max: Callback de progreso (valor máximo).

    Returns:
        Path con los frames extraídos, o None si falló.
    """
    def log(msg: str):
        if logger:
            logger(msg)

    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        log("❌ yt-dlp no está instalado.  Ejecuta: pip install yt-dlp")
        return None

    fmt_ydl = _QUALITY_MAP.get(quality, _QUALITY_MAP["Mejor disponible"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "video.mp4"

        ydl_opts = {
            "format":   fmt_ydl,
            "outtmpl":  str(tmp_path.with_suffix("")),
            "quiet":    True,
            "no_warnings": True,
        }

        log(f"⬇️  Descargando: {url}")
        log(f"   Calidad: {quality}")

        try:
            import yt_dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info    = ydl.extract_info(url, download=True)
                nombre  = info.get("title", "youtube_video")
        except Exception as e:
            log(f"❌ Error al descargar: {e}")
            return None

        # El archivo descargado puede tener extensión distinta
        candidates = list(Path(tmpdir).glob("video.*"))
        if not candidates:
            log("❌ No se encontró el archivo descargado.")
            return None

        video_file = candidates[0]
        log(f"✅ Descarga completada: {video_file.name}")

        return extract_local(
            video_file, nombre, base_out,
            fps_obj=fps_obj, fmt=fmt, quality=quality_jpeg,
            t_start_s=t_start_s, t_end_s=t_end_s,
            logger=logger,
            progress_set=progress_set,
            progress_set_max=progress_set_max,
        )
