# yolonote-dataset
# YOLO Annotator Pro

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-41CD52?style=flat-square&logo=qt&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-GPL/Commercial-green?style=flat-square)


**Herramienta de anotación de imágenes para entrenar modelos YOLO.**  
Gestiona datasets completos: anota, organiza splits y exporta listos para entrenamiento.

![Screenshot placeholder](https://github.com/melero99/yolonote-dataset/blob/main/image1.png)
![Screenshot placeholder](https://github.com/melero99/yolonote-dataset/blob/main/image2.png)

</div>

---

## ¿Qué es?

YOLO Annotator Pro es una aplicación de escritorio para crear y gestionar datasets de detección de objetos en formato YOLO. Cubre todo el flujo de trabajo: desde la importación de imágenes (o extracción de frames de vídeo y streams de YouTube) hasta la exportación del dataset estructurado con su `data.yaml`, listo para pasarle directamente a YOLOv8, YOLOv5 o cualquier variante compatible.

---

## Características

### Anotación
- **Bounding boxes** con clic y arrastre sobre la imagen
- **Zoom con rueda del ratón** centrado en el cursor, de 10% a 2000%
- **Pan** con botón central del ratón para navegar imágenes grandes
- **Undo** por capas (`Ctrl+Z`) y eliminación por doble clic sobre una caja
- **Selección de clase por tecla numérica** (1–9) para anotar sin levantar la mano del teclado

### Gestión de clases
- Añade y elimina clases con nombre y color personalizable
- Las clases se identifican por ID y se renderizan con su color sobre el canvas

### Organización del dataset
- Asigna cada imagen a `train`, `valid` o `test` con un solo clic
- **Auto-split aleatorio** configurable por porcentaje (ej. 80/10/10)
- Marca imágenes como **verificadas** para control de calidad
- Estadísticas en tiempo real: total, anotadas, verificadas, por split y por clase

### Exportación
- Genera la estructura de carpetas estándar YOLO:
  ```
  dataset/
  ├── train/images/  train/labels/
  ├── valid/images/  valid/labels/
  ├── test/images/   test/labels/
  └── data.yaml
  ```
- Exporta `data.yaml` listo para YOLOv8/v5
- Exporta `classes.txt` compatible con LabelImg y otras herramientas
- Opción de copiar imágenes o generar solo los labels

### Extractor de frames integrado
- Extrae frames de vídeos locales (MP4, MKV, AVI, MOV, WebM…) a la cadencia elegida
- Control de FPS de extracción, formato (JPG/PNG), calidad JPEG y ancho mínimo
- **Rango de tiempo configurable** en formato `mm:ss`: extrae solo el segmento que te interesa, no el vídeo entero
- Añade automáticamente los frames extraídos al proyecto abierto

### Descarga y extracción desde YouTube
- Descarga vídeos de YouTube vía `yt-dlp` y extrae los frames **directamente, sin guardar el vídeo completo en disco**
- **Selección de resolución:** Mejor disponible, 1080p, 720p, 480p, 360p, 240p, 144p
- **Con ffmpeg:** accede a streams DASH separados (vídeo + audio) para obtener 1080p/720p real. Los frames se extraen mediante pipe de ffmpeg sin pasar por disco
- **Sin ffmpeg:** descarga el mejor stream progresivo disponible (hasta ~480p)
- **Rango de tiempo:** extrae solo entre `mm:ss` de inicio y `mm:ss` de fin; con ffmpeg el salto al punto de inicio es instantáneo (`-ss` nativo)
- Detección automática de ffmpeg: busca en el PATH del sistema y en rutas típicas de Windows (`C:\ffmpeg\bin`, `C:\Program Files\ffmpeg\bin`…) y actualiza el PATH del proceso si lo encuentra
- El progreso de descarga y extracción se muestra en el log de la propia ventana, sin necesidad de tener la consola abierta

### Proyectos persistentes
- Guarda y carga proyectos en formato `.yannotator` (JSON)
- Recuerda la imagen donde se dejó el trabajo, los splits y todas las anotaciones

---

## Instalación

```bash
pip install PyQt5 opencv-python
python yolo_annotator.py
```

Para soporte de YouTube (opcional):
```bash
pip install yt-dlp
```

Para descargas en 1080p/720p desde YouTube es necesario tener **ffmpeg** instalado y accesible. En Windows:
```bash
winget install Gyan.FFmpeg
```

**Requisitos:** Python 3.10+

---

## Instalador para Windows

En la carpeta [`build/`](build/) encontrarás los archivos para generar un `.exe` portable o un instalador con asistente:

```bash
# En Windows, doble clic en:
build.bat
```

El script instala las dependencias, compila con PyInstaller y, si detecta Inno Setup, genera el instalador `.exe`. Ver [`build/README_BUILD.md`](build/README_BUILD.md) para más detalles.

---

## Atajos de teclado

| Acción | Atajo |
|---|---|
| Imagen anterior / siguiente | `←` / `→` |
| Seleccionar clase | `1` – `9` |
| Deshacer última caja | `Ctrl+Z` |
| Guardar proyecto | `Ctrl+S` |
| Nuevo / Abrir proyecto | `Ctrl+N` / `Ctrl+O` |
| Extractor de frames | `Ctrl+E` |
| Eliminar caja | Doble clic sobre ella |
| Zoom in / out | Rueda del ratón |
| Zoom fijo | `+` / `-` |
| Ajustar a ventana | `0` |
| Pan (mover vista) | Botón central + arrastrar |

---

## Flujo de trabajo típico

```
1. Archivo → Nuevo proyecto
2. Archivo → Añadir carpeta de imágenes
           (o Ctrl+E → archivo local para extraer frames de un vídeo)
           (o Ctrl+E → URL de YouTube para extraer frames sin descargar el vídeo)
3. Panel izquierdo → Añadir clases
4. Seleccionar clase + dibujar bounding boxes
5. Asignar cada imagen a TRAIN / VALID / TEST
           (o Proyecto → Auto-asignar splits)
6. Archivo → Exportar dataset
```

---

## Formato de exportación

Cada label se genera en formato YOLO normalizado:

```
<class_id> <cx> <cy> <width> <height>
```

Todos los valores entre 0 y 1, relativos al tamaño de la imagen.

---

## Dependencias

| Paquete | Uso |
|---|---|
| `PyQt5` | Interfaz gráfica |
| `opencv-python` | Lectura de imágenes y vídeo |
| `yt-dlp` *(opcional)* | Descarga de vídeos de YouTube |
| `ffmpeg` *(opcional, binario del sistema)* | Streams 1080p/720p y pipe de frames desde YouTube |
| `pyinstaller` *(build)* | Generación del ejecutable |

---

## Contribuir

Los pull requests son bienvenidos. Para cambios grandes, abre primero un issue para discutir qué te gustaría cambiar.

---

## Licencia

[GPL](LICENSE) [COMMERCIAL](COMMERCIAL_LICENSE)
