# Video Enhancement Pipeline

> Toma un video → duplica resolución → duplica FPS → limpia audio.

```
Entrada: 2240×1260 @ 25fps  →  Salida: 4480×2520 @ 50fps + audio mejorado
```

## Requisitos

- **[Video2X](https://github.com/k4yt3x/video2x)** — Real-ESRGAN (upscale) + RIFE (interpolación)
- **ffmpeg** + **ffprobe** — split, concat, audio
- **GPU con Vulkan** — NVIDIA, AMD o Intel

### Instalar Video2X

```bash
# AppImage (cualquier distro Linux)
wget https://github.com/k4yt3x/video2x/releases/latest/download/Video2X-x86_64.AppImage
chmod +x Video2X-x86_64.AppImage
export V2X_BIN="$PWD/Video2X-x86_64.AppImage"

# Arch Linux
yay -S video2x
```

## Uso

```bash
# Básico — output junto al input
./enhance.sh video.mp4

# Especificar output
./enhance.sh video.mp4 video_4k_50fps.mp4

# Con audio externo (sidecar .m4a de Zoom, etc.)
./enhance.sh video.mp4 output.mp4 audio.m4a
```

### Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `V2X_BIN` | `video2x` | Ruta al binario/AppImage de Video2X |
| `V2X_UPSCALE_FACTOR` | `2` | Multiplicador de resolución |
| `V2X_UPSCALE_MODEL` | `realesr-animevideov3` | Modelo Real-ESRGAN |
| `V2X_INTERP_FACTOR` | `2` | Multiplicador de FPS |
| `V2X_INTERP_MODEL` | `rife-v4.6` | Modelo RIFE |
| `V2X_GPU` | `0` | Índice de dispositivo Vulkan |
| `V2X_GPU_WORKERS` | `4` | Workers paralelos por chunk |
| `CHUNK_MINUTES` | `15` | Minutos por chunk |
| `AUDIO_FILTER` | `highpass+anlmdn+loudnorm+alimiter` | Filtro ffmpeg de audio |

### Ejemplos

```bash
# Más workers para saturar la GPU
V2X_GPU_WORKERS=6 ./enhance.sh video.mp4

# AppImage custom
V2X_BIN=./Video2X-x86_64.AppImage ./enhance.sh video.mp4

# Solo upscale sin interpolación
V2X_INTERP_FACTOR=1 ./enhance.sh video.mp4

# Upscale ×4 (requiere más VRAM)
V2X_UPSCALE_FACTOR=4 ./enhance.sh video.mp4
```

## Qué hace el script

```
1. Split      →  ffmpeg divide el video en chunks de N min (-c copy, sin re-encode)
2. Upscale    →  Video2X: realesrgan -s N  (por chunk, en paralelo)
3. Interpolar →  Video2X: rife -m N        (por chunk, en paralelo)
4. Audio      →  ffmpeg aplica filtro de limpieza (en paralelo con pasos 2-3)
5. Concat     →  ffmpeg concatena chunks + muxa audio mejorado
```

Video2X ejecuta un procesador por invocación, así que upscale e interpolación
son dos pasadas separadas por chunk. Los intermedios de upscale se eliminan
automáticamente después de interpolar.

### Pipeline de audio

```
highpass=f=80              → elimina rumble <80Hz
anlmdn=s=7:p=0.002:m=15   → denoising no-local means
loudnorm=I=-16:TP=-1.5     → normalización EBU R128
alimiter=limit=0.95        → limiter para evitar clipping
```

## Hardware de referencia

| Componente | Modelo | Notas |
|---|---|---|
| CPU | Ryzen 9 9950X3D | ffmpeg encode/decode |
| GPU0 | RTX 5070 Ti 16GB | 4 workers ESRGAN |
| GPU1 | RTX 2060 6GB | 1 worker (PCIe ×4) |
| RAM | 128 GB DDR5 | — |

Con 5 workers totales en producción: **~8.5 FPS** para upscale ×2.

## Estructura

```
.
├── enhance.sh       ← el script (esto es todo)
├── videos/          ← videos de entrada
├── enhanced/        ← outputs
├── README.md
├── TODO.md
└── .gitignore
```

## Lección aprendida

Se intentó un pipeline custom en Python (~4500 LOC, 12 archivos, ~30h):
ESRGAN PyTorch, RIFE ncnn, streaming NVENC, scheduler CCD-aware, face-adaptive
blending, perfiles de audio, progreso resumible por chunk...

Al final, **Video2X con workers encolados hizo lo mismo con 0 líneas de código**.
La única adición útil fue el filtro de audio de ffmpeg (~1 línea).

> Si existe una herramienta open-source madura para tu problema, úsala primero.
