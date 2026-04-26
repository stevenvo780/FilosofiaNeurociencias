# FilosofiaNeurociencias

Guía operativa del proyecto para producir:

- video con **doble resolución + doble FPS**,
- audio final **bueno de verdad** (`*_audio_mejorado.wav`),
- subtítulos en **español** y **inglés**,
- y, si se quiere, un **MKV final** con todo muxeado.

## Estado actual del proyecto

### Qué quedó como resultado final válido

El audio que finalmente **sí quedó mejor** y se toma como referencia es:

- `*_audio_mejorado.wav`

En este repo, el flujo canónico para audio es el modo:

- `AUDIO_MODE=safe`

Ese flujo fue el que dio el mejor equilibrio entre limpieza, inteligibilidad y ausencia de artefactos.

### Qué NO tomar como referencia final

Estos flujos existieron como pruebas o experimentos, pero **no son la referencia final**:

- `*_audio_ia_deepfilter.wav`
- `*_audio_ia_multietapa.wav`
- scripts históricos de DeepFilter/Resemble para pruebas puntuales

No se borran de la historia del proyecto, pero **no son lo que conviene regenerar por defecto**.

## Salidas canónicas por charla

La estructura esperada por charla es esta:

```text
output/charlaN/
├── charlaN.mp4
├── charlaN_audio_mejorado.wav
├── charlaN.es.srt
├── charlaN.en.srt
└── charlaN_final.mkv
```

> Si algún archivo final fue borrado manualmente, se puede regenerar con los scripts del repo.

## Scripts que sí mandan aquí

### `enhance.sh`

Genera video con:

- resolución ×2
- FPS ×2
- conserva el audio original del input

Internamente usa:

- **Video2X**
- **Real-ESRGAN** para upscale
- **RIFE** para interpolación

### `scripts/process_charlas_gpu.sh`

Es el script recomendado para las charlas segmentadas (`output/charla*.mp4`).

Hace esto:

1. mejora el audio,
2. genera subtítulos,
3. traduce cuando hace falta,
4. crea `*_final.mkv`.

El modo recomendado es:

- `AUDIO_MODE=safe`

Ese modo usa el filtro conservador:

```text
highpass=f=70,lowpass=f=12000,afftdn=nr=6:nf=-35,loudnorm=I=-24:LRA=11:TP=-3:linear=true
```

### Scripts históricos o experimentales

Estos sirven como referencia técnica, pero no como flujo canónico por defecto:

- `scripts/run_audio_multistage_cuda_docker.sh`
- `scripts/batch_audio_multistage_cuda.sh`
- `scripts/run_deepfilter_cuda_docker.sh`
- `audio_df_parallel.sh`
- `enhance_audio_after.sh`
- `scripts/enhance_audio_resemble.py`
- `scripts/denoise_audio_resemble.py`

## Requisitos

### Para video x2 / 50 fps

- `ffmpeg`
- `ffprobe`
- `Video2X-x86_64.AppImage` o `video2x`
- GPU con soporte Vulkan

### Para audio + subtítulos

- `ffmpeg`
- Docker
- runtime de NVIDIA para Docker
- GPU CUDA para `whisper.cpp`

El script `scripts/process_charlas_gpu.sh` bootstrappea automáticamente `whisper.cpp` mediante:

- `scripts/bootstrap_whispercpp_cuda.sh`

## Flujo recomendado de trabajo

### Opción A — Sólo sacar audio mejorado + subtítulos + MKV final por charla

Si ya tienes `output/charla1.mp4`, `output/charla2.mp4`, etc., este es el flujo más útil y más estable:

```bash
DELIVER_DIR="$PWD/output" \
AUDIO_MODE=safe \
./scripts/process_charlas_gpu.sh
```

Esto genera por charla:

- `*_audio_mejorado.wav`
- `*.es.srt`
- `*.en.srt`
- `*_final.mkv`

Si sólo quieres una charla concreta:

```bash
INPUT_PATTERN="$PWD/output/charla4.mp4" \
DELIVER_DIR="$PWD/output" \
AUDIO_MODE=safe \
./scripts/process_charlas_gpu.sh
```

### Opción B — Hacer primero el video con doble calidad

Para producir un video con resolución ×2 y FPS ×2:

```bash
./enhance.sh input.mp4 output_4K50.mkv
```

Ejemplo real del repo:

```bash
./enhance.sh videos/GMT20260320-130023_Recording_2240x1260.mp4 output/GMT20260320-130023_Recording_2240x1260_4K50.mkv
```

Qué hace `enhance.sh`:

1. divide el video en chunks,
2. hace upscale con Real-ESRGAN,
3. interpola cuadros con RIFE,
4. concatena el resultado,
5. conserva el audio original del input.

### Opción C — Combinar ambas cosas: video x2 + audio mejorado + subtítulos

Ese resultado final se produce en dos etapas:

1. generar el video mejorado visualmente con `enhance.sh`,
2. generar `*_audio_mejorado.wav` y subtítulos,
3. hacer el mux final.

Ejemplo de mux manual:

```bash
ffmpeg -y \
	-i output/charla4/charla4_4K50.mkv \
	-i output/charla4/charla4_audio_mejorado.wav \
	-i output/charla4/charla4.es.srt \
	-i output/charla4/charla4.en.srt \
	-map 0:v:0 -map 1:a:0 -map 2:0 -map 3:0 \
	-c:v copy \
	-c:a aac -b:a 192k \
	-c:s srt \
	-metadata:s:s:0 language=spa \
	-metadata:s:s:0 title="Español" \
	-metadata:s:s:1 language=eng \
	-metadata:s:s:1 title="English" \
	-shortest \
	output/charla4/charla4_4K50_final.mkv
```

## Variables útiles

### `enhance.sh`

| Variable | Default | Uso |
|---|---|---|
| `V2X_BIN` | `./Video2X-x86_64.AppImage` | Binario de Video2X |
| `V2X_UPSCALE_FACTOR` | `2` | Multiplicador de resolución |
| `V2X_INTERP_FACTOR` | `2` | Multiplicador de FPS |
| `GPU0_WORKERS` | `3` | Workers en GPU rápida |
| `GPU1_WORKERS` | `1` | Workers en GPU secundaria |
| `CHUNK_MINUTES` | `15` | Duración de chunk |

### `scripts/process_charlas_gpu.sh`

| Variable | Default | Uso |
|---|---|---|
| `INPUT_PATTERN` | `output/charla*.mp4` | Charlas de entrada |
| `DELIVER_DIR` | `output` | Carpeta de salida |
| `AUDIO_MODE` | `safe` | **Modo recomendado** |
| `FORCE_REGEN_AUDIO` | `0` | Rehacer audio |
| `FORCE_REGEN_ASR` | `0` | Rehacer transcripción |
| `FORCE_REMUX_FINAL` | `0` | Rehacer sólo el MKV final |

Si quieres regenerar sólo el MKV final sin rehacer audio ni subtítulos:

```bash
DELIVER_DIR="$PWD/output" \
FORCE_REMUX_FINAL=1 \
./scripts/process_charlas_gpu.sh
```

## Decisión operativa para el futuro

Si mañana hay que volver a producir todo, la recomendación es:

1. **Video**: usar `enhance.sh`.
2. **Audio/Subtítulos**: usar `scripts/process_charlas_gpu.sh` con `AUDIO_MODE=safe`.
3. **Resultado de audio a conservar**: `*_audio_mejorado.wav`.
4. **No volver a tomar como referencia final** los audios IA multi-etapa salvo que se quiera reabrir la investigación.

## Historial y contexto

Para un resumen de qué se probó, qué salió mal y qué terminó quedándose, ver:

- `PROGRESO.md`
