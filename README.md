# Video Enhancement Pipeline — FilosofiaNeurociencias

> Pipeline de mejora de video para una grabación Zoom.
> **Entrada**: 2240×1260 @ 25fps H.264 → **Salida**: 4480×2520 @ 50fps HEVC 4K
> **Hardware**: Ryzen 9 9950X3D · RTX 5070 Ti · RTX 2060 · 128 GB DDR5

---

## 1. Arquitectura del Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE v8 — 4 etapas                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────┐    ┌──────────────────┐    ┌──────────────────────────────┐  │
│  │  EXTRACT   │───►│  RIFE (1260p)    │───►│  ESRGAN (4K) dual-GPU batch │  │
│  │  ffmpeg    │    │  Vulkan GPU1     │    │  CUDA GPU0 (bs=8)           │  │
│  │  CPU       │    │  rife-ncnn-vk    │    │  CUDA GPU1 (bs=4)           │  │
│  └────────────┘    └──────────────────┘    └──────────────┬───────────────┘  │
│                                                           │                  │
│                                                   ┌───────▼──────────────┐   │
│                                                   │  NVENC (HEVC 4K)     │   │
│                                                   │  hevc_nvenc GPU0     │   │
│                                                   │  Streaming via pipe  │   │
│                                                   └──────────────────────┘   │
│                                                                              │
│  Audio: procesado en paralelo en CPU (afftdn + loudnorm + alimiter)          │
│  Intermedios: tmpfs /tmp (RAM, zero I/O disco)                               │
│  Progreso: resumible por chunk (progress.json)                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Flujo por chunk

1. **Extract** — ffmpeg decodifica un segmento de N segundos a frames RGB en RAM
2. **RIFE** — rife-ncnn-vulkan interpola 2× a resolución original (1260p), duplicando frames
3. **ESRGAN** — modelo de super-resolución sube de 1260p a 4K (dual-GPU con distribución dinámica)
4. **NVENC** — codifica HEVC 4K vía pipe stdin con reorder buffer para frames out-of-order

---

## 2. Hardware del Sistema

### CPU — AMD Ryzen 9 9950X3D

| Spec | Valor |
|---|---|
| Arquitectura | Zen 5 + 3D V-Cache |
| Cores / Threads | 16C / 32T |
| L3 Cache | 128 MB (dual CCD) |
| RAM | 128 GB DDR5 |

### GPU0 — NVIDIA RTX 5070 Ti (Blackwell)

| Recurso | Valor |
|---|---|
| VRAM | 16 GB GDDR7 |
| CUDA Cores | ~8960 |
| Compute Cap | 12.0 |
| Rol | ESRGAN principal + NVENC |

### GPU1 — NVIDIA RTX 2060 (Turing)

| Recurso | Valor |
|---|---|
| VRAM | 6 GB GDDR6 |
| CUDA Cores | 1920 |
| Compute Cap | 7.5 |
| Rol | RIFE Vulkan |
| ⚠️ | PCIe negociando x4 (límite HW) |

---

## 3. Software Stack

| Componente | Versión | Uso |
|---|---|---|
| Python | 3.x | Pipeline principal |
| PyTorch | 2.10.0 + CUDA 12.8 | ESRGAN inference (fp16) |
| Vulkan | 1.4.321 | RIFE interpolación |
| spandrel | 0.4.2 | Carga de modelos ESRGAN |
| FFmpeg | 7.1.1 | Extract, encode, audio |
| RIFE | rife-ncnn-vulkan 20221029 | Interpolación de frames |

---

## 4. Estructura de Archivos

```
FilosofiaNeurociencias/
├── enhance/                        ← Paquete principal del pipeline
│   ├── __init__.py
│   ├── config.py                   ← Configuración centralizada (env vars + defaults)
│   ├── esrgan.py                   ← Motor ESRGAN dual-GPU con batches dinámicos
│   ├── pipeline.py                 ← Pipeline 4 etapas (extract→rife→esrgan→nvenc)
│   ├── rife_backend.py             ← Backend abstracto RIFE (ncnn + torch)
│   ├── rife_torch_model.py         ← Implementación IFNet oficial para backend torch
│   ├── ffmpeg_utils.py             ← Utilidades ffmpeg (extract, audio)
│   ├── progress.py                 ← Tracking de progreso resumible por chunk
│   ├── profiles.py                 ← Perfiles visuales, audio, scheduler, RIFE backend
│   ├── models.py                   ← Registro de modelos con auto-download y SHA256
│   ├── visual_eval.py              ← Evaluación visual ROI + blending facial
│   ├── audio_profiles.py           ← Perfiles de audio y utilidades A/B
│   └── scheduler.py                ← Afinidad CPU (taskset/ionice/chrt) por rol
│
├── scripts/                        ← Scripts de ejecución y herramientas
│   ├── run.py                      ← CLI principal
│   ├── process_production.sh       ← Wrapper de producción con env vars de perfil
│   ├── benchmark_runner.py         ← Runner de benchmarks con instrumentación
│   ├── gate_acceptance.py          ← Gate de aceptación (throughput + calidad)
│   ├── audio_ab_bench.py           ← Benchmark A/B de perfiles de audio
│   └── test_components.py          ← Tests de componentes
│
├── enhanced/                       ← Output y artefactos del pipeline
│   ├── models/                     ← Pesos de modelos ESRGAN y RIFE
│   └── audio_bench/                ← Resultados de benchmark de audio
│
├── videos/                         ← Material fuente
│   ├── GMT20260320-130023_Recording_2240x1260.mp4   ← Video de entrada
│   ├── GMT20260320-130023_Recording.m4a             ← Audio separado
│   └── enhanced/                   ← Outputs y directorios de trabajo
│
├── README.md                       ← Este archivo
└── TODO.md                         ← Tareas pendientes de optimización
```

---

## 5. Video de Entrada

| Propiedad | Valor |
|---|---|
| Archivo | `GMT20260320-130023_Recording_2240x1260.mp4` |
| Video codec | H.264 (AVC) |
| Resolución | 2240 × 1260 |
| FPS | 25 |
| Audio | AAC 48kHz stereo |
| Duración | ~7.4 horas |

### Salida esperada

| Propiedad | Valor |
|---|---|
| Video codec | HEVC (hevc_nvenc) |
| Resolución | 4480 × 2520 |
| FPS | 50 |
| Video bitrate | 20 Mbps (CQ 19) |
| Audio | AAC 256kbps (procesado con perfil `natural`) |

---

## 6. Configuración

Toda la configuración se controla via variables de entorno en `enhance/config.py`.

### Variables principales

| Variable | Default | Descripción |
|---|---|---|
| `ENHANCE_CHUNK_SECONDS` | `15` | Duración de cada chunk |
| `ENHANCE_GPU0_BATCH` | `8` | Batch size GPU0 (5070 Ti) |
| `ENHANCE_GPU1_BATCH` | `4` | Batch size GPU1 (2060) |
| `ENHANCE_ESRGAN_GPUS` | `0` | GPUs para ESRGAN |
| `ENHANCE_RIFE_GPU` | `1` | GPU para RIFE Vulkan |
| `ENHANCE_RIFE_THREADS` | `1:8:4` | Threads RIFE (j:p:t) |
| `ENHANCE_PIPELINE_DEPTH` | `2` | Profundidad del pipeline |
| `ENHANCE_NVENC_PRESET` | `p1` | Preset NVENC |
| `ENHANCE_NVENC_CQ` | `19` | Calidad constante NVENC |
| `ENHANCE_NVENC_BITRATE` | `20M` | Bitrate objetivo |

### Variables de perfil

| Variable | Default | Descripción |
|---|---|---|
| `ENHANCE_VISUAL_PROFILE` | `None` | Perfil visual |
| `ENHANCE_AUDIO_PROFILE` | `None` | Perfil de audio |
| `ENHANCE_SCHEDULER_PROFILE` | `None` | Perfil de scheduler CPU |
| `ENHANCE_RIFE_BACKEND` | `None` | Backend RIFE |

---

## 7. Uso

### Ejecución básica

```bash
python3 scripts/run.py videos/GMT20260320-130023_Recording_2240x1260.mp4 \
  --outdir enhanced/
```

### Producción (con perfiles)

```bash
bash scripts/process_production.sh
```

Usa por defecto: `quality` visual, `natural` audio, `production` scheduler, `baseline` RIFE.

### Benchmark instrumentado

```bash
python3 scripts/benchmark_runner.py \
  --input videos/GMT20260320-130023_Recording_2240x1260.mp4 \
  --tag test_v1 \
  --slice-start 60 --slice-duration 60
```

### Gate de aceptación

```bash
python3 scripts/gate_acceptance.py enhanced/logs/bench_*/summary.json
```

---

## 8. Perfiles

### Visual

| Perfil | Modelo | Face Adaptive |
|---|---|---|
| `baseline` | anime_baseline | No |
| `quality` | realesrgan_x2plus | Sí |
| `production` | realesrgan_x2plus | Sí |
| `face_preserve` | realesrgan_x2plus | Sí |

### Audio

| Perfil | Filtros |
|---|---|
| `baseline` | afftdn + loudnorm + dynaudnorm |
| `natural` | afftdn + loudnorm + alimiter |
| `production` | highpass + anlmdn + dialoguenhance + loudnorm + alimiter |

### Scheduler

| Perfil | Estrategia |
|---|---|
| `baseline` | Sin afinidad |
| `production` | CCD split + ionice + chrt |

---

## 9. Rendimiento

### Mejor benchmark sostenido

| Métrica | Valor |
|---|---|
| throughput | 0.4198× realtime (300s) |
| effective_fps | 24.0 |
| GPU0 avg | 70.8% |
| GPU1 avg | 62.3% |
| CPU avg | 30.9% |

### Speedup acumulado v1→v8: 6.6×

---

## 10. Decisiones Técnicas Cerradas

| Decisión | Razón |
|---|---|
| CPU ESRGAN deshabilitado | Destruye rendimiento GPU (-29%) |
| RIFE a 1260p (no 4K) | 3.8× más rápido, sin pérdida de calidad |
| CPU_SHARE = 0 | Benchmarked: siempre peor con CPU worker |
| HEVC como codec | Balance calidad/velocidad |
| tmpfs para intermedios | Elimina I/O de disco |
| Perfil audio `natural` | Sin dynaudnorm, voz más natural |
