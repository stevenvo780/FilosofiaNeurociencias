# Video Enhancement Pipeline — FilosofiaNeurociencias

> Pipeline de mejora de video para grabaciones Zoom.  
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
│  Audio: procesado en paralelo en CPU (afftdn + loudnorm + dynaudnorm)        │
│  Intermedios: tmpfs /tmp (RAM, zero I/O disco)                               │
│  Progreso: resumible por chunk (progress.json)                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Flujo por chunk

1. **Extract** — ffmpeg decodifica un segmento de N segundos a frames RGB en RAM (o PNGs en tmpfs para RIFE)
2. **RIFE** — rife-ncnn-vulkan interpola 2× a resolución original (1260p), produciendo el doble de frames
3. **ESRGAN** — modelo de super-resolución sube de 1260p a 4K. Dual-GPU con distribución dinámica de batches
4. **NVENC** — codifica HEVC 4K vía pipe stdin. Reorder buffer para manejar frames out-of-order de dual-GPU

El pipeline soporta streaming completo: RIFE→ESRGAN→NVENC se ejecutan como pipeline encadenado sin materializar todos los frames 4K en RAM.

---

## 2. Hardware del Sistema

### CPU — AMD Ryzen 9 9950X3D

| Spec | Valor |
|---|---|
| Arquitectura | Zen 5 + 3D V-Cache |
| Cores / Threads | 16C / 32T |
| Frecuencia Max | 5803 MHz |
| L3 Cache | 128 MB (2 grupos CCD, 3D V-Cache) |
| RAM | 128 GB DDR5 |
| Swap | 191 GB (zram 64 GB + md127 128 GB) |

### GPU0 — NVIDIA RTX 5070 Ti (Blackwell)

| Recurso | Valor |
|---|---|
| CUDA Cores | ~8960 |
| Tensor Cores | 280 (5ª gen) |
| NVENC | 2 sesiones |
| NVDEC | 1 |
| VRAM | 16 GB GDDR7, ~504 GB/s |
| PCIe | Gen 5 × 16 (~64 GB/s) |
| Compute Cap | 12.0 |
| SM Count | 70 |

### GPU1 — NVIDIA RTX 2060 (Turing)

| Recurso | Valor |
|---|---|
| CUDA Cores | 1920 |
| Tensor Cores | 240 (1ª gen) |
| NVENC | 1 sesión |
| NVDEC | 1 |
| VRAM | 6 GB GDDR6 |
| PCIe | Gen 3 max x16, **negociando x4** ⚠️ |
| Compute Cap | 7.5 |
| Display | Monitor conectado aquí |

### Almacenamiento

| Dispositivo | Tipo | Montaje | Uso |
|---|---|---|---|
| NVMe RAID0 (×3) | NVMe | `/home` (md127) | Código, videos, output |
| tmpfs | RAM | `/tmp` (62 GB) | Frames intermedios — zero I/O disco |

---

## 3. Software Stack

| Componente | Versión | Uso |
|---|---|---|
| Python | 3.x | Pipeline principal |
| PyTorch | 2.10.0 + CUDA 12.8 | ESRGAN inference (fp16) |
| cuDNN | 9.1.002 | Convoluciones optimizadas |
| Vulkan | 1.4.321 | RIFE interpolación |
| spandrel | 0.4.2 | Carga de modelos ESRGAN |
| OpenCV | 4.13.0 | Lectura/escritura de imágenes |
| NumPy | 2.4.3 | Manipulación de arrays |
| FFmpeg | 7.1.1 | Extract, encode, audio |
| RIFE | rife-ncnn-vulkan 20221029 | Interpolación de frames via Vulkan |

---

## 4. Estructura de Archivos

```
FilosofiaNeurociencias/
├── enhance/                        ← Paquete principal del pipeline
│   ├── __init__.py
│   ├── config.py                   ← Configuración centralizada (env vars + defaults)
│   ├── esrgan.py                   ← Motor ESRGAN dual-GPU con batches dinámicos
│   ├── pipeline.py                 ← Pipeline 4 etapas (extract→rife→esrgan→nvenc)
│   ├── rife.py                     ← Wrapper de rife-ncnn-vulkan
│   ├── ffmpeg_utils.py             ← Utilidades ffmpeg (extract, audio)
│   ├── progress.py                 ← Tracking de progreso resumible por chunk
│   ├── profiles.py                 ← Perfiles visuales, audio, scheduler, RIFE backend
│   ├── models.py                   ← Registro de modelos con auto-download y SHA256
│   ├── visual_eval.py              ← Evaluación visual ROI + blending híbrido/facial
│   ├── audio_profiles.py           ← Perfiles de audio y utilidades A/B
│   ├── scheduler.py                ← Afinidad CPU (taskset/ionice/chrt) por rol
│   ├── rife_backend.py             ← Backend abstracto RIFE (ncnn + torch stub)
│   ├── scripts/
│   │   ├── benchmark_isolated.py   ← Benchmark aislado de componentes
│   │   ├── run_test.sh … run_test3.sh
│   │   └── test_monitor.sh
│   └── logs/                       ← Logs históricos de benchmarks v1–v8
│
├── scripts/                        ← Scripts de ejecución y herramientas
│   ├── run.py                      ← CLI principal (9 args de perfil)
│   ├── benchmark_runner.py         ← Runner de benchmarks con instrumentación completa
│   ├── gate_acceptance.py          ← Gate de aceptación (throughput + calidad)
│   ├── audio_ab_bench.py           ← Benchmark A/B de perfiles de audio
│   ├── bench_batch.py              ← Benchmark batch
│   ├── process_production.sh       ← Wrapper de producción con env vars de perfil
│   ├── test_components.py          ← Tests de componentes
│   └── test_cpu_esrgan.py          ← Tests de ESRGAN en CPU
│
├── enhanced/                       ← Output procesado
│   ├── logs/                       ← Logs de corridas de producción y benchmarks
│   │   ├── bench_baseline_v1_*/    ← Benchmark instrumentado con gpu.csv, mpstat, etc.
│   │   └── usage_bench_*/          ← Corrida instrumentada de uso
│   ├── work_*/                     ← Directorios de trabajo por video
│   │   ├── chunk_NNNN/output.mp4   ← Chunks procesados
│   │   ├── progress.json           ← Estado resumible
│   │   └── concat.txt              ← Lista para concatenar
│   └── *.mp4                       ← Videos finales
│
├── videos/                         ← Material fuente
│   ├── *.mp4                       ← Videos de entrada
│   └── *.m4a                       ← Audio separado
│
├── backups/                        ← Backups de archivos modificados
├── README.md                       ← Este archivo
├── TODO.md                         ← Tareas pendientes de optimización
├── OPTIMIZATION_REPORT.md          ← Reporte original de auditoría
└── SECOND_VIDEO_RESTRUCTURE_PLAN.md ← Plan maestro del segundo video
```

---

## 5. Configuración

Toda la configuración se controla via variables de entorno en `enhance/config.py`.

### Variables principales

| Variable | Default | Descripción |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1` | GPUs visibles |
| `ENHANCE_CHUNK_SECONDS` | `15` | Duración de cada chunk en segundos |
| `ENHANCE_GPU0_BATCH` | `8` | Batch size para GPU0 (5070 Ti) |
| `ENHANCE_GPU1_BATCH` | `4` | Batch size para GPU1 (2060) |
| `ENHANCE_ESRGAN_GPUS` | `0` | GPUs para ESRGAN (separadas por coma) |
| `ENHANCE_RIFE_GPU` | `1` | GPU para RIFE Vulkan |
| `ENHANCE_RIFE_THREADS` | `1:8:4` | Threads RIFE (j:p:t) |
| `ENHANCE_RIFE_STREAM_WINDOW` | `192` | Ventana de streaming RIFE→ESRGAN |
| `ENHANCE_RIFE_MIN_WINDOW` | `64` | Mínimo de frames por ventana |
| `ENHANCE_PIPELINE_DEPTH` | `2` | Profundidad del pipeline de chunks |
| `ENHANCE_MAX_EXTRACT_BYTES_IN_FLIGHT` | `6 GiB` | Budget de extracción en vuelo |
| `ENHANCE_MAX_RIFE_READY_BYTES` | `3 GiB` | Budget de frames RIFE listos |
| `ENHANCE_NVENC_PRESET` | `p1` | Preset NVENC (p1=fastest) |
| `ENHANCE_NVENC_CQ` | `19` | Calidad constante NVENC |
| `ENHANCE_NVENC_BITRATE` | `20M` | Bitrate objetivo |

### Variables de perfil

| Variable | Default | Descripción |
|---|---|---|
| `ENHANCE_VISUAL_PROFILE` | `None` | Perfil visual (baseline, quality, etc.) |
| `ENHANCE_AUDIO_PROFILE` | `None` | Perfil de audio |
| `ENHANCE_SCHEDULER_PROFILE` | `None` | Perfil de scheduler CPU |
| `ENHANCE_RIFE_BACKEND` | `None` | Backend RIFE (ncnn, torch) |

### Variables PyTorch

| Variable | Default | Descripción |
|---|---|---|
| `ENHANCE_TORCH_COMPILE` | `0` | Activar `torch.compile()` |
| `ENHANCE_CUDNN_BENCHMARK` | `0` | cuDNN benchmark mode |
| `ENHANCE_CUDA_MATMUL_ALLOW_TF32` | `0` | Permitir TF32 en matmul |
| `ENHANCE_CUDNN_ALLOW_TF32` | `1` | Permitir TF32 en cuDNN |
| `ENHANCE_ESRGAN_PINNED_STAGING` | `0` | Staging con pinned memory experimental |

---

## 6. Uso

### Ejecución básica

```bash
cd /home/stev/Descargas/FilosofiaNeurociencias
python3 scripts/run.py \
  --input videos/GMT20260320-130023_Recording_2240x1260.mp4 \
  --output enhanced/output_4k_50fps.mp4 \
  --rife
```

### Con perfiles

```bash
python3 scripts/run.py \
  --input videos/input.mp4 \
  --output enhanced/output.mp4 \
  --rife \
  --visual-profile baseline \
  --audio-profile baseline \
  --scheduler-profile baseline \
  --rife-backend ncnn
```

### Benchmark instrumentado

```bash
python3 scripts/benchmark_runner.py \
  --input videos/input.mp4 \
  --tag test_v1 \
  --slice-start 60 --slice-duration 60 \
  --visual-profile baseline \
  --audio-profile baseline \
  --scheduler-profile baseline
```

### Gate de aceptación

```bash
python3 scripts/gate_acceptance.py \
  enhanced/logs/bench_baseline_v1_*/summary.json
```

### Audio A/B benchmark

```bash
python3 scripts/audio_ab_bench.py \
  --input videos/audio.m4a \
  --start 60 --duration 30
```

### Producción

```bash
bash scripts/process_production.sh
```

---

## 7. Sistema de Perfiles

El sistema de perfiles (`enhance/profiles.py`) permite configurar cuatro ejes de forma declarativa:

### Visual Profiles

| Perfil | Modelo | Downscale | Hybrid Detail | Face Adaptive |
|---|---|---|---|---|
| `baseline` | anime_baseline | 0.5× | 0.0 | No |
| `fast` | anime_baseline | 0.5× | 0.0 | No |
| `real_x2` / `real_x2plus` | realesrgan_x2plus | 1.0× | 0.0 | No |
| `real_x4plus` | realesrgan_x4plus | 0.5× | 0.08 | No |
| `hybrid_detail` | realesrgan_x4plus | 0.5× | 0.12 | No |
| `face_adaptive` | realesrgan_x2plus | 1.0× | 0.2 | Sí |
| `quality` | realesrgan_x2plus | 1.0× | 0.15 | Sí |
| `face_preserve` | realesrgan_x2plus | 1.0× | 0.25 | Sí |
| `production` | realesrgan_x2plus | 1.0× | 0.15 | Sí |

### Audio Profiles

| Perfil | Filtros |
|---|---|
| `baseline` | afftdn + loudnorm + dynaudnorm |
| `conservative` | anlmdn + loudnorm + alimiter |
| `voice` | anlmdn + dialoguenhance + speechnorm + alimiter |
| `natural` | afftdn + loudnorm + alimiter |
| `voice_natural` / `lecture_natural` / `production` | highpass + anlmdn + dialoguenhance + loudnorm + alimiter |

### Scheduler Profiles

| Perfil | Estrategia |
|---|---|
| `baseline` | Sin afinidad |
| `split_l3_a` | CCD0 para extract/audio, CCD1 para coordinación |
| `split_l3_b` | Inverso |
| `production` | CCD split + ionice + chrt para producción |

### RIFE Backends

| Backend | Implementación |
|---|---|
| `ncnn` | rife-ncnn-vulkan (producción) |
| `torch` | PyTorch (stub, futuro) |

---

## 8. Modelo de Registro

`enhance/models.py` gestiona modelos con:

- Auto-descarga desde URLs pre-configuradas
- Verificación SHA256 de integridad
- Resolución por nombre clave
- Directorio configurable (`ENHANCE_MODELS_DIR`)

---

## 9. Observabilidad

### Métricas por chunk (chunk_metrics.jsonl)

Cada chunk emite un registro JSON con:

- `extract_seconds`, `extract_fps`, `extract_bytes`
- `rife_seconds`, `rife_fps`, `rife_frames`
- `readback_seconds`, `readback_fps`
- `esrgan_fill_seconds`, `esrgan_h2d_seconds`, `esrgan_infer_seconds`, `esrgan_d2h_seconds`
- `esrgan_writer_wait_seconds`
- `encode_seconds`, `effective_fps`
- `window_avg_frames`, `window_max_frames`
- `extract_peak_bytes`, `rife_ready_peak_bytes`, `nvenc_peak_frames`

### Instrumentación del benchmark runner

El runner (`scripts/benchmark_runner.py`) captura automáticamente:

- `nvidia-smi dmon` → `gpu.csv`
- `mpstat -P ALL 1` → `mpstat.log`
- `iostat -x 1` → `iostat.log`
- `free -h` cada 2s → `memory.log`
- `hardware.json` — specs del sistema
- `summary.json` — resultado global

### Gate de aceptación

`scripts/gate_acceptance.py` valida:

- `throughput_ratio >= 0.40`
- `effective_fps >= 20.0`
- Ausencia de zombies
- Output válido
- PCIe warning si GPU1 negocia x4

---

## 10. Benchmarks Históricos

### ESRGAN aislado (200 frames, 2240×1260 → 4480×2520)

| Config | FPS |
|---|---|
| GPU0 sola | 42.2 |
| GPU1 sola | 13.0 |
| **GPU0 + GPU1** | **53.6** |
| GPU0 + GPU1 + CPU | 38.2 ❌ (CPU destruye rendimiento GPU) |

### Pipeline completo (evolución v1→v8)

| Versión | ESRGAN FPS | Total chunk | Cambio clave |
|---|---|---|---|
| v1 | 17.2 | ~493s | Baseline original |
| v3 | 22.7 | ~75s | RIFE primero a 1260p |
| v8 | **35.3** | **~75s** | `.contiguous()` en GPU |

**Speedup total v1→v8: 6.6×**

### Baseline productivo actual (bench_baseline_v1)

| Métrica | Valor |
|---|---|
| throughput_ratio | **0.404×** realtime |
| effective_fps | **21.16** |
| chunk promedio | **35.46s** |
| RIFE fps | 21.1 |
| ESRGAN fps | 35.3 |
| NVENC fps | 48.2 |

---

## 11. Decisiones Técnicas Cerradas

| Decisión | Razón |
|---|---|
| CPU ESRGAN deshabilitado | Destruye rendimiento GPU (-29%) |
| RIFE a 1260p (no 4K) | 3.8× más rápido, sin pérdida de calidad |
| CPU_SHARE = 0 | Benchmarked: siempre peor con CPU worker |
| HEVC como codec | Buen balance calidad/velocidad |
| tmpfs para intermedios | Elimina I/O de disco |
| RT Cores fuera de scope | No aplican a super-resolución |

---

## 12. Utilización Real del Hardware (baseline_v1)

| Recurso | Uso promedio | Pico | Estado |
|---|---|---|---|
| GPU0 compute | 46.6% | 100% | Subalimentada — 49% del tiempo <30% |
| GPU0 NVENC | ~4% | 15% | Activo cuando toca, no es cuello |
| GPU1 compute | 59.1% | 100% | Solo Vulkan para RIFE |
| GPU1 NVENC | 0% | 0% | Completamente idle |
| CPU global | 10.2% | 61% | Masivamente subutilizado |
| RAM | 44–54 GB / 123 GB | 54 GB | Holgada |
| I/O disco | <1% | 6% | No es cuello |

### Cuello de botella #1: Transfer D2H en ESRGAN

```
esrgan_d2h_seconds:    23.87s promedio (85% del tiempo ESRGAN)
esrgan_infer_seconds:   0.11s promedio (compute real — negligible)
esrgan_h2d_seconds:     1.74s promedio
esrgan_fill_seconds:    2.31s promedio
idle_between_stages:    ~7.5s por chunk
```

La GPU procesa el batch en **0.11s** pero la transferencia de vuelta a CPU toma **23.87s**. La GPU está idle esperando a PCIe durante ese tiempo.

---

## 13. Entrada / Salida

### Entrada

| Propiedad | Valor |
|---|---|
| Video codec | H.264 (AVC) |
| Resolución | 2240 × 1260 |
| FPS | 25 |
| Video bitrate | 354 kbps |
| Audio | AAC 48kHz stereo 75kbps |
| Duración | ~7.4 horas |

### Salida

| Propiedad | Valor |
|---|---|
| Video codec | HEVC (hevc_nvenc) |
| Resolución | 4480 × 2520 |
| FPS | 50 |
| Video bitrate | 20 Mbps (CQ 19) |
| Audio | AAC 256kbps (procesado) |
