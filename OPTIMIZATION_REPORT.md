# Video Enhancement Pipeline — Reporte Completo de Optimización

> **Fecha**: 2026-03-30  
> **Sistema**: Ryzen 9 9950X3D / RTX 5070 Ti / RTX 2060 / 128GB DDR5  
> **Video**: Grabación Zoom 7.4h, 2240×1260@25fps → 4480×2520@50fps

---

## 1. Hardware del Sistema

### 1.1 CPU — AMD Ryzen 9 9950X3D
| Spec | Valor |
|---|---|
| Arquitectura | Zen 5 + 3D V-Cache |
| Cores / Threads | 16C / 32T |
| Frecuencia Max | 5803 MHz |
| L3 Cache | 128 MB (3D V-Cache) |
| RAM | 128 GB DDR5 |
| Swap | 191 GB |

### 1.2 GPU0 — NVIDIA RTX 5070 Ti (Blackwell)
| Chip | Cantidad | Descripción |
|---|---|---|
| **CUDA Cores** | ~8960 | Procesamiento paralelo general |
| **Tensor Cores** | 280 (5th gen) | Aceleración de matrices (fp16/tf32/int8) |
| **RT Cores** | 70 (4th gen) | Ray tracing en hardware |
| **NVENC** | 2 sessions | Codificación de video por hardware (H.264/HEVC/AV1) |
| **NVDEC** | 1 | Decodificación de video por hardware |
| **VRAM** | 16 GB GDDR7 | 14001 MHz, ~504 GB/s bandwidth |
| **PCIe** | Gen 5 × 16 | ~64 GB/s bidireccional |
| **TDP** | 350W | |
| **Compute Cap** | 12.0 | Blackwell |
| **SM Count** | 70 | Stream Multiprocessors |

### 1.3 GPU1 — NVIDIA RTX 2060 (Turing)
| Chip | Cantidad | Descripción |
|---|---|---|
| **CUDA Cores** | 1920 | Procesamiento paralelo general |
| **Tensor Cores** | 240 (1st gen) | Aceleración de matrices (fp16) |
| **RT Cores** | 30 (1st gen) | Ray tracing en hardware |
| **NVENC** | 1 session | Codificación de video por hardware |
| **NVDEC** | 1 | Decodificación de video por hardware |
| **VRAM** | 6 GB GDDR6 | 7001 MHz |
| **PCIe** | Gen 3, Max x16, **Negociando x4** | ~4 GB/s (debería ser 16 GB/s) |
| **TDP** | 172.5W | |
| **Compute Cap** | 7.5 | Turing |
| **SM Count** | 30 | |
| **Display** | Monitor conectado aquí | (Active + Attached) |

### 1.4 Almacenamiento
| Dispositivo | Tamaño | Tipo | Montaje |
|---|---|---|---|
| **tmpfs** | 62 GB (RAM) | tmpfs | `/tmp` — frames intermedios, zero I/O disco |
| sda | 931 GB | HDD NTFS | /media/stev/Documentos |
| sdb | 931 GB | HDD NTFS | /media/stev/Data2 |
| sdc | 1.8 TB | HDD NTFS | /media/stev/Juegos |
| sdd | 596 GB | HDD NTFS | /media/stev/External |

### 1.5 Media de Entrada
| Propiedad | Valor |
|---|---|
| Video codec | H.264 (AVC) |
| Resolución | 2240 × 1260 |
| FPS | 25 |
| Video bitrate | 354 kbps |
| Audio codec | AAC |
| Audio sample rate | 48000 Hz |
| Audio channels | 2 (stereo) |
| Audio bitrate | 75 kbps (baja calidad) |
| Duración | ~7.4 horas |
| Audio separado | GMT20260320-130023_Recording.m4a |

---

## 2. Software Stack

| Componente | Versión | Uso |
|---|---|---|
| PyTorch | 2.10.0 + CUDA 12.8 | ESRGAN inference |
| CUDA Toolkit | 12.4 | Compilación/runtime |
| cuDNN | 9.1.002 | Convoluciones optimizadas |
| Vulkan | 1.4.321 | RIFE interpolación |
| spandrel | 0.4.2 | Carga de modelos ESRGAN |
| OpenCV | 4.13.0 | Lectura/escritura imágenes |
| NumPy | 2.4.3 | Arrays |
| FFmpeg | 7.1.1 | Extract/encode video, audio |
| RIFE | rife-ncnn-vulkan 20221029 | Frame interpolation via Vulkan |

### FFmpeg Capacidades Hardware
| Capacidad | Codecs Disponibles | ¿Se usa? |
|---|---|---|
| **NVENC** (encode) | h264, hevc, av1 | ✅ hevc_nvenc (solo GPU0) |
| **NVDEC** (decode) | h264, hevc, av1, vp9, mpeg | ❌ No se usa |
| **Vulkan** | Disponible | ✅ Vía RIFE |
| **OpenCL** | Disponible | ❌ No se usa |

---

## 3. Pipeline Actual (v8)

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE v8 ACTUAL                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐ │
│  │ EXTRACT  │───►│ RIFE (1260p) │───►│ ESRGAN (4K) batch │ │
│  │ ffmpeg   │    │ Vulkan GPU0  │    │ CUDA GPU0+GPU1    │ │
│  │ CPU      │    │ 21.1 fps     │    │ 35.3 fps          │ │
│  │ 1.8s     │    │ 35.9s        │    │ 21.3s             │ │
│  └──────────┘    └──────────────┘    └─────────┬──────────┘ │
│                                                │            │
│                                        ┌───────▼──────────┐ │
│                                        │ NVENC (HEVC 4K)  │ │
│                                        │ ASIC GPU0 only   │ │
│                                        │ 48.2 fps         │ │
│                                        │ 15.6s            │ │
│                                        └──────────────────┘ │
│                                                             │
│  Total: ~75s por 15s de video (0.19x realtime)              │
│  Audio: ❌ NO SE PROCESA                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Auditoría de Utilización — Chip por Chip

### 4.1 RTX 5070 Ti — 70 SMs, Blackwell

| Chip/Unidad | ¿Se usa? | Detalles | Potencial no explotado |
|---|---|---|---|
| **CUDA Cores** | ✅ Sí | ESRGAN fp16 inference, batch=8 | Batch podría ser 16 (solo usa 5GB de 16GB) |
| **Tensor Cores** (280) | ⚠️ **PARCIAL** | PyTorch fp16 los usa implícitamente para matmul, PERO: `allow_tf32=False`, `cudnn.benchmark=False` | **Activar TF32 + cudnn.benchmark = +15-25% FPS** |
| **RT Cores** (70) | ❌ No | Ray tracing no aplica a super-resolución ni interpolación | N/A — estos cores literalmente no sirven para este flujo |
| **NVENC #1** | ✅ Sí | Codifica HEVC 4K a 48 fps | Ya al límite del ASIC |
| **NVENC #2** | ❌ No | La 5070Ti tiene 2 encoders, solo usamos 1 | **Podría encode paralelo** |
| **NVDEC** | ❌ No | ffmpeg decodifica H.264 en CPU | **Decode HW = CPU libre para audio** |
| **VRAM** | ⚠️ 5/16 GB | batch=8 frames de 1260p → ~5GB | **batch=16 usaría ~10GB** |
| **Mem BW** | ⚠️ Parcial | 504 GB/s disponible | `.contiguous()` en GPU aprovecha esto |

### 4.2 RTX 2060 — 30 SMs, Turing

| Chip/Unidad | ¿Se usa? | Detalles | Potencial no explotado |
|---|---|---|---|
| **CUDA Cores** | ✅ Sí | ESRGAN fp16 inference, batch=4 | Al 96-98% durante ESRGAN |
| **Tensor Cores** (240) | ⚠️ **PARCIAL** | Turing gen1, fp16 implícito | **cudnn.benchmark mejoraría ~10%** |
| **RT Cores** (30) | ❌ No | N/A | N/A |
| **NVENC** | ❌ **No** | Completamente idle | **Encode paralelo para chunks alternos** |
| **NVDEC** | ❌ **No** | No se usa | Menor impacto |
| **VRAM** | ⚠️ 1.5/6 GB | batch=4 → ~1.5GB | **batch=8 usaría ~3GB** |
| **PCIe** | ⚠️ x4 actual | Negocia x4 de x16 max | **Problema físico (BIOS/slot/cable)** |

### 4.3 Ryzen 9 9950X3D — 16C/32T

| Recurso | ¿Se usa? | Cuánto | Potencial |
|---|---|---|---|
| **Cores** | ⚠️ ~4-8T | ffmpeg extract, numpy stack, PNG I/O | **Audio enhance podría usar 16+ threads** |
| **128MB L3 V-Cache** | ❌ Subutilizado | Ideal para procesamiento de señal | **FFT-based audio denoising** |
| **128GB RAM** | ⚠️ ~10GB | Frames en tmpfs | **Más chunks en RAM simultáneos** |

### 4.4 Configuración PyTorch Actual

| Setting | Valor actual | Óptimo | Impacto |
|---|---|---|---|
| `allow_tf32 (matmul)` | **False** ❌ | True | +10-15% FPS en matmul |
| `allow_tf32 (cudnn)` | True ✅ | True | Ya correcto |
| `cudnn.benchmark` | **False** ❌ | True | +10-20% FPS en conv2d |
| `cudnn.enabled` | True ✅ | True | Ya correcto |
| `torch.compile()` | **No usado** ❌ | `reduce-overhead` | +10-20% FPS (kernel fusion) |
| `CUDA_VISIBLE_DEVICES` | `0,1` ✅ | `0,1` | Correcto |

---

## 5. Evolución de Rendimiento (Benchmarks Verificados)

### 5.1 ESRGAN Aislado (200 frames, 2240×1260 → 4480×2520)

| Config | FPS | GPU0 avg | GPU1 avg |
|---|---|---|---|
| GPU0 sola | 42.2 | 85% | — |
| GPU1 sola | 13.0 | — | 92% |
| **GPU0 + GPU1** | **53.6** | 58% | 72% |
| GPU0 + GPU1 + CPU | 38.2 ❌ | 62% | 68% |

> **Lección**: CPU worker destruye rendimiento GPU (-29%). Deshabilitado permanentemente.

### 5.2 RIFE Vulkan (20 frames benchmark)

| Config | Resolución | Tiempo | FPS | Speedup |
|---|---|---|---|---|
| Dual GPU (0,1) | 4480×2520 | 27.2s | 0.74 | 1× |
| Solo GPU0 | 4480×2520 | 25.6s | 0.78 | 1.06× |
| **Solo GPU0** | **2240×1260** | **7.2s** | **2.78** | **3.8×** |

> **Lección**: RIFE a resolución original es 3.8× más rápido. La 2060 aporta solo 1.5s de 27s en RIFE.

### 5.3 Pipeline Completo (15s de video, 375 input → 750 output frames)

| Versión | ESRGAN FPS | RIFE | NVENC | Total | Cambio clave |
|---|---|---|---|---|---|
| **v1** (original) | 17.2 | 435s (4K) | — | ~493s | Baseline |
| **v3** (pipeline reorder) | 22.7 | 34.5s (1260p) | 40s (p6) | ~75s | RIFE primero a 1260p |
| **v4** (batch mode) | 23.8 | 34.5s | 12.7s (p1) | ~79s | Sin streaming backpressure |
| **v5** (bulk np.stack) | 36.5 | 34.5s | 21.0s ❌ | ~91s | np.stack pero frames no contiguas |
| **v7** (+ .copy()) | 21.1 ❌ | 35.5s | 48.1s | ~87s | .copy() arregla NVENC pero mata ESRGAN |
| **v8** (.contiguous() GPU) | **35.3** | **35.9s** | **48.2** | **~75s** | **Contiguous en GPU = ambos rápidos** |

> **Speedup total v1→v8: 6.6×**

### 5.4 GPU Utilización por Fase (v8 verificado)

**Fase RIFE (SEC 7-38, 32s):**
```
GPU0: 18-31%  (Vulkan compute, nvidia-smi reporta bajo)
GPU1: 19%     (idle, display compositor)
```

**Fase ESRGAN (SEC 41-59, 19s):**
```
GPU0: 91→100→100→100→79→32→88→100→72→50→85→100→100→62→37→69→100→100→100
GPU1: 13→89→96→96→96→97→97→98→97→98→98→97→97→98→98→98→96→97→98
```
- **GPU0**: Picos de **100%** con oscilaciones (procesa batches tan rápido que nvidia-smi no captura todos)
- **GPU1**: **96-98% sostenido** ✅

**Fase NVENC (SEC 60-83, 16s):**
```
GPU0: 4-12%   (NVENC usa ASIC dedicado, no CUDA cores)
GPU1: 17-40%  (display compositor + idle)
```

---

## 6. Cuellos de Botella Identificados — Priorizado

### 🔴 Prioridad ALTA

| # | Bottleneck | Impacto | Solución |
|---|---|---|---|
| 1 | **Tensor Cores suboptimizados** | ESRGAN usa kernels conv2d genéricos | Activar `cudnn.benchmark=True` + `allow_tf32=True` |
| 2 | **Sin torch.compile()** | Overhead de Python entre operaciones | `torch.compile(model, mode="reduce-overhead")` |
| 3 | **NVENC GPU1 idle** | Solo GPU0 encoda, GPU1 NVENC 100% idle | Dual NVENC: chunks pares/impares |
| 4 | **RIFE y ESRGAN secuenciales** | Cuando RIFE corre, GPUs idle en CUDA; cuando ESRGAN corre, Vulkan idle | Overlap: RIFE chunk N+1 mientras ESRGAN chunk N |

### 🟡 Prioridad MEDIA

| # | Bottleneck | Impacto | Solución |
|---|---|---|---|
| 5 | **NVDEC no utilizado** | CPU decodifica H.264 (menor, extract solo 1.8s) | `-hwaccel cuda -c:v h264_cuvid` |
| 6 | **VRAM subutilizada** | 5/16 GB en 5070Ti, 1.5/6 GB en 2060 | Subir batch sizes |
| 7 | **Audio no procesado** | 75kbps AAC calidad baja | ffmpeg denoising + normalization en CPU |
| 8 | **PCIe x4 en RTX 2060** | Negocia x4 de x16 max | Verificar BIOS / slot físico (hardware) |

### 🟢 No aplicable

| Recurso | Razón |
|---|---|
| **RT Cores** (ambas GPUs) | Ray tracing no tiene utilidad en super-resolución ni interpolación de frames |
| **OpenCL** | PyTorch usa CUDA, no OpenCL |
| **AV1 NVENC** | HEVC es suficiente, AV1 encode es más lento |

---

## 7. Estructura de Archivos Actual

```
FilosofiaNeurociencias/
├── enhance/
│   ├── __init__.py
│   ├── config.py          ← Configuración del pipeline
│   ├── esrgan.py          ← Motor ESRGAN dual-GPU
│   ├── pipeline.py        ← Pipeline principal (extract → rife → esrgan → nvenc)
│   ├── rife.py            ← Wrapper de rife-ncnn-vulkan
│   ├── ffmpeg_utils.py    ← Utilidades ffmpeg
│   ├── progress.py        ← Tracking de progreso
│   ├── SYSTEM_SPECS.md    ← Specs del hardware
│   ├── logs/              ← Todos los logs de benchmarks
│   │   ├── metrics.log .. metrics_v7.log
│   │   ├── run_test.log .. run_pipeline_v8.log
│   └── scripts/           ← Scripts de benchmark y test
│       ├── benchmark_isolated.py
│       ├── run_test.sh .. run_test3.sh
│       └── test_monitor.sh
├── enhanced/              ← Output de videos procesados
│   └── test_trim_ai_50fps.mp4
├── scripts/
│   ├── run.py             ← Script principal de ejecución
│   └── ...
└── videos/
    ├── GMT20260320-130023_Recording_2240x1260.mp4  ← Video fuente
    ├── GMT20260320-130023_Recording.m4a             ← Audio fuente
    └── test_trim.mp4
```

---

## 8. Plan de Optimización — 3 Fases

### Fase 1: Activar Chips Dormidos (sin cambio de arquitectura)

**Estimación: ESRGAN de 35 fps → 45-55 fps**

#### 1.1 Tensor Cores + cuDNN Benchmark
```python
# Agregar en config.py ANTES de importar torch
import torch
torch.backends.cuda.matmul.allow_tf32 = True      # ACTUALMENTE: False
torch.backends.cudnn.benchmark = True              # ACTUALMENTE: False
```
- `cudnn.benchmark=True`: PyTorch probará todos los kernels de convolución al inicio y seleccionará el más rápido para cada tamaño de tensor. En la 5070 Ti, esto activará kernels que usan Tensor Cores explícitamente.
- `allow_tf32=True`: Permite que las multiplicaciones de matrices usen TF32 (19 bits de precisión) en vez de fp32, usando Tensor Cores. Para ESRGAN en fp16 el impacto es menor, pero las operaciones intermedias (como interpolate) se benefician.

#### 1.2 torch.compile() del modelo
```python
# En esrgan.py, después de cargar el modelo
net = torch.compile(net, mode="reduce-overhead", fullgraph=True)
```
- Funde múltiples operaciones CUDA en un solo kernel (elimina launches de kernels individuales)
- Elimina overhead de Python entre operaciones
- Genera código CUDA optimizado para la GPU específica
- Requiere un "warmup" pass (primera inferencia es lenta, resto son rápidas)

#### 1.3 Batch sizes más grandes
```python
# En config.py
GPU0_BATCH = 16  # 5070 Ti: 16GB disponible, actualmente solo usa ~5GB con batch=8
GPU1_BATCH = 8   # 2060: 6GB disponible, actualmente solo usa ~1.5GB con batch=4
```
- Menos lanzamientos de kernels CUDA por chunk
- Mejor amortización de latencias de transferencia CPU↔GPU
- La VRAM adicional está completamente libre

#### 1.4 NVDEC para extracción
```python
# En pipeline.py, cambiar extract de:
cmd = ["ffmpeg", "-ss", str(start), "-t", str(dur), "-i", str(src), ...]
# A:
cmd = ["ffmpeg", "-hwaccel", "cuda", "-c:v", "h264_cuvid", 
       "-ss", str(start), "-t", str(dur), "-i", str(src), ...]
```
- Decodificación H.264 por hardware (NVDEC ASIC)
- Libera CPU para otras tareas (audio enhancement)
- Impacto en extract: ~1.8s → ~0.5s

---

### Fase 2: Overlap + Dual NVENC

**Estimación: Tiempo total -40% adicional**

#### 2.1 Dual NVENC
```
Chunk 0: → NVENC GPU0
Chunk 1: → NVENC GPU1  (en paralelo con ESRGAN del chunk 2)
Chunk 2: → NVENC GPU0
```
- Ambos ASICs NVENC trabajan simultáneamente
- La RTX 2060 tiene NVENC Turing (soporta HEVC 4K)
- Throughput NVENC: de ~48fps → ~90fps efectivos

#### 2.2 Overlap RIFE/ESRGAN entre chunks
```
Tiempo →
Chunk 0: [RIFE GPU1]──────►[ESRGAN GPU0+GPU1]──►[NVENC GPU0]
Chunk 1:                    [RIFE GPU1]────────►[ESRGAN GPU0+GPU1]──►[NVENC GPU1]
Chunk 2:                                        [RIFE GPU1]────────►[ESRGAN ...]
```
- RIFE se mueve a **GPU1 sola** (Vulkan)
- Mientras ESRGAN procesa chunk N en ambas GPUs, RIFE procesa chunk N+1 en GPU1 vía Vulkan
- RIFE Vulkan y ESRGAN CUDA pueden coexistir en la misma GPU (son APIs distintas)
- El overlap elimina los ~36s de RIFE como etapa bloqueante

---

### Fase 3: Audio Enhancement

**Impacto: Cero costo adicional (corre en CPU paralelo)**

#### 3.1 Pipeline de audio
```
Audio original (AAC 48kHz 75kbps)
  ↓
  ffmpeg decode
  ↓
  afftdn (FFT-based noise reduction)
  ↓
  loudnorm (EBU R128 normalization)
  ↓
  dynaudnorm (dynamic normalization)
  ↓
  Re-encode AAC 256kbps o OPUS 128kbps
```

#### 3.2 Ejecución paralela
- Audio se procesa en un thread separado desde el inicio del pipeline
- Usa CPU (que está 90% idle durante GPU work)
- El 128MB L3 V-Cache del 9950X3D es ideal para FFT
- Se muxa con el video al final: `ffmpeg -i video.mp4 -i audio_enhanced.m4a -c copy output.mp4`

---

## 9. Estimación de Tiempos Finales

### Para 15 segundos de video (375 frames → 750 frames 4K@50fps)

| Configuración | ESRGAN | RIFE | NVENC | Total |
|---|---|---|---|---|
| **v8 actual** | 21.3s (35fps) | 35.9s | 15.6s | **~75s** |
| **+ Fase 1** (tensor/compile/batch) | ~14s (50fps) | 35.9s | 15.6s | **~68s** |
| **+ Fase 2** (overlap+dual nvenc) | ~14s | *overlap* | ~8s | **~25s** |
| **+ Fase 3** (audio paralelo) | — | — | — | **+0s** |

### Para video completo de 7.4 horas (666,000 frames → 1,332,000 frames)

| Configuración | Tiempo estimado | Ratio |
|---|---|---|
| **v1 original** | ~243 horas | 0.03× |
| **v8 actual** | ~37 horas | 0.19× |
| **+ Fase 1** | ~33 horas | 0.22× |
| **+ Fases 1+2** | **~12 horas** | **0.61×** |
| **Teórico máximo** | ~6 horas | 1.2× (realtime) |

---

## 10. Resumen Ejecutivo

### Lo que funciona bien ✅
- ESRGAN dual-GPU a 35.3 fps con GPU1 al 96-98%
- RIFE a resolución original (12× más rápido que a 4K)
- NVENC p1 a 48.2 fps
- Pipeline end-to-end estable con progreso resumible
- Frames en tmpfs (RAM) eliminan I/O de disco

### Lo urgente por hacer 🔴
1. Activar Tensor Cores (`cudnn.benchmark=True`, `allow_tf32=True`)
2. `torch.compile()` del modelo ESRGAN
3. Overlap RIFE/ESRGAN entre chunks (elimina 36s bloqueantes)
4. Dual NVENC (usar el encoder idle de la 2060)

### Lo importante por hacer 🟡
5. Audio enhancement (el audio original es 75kbps — calidad muy baja)
6. Aumentar batch sizes (VRAM desperdiciada)
7. NVDEC para extract (libera CPU)

### Lo que NO se puede mejorar por software 🔒
- RT Cores: no aplican a este flujo de trabajo
- PCIe x4 en 2060: problema físico (BIOS/slot)
- NVENC speed: es un ASIC, 48fps es su tope a 4K HEVC
