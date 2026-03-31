# System Specs & Bottleneck Analysis

## Hardware

### CPU
| Spec | Valor |
|---|---|
| Modelo | AMD Ryzen 9 9950X3D |
| Cores/Threads | 16C / 32T |
| Frecuencia Max | 5803 MHz |
| Arquitectura | Zen 5 (3D V-Cache) |
| Socket | AM5 |

### RAM
| Spec | Valor |
|---|---|
| Total | 128 GB DDR5 |
| Disponible | 92 GB |
| Swap | 191 GB |

### GPU 0 — RTX 5070 Ti
| Spec | Valor |
|---|---|
| Arquitectura | Blackwell |
| VRAM | 16 GB GDDR7 |
| Clock Max GPU | 3105 MHz |
| Clock Max Mem | 14001 MHz |
| TDP | 350W |
| PCIe | **Gen 5 × 16** (~64 GB/s) |
| NVENC Sessions | Soporta multi-session |
| Bus ID | 00:01:00.0 |

### GPU 1 — RTX 2060
| Spec | Valor |
|---|---|
| Arquitectura | Turing |
| VRAM | 6 GB GDDR6 |
| Clock Max GPU | 2100 MHz |
| Clock Max Mem | 7001 MHz |
| TDP | 172.5W |
| PCIe | **Gen 3, Max x16, NEGOCIANDO x4** (~4 GB/s) |
| Bus ID | 00:11:00.0 |
| Display | **Attached + Active** (monitor conectado aquí) |

> [!WARNING]
> **La RTX 2060 negocia solo 4 lanes en vez de 16.** Esto es un problema FÍSICO del slot/riser o BIOS. El motherboard reporta x16 max pero el link actual opera a x4. Esto reduce el ancho de banda de 16 GB/s (Gen3 x16) a **4 GB/s (Gen3 x4)**. Esto puede contribuir al 70% de utilización en vez de 100%.

### Almacenamiento
| Dispositivo | Tamaño | Tipo | Montaje |
|---|---|---|---|
| tmpfs | 62 GB | RAM | /tmp (frames en RAM, cero I/O disco) |
| sda | 931 GB | HDD (NTFS) | /media/stev/Documentos |
| sdb | 931 GB | HDD (NTFS) | /media/stev/Data2 |
| sdc | 1.8 TB | HDD (NTFS) | /media/stev/Juegos |
| sdd | 596 GB | HDD (NTFS) | /media/stev/External |

> [!NOTE]
> No hay SSD detectado como disco de datos separado. El sistema corre desde ¿NVMe? (no listado por lsblk, posiblemente montaje root). Tmpfs de 62GB es suficiente para el pipeline.

---

## Software

| Componente | Versión |
|---|---|
| PyTorch | 2.10.0 + CUDA 12.8 |
| CUDA Toolkit | 12.4 |
| Vulkan | 1.4.321 |
| spandrel | 0.4.2 |
| OpenCV | 4.13.0 |
| NumPy | 2.4.3 |
| FFmpeg | 7.1.1 |
| RIFE | rife-ncnn-vulkan 20221029 (Vulkan) |

### FFmpeg Capacidades Hardware
- **NVENC**: h264, hevc, av1 ✅
- **NVDEC (cuvid)**: h264, hevc, av1, vp9, etc. ✅
- **Vulkan**: Disponible ✅
- **OpenCL**: Disponible ✅

---

## Qué se usa vs qué NO se usa

### ✅ Hardware Utilizado
| Recurso | Cómo se usa | Utilización |
|---|---|---|
| RTX 5070 Ti CUDA | ESRGAN (PyTorch, batch=8) | 83-100% durante ESRGAN |
| RTX 2060 CUDA | ESRGAN (PyTorch, batch=4) | 67-76% durante ESRGAN |
| RTX 5070 Ti Vulkan | RIFE interpolación | ~16-36% (Vulkan reporta distinto) |
| NVENC (5070 Ti) | Codificación HEVC | ~10% (ASIC dedicado) |
| CPU (16 cores) | Extract ffmpeg, PNG R/W | Variable |
| tmpfs (62GB) | Frames intermedios | ~10GB pico |

### ❌ Hardware NO Utilizado
| Recurso | Potencial | Por qué no se usa |
|---|---|---|
| **NVDEC** (ambas GPUs) | Decodificación por HW | ffmpeg extrae con CPU, no usa cuvid |
| **NVENC 2060** | Encode paralelo | Solo se encoda en GPU0 |
| **RTX 2060 Vulkan** | RIFE dual-GPU | Benchmark mostró irrelevante (+6%) |
| **CPU para ESRGAN** | Inference en CPU | Benchmark mostró que DESTRUYE GPU perf |
| **RAM (92GB libres)** | Buffers masivos | Solo se usan ~10GB |
| **32 threads CPU** | Solo ~4-16 activos | ffmpeg extract no paraleliza mucho |
| **AV1 NVENC** | Mejor compresión | Usando HEVC, av1 sería más lento |

---

## Cuellos de Botella Identificados

### 1. 🔴 RTX 2060 PCIe x4 (CRÍTICO — HARDWARE)
- **Síntoma**: GPU1 nunca pasa de 76% en ESRGAN
- **Causa**: Link PCIe negocia x4 en vez de x16
- **Impacto**: Ancho de banda reducido 4x (4 GB/s vs 16 GB/s)
- **Solución**: Verificar slot BIOS settings, riser cable, o posición física del slot

### 2. 🟡 GPU0 cae a 0% al final de ESRGAN (DISEÑO)
- **Síntoma**: La 5070 Ti termina su porción del queue y espera a la 2060
- **Causa**: La 5070 Ti es ~3x más rápida → agota frames primero
- **Impacto**: ~30% del tiempo ESRGAN la 5070 está idle
- **Solución posible**: Cuando la 5070Ti termina, que robe frames de la cola de la 2060

### 3. 🟡 RIFE secuencial (DISEÑO)
- **Síntoma**: RIFE bloquea el pipeline 34s
- **Causa**: RIFE Vulkan es un proceso externo single-shot
- **Impacto**: Mientras RIFE corre, ESRGAN está idle, y viceversa
- **Solución posible**: Overlap RIFE chunk N con ESRGAN chunk N-1

### 4. 🟢 NVDEC no se usa para decode (OPORTUNIDAD)
- **Síntoma**: CPU decodifica con ffmpeg software
- **Causa**: No se pasó `-hwaccel cuda -c:v h264_cuvid`
- **Impacto**: Menor (extract solo toma 1.6s)

### 5. 🟡 ESRGAN pinned memory copy (CPU BOTTLENECK)
- **Síntoma**: GPU0 oscila 100→0→100 incluso en batch mode
- **Causa**: `pinned[buf].copy_(torch.from_numpy(frame))` bloquea el thread
- **Impacto**: GPU espera a que CPU prepare el siguiente batch

---

## Velocidades Medidas (por etapa)

| Etapa | FPS | Tiempo (15s video) | Bottleneck |
|---|---|---|---|
| Extract | 236 | 1.6s | — |
| RIFE (1260p) | 21.8 | 34.5s | Vulkan compute |
| ESRGAN (batch) | 23.8 | 31.6s | CPU→GPU copy |
| NVENC (p1) | 58.8 | 12.7s | — |

**Cuello de botella actual: RIFE y ESRGAN están parejos (~34s vs ~32s), NVENC ya no es bottleneck.**
