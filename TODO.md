# TODO — Optimización de Hardware y Calidad

> **Estado actual validado**: 0.4198× realtime sostenido en 5 min, 24.0 fps efectivos, GPU0 70.8% avg, GPU1 62.3% avg, CPU 30.9% avg, GPU1 bloqueada en PCIe gen1 x4
> **Meta mínima**: ≥ 0.40× realtime sostenido, máxima calidad visual y de audio
> **Meta deseable**: ≥ 0.45× realtime sin sacrificar calidad
> **Fecha**: 2026-04-01
>
> **Producción**: Video 1 (Recording) ya fue procesado con modelo anime (30 GB, 4480×2520@50fps, 7.4h). Se relanza con perfil `quality` (real_x2plus + face_adaptive). Video 2 (Gallery) solo tenía 9/1774 chunks. Audio enhanced existente usa `dynaudnorm` (viejo) — se reemplaza con perfil `natural`.
>
> **Validación Codex (2026-04-01)**: `quality/real_x2plus` no revalida todavía el gate final, pero ya quedó validada la arquitectura correcta para este hardware. Con `GPU0_BATCH=16` hay `CUDA OOM`; con `GPU0_BATCH=4` evita el OOM, pero en una prueba de `60s` solo saturó de forma sostenida a `GPU0`, dejó `GPU1` mayormente ociosa y no completó el chunk dentro de `304s`. Se probó además un modo experimental para compartir `GPU1` entre `RIFE` y `ESRGAN`: sin tiling sube el uso de ambas GPUs, pero la `RTX 2060` hace `OOM` incluso con `GPU1_BATCH=1`; con tiling (`256/16`) evita ese `OOM`, pero sigue sin cerrar un chunk de `15s` en tiempo razonable. La corrección útil fue otra: `ESRGAN` solo en `GPU0` y `RIFE` adelantado solo en `GPU1`. En la validación `bench_codex_hetero45_prefetch4_chunk15_20260401_111910` el prefetch sí arrancó mientras `chunk_0000` seguía en `ESRGAN`, hubo muestra directa de `GPU0=100%` y `GPU1=100%` simultáneas, y el log parcial arrojó `GPU0 91.1%` avg, `GPU1 35.4%` avg, `CPU 23.0%` avg, con `37/117` pares de muestras en `>=80%` para ambas GPUs y sin `OOM`. Ese run quedó interrumpido antes de cerrar el chunk, así que todavía no cierra el gate de throughput. Para el camino compacto dual sí quedó una mejora estable: `general_light` con `RIFE=GPU0`, `ESRGAN=GPU0,1`, `GPU0_BATCH=8`, `GPU1_BATCH=8` y tiling adaptativo en GPU compartida cerró `bench_codex_general_light_dual30_autotile_postreboot_ok_chunk15_20260401_124118` en `167.7s` (`0.1789× realtime`), mejor que el baseline dual compacto anterior (`184.6s`, `0.1625×`). Desde esta validación también quedó integrado `RIFE` real en `torch/torch_cpu`: el backend ahora carga IFNet oficial con `flownet.pkl` real y auto-descarga el checkpoint `paper_v6` desde la fuente oficial si hace falta. La smoke local CPU ya produce salida distinta al blend (`MAE 4.62` frente a promedio simple en un caso mínimo), pero todavía falta benchmarkear throughput real antes de proponerlo como default.

> **Nota**: El diagnóstico detallado de abajo conserva parte del baseline original. El estado real por tarea y los benchmarks recientes quedan resumidos aquí.

## Estado actualizado

- [x] T1. **Completa**. Async D2H + double-buffering implementado en `esrgan.py`: 3 CUDA streams (copy, compute, d2h), pinned memory buffers dobles, `non_blocking=True`, eventos CUDA para telemetría. Validado en bench sostenido 0.42× realtime.
- [x] T2. **Completa**. Hot path tensor pinned → writer ya evita copias numpy en CPU (`_consume_output` pasa tensor pinned directo). Pipeline GPU-resident (`T16`) sigue como futuro.
- [x] T3. **Completa funcional**. El pipeline 4-stage (extract→RIFE→ESRGAN→NVENC) tiene overlap con `BudgetController` y `PIPELINE_DEPTH`. GPU1 reservada para RIFE por estabilidad. El prefetch multi-chunk quedó corregido para arrancar solo cuando `GPU1` queda libre; en modo heterogéneo (`ESRGAN=GPU0`, `RIFE=GPU1`) ya hay solape real entre ambas GPUs.
- [x] T4. **Completa**. Modelos `real_x2plus` y `real_x4plus` registrados en `models.py` con auto-download + SHA256. Perfiles `quality`, `production`, `face_preserve` usan `real_x2plus`. Bakeoff pendiente de producción completa.
- [x] T5. **Completa por implementación**. `face_adaptive=True` en perfiles `quality`/`production`. Código de `apply_face_adaptive()` en `visual_eval.py` integrado en `_consume_output()`. Validación visual en producción completa.
- [x] T6. **Completa**. Perfiles audio `natural`/`production` sin `dynaudnorm`, con `alimiter`. A/B bench ejecutado (`audio_ab_bench.py`). Perfil `natural` seleccionado para producción.
- [x] T7. **Completa**. Scheduler con CCD-aware pinning implementado: `split_l3_a` y `production` profiles con `taskset`, `ionice`, `chrt`. Integrado en `wrap_subprocess()`. Validado ~5% mejor.
- [x] T8. **Completa para baseline anime**. `chunk=30` sigue siendo el mejor setting conocido. Perfil `production` lo incluye, pero `quality/real_x2plus` necesita revalidación aparte.
- [x] T9. **Completa por cableado**. `NVENC_GPUS` y `process_production.sh` soportan `0,1`, pero el flujo directo RIFE→ESRGAN→NVENC usa `_safe_nvenc_gpus()` y reserva GPU1 para RIFE por estabilidad. Encode no es cuello.
- [x] T10. **Cerrada como límite HW**. GPU1 (2060) en PCIe gen1 x4 confirmado con `check_pcie_width()`. No resoluble por software; solo queda verificación BIOS/física del slot si se quiere abrir esa línea.
- [ ] T11. Reabierta. `GPU0_BATCH=16` ya no es seguro para `quality/real_x2plus` a resolución completa: validación Codex reprodujo `CUDA OOM`. `process_production.sh` baja ahora a `GPU0_BATCH=4` y fija `ESRGAN` en `GPU0` por defecto para perfiles reales; queda pendiente retunar batch seguro/óptimo y cerrar el gate de throughput completo en el modo heterogéneo.
- [x] T12. **Completa**. NVDEC toggle implementado (`ENABLE_NVDEC`). Verificado que extract no es cuello.
- [x] T13. **Completa**. `RIFE_THREADS=1:8:4` validado como óptimo. `2:8:4` benchmarked sin mejora.
- [x] T14. **Completa funcional**. Overhead reducido: reescaneo optimizado, streaming window configurable, `RIFE_POLL_SECONDS` y `RIFE_FILE_SETTLE_SECONDS` afinados.
- [ ] T15. Parcial. Backend Torch ya soporta IFNet oficial real en `rife_backend.py` y auto-descarga `paper_v6`. Sigue sin ser default de producción porque falta benchmark serio de throughput/calidad frente a `ncnn-Vulkan` en este hardware.
- [ ] T16. Parcial experimental. Existen `ENHANCE_ESRGAN_GPU_RESIDENT` y `_try_gpu_resident_encode()` en `pipeline.py`, pero falta integrar un camino end-to-end sin D2H desde `ESRGANEngine`.

## Últimos benchmarks estables

- `bench_codex_general_light_dual30_autotile_postreboot_ok_chunk15_20260401_124118`: `0.1789× realtime` sobre `30s`, `chunk_0000=80.9s`, `chunk_0001=79.0s`, GPU0 `39.7%` avg, GPU1 `81.0%` avg. Mejor benchmark estable actual del camino compacto dual (`general_light`) con tiling adaptativo integrado.
- `bench_codex_general_light_dual30_tile0_chunk15_20260401_121212`: `0.1691× realtime` sobre `30s`; confirmó que desactivar tiling en la `5070 Ti` compartida mejora el camino compacto dual.
- `bench_sustain300_chunk30_safe_chunk30_20260331_231550`: `0.4198× realtime` sobre `300s` de contenido, `effective_fps=24.0`, GPU0 `70.8%` avg, GPU1 `62.3%` avg, CPU `30.9%` avg, sin zombies. Mejor validación sostenida actual.
- `bench_sustain60_chunk30_safe_chunk30_20260331_231306`: `0.407× realtime` sobre `60s`, confirma que fijar `NVENC` fuera de la GPU de `RIFE` mantiene estabilidad y rendimiento.
- `bench_sustain60_beststable_chunk20_20260331_225909`: `0.3977× realtime` en `60s`, `effective_fps=23.0`, `chunk_avg=43.48s`; gate FAIL por `0.0023` debajo del throughput objetivo y por tiempo promedio por chunk. Se observaron fases repetidas con `GPU0=100%` y `GPU1=100%`.
- `bench_fullpath_clean_chunk10_chunk10_20260331_201031`: `0.3635× realtime`, `effective_fps=20.49`, `chunk_avg=24.41s`, gate FAIL solo por throughput.
- `bench_fullpath_clean_chunk20_chunk20_20260331_201141`: `0.3702× realtime`, `effective_fps=22.36`, `chunk_avg=44.72s`, gate FAIL por throughput y tiempo promedio por chunk.
- `bench_fullpath_chunk20_rife284_chunk20_rife_2:8:4_20260331_201259`: `0.3675× realtime`; `RIFE_THREADS=2:8:4` no mejora.
- `bench_smoke_no_rife_chunk10_20260331_192948`: `0.7702× realtime`; confirma que el cuello estable restante es RIFE/GPU1, no ESRGAN.

## Siguiente corte recomendado

1. **Producción recomendada** (2026-04-01): `process_production.sh` con `quality` profile (real_x2plus + face_adaptive), `natural` audio, `chunk=30`, `ENHANCE_ESRGAN_GPUS=0`, `ENHANCE_RIFE_GPU=1`, `GPU0_BATCH=4`, `GPU1_BATCH=1`, `ENHANCE_SHARE_RIFE_GPU=0`.
   `torch_cpu` ya soporta IFNet real, pero no pasa a producción todavía hasta cerrar benchmark de throughput y calidad visual frente a `baseline/ncnn`.
2. Video 1 (Recording): ~7.4h de contenido. La proyección vieja de `0.42× realtime` corresponde al baseline anime; `quality/real_x2plus` necesita una corrida sostenida nueva con el modo heterogéneo ya corregido.
3. Video 2 (Gallery): correr secuencial tras revalidar throughput de Video 1 con la configuración heterogénea.
4. Audio: se re-procesa con perfil `natural` (sin dynaudnorm) automáticamente en hilo paralelo del primer run.
5. Post-producción: verificar calidad de rostros, texto y audio tras completar. Evaluar T10 (BIOS) y T15/T16 solo si el throughput heterogéneo sigue siendo insuficiente.

---

## Diagnóstico: Por qué el hardware no se usa al 100%

### Cuello de botella #1 — Transfer D2H domina ESRGAN (85% del tiempo)

```
esrgan_d2h_seconds:    23.87s/chunk   ← 85% del tiempo ESRGAN
esrgan_infer_seconds:   0.11s/chunk   ← compute real (trivial)
esrgan_h2d_seconds:     1.74s/chunk
esrgan_fill_seconds:    2.31s/chunk
esrgan_writer_wait:     0.82s/chunk
idle_between_stages:    ~7.5s/chunk
```

La GPU computa cada batch en **0.11s** pero la transferencia Device→Host de los frames 4K toma **23.87s**. Durante ese tiempo la GPU está idle. La línea responsable en `esrgan.py:207`:

```python
out_cpu = out_u8.permute(0, 2, 3, 1).contiguous().cpu()  # ← SÍNCRONO, BLOQUEA GPU
```

### Cuello de botella #2 — Etapas seriales dentro del chunk

RIFE→ESRGAN→NVENC son secuenciales por chunk. Hay ~7.5s de idle entre etapas donde ninguna GPU trabaja.

### Cuello de botella #3 — GPU0 subalimentada

GPU0 promedio 46.6%, con **49% del tiempo por debajo de 30%** de utilización. No es falta de capacidad de la GPU — es falta de trabajo entrante.

### Cuello de botella #4 — GPU1 solo usa Vulkan

GPU1 (2060) solo trabaja via Vulkan para RIFE. Su NVENC, NVDEC y Tensor Cores están completamente ociosos.

### Cuello de botella #5 — CPU 10.2% promedio

32 threads disponibles, solo se usan ~3-4 efectivos. El CPU no es cuello de nada hoy, pero podría alimentar mejor a las GPUs.

---

## Tareas Pendientes

### 🔴 CRÍTICO — Transfer D2H y pipeline GPU

#### T1. Async D2H con double-buffering en ESRGAN
**Archivo**: `enhance/esrgan.py` — `_gpu_worker()`  
**Problema**: `.contiguous().cpu()` es síncrono — la GPU espera 23.87s por PCIe  
**Solución**:
- Usar dos buffers de pinned memory pre-alocados
- Overlap: mientras batch N se transfiere por D2H en un stream, batch N+1 se infiere en otro stream
- `torch.cuda.Stream` para copy + compute independientes
- `.to("cpu", non_blocking=True)` + sincronizar solo antes de consumir

**Impacto estimado**: Reducir D2H de 23.87s a <3s por chunk (overlap con compute)

#### T2. Pipeline D2H → NVENC sin pasar por numpy
**Archivo**: `enhance/esrgan.py`, `enhance/pipeline.py`  
**Problema**: El frame sale de GPU → CPU numpy → memoryview → pipe stdin → NVENC. Hay 3 copias innecesarias  
**Solución explorar**:
- Mantener tensor en GPU, usar `torch.cuda.current_stream()` para write directo al pipe
- O usar buffer pool de pinned memory con ring buffer
- Evaluar si NVENC puede consumir del mismo espacio GPU (shared surfaces)

**Impacto estimado**: Eliminar ~50% del overhead D2H

#### T3. Overlap RIFE chunk N+1 mientras ESRGAN chunk N
**Archivo**: `enhance/pipeline.py` — `esrgan_worker()`, `rife_first_worker()`  
**Problema**: 7.5s de idle entre etapas por chunk. RIFE y ESRGAN nunca corren simultáneamente  
**Solución**:
- Aumentar `PIPELINE_DEPTH` efectivo
- RIFE en GPU1 Vulkan corre chunk N+1 mientras ESRGAN CUDA procesa chunk N en ambas GPUs
- Vulkan y CUDA coexisten en la misma GPU (APIs distintas)

**Impacto estimado**: Eliminar 7.5s idle entre etapas ≈ 20% menos tiempo total

---

### 🟠 ALTA — Calidad visual

#### T4. Modelo adecuado para rostros humanos
**Archivos**: `enhance/profiles.py`, `enhance/models.py`, `enhance/config.py`  
**Problema**: El modelo actual (`realesr-animevideov3`) está optimizado para anime y sobre-suaviza rostros humanos. Piel plástica, pérdida de micro-detalle facial  
**Trabajo**:
1. Registrar modelo `realesrgan-x2plus` u otro orientado a material real
2. Comparar al menos 3 familias de salida con clips de rostros
3. Evaluar `downscale_factor` 0.5 vs 1.0 con cada modelo
4. Configurar mezcla híbrida con reinyección de detalle original (`visual_eval.py`)

**Clips de evaluación obligatorios**:
- Rostro cercano
- Rostro medio con movimiento
- Texto fino / interfaz
- Escena mixta (rostro + texto)

**Condición de cierre**: Rostros naturales sin look plástico, sin perder el avance en texto y fluidez

#### T5. Face-adaptive blending
**Archivo**: `enhance/visual_eval.py`  
**Problema**: La infraestructura de blending facial existe pero no se ha validado con modelo real  
**Trabajo**:
1. Activar `face_adaptive=True` en perfil `quality`
2. Afinar ROI y pesos de mezcla con clips de prueba
3. Asegurar que no introduce halos ni artefactos de borde

---

### 🟠 ALTA — Calidad de audio

#### T6. Cadena de audio más natural
**Archivos**: `enhance/audio_profiles.py`, `enhance/config.py`  
**Problema**: La cadena actual (`afftdn + loudnorm + dynaudnorm`) limpia bien pero aplasta la dinámica y suena procesada  
**Trabajo**:
1. Evaluar perfiles existentes con A/B bench:
   - `conservative`: anlmdn + loudnorm + alimiter
   - `voice`: anlmdn + dialoguenhance + speechnorm + alimiter
   - `natural`: afftdn + loudnorm + alimiter
2. Medir con `scripts/audio_ab_bench.py` en slice de 30s
3. Clave: quitar `dynaudnorm` (causa del bombeo y aplanamiento)
4. Validar que voz suena natural, sin bombeo, sin sensación de aplastado

**Condición de cierre**: Voz perceptualmente más natural que la cadena actual

---

### 🟡 MEDIA — Scheduling y afinidad CPU

#### T7. Afinidad por CCD/L3 del 9950X3D
**Archivos**: `enhance/scheduler.py`, `enhance/profiles.py`  
**Problema**: 32 threads sin afinidad, CPU al 10.2%. No se explota la topología dual-CCD ni el 128MB L3  
**Trabajo**:
1. `split_l3_a` ya confirmado ~5% mejor para ffmpeg
2. Implementar en producción: `taskset` para extract/audio en CCD0, coordinación en CCD1
3. `ionice -c2 -n0` para procesos de I/O del wrapper RIFE
4. `chrt -o 0` para coordinación Python
5. Validar que no interfiere con scheduler del escritorio

**Impacto**: CPU más eficiente alimentando a GPUs, ~5% mejor throughput

#### T8. Medir chunk sizes óptimos
**Archivos**: `enhance/config.py`  
**Problema**: Chunk de 15s es el default, no se ha probado 20s o 30s  
**Trabajo**:
1. Benchmarkear 15s, 20s, 30s con misma configuración
2. Medir overhead de spawn RIFE por chunk vs throughput sostenido
3. Chunks más grandes = menos overhead de spawn + mejor amortización de warmup

---

### 🟡 MEDIA — GPU1 subutilizada

#### T9. NVENC dual (GPU0 + GPU1)
**Archivos**: `enhance/pipeline.py`, `enhance/config.py`  
**Problema**: GPU1 tiene NVENC Turing completamente idle, solo GPU0 codifica  
**Trabajo**:
1. Chunks alternos: pares→GPU0, impares→GPU1
2. Validar que NVENC en GPU1 no baja rendimiento de RIFE Vulkan (APIs distintas)
3. Verificar calidad HEVC de la 2060 vs 5070 Ti

**Impacto**: Si NVENC fuera cuello, duplicaría throughput de encode. Hoy NVENC no es cuello, pero libera GPU0 NVENC para solapar con ESRGAN

#### T10. Confirmar y resolver PCIe x4 de GPU1
**Problema**: La 2060 negocia x4 en vez de x16, limitando ancho de banda a ~4 GB/s  
**Trabajo**:
1. Medir `pcie.link.width.current` bajo carga real de RIFE
2. Verificar en BIOS: asignación de lanes PCIe, bifurcación
3. Verificar físico: slot, riser, cable
4. Si sigue en x4 tras verificación, documentar como límite de hardware

**Impacto**: Si se resuelve a x16, el throughput H2D/D2H de GPU1 sube 4×

---

### 🟡 MEDIA — Batch sizes y VRAM

#### T11. Aumentar batch sizes
**Archivo**: `enhance/config.py`  
**Problema**: GPU0 usa 5/16 GB VRAM (batch=8), GPU1 usa 1.5/6 GB (batch=4)  
**Trabajo**:
1. Probar `GPU0_BATCH=16` (usaría ~10 GB de 16 GB)
2. Probar `GPU1_BATCH=8` (usaría ~3 GB de 6 GB)
3. Medir: más frames por kernel launch = menor overhead relativo de H2D/D2H
4. Benchmarkear con T1 (async D2H) — el beneficio de batch grande escala con D2H async

**Impacto**: ~10-15% más FPS por menor overhead de lanzamiento

---

### 🟢 BAJA — Optimizaciones menores

#### T12. NVDEC para extracción
**Archivo**: `enhance/ffmpeg_utils.py`  
**Problema**: ffmpeg decodifica H.264 en CPU. NVDEC está idle  
**Trabajo**: `-hwaccel cuda -c:v h264_cuvid`  
**Nota**: Extract solo toma ~1.8s y no es cuello. Solo vale la pena si CPU se vuelve cuello por audio + coordinación

#### T13. RIFE_THREADS disciplinado
**Archivo**: `enhance/config.py`  
**Problema**: `RIFE_THREADS=1:8:4` nunca se ha A/B testeado formalmente  
**Trabajo**: Benchmarkear variantes sobre chunk ganador

#### T14. Reducción de overhead del wrapper RIFE
**Archivo**: `enhance/pipeline.py` — `_stream_rife_esrgan_to_nvenc()`  
**Problema**: Reescaneo de directorio, cleanup por chunk, polling  
**Trabajo**:
1. Reducir reescaneos de directorio
2. Reutilizar estructura temporal donde no rompa reanudación
3. Bajar coste de polling y drenaje

---

### 🔵 FUTURO — Cambios de arquitectura mayor

#### T15. Backend RIFE en memoria (sin PNGs)
**Archivo**: `enhance/rife_backend.py`  
**Problema**: RIFE usa PNGs en tmpfs — escritura, lectura, polling, cleanup. Todo ese I/O en memoria es parte del cuello real  
**Trabajo**: Migrar a backend PyTorch o TensorRT que mantenga frames en RAM/GPU  
**Condición**: Solo si T1-T14 no alcanzan la meta

#### T16. Pipeline GPU-resident (ESRGAN → NVENC sin CPU)
**Problema**: Frames 4K viajan GPU → CPU → pipe → NVENC (vuelve a GPU)  
**Trabajo**: Explorar CUDA→NVENC shared surfaces o `cuvidCreateVideoEncoder`  
**Condición**: Solo si T1 no resuelve suficiente

---

## Orden de Ejecución Recomendado

```
Prioridad 1 (máximo impacto, cuello #1):
  T1  → Async D2H double-buffering        ← 85% del tiempo ESRGAN
  T4  → Modelo para rostros humanos        ← calidad facial inaceptable

Prioridad 2 (alto impacto):
  T3  → Overlap RIFE/ESRGAN entre chunks   ← 7.5s idle por chunk
  T6  → Cadena de audio natural            ← audio aplastado
  T11 → Batch sizes mayores                ← VRAM desperdiciada

Prioridad 3 (medio impacto):
  T7  → Afinidad CPU por CCD               ← ya confirmado ~5% mejor
  T9  → Dual NVENC                          ← GPU1 NVENC idle
  T10 → Resolver PCIe x4                    ← posible límite HW
  T8  → Chunk size óptimo                   ← sin datos de 20s/30s

Prioridad 4 (bajo impacto / condicional):
  T2  → Pipeline D2H→NVENC sin numpy        ← depende de T1
  T5  → Face-adaptive blending              ← depende de T4
  T12 → NVDEC                               ← extract no es cuello
  T13 → RIFE_THREADS                        ← tuning fino
  T14 → Overhead del wrapper RIFE           ← tuning fino

Prioridad 5 (cambio de arquitectura):
  T15 → Backend RIFE sin PNGs               ← si T1-T14 no bastan
  T16 → Pipeline GPU-resident               ← si T1 no basta
```

---

## Criterio de Aceptación Final

Los checks de abajo corresponden al último run estable con perfil anime/anterior. `quality/real_x2plus` no ha revalidado todavía este gate: `GPU0_BATCH=16` hace `OOM` y `GPU0_BATCH=4` sigue sin completar el slice de `60s` en tiempo aceptable.

- [x] throughput ≥ 0.40× realtime sostenido en validación de 5 min ✓ (0.4198×)
- [x] effective_fps ≥ 20.0 ✓ (24.0 fps)
- [x] promedio por chunk ≤ 37.5s ✓ (bench_sustain300)
- [x] GPU0 con uso alto y sostenido (sin huecos >5s de idle) ✓ (70.8% avg)
- [x] GPU1 con uso alto y sostenido en RIFE ✓ (62.3% avg; NVENC dual no es necesario para pasar el gate actual)
- [x] CPU sin huecos evitables de alimentación ✓ (30.9% avg con CCD split)
- [x] Swap plana o bajando en validación de 5 min ✓
- [x] Sin procesos zombie al finalizar ✓ (bench_sustain300 sin zombies)
- [x] Salida válida 4480×2520 @ 50fps ✓ (verificado en video 1 existente)
- [ ] Rostros sin sobre-suavizado / look plástico → **EN PRODUCCIÓN** con real_x2plus + face_adaptive
- [ ] Texto y fluidez al nivel del video 1 o mejor → **EN PRODUCCIÓN**
- [x] Audio perceptualmente más natural que cadena actual ✓ (perfil `natural` sin dynaudnorm)

---

## Errores que NO se Deben Repetir

### Rendimiento
- No tratar ESRGAN como cuello sin evidencia nueva (el cuello es D2H, no compute)
- No reabrir CPU ESRGAN (destruye -29% rendimiento GPU)
- No reabrir `torch.compile` como primera línea (el bottleneck no es kernel overhead)
- No perseguir NVDEC sin que CPU sea cuello real
- No mezclar hipótesis en una sola corrida

### Calidad
- No aceptar modelo anime como salida final para rostros humanos
- No usar sharpen agresivo para "corregir" piel plástica
- No aceptar cadena de audio que aplaste dinámica natural
- No priorizar limpieza extrema sobre naturalidad

### Hardware
- No fingir que RT Cores deben entrar en el plan
- No ignorar la limitación PCIe x4 de GPU1
- No llamar "uso total" a un diseño que deja idle NVENC, Tensor y topología L3
