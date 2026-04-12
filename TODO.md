# TODO — Optimización de Hardware y Calidad

> **Video**: `GMT20260320-130023_Recording_2240x1260.mp4` (~7.4h, grabación Zoom)
> **Meta**: ≥ 0.40× realtime sostenido, máxima calidad visual y de audio
> **Fecha**: 2026-04-01

---

## Estado de Tareas

### Completadas

- [x] **T1. Async D2H + double-buffering** — `esrgan.py`: 3 CUDA streams, pinned memory, `non_blocking=True`. Validado 0.42× realtime.
- [x] **T2. Hot path tensor pinned → writer** — Evita copias numpy en CPU. Pipeline GPU-resident (T16) queda como futuro.
- [x] **T3. Overlap RIFE/ESRGAN entre chunks** — Pipeline 4-stage con `BudgetController` y `PIPELINE_DEPTH`. ESRGAN=GPU0, RIFE=GPU1 con prefetch corregido.
- [x] **T4. Modelo para rostros humanos** — `real_x2plus` y `real_x4plus` registrados con auto-download + SHA256. Perfiles `quality`/`production` usan `real_x2plus`.
- [x] **T5. Face-adaptive blending** — `face_adaptive=True` en perfiles `quality`/`production`. Integrado en `_consume_output()`.
- [x] **T6. Cadena de audio natural** — Perfiles `natural`/`production` sin `dynaudnorm`, con `alimiter`. A/B bench ejecutado. Perfil `natural` seleccionado.
- [x] **T7. Afinidad CPU por CCD** — Scheduler CCD-aware con `taskset`, `ionice`, `chrt`. Validado ~5% mejor.
- [x] **T8. Chunk size óptimo** — `chunk=30` es el mejor setting. Incluido en perfil `production`.
- [x] **T9. NVENC dual** — Soportado pero no necesario; NVENC no es cuello de botella.
- [x] **T10. PCIe x4 de GPU1** — Confirmado como límite HW. No resoluble por software.
- [x] **T12. NVDEC toggle** — Implementado (`ENABLE_NVDEC`). Extract no es cuello.
- [x] **T13. RIFE_THREADS** — `1:8:4` validado como óptimo.
- [x] **T14. Overhead wrapper RIFE** — Reescaneo optimizado, streaming window y polling afinados.

### Pendientes

- [ ] **T11. Batch sizes para quality/real_x2plus** — `GPU0_BATCH=16` hace OOM a resolución completa. Default actual: `GPU0_BATCH=4`. Pendiente retunar batch óptimo seguro.
- [ ] **T15. Backend RIFE torch** — IFNet oficial ya funciona en `rife_backend.py` con auto-descarga `paper_v6`. Falta benchmark de throughput/calidad frente a ncnn-Vulkan antes de hacerlo default.
- [ ] **T16. Pipeline GPU-resident** — Experimental. `ENHANCE_ESRGAN_GPU_RESIDENT` existe pero falta camino end-to-end sin D2H.

---

## Producción Recomendada

```bash
bash scripts/process_production.sh
```

Configuración por defecto:
- **Visual**: `quality` (real_x2plus + face_adaptive)
- **Audio**: `natural` (afftdn + loudnorm + alimiter)
- **Scheduler**: `production` (CCD split + ionice + chrt)
- **RIFE**: `baseline` (ncnn-Vulkan)
- **Chunk**: 30s
- **ESRGAN**: solo GPU0, batch=4
- **RIFE**: GPU1

---

## Mejor Benchmark Sostenido

```
bench_sustain300_chunk30_safe:
  throughput    = 0.4198× realtime (300s)
  effective_fps = 24.0
  GPU0          = 70.8% avg
  GPU1          = 62.3% avg
  CPU           = 30.9% avg
  zombies       = 0
```

---

## Criterio de Aceptación

- [x] throughput ≥ 0.40× realtime sostenido (5 min) ✓
- [x] effective_fps ≥ 20.0 ✓
- [x] promedio por chunk ≤ 37.5s ✓
- [x] Sin procesos zombie ✓
- [x] Salida 4480×2520 @ 50fps válida ✓
- [x] Audio natural sin dynaudnorm ✓
- [ ] Rostros sin sobre-suavizado → en producción con real_x2plus + face_adaptive
- [ ] Gate completo para quality/real_x2plus (T11)

---

## Errores que NO se Deben Repetir

- No tratar ESRGAN como cuello sin evidencia (el cuello es D2H, no compute)
- No reabrir CPU ESRGAN (-29% rendimiento GPU)
- No reabrir `torch.compile` como primera línea
- No aceptar modelo anime para rostros humanos
- No usar dynaudnorm para audio (aplasta dinámica)
- No ignorar PCIe x4 de GPU1
- No mezclar hipótesis en una sola corrida
