# TODO — Tareas Pendientes

> **Video**: `GMT20260320-130023_Recording_2240x1260.mp4` (~7.4h, grabación Zoom)
> **Meta**: ≥ 0.40× realtime sostenido, máxima calidad visual y de audio

---

## Pendientes

- [ ] **Batch sizes para quality/real_x2plus** — `GPU0_BATCH=16` hace OOM a resolución completa. Default actual: `GPU0_BATCH=4`. Retunar batch óptimo seguro.
- [ ] **Backend RIFE torch** — IFNet oficial funciona (`rife_backend.py`, auto-descarga `paper_v6`). Falta benchmark throughput/calidad vs ncnn-Vulkan antes de hacerlo default.
- [ ] **Ejecutar producción completa** — `bash scripts/process_production.sh` para las 7.4h.

---

## Decisiones Cerradas

| Decisión | Razón |
|---|---|
| CPU ESRGAN deshabilitado | Destruye rendimiento GPU (-29%) |
| RIFE a 1260p (no 4K) | 3.8× más rápido, sin pérdida de calidad |
| `torch.compile` off | Sin ganancia medible en este pipeline |
| GPU sharing off | OOM en RTX 2060 (6GB, PCIe x4) |
| Perfil audio `natural` | Sin dynaudnorm, voz más natural |
| tmpfs para intermedios | Elimina I/O de disco |

---

## Mejor Benchmark Sostenido

```
bench_sustain300_chunk30_safe:
  throughput    = 0.4198× realtime (300s)
  effective_fps = 24.0
  GPU0          = 70.8% avg
  GPU1          = 62.3% avg
  CPU           = 30.9% avg
```

---

## Errores que NO se Deben Repetir

- No tratar ESRGAN como cuello sin evidencia (el cuello es D2H, no compute)
- No reabrir CPU ESRGAN (-29% rendimiento GPU)
- No reabrir `torch.compile` como primera línea
- No aceptar modelo anime para rostros humanos
- No usar dynaudnorm para audio (aplasta dinámica)
- No ignorar PCIe x4 de GPU1
- No mezclar hipótesis en una sola corrida
