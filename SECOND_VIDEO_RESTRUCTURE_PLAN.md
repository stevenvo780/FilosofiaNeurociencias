# Plan Maestro de Calidad y Aprovechamiento Total del Hardware

Fecha: 2026-03-31

## 1. Criterio Rector

El objetivo no es llenar la máquina con carga artificial.

El objetivo es este, en este orden:

1. obtener la mejor calidad final que esta PC pueda sostener para este material
2. llenar con trabajo útil todos los motores de hardware que sí aporten a esa calidad
3. eliminar esperas evitables sin introducir artefactos, inestabilidad ni sobreprocesado

Este plan considera éxito solo el uso útil máximo del hardware relevante.

No se considera éxito:

- subir ocupación a costa de peor calidad facial o peor audio
- generar CPU spin o RAM muerta solo para “ver uso”
- perseguir bloques que no aportan al problema, como `RT cores`

## 1.1 Resumen Ejecutivo del Estudio

Este es el resultado directo del estudio actual del pipeline.

| Área | Estado actual | Conclusión dura |
|---|---|---|
| calidad visual | texto y fluidez mejoraron, caras no | la calidad facial actual no es aceptable como salida final |
| calidad de audio | mejor que el original, pero procesada de más | la cadena actual de audio no está cerrada |
| throughput | `0.38x realtime` | todavía falta para la meta mínima |
| GPU0 | sí trabaja fuerte, pero no sostenida todo el tiempo | está subalimentada por la arquitectura del pipeline |
| GPU1 | solo se aprovecha de verdad por Vulkan para `RIFE` | la tarjeta sigue lejos de estar explotada como tarjeta completa |
| CPU | `~40.1%` de uso medio | hay margen útil grande en afinidad, caché y scheduling |
| I/O de bloque | muy bajo | el disco no es el cuello principal |
| I/O en memoria / PNG / pipes | alto impacto | sí forma parte del cuello real |
| PCIe GPU1 | `x4` bajo carga | posible límite físico real |

## 1.2 Conclusiones Duras que Sí o Sí Debe Reflejar el Plan

1. El pipeline actual no usa “toda la PC”; usa bien GPU0 y usa parcialmente GPU1 y CPU.
2. El cuello actual no es el disco.
3. El cuello actual tampoco es `NVENC`.
4. El extractor real no es lento; se bloquea por backpressure downstream.
5. El contrato por PNG sigue siendo parte central del problema de uso y de espera.
6. La 2060 sigue demasiado desaprovechada:
   - no usa `NVENC`
   - no usa `NVDEC`
   - no usa su camino CUDA/Tensor en producción
7. La calidad actual todavía falla en dos puntos de primer nivel:
   - rostros demasiado suavizados
   - audio demasiado procesado
8. Si se quiere acercarse al uso útil máximo del equipo, el plan ya no puede ser solo de tuning de `RIFE`; tiene que cubrir:
   - calidad visual
   - calidad de audio
   - scheduler CPU
   - I/O en memoria
   - PCIe de GPU1
   - reaprovechamiento real de GPU1

## 1.3 Tabla Rápida de Uso Actual vs Meta Útil

| Componente | Uso actual | Meta útil | Brecha visible |
|---|---:|---:|---|
| GPU0 compute | `47.4%` promedio | alto y sostenido | grande |
| GPU0 NVENC | `4.2%` promedio | sostenido cuando encode toque | baja prioridad |
| GPU1 compute | `55.1%` promedio, solo vía Vulkan | aprovechar más motores útiles | grande |
| GPU1 NVENC | `0%` | abierto a uso si no daña `RIFE` | total |
| CPU | `40.1%` promedio | `50-70%` útil y bien repartido | media |
| CPU iowait | `~0.60%` | bajo | ya está bien |
| disco `/home` | `md127 ~1.09%` promedio | bajo | no perseguir |
| tmpfs `/tmp` | holgado | backlog útil, no memoria muerta | media |
| swap | estable en `18 GiB` | estable o menor | aceptable, pero mejorable |
| PCIe GPU1 | `x4` | idealmente `x16`, mínimo entendido | alta |

## 2. Baseline Operativo Válido

Las decisiones de este plan se apoyan en el pipeline actual instrumentado y en los logs ya generados del repo.

Baseline productivo actual confirmado:

- entrada: `2240x1260 @ 25 fps`
- salida: `4480x2520 @ 50 fps`
- `RIFE` en GPU1 vía `rife-ncnn-vulkan`
- `ESRGAN + NVENC` en GPU0
- `chunk = 15s`
- `RIFE_STREAM_WINDOW = 192`
- `RIFE_MIN_WINDOW = 64`
- control por budgets:
  - `MAX_EXTRACT_BYTES_IN_FLIGHT = 6 GiB`
  - `MAX_RIFE_READY_BYTES = 3 GiB`
  - `MAX_ESRGAN_READY_FRAMES = 192`
  - `MAX_NVENC_BUFFERED_FRAMES = 8`

Resultado largo de referencia ya validado:

- bench de `5 min`
- throughput global: `0.38x realtime`
- promedio por chunk: `39.645s`
- `effective_fps`: `18.922`
- brecha contra meta `0.40x`: `2.145s` por chunk

Restricciones de calidad ya observadas en el video 1:

- textos y bordes mejoraron
- la fluidez mejoró
- los rostros se sobre suavizan
- la piel queda plástica
- el audio mejora, pero sigue demasiado procesado

## 2.1 Corrida Instrumentada de Referencia

Corrida adicional ejecutada para este plan:

- slice: `60s -> 120s`
- duración del slice: `60s`
- audio: activo
- configuración: baseline productivo actual
- salida: válida
- logs crudos: `enhanced/logs/usage_bench_20260331_160111`

Esta corrida corta no sustituye el baseline largo de `5 min`.

Su propósito es otro:

- medir curvas de uso reales
- cuantificar brecha por recurso
- identificar si el límite viene de cómputo, espera o hardware físico

Archivos relevantes de esa corrida:

- `enhanced/logs/usage_bench_20260331_160111/pipeline.log`
- `enhanced/logs/usage_bench_20260331_160111/chunk_metrics.jsonl`
- `enhanced/logs/usage_bench_20260331_160111/gpu.csv`
- `enhanced/logs/usage_bench_20260331_160111/mpstat.log`
- `enhanced/logs/usage_bench_20260331_160111/pidstat.log`
- `enhanced/logs/usage_bench_20260331_160111/memory.log`
- `enhanced/logs/usage_bench_20260331_160111/iostat.log`

## 2.2 Tabla de Etapas Reales del Pipeline

| Métrica | Valor medido | Lectura correcta |
|---|---:|---|
| throughput global | `0.38x realtime` | Sigue por debajo de la meta |
| `total_seconds` promedio | `37.812s` | Faltan `0.312s` para `37.5s` en esta corrida corta |
| `effective_fps` promedio | `19.837` | Falta `0.163 fps` para `20.0` |
| `RIFE` promedio | `37.258s` | Sigue dominando el chunk |
| `RIFE fps` promedio | `20.132` | Muy cerca del mínimo operativo, pero aún no suficiente |
| `readback` promedio | `2.013s` | No es el cuello principal |
| `ESRGAN` promedio | `28.478s` | Trabajo pesado, pero ya solapado dentro del chunk |
| `encode` promedio | `2.354s` | No es el cuello principal |
| `window_avg_frames` | `131.25` | El streaming ya evita microventanas patológicas |
| `window_max_frames` | `192` | Se está usando el techo configurado |
| `extract_peak_bytes` | `5.91 GiB` | Budget alto pero controlado |
| `rife_ready_peak_bytes` | `1.51 GiB` | Backlog contenido |
| `nvenc_peak_frames` | `8` | El buffer de NVENC está saturando el máximo configurado |

## 2.3 Curva Real de Extract vs Backpressure

La corrida instrumentada dejó un hallazgo importante:

| Grupo de chunks | `extract_seconds` promedio | `extract_fps` promedio | Interpretación |
|---|---:|---:|---|
| chunks `0-1` | `4.329s` | `86.626 fps` | Coste real de extracción |
| chunks `2-3` | `37.583s` | `9.978 fps` | No es extract más lento; es extract bloqueado por backpressure |

Conclusión:

- el extractor no es el cuello de cómputo
- el extractor está esperando a downstream
- perseguir más optimización de decode antes de resolver ese downstream sería una mala prioridad

## 2.4 Tabla de Uso Real por Recurso y Brecha

En esta tabla se distinguen dos conceptos:

- `brecha literal a 100%`: cuánto falta para un uso bruto del medidor
- `brecha útil`: cuánto falta para el nivel de ocupación que sí aportaría valor real al pipeline

| Recurso | Uso real medido | Pico | Brecha literal a `100%` | Brecha útil | Lectura |
|---|---:|---:|---:|---:|---|
| GPU0 `utilization.gpu` | `47.4%` | `100%` | `52.6 pts` | `~35-45 pts` | GPU0 sí llega a tope, pero no de forma sostenida |
| GPU0 `utilization.encoder` | `4.2%` | `15%` | `95.8 pts` | baja prioridad | El encode existe, pero no domina el tiempo total |
| GPU0 `utilization.decoder` | `0%` | `0%` | `100 pts` | condicionada | `NVDEC` sigue fuera por decisión basada en evidencia |
| GPU1 `utilization.gpu` | `55.1%` | `100%` | `44.9 pts` | `~30-40 pts` | GPU1 trabaja por Vulkan, pero lejos de un uso total útil sostenido |
| GPU1 `utilization.encoder` | `0%` | `0%` | `100 pts` | abierta | `NVENC` de la 2060 sigue totalmente ocioso |
| GPU1 `utilization.decoder` | `0%` | `0%` | `100 pts` | abierta | `NVDEC` de GPU1 no participa |
| CPU global | `40.1%` | `72.1%` | `59.9 pts` | `~20-35 pts` | Hay bastante CPU libre, pero hoy no está bien orquestada |
| RAM total | `50-56 GiB / 123 GiB` | `56 GiB` | `~54-59%` libre | no perseguir | Llenar RAM no es un objetivo |
| tmpfs `/tmp` | `~12 GiB / 62 GiB` | `12 GiB` | `~80%` libre | no perseguir | Hay margen; no es cuello |
| swap | `18 GiB` estable | `18 GiB` | n/a | sin brecha crítica | No crece durante la corrida |
| GPU1 PCIe width | `x4 / x16 max` | `x4` | `75%` de ancho sin usar | alta | Posible cuello físico real |
| GPU1 PCIe gen | `avg 2.5 / max 3` | `3` | variable | media | El link sí sube bajo carga, pero el ancho sigue en `x4` |

## 2.5 Lecturas Críticas que el Plan Debe Respetar

1. GPU0 no está “mal” porque su pico no llegue siempre a `100%`; está subalimentada por el contrato downstream/upstream.
2. GPU1 no está realmente explotada como tarjeta completa; solo está explotada como worker Vulkan.
3. El CPU no está saturado, pero tampoco está afinado. Hay margen útil en afinidad, caché y scheduler.
4. El extractor real ya es razonablemente rápido; lo que se ve lento en los chunks tardíos es espera, no decode.
5. `NVENC` en GPU0 no es el cuello hoy.
6. `NVDEC` no debe reabrirse por intuición; solo si cambia el contrato o si el CPU pasa a ser cuello real.
7. La telemetría fina interna de `ESRGAN` todavía no es completamente fiable por subfase en el camino por defecto.
   - En la ruta actual de PyTorch, parte del trabajo asíncrono acaba colapsado en el tiempo de sincronización/D2H.
   - Por ahora, los tiempos de chunk completos sí son fiables; la división microinterna todavía no debe tomarse como verdad absoluta.

## 2.6 Resumen Actual del Uso del PC

Este es el resumen ejecutivo del estado actual, usando la corrida instrumentada y el baseline largo.

| Componente | Estado actual | Veredicto |
|---|---|---|
| GPU0 compute | útil pero irregular | falta sostener ocupación |
| GPU0 NVENC | activo | no es cuello principal |
| GPU0 NVDEC | apagado | correcto por ahora |
| GPU1 Vulkan | sí se usa para `RIFE` | bien, pero insuficiente para hablar de “uso total” |
| GPU1 CUDA/Tensor | apagado | brecha real del plan |
| GPU1 NVENC | apagado | brecha real del plan |
| CPU cores | uso medio `~40.1%` | hay margen útil sin explotar |
| CPU iowait | `~0.60%` promedio | el disco no está frenando al pipeline |
| RAM | holgada | no es cuello |
| tmpfs `/tmp` | holgado | no es cuello de capacidad |
| swap | estable | aceptable, pero no deseable para producción larga |
| I/O de bloque | bajo | no es el cuello principal |
| I/O en memoria / PNG / pipes | alto impacto | sí forma parte del problema real |
| PCIe GPU1 | `x4` | posible límite físico importante |
| calidad facial | insuficiente | bloqueo de calidad |
| calidad de audio | insuficiente | bloqueo de calidad |

## 2.7 Meta de Uso Objetivo por Componente

La meta no es `100%` literal en todo.

La meta es esta:

| Componente | Meta útil | Condición de éxito |
|---|---|---|
| GPU0 compute | alta y sostenida | sin huecos largos entre ventanas |
| GPU0 NVENC | sostenido cuando corresponda | sin `writer_wait` patológico |
| GPU1 | al menos una de estas dos rutas: `RIFE` más eficiente o `RIFE + NVENC` o backend nuevo con CUDA/Tensor | que deje de estar medio ociosa como tarjeta completa |
| CPU | `50-70%` útil y bien repartido | sin dejar esperando a GPU0/GPU1/audio |
| I/O de bloque | bajo | no perseguir uso por uso |
| I/O tmpfs / memoria | optimizado | menos churn de PNG y menos copias inútiles |
| PCIe GPU1 | idealmente `x16`, mínimo validado y entendido | sin ignorar un posible límite físico |
| RAM/swap | backlog útil con swap plana | sin memoria muerta |
| calidad visual | rostros naturales | no sacrificar caras por throughput |
| calidad de audio | voz natural e inteligible | no sacrificar naturalidad por limpieza agresiva |

Conclusión operativa:

- `disco al 100%` no es la meta
- `RAM al 100%` no es la meta
- `RT cores al 100%` no es la meta
- sí son meta: GPU0, GPU1, CPU, caches, tmpfs, pipes, PCIe y media engines útiles cuando aporten tiempo total o calidad

## 3. Estudio del Uso Real del Hardware

## 3.1 CPU y Memoria

Sistema confirmado:

- CPU: `Ryzen 9 9950X3D`, `16C/32T`
- L3: `128 MiB` en `2` grupos de caché
- governor: `performance`
- `taskset`, `perf`, `nsys`, `ionice`, `chrt`, `pidstat`, `iostat`, `mpstat`: disponibles
- `numactl`: no disponible
- `/tmp`: `tmpfs` de `62 GiB`
- swap activa:
  - `zram0 64 GiB`
  - `md127p3 128 GiB`

Uso real hoy:

- `ffmpeg` hace extracción por CPU
- el wrapper de `RIFE` consume CPU en escritura PNG, polling, lectura PNG y cleanup
- el audio también cae en CPU
- no existe afinidad por CCD/L3
- no existe política de prioridad de procesos
- no existe separación explícita entre hilos de extract, audio, polling y coordinación

Capacidad útil todavía no explotada:

- afinidad por CCD/L3 usando `taskset`
- priorización con `chrt` e `ionice`
- mejor uso del CPU para audio en paralelo y para alimentar mejor a GPU0/GPU1
- mejor disciplina de swap y de higiene de memoria antes de una corrida larga

## 3.2 GPU0 — RTX 5070 Ti

Estado confirmado:

- `PCIe Gen5 x16`
- `16 GiB` VRAM
- usada hoy para:
  - `ESRGAN` por CUDA
  - `Tensor cores` implícitos vía fp16 en PyTorch
  - `NVENC` del chunk actual

Uso real hoy:

- GPU0 sí hace el trabajo pesado de upscale
- `NVENC` sí se usa
- `NVDEC` no se usa
- no hay segunda sesión NVENC planificada en el flujo actual
- el pipeline productivo actual fija `ENHANCE_ESRGAN_GPUS=0`

Capacidad útil todavía no explotada:

- segundo motor de encode si el scheduler lo vuelve rentable
- `NVDEC` solo si cambia el contrato de extracción y deja de penalizar el camino `rawvideo`
- afinado de ocupación sostenida por mejor scheduler y mejor backlog útil

## 3.3 GPU1 — RTX 2060

Estado confirmado:

- `6 GiB` VRAM
- `PCIe width current = x4`
- `PCIe width max = x16`
- `PCIe gen current = 1` en reposo
- `PCIe gen max = 3`
- usada hoy para:
  - `RIFE` vía Vulkan

Uso real hoy:

- GPU1 no se usa por CUDA en producción
- sus `Tensor cores` no participan en el camino productivo actual
- su `NVENC` está ocioso
- su `NVDEC` está ocioso

Capacidad útil todavía no explotada:

- confirmar el link PCIe real bajo carga
- usar `NVENC` de GPU1 si no interfiere con `RIFE`
- usar `Tensor cores` de GPU1 solo si `RIFE` migra de `ncnn-vulkan` a backend Torch/TensorRT o si se abre helper scheduling posterior

Conclusión dura:

Hoy GPU1 está aprovechada solo en su camino Vulkan.

Si el objetivo es usar de verdad CUDA/Tensor/NVENC de la 2060, el plan tiene que contemplar una reestructuración posterior a `RIFE`, no solo tuning superficial.

## 3.4 Media Engines

Confirmado en la máquina:

- encoders disponibles:
  - `hevc_nvenc`
  - `h264_nvenc`
  - `av1_nvenc`
- decoders disponibles:
  - `h264_cuvid`
  - `hevc_cuvid`
  - `av1_cuvid`

Uso real hoy:

- encode: `hevc_nvenc` en GPU0
- decode: CPU
- audio: CPU

Conclusión:

- `NVENC` sí es parte del plan
- `NVDEC` no se reabre por defecto porque el propio código ya documenta que ese camino fue peor para esta extracción
- `AV1 NVENC` no es prioridad para esta corrida porque la meta principal es calidad + tiempo total, no bitrate mínimo

## 3.5 Tensor Cores y RT Cores

Estado real:

- Tensor cores de GPU0 sí participan indirectamente en `ESRGAN` fp16
- Tensor cores de GPU1 no participan en producción actual
- `RT cores` no tienen utilidad directa en este pipeline

Decisión:

- `RT cores` quedan explícitamente fuera del objetivo técnico
- no se pierde tiempo intentando “usarlos por usarlos”
- el uso de `Tensor cores` en GPU1 queda subordinado a un cambio de backend o a un scheduler nuevo

## 3.6 I/O de Bloque

Estado confirmado:

- fuente y repo: `/home` sobre `md127p2`
- `md127` vive sobre `raid0` de `nvme1n1 + nvme2n1 + nvme3n1`
- temporales del pipeline: `/tmp` en `tmpfs`

Métrica real del bench instrumentado:

- `md127 %util` promedio: `~1.09%`
- `md127 %util` pico: `6.0%`
- `CPU iowait` promedio: `~0.60%`
- `CPU iowait` pico: `2.68%`

Conclusión:

- el I/O de bloque no es el cuello principal
- no falta ancho de disco para esta versión del pipeline
- optimizar disco antes de tocar el contrato PNG/memoria sería mala prioridad

## 3.7 I/O en tmpfs, PNG y Pipes

Este sí es I/O importante para el pipeline.

No es I/O de disco, pero sí es I/O real:

- escritura de PNGs para alimentar `RIFE`
- lectura de PNGs listos desde `RIFE`
- escritura al pipe de `NVENC`
- copias CPU↔RAM↔GPU↔pipe

Evidencia indirecta del bench:

- el extractor real hace `~86.6 fps` cuando está libre
- luego cae a `~10 fps` por backpressure, no por decode
- `window_max_frames = 192`
- `nvenc_peak_frames = 8`
- `rife_ready_peak_bytes = 1.51 GiB`

Conclusión:

- el contrato por PNG sigue siendo una fuente central de espera
- el I/O en memoria y en pipes sí pertenece al cuello real
- esta es la razón por la que el plan insiste en reducir overhead del wrapper y, si hace falta, migrar `RIFE` a backend en memoria

## 3.8 I/O PCIe

Este también es I/O relevante.

Estado confirmado:

- GPU0: `Gen5 x16`
- GPU1: bajo carga sube hasta `Gen3`, pero se queda en `x4`

Conclusión:

- GPU1 no solo tiene una brecha de software
- también puede tener un límite físico de transferencia
- el plan debe tratar el bus de la 2060 como un factor real, no como nota al pie

## 4. Recursos que Sí se Usan y Recursos que Faltan

## 4.1 Ya aprovechados de forma útil

- CUDA + Tensor implícito en GPU0 para `ESRGAN`
- Vulkan compute en GPU1 para `RIFE`
- `NVENC` en GPU0
- CPU para extracción, wrapper y audio
- tmpfs para intermedios
- budgets de memoria en vuelo
- audio en paralelo desde el inicio

## 4.2 Parcialmente aprovechados

- CPU del `9950X3D`
- L3 por CCD
- VRAM total de GPU0
- ancho de banda de GPU1 por posible limitación PCIe
- `NVENC` total disponible del sistema

## 4.3 No aprovechados todavía

- `NVENC` de GPU1
- `NVDEC`
- `Tensor cores` de GPU1
- afinidad por CCD/L3
- scheduler de procesos del SO
- pipeline visual adaptativo para caras
- pipeline de audio orientado a naturalidad

## 5. Brecha Técnica Exacta

Meta mínima de rendimiento:

- throughput `>= 0.40x realtime`
- `effective_fps >= 20.0`
- promedio por chunk `<= 37.5s`

Estado actual:

- `39.645s` por chunk
- brecha: `2.145s`
- mejora necesaria: `~5.41%`

Interpretación correcta:

- la brecha es pequeña en porcentaje
- la brecha no justifica degradar calidad
- el cuello de botella sigue estando en el camino `RIFE + wrapper + contrato PNG`
- el siguiente salto de uso de hardware no sale de exprimir más `ESRGAN` a ciegas

## 6. Objetivos Medibles por Recurso

## 6.1 Calidad

No se permite relanzar el segundo video largo hasta cumplir esto:

- rostros sin look plástico evidente
- microdetalle mejor en ojos, boca, cejas y contorno facial
- texto y bordes no peor que el video 1
- audio más natural que la cadena actual
- sin halos notorios ni sharpen agresivo

## 6.2 Rendimiento

Objetivos de uso útil:

- GPU0:
  - compute alto y sostenido durante ventanas `ESRGAN`
  - `NVENC` activo sin generar cola excesiva ni `writer_wait` patológico
- GPU1:
  - `RIFE` sostenido sin esperas de input/output
  - si se abre `NVENC` en GPU1, solo si no baja `RIFE`
- CPU:
  - ocupación útil para extract, audio, I/O y coordinación
  - sin busy-wait evitable
  - sin mezclar cargas sensibles a caché al azar
- RAM/tmpfs:
  - backlog útil, no acumulación muerta
  - sin crecimiento de swap durante la validación larga

## 7. Decisiones Cerradas

Estas decisiones quedan cerradas salvo evidencia nueva fuerte y medible:

- `CPU ESRGAN` queda fuera
- `torch.compile` no vuelve a ser primera línea de trabajo
- `TF32` no vuelve a ser primera línea de trabajo
- `NVDEC` permanece desactivado para la extracción actual por `rawvideo`
- `HEVC NVENC` sigue siendo el codec productivo
- el modelo `anime` actual no se considera aceptable para rostros humanos como salida final
- `RT cores` quedan fuera del alcance técnico de este flujo

## 8. Plan de Trabajo

## 8.1 Fase 0. Higiene de Corrida y Observabilidad

Objetivo:

- que cada experimento mida trabajo real y no basura residual del sistema

Trabajo:

1. matar procesos viejos del pipeline antes de cada benchmark
2. limpiar directorios temporales y trabajo parcial
3. confirmar que no queden `python3`, `ffmpeg` ni `rife-ncnn-vulkan` zombies
4. si `zram` sigue alta en reposo, preferir arranque limpio antes de la corrida larga
5. levantar observabilidad estándar para cada bench:
   - `nvidia-smi dmon`
   - `pidstat`
   - `mpstat`
   - `iostat`
   - `perf stat`
6. reservar `nsys` para dos capturas cortas y representativas, no para cada prueba

Salida obligatoria:

- baseline limpia
- logs comparables
- sin ruido de sesiones previas

## 8.2 Fase 1. Verificación de Hardware Antes de Tocar Software

Objetivo:

- resolver si GPU1 está limitada por hardware o solo por software

Trabajo:

1. medir `pcie.link.gen.current` y `pcie.link.width.current` de GPU1 bajo carga real de `RIFE`
2. confirmar si la 2060 sube a `Gen3` y si sigue clavada en `x4`
3. si bajo carga sigue lejos de `Gen3 x16`, abrir línea de remediación física:
   - slot
   - BIOS
   - riser/cable
   - lane sharing

Regla:

- si GPU1 sigue realmente limitada por bus, ese problema sube a prioridad alta
- no tiene sentido diseñar un plan de “uso total de la torre” ignorando un estrangulamiento físico

## 8.3 Fase 2. Calidad Base Visual

Objetivo:

- fijar una ruta visual que supere al video 1 antes de perseguir más throughput

Problema confirmado:

- el pipeline actual usa solo `/tmp/realesr-animevideov3.pth`
- el camino actual hace `downscale 0.5x` antes del modelo

Trabajo obligatorio:

1. introducir al menos un modelo orientado a material real, idealmente `x2`
2. comparar tres familias de salida:
   - baseline actual
   - modelo real sin sesgo anime
   - mezcla híbrida con reinyección de detalle del original
3. reevaluar el `downscale 0.5x`
4. si los rostros siguen castigados, abrir camino adaptativo:
   - máscara facial suave
   - mezcla menos agresiva en piel
   - preservación selectiva de luma y detalle original

Decisión técnica:

- no se usará sharpen agresivo como solución principal
- si hace falta sharpen, será mínimo y posterior al rescate de detalle real

Clips obligatorios:

- rostro cercano
- rostro medio con movimiento
- texto fino / interfaz
- escena mixta con rostro + texto

Condición de cierre:

- la variante ganadora mejora caras sin perder el avance en texto y fluidez

## 8.4 Fase 3. Calidad Base de Audio

Objetivo:

- sustituir la cadena actual por una salida más natural, manteniendo inteligibilidad

Cadena actual:

- `afftdn + loudnorm + dynaudnorm`

Problema confirmado:

- limpia y nivela, pero aplanando demasiado el resultado

Trabajo obligatorio:

1. montar matriz A/B con cadenas que usen filtros ya disponibles localmente:
   - `anlmdn`
   - `dialoguenhance`
   - `speechnorm`
   - `deesser`
   - `alimiter`
   - `highpass`
   - `loudnorm`
2. probar al menos estas rutas:
   - conservadora: denoise leve + loudnorm + limiter
   - voz: denoise orientado a voz + dialog enhancement + normalización conservadora
   - naturalidad: quitar `dynaudnorm` y sustituirlo por control más suave
3. medir si `AUDIO_THREADS=24` realmente ayuda o solo mete contención
4. pinnear el proceso de audio a un CCD concreto para aprovechar CPU sin contaminar el resto

Condición de cierre:

- voz más natural
- sin bombeo evidente
- sin sensación de audio aplastado

## 8.5 Fase 4. Afinidad, Caché y Scheduler del Sistema

Objetivo:

- dejar de tratar el `9950X3D` como una masa uniforme de hilos

Problema confirmado:

- hay dos grupos L3 y hoy no se aprovechan
- no hay `taskset`
- no hay `ionice`
- no hay `chrt`

Trabajo:

1. introducir lanzamiento afinado por `taskset`
2. separar al menos dos dominios de CPU:
   - dominio A: extract, audio, ffmpeg
   - dominio B: coordinación Python, polling `RIFE`, merge y colas
3. medir qué grupo L3 deja mejores resultados para extract/audio
4. fijar prioridades:
   - `ionice` para reducir latencia de tmpfs/FS del wrapper
   - `chrt` solo si mejora estabilidad sin romper el escritorio
5. revisar si `PIPELINE_DEPTH=2` sigue siendo óptimo una vez afinada la afinidad

Regla:

- la meta no es saturar el CPU a 100%
- la meta es que el CPU no deje esperando a las GPU ni al audio

## 8.6 Fase 5. Amortización del Coste de `RIFE`

Objetivo:

- sacar la mejora faltante sin degradar calidad ni reabrir cuellos ya resueltos

Trabajo:

1. reabrir el tamaño de chunk:
   - `15s`
   - `20s`
   - `30s`
2. reabrir `RIFE_THREADS` de forma disciplinada sobre el chunk ganador
3. medir por separado:
   - `rife_spawn_seconds`
   - `rife_compute_seconds`
   - `rife_drain_seconds`
   - `rife_cleanup_seconds`
4. eliminar overhead del wrapper:
   - reducir reescaneos
   - evitar cleanup caro por chunk
   - reutilizar estructura temporal donde no rompa la reanudación
   - bajar el coste de polling y de drenaje de salida

Conclusión esperada:

- aquí debe salir la mayor parte del `~5.41%` que falta

## 8.7 Fase 6. Reaprovechamiento de GPU1 Más Allá de Vulkan

Objetivo:

- usar más partes útiles de la 2060 sin perjudicar la calidad ni el throughput total

Trabajo:

1. abrir prueba controlada de `NVENC` en GPU1 para chunks alternos o encode solapado
2. aceptar esa ruta solo si:
   - no baja `RIFE`
   - no añade jitter de pipeline
   - no degrada estabilidad
3. no abrir helper `ESRGAN` en GPU1 mientras `RIFE` siga viviendo ahí y sea el cuello
4. si el cuello sigue siendo `RIFE` y GPU1 aún tiene motores clave ociosos, abrir la ruta mayor:
   - migrar `RIFE` a backend en memoria
   - priorizar backend que sí use CUDA/Tensor en GPU1

Interpretación correcta:

- con `rife-ncnn-vulkan`, la 2060 no va a usar sus `Tensor cores`
- si el objetivo es exprimir realmente GPU1, el plan tiene que contemplar migración de backend

## 8.8 Fase 7. Media Engines

Objetivo:

- usar `NVENC` y `NVDEC` solo cuando aporten tiempo total real

Trabajo:

1. mantener `HEVC NVENC` como ruta principal
2. evaluar `NVENC` de GPU1 solo en pruebas separadas y con observabilidad
3. mantener `NVDEC` apagado mientras el camino `rawvideo` siga ganando
4. reabrir `NVDEC` únicamente si:
   - cambia el contrato de extracción
   - el CPU se vuelve cuello por audio + extract + wrapper
   - la medición real lo vuelve favorable

## 8.9 Fase 8. Ruta de Backend Nuevo para `RIFE`

Esta fase solo se abre si tras las fases anteriores no se alcanza la meta.

Objetivo:

- eliminar el contrato por PNG
- usar más hardware útil de GPU1

Opciones:

1. `RIFE` en PyTorch
2. `RIFE` en TensorRT
3. proceso persistente en memoria

Razón:

- es la única ruta que puede convertir GPU1 en algo más que un worker Vulkan con filesystem intermedio
- es también la ruta que abre el uso real de `Tensor cores` de GPU1

## 9. Orden Exacto de Ejecución

1. higiene de corrida y baseline observada
2. verificación PCIe real de GPU1 bajo carga
3. baseline de calidad visual
4. baseline de calidad de audio
5. afinidad CPU / scheduler / backlog
6. chunk size
7. `RIFE_THREADS`
8. reducción del overhead del wrapper
9. `NVENC` adicional en GPU1 si sigue siendo rentable
10. validación larga `60s`
11. validación larga `5 min`
12. si aún falta, ruta de backend nuevo para `RIFE`
13. solo después: render largo del segundo video con `--clean`

## 10. Observabilidad Oficial del Plan

Cada experimento importante debe dejar:

- log del pipeline
- `nvidia-smi dmon`
- `pidstat`
- `mpstat`
- `iostat`
- extracto de `perf stat`

Y, para dos pruebas cortas representativas:

- un perfil `nsys` del camino `ESRGAN`
- un perfil `nsys` o equivalente del camino `RIFE + wrapper`

Métricas mínimas por chunk:

- `total_seconds`
- `effective_fps`
- `extract_seconds`
- `rife_seconds`
- `rife_spawn_seconds`
- `rife_compute_seconds`
- `rife_drain_seconds`
- `rife_cleanup_seconds`
- `readback_seconds`
- `esrgan_seconds`
- `encode_seconds`
- `window_avg_frames`
- `window_max_frames`
- `extract_peak_bytes`
- `rife_ready_peak_bytes`
- `nvenc_peak_frames`

## 11. Criterio de Aceptación

No se relanza la corrida larga final hasta cumplir todo esto:

- throughput `>= 0.40x realtime`
- `effective_fps >= 20.0`
- promedio por chunk `<= 37.5s`
- GPU0 y GPU1 llenas con trabajo útil, no solo con picos aislados
- CPU sin huecos evitables de alimentación
- swap plana o bajando en `5 min`
- sin zombies al final
- salida válida `4480x2520 @ 50 fps`
- rostros sin sobre difuminado evidente
- texto y fluidez al nivel del video 1 o mejor
- audio perceptualmente superior a la cadena actual

Objetivo deseable, no mínimo:

- `0.45x realtime` si no exige renunciar a calidad facial o naturalidad del audio

## 12. Errores que No se Deben Repetir

## 12.1 Rendimiento

1. Volver a tratar `ESRGAN` como cuello principal sin evidencia nueva.
2. Reabrir `CPU ESRGAN`.
3. Reactivar por defecto la ruta experimental de `pinned staging`.
4. Reabrir `torch.compile` o `TF32` como primera línea de tuning.
5. Lanzar dos renders largos en paralelo.
6. Reutilizar chunks viejos cuando cambie la arquitectura.
7. Mezclar varias hipótesis en una sola corrida.
8. Perseguir `NVDEC` por intuición en el contrato actual.

## 12.2 Calidad visual

1. Dar por buena la calidad facial del video 1.
2. Seguir usando el modelo `anime` como salida final sin comparación seria.
3. Usar sharpen agresivo para “corregir” piel plástica.
4. Mejorar texto y fluidez ignorando caras.

## 12.3 Audio

1. Dar por cerrada la cadena `afftdn + loudnorm + dynaudnorm`.
2. Priorizar limpieza extrema sobre naturalidad.
3. Usar todos los hilos del CPU para audio sin medir contención real.

## 12.4 Hardware

1. Fingir que `RT cores` deben entrar en el plan aunque no aporten a este flujo.
2. Ignorar la posible limitación PCIe de GPU1.
3. Llamar “uso total del PC” a un diseño que deja ocioso `NVENC` de GPU1, `Tensor` de GPU1 y la topología L3 del CPU sin siquiera evaluarlos.

## 13. Resultado Esperado del Plan

Al cerrar este plan, el pipeline debe quedar con estas propiedades:

- calidad visual claramente mejor que la del video 1 en rostros
- audio más natural
- uso útil máximo de GPU0, GPU1, CPU, tmpfs y media engines relevantes
- tiempos muertos del wrapper de `RIFE` recortados
- criterio claro de qué bloques de hardware sí merecen trabajo y cuáles no
- una base suficientemente fuerte para decidir si basta con tuning o si toca migrar `RIFE` a backend nuevo
