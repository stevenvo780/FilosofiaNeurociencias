# Plan Real de Reestructuración para Acelerar el Segundo Video

Fecha: 2026-03-31

## Resumen Ejecutivo

El plan anterior era correcto en la intuición, pero insuficiente en profundidad.

El cuello real no es solo "`ESRGAN` va más lento que `RIFE`". El cuello real es más duro:

1. El pipeline sigue organizado por `chunk` completo.
2. `RIFE` sigue encapsulado como binario externo con ABI de carpetas PNG.
3. `ESRGAN` todavía hace demasiadas copias, reasignaciones y sincronizaciones por batch.
4. La memoria en vuelo no está modelada por bytes, solo por cantidad de elementos en cola.
5. La coordinación actual está hecha por etapas rígidas, no por recursos y backlog útil.

Conclusión:

- Si solo afinamos colas, batches o threads, la mejora será marginal.
- Si queremos de verdad exprimir la PC, hay que reestructurar la frontera entre `extract`, `RIFE`, `ESRGAN` y `NVENC`.

Este documento reemplaza el plan anterior y divide el trabajo en tres rutas:

- `Ruta conservadora`: cirugía fuerte sin cambiar el backend de `RIFE`
- `Ruta profunda`: streaming real intra-chunk con scheduler por recurso
- `Ruta máxima`: eliminar el ABI por PNG de `RIFE`

## Estado Real del Pipeline Actual

Archivos implicados:

- [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py)
- [enhance/rife.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/rife.py)
- [enhance/esrgan.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/esrgan.py)
- [enhance/ffmpeg_utils.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/ffmpeg_utils.py)

Métricas reales medidas en producción y slices:

- `Extract`: `~105-200 fps`
- `RIFE input write`: `~400-635 fps`
- `RIFE`: `~29-32 fps`
- `RIFE readback`: `~420-570 fps`
- `ESRGAN`: `~20.9-24.8 fps`
- throughput global estable reciente: `~0.32x realtime`

Lectura correcta:

- `RIFE` ya no es el cuello dominante.
- `ESRGAN` limita el throughput total.
- Pero el pipeline no está realmente "ocupando mal la GPU" por un único motivo.
- Está perdiendo tiempo en barreras de arquitectura, especialmente entre `RIFE` y `ESRGAN`.

## Autocrítica del Plan Anterior

El plan anterior fue flojo en estos puntos:

1. No trató como prioridad máxima el ABI de `RIFE` basado en carpetas PNG.
2. No cuantificó la memoria real en vuelo.
3. No separó "optimización del pipeline actual" de "reescritura estructural".
4. No identificó con suficiente dureza el doble y triple copiado del datapath.
5. Puso demasiado pronto la idea de usar la `RTX 2060` para ayudar a `ESRGAN`.
6. No definió presupuestos de backlog en bytes ni criterios de backpressure por recurso.

## Cuellos Exactos del Código

### 1. Barrera extract -> RIFE

Hoy se hace esto:

- `ffmpeg -> RAM` en [enhance/ffmpeg_utils.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/ffmpeg_utils.py#L86)
- `RAM -> PNG` en [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py#L277)

Eso significa:

- se materializa el chunk entero en RAM
- luego se vuelve a serializar el chunk entero a PNG
- luego `RIFE` vuelve a leerlo

Eso no es un detalle. Es una frontera de ABI costosa.

### 2. Barrera RIFE -> ESRGAN

Hoy `RIFE`:

- se ejecuta con `subprocess.run(...)` en [enhance/rife.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/rife.py#L14)
- no devuelve control hasta terminar todo el chunk

Y después recién:

- se listan los PNGs
- se releen todos a RAM en [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py#L295)
- se entrega el chunk completo a `ESRGAN`

Esa es la barrera más dañina del diseño actual.

### 3. Cuello interno de ESRGAN

Dentro de `ESRGAN`, por batch, hoy ocurre:

- `np.stack(...)` en [enhance/esrgan.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/esrgan.py#L126)
- copia H2D
- `interpolate(scale_factor=0.5)` en GPU
- inferencia
- salida D2H en [enhance/esrgan.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/esrgan.py#L142)
- `.numpy()`
- copia adicional en `_ReorderWriter.on_frame()` en [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py#L64)

Eso explica por qué la `5070 Ti` no permanece llena todo el tiempo aunque `ESRGAN` siga siendo el cuello.

### 4. Modelo de colas demasiado pobre

Hoy el pipeline usa:

- `Queue(maxsize=C.PIPELINE_DEPTH)` en [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py#L453)

Eso es insuficiente, porque un item de cola puede significar:

- un chunk de entrada `~2.96 GiB`
- un chunk post-RIFE `~5.91 GiB`
- o mucho más si se materializa upscale

No se puede gobernar memoria real con "cantidad de chunks".

## Modelo de Memoria que Debe Gobernar el Diseño

Para `2240x1260`, `25 fps`, `15s`:

- frame de entrada RGB: `8,467,200 bytes`
- chunk extraído de `375` frames: `~2.96 GiB`
- chunk post-RIFE de `749` frames: `~5.91 GiB`
- frame 2x RGB: `33,868,800 bytes`
- chunk 2x completo de `749` frames: `~23.63 GiB`

Conclusión:

- El programa nunca debe volver a materializar un chunk 2x completo en RAM.
- El backlog no se debe configurar por cantidad de chunks.
- El backlog se debe configurar por bytes y por frames.

## Objetivo Correcto

El objetivo no es "ver CPU, RAM y dos GPU al 100% todo el tiempo" como KPI aislado.

El objetivo correcto es:

- saturar de forma sostenida el recurso limitante
- mantener backlog útil en los demás recursos
- evitar burbujas entre etapas
- no gastar RAM en datos que todavía no se van a consumir

Dicho eso, sí hay margen real de mejora:

- más ocupación útil de la `5070 Ti`
- menos huecos en la `2060`
- más trabajo útil del CPU en decode, staging y prefetch

## Arquitectura Objetivo

### Principio 1. Chunk lógico, ventana operativa

Mantener:

- chunk lógico de `15s` para reanudación, merge, progreso y fallos

Cambiar a:

- ventana operativa de `64-96` frames interpolados

Eso permite:

- arrancar `ESRGAN` antes
- solapar `RIFE` y `ESRGAN` de verdad
- contener la memoria en vuelo

### Principio 2. Scheduler por recurso

El sistema no debe pensar en "etapas lineales".

Debe pensar en recursos:

- decode CPU
- PNG writer o extractor directo
- `RIFE` en `GPU1`
- readback de ventanas
- `ESRGAN` en `GPU0`
- `NVENC`
- audio

Cada recurso debe consumir la siguiente unidad lista según:

- disponibilidad del recurso
- backlog aguas abajo
- presupuesto de memoria

### Principio 3. Presupuesto explícito de memoria

Se deben introducir budgets configurables:

- `MAX_EXTRACT_BYTES_IN_FLIGHT`
- `MAX_RIFE_READY_BYTES`
- `MAX_ESRGAN_READY_FRAMES`
- `MAX_NVENC_BUFFERED_FRAMES`

No más `Queue(maxsize=2)` como mecanismo principal.

## Rutas de Reestructuración

## Ruta Conservadora

Objetivo:

- obtener mejora importante sin reemplazar `rife-ncnn-vulkan`

### Paso C1. Eliminar `extract -> RAM -> PNG`

Archivos:

- [enhance/ffmpeg_utils.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/ffmpeg_utils.py)
- [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py)

Cambio:

- para el camino con `RIFE`, dejar de usar `extract_frames_to_ram(...)`
- usar extracción directa a PNG para alimentar `RIFE`

Razón:

- hoy decodificamos a RAM para luego serializar el mismo chunk a PNG
- eso gasta CPU, RAM y tiempo sin valor añadido

Impacto esperado:

- menor presión de RAM
- menor churn de GC
- CPU mejor usada

### Paso C2. Convertir RIFE a ejecución observable

Archivos:

- [enhance/rife.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/rife.py)
- [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py)

Cambio:

- cambiar `subprocess.run(...)` por `subprocess.Popen(...)`
- exponer PID, estado y finalización
- permitir polling del directorio de salida

Razón:

- sin esto no existe streaming intra-chunk real

Impacto esperado:

- no mejora solo por sí mismo
- habilita la mejora importante del siguiente paso

### Paso C3. Liberar ventanas contiguas desde la salida de RIFE

Archivos:

- [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py)

Cambio:

- vigilar `rife_out`
- detectar secuencias contiguas ya completas
- leer solo ventanas listas
- enviar esas ventanas a `ESRGAN`
- no esperar el chunk entero

Regla:

- una ventana solo se libera si todos sus índices existen y son legibles

Impacto esperado:

- adelantar el arranque de `ESRGAN` por chunk
- reducir mucho las burbujas entre `GPU1` y `GPU0`

### Paso C4. Rehacer la gobernanza de memoria

Archivos:

- [enhance/config.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/config.py)
- [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py)

Cambio:

- introducir budgets por bytes y frames
- bloquear `extract` cuando el presupuesto de entrada esté lleno
- bloquear `readback` cuando el backlog de `ESRGAN` sea excesivo

Impacto esperado:

- menos swap
- menos picos de RAM
- pipeline más estable

## Ruta Profunda

Objetivo:

- reestructurar el pipeline actual sin cambiar aún de backend de interpolación

### Paso P1. Scheduler por estados y recursos

Archivos:

- [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py)

Estado mínimo por chunk:

- `decode_pending`
- `decode_ready`
- `rife_running`
- `rife_window_ready`
- `esrgan_running`
- `encode_running`
- `done`

Cambio:

- abandonar la coordinación principal por colas stage-to-stage
- introducir un scheduler central con colas internas por recurso

Razón:

- hoy las etapas se pasan sentinels y listas completas
- eso es frágil y no refleja el uso real de hardware

### Paso P2. Rehacer el datapath de ESRGAN

Archivos:

- [enhance/esrgan.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/esrgan.py)
- [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py)

Cambio:

- prealocar batch buffers CPU contiguos
- usar pinned staging real
- usar dos `CUDA streams`
- separar métricas de:
  - `stack/fill`
  - `H2D`
  - `downscale`
  - `infer`
  - `D2H`
  - `writer wait`

Razón:

- hoy no sabemos cuánto tiempo exacto se va en compute y cuánto en mover datos
- sin esa separación, optimizar `ESRGAN` es disparar a ciegas

Impacto esperado:

- mayor ocupación útil de la `5070 Ti`
- subida real de fps en el hot path

### Paso P3. Quitar la copia redundante del writer

Archivos:

- [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py)

Cambio:

- rediseñar `_ReorderWriter` para trabajar con ownership de buffer
- evitar `np.ascontiguousarray(...).copy()` por frame cuando el frame ya venga contiguo y transferido

Razón:

- ahora mismo se vuelve a copiar cada frame en [enhance/pipeline.py](/home/stev/Descargas/FilosofiaNeurociencias/enhance/pipeline.py#L67)

Impacto esperado:

- menor uso de memoria
- menor CPU por frame
- menor latencia hacia `NVENC`

## Ruta Máxima

Objetivo:

- eliminar el ABI por PNG de `RIFE`

### Paso M1. Sustituir el backend de RIFE

Opciones:

- backend `torch`
- backend `TensorRT`
- backend que acepte frames/tensores en memoria

Requisito:

- entrada en memoria
- salida en memoria
- control fino de batches o ventanas

Razón:

- mientras `RIFE` dependa de directorios PNG, la orquestación fina siempre estará limitada

Impacto esperado:

- desaparición de la barrera más costosa del pipeline
- mejor uso simultáneo de CPU, RAM y ambas GPU
- arquitectura mucho más limpia

### Paso M2. Pipeline totalmente en memoria

Flujo objetivo:

- `ffmpeg decode -> frame ring`
- `RIFE in-memory -> output ring`
- `ESRGAN in-memory -> NVENC pipe`

Persistencia:

- chunk logical metadata a disco
- no frames intermedios a disco salvo fallback o debug

Esta es la arquitectura que más se acerca a "usar la máquina de verdad".

## Qué NO Debe Volver a Hacerse

- no volver a materializar un chunk 2x completo en RAM
- no confiar en `Queue(maxsize=n)` como control principal de memoria
- no mezclar dos renders largos a la vez
- no usar helper mode de la `2060` antes de quitar barreras mayores
- no repetir `GPU0_BATCH=12`
- no reactivar por defecto `torch.compile`, `allow_tf32` o `cudnn.benchmark=True` en esta máquina

## Priorización Correcta

Orden correcto:

1. instrumentación fina y budgets
2. eliminar `extract -> RAM -> PNG`
3. convertir `RIFE` a `Popen`
4. liberar ventanas contiguas intra-chunk
5. scheduler por recurso
6. cirugía interna de `ESRGAN`
7. evaluar si todavía tiene sentido que la `2060` ayude a `ESRGAN`
8. decidir si se justifica cambiar de backend de `RIFE`

## Criterios de Aceptación por Etapa

### Aceptación mínima

- throughput `> 0.35x realtime`
- RAM estable, sin crecimiento progresivo
- sin colgados ni deadlocks

### Aceptación buena

- `RIFE >= 30 fps`
- `ESRGAN >= 26 fps`
- throughput `>= 0.40x realtime`

### Aceptación excelente

- `ESRGAN >= 28 fps`
- throughput `>= 0.45x realtime`
- GPU0 con ocupación alta y sin burbujas largas
- GPU1 con menos huecos visibles

## Qué Haría Justo Después de Terminar el Video 1

1. añadir telemetría por subfase y por bytes
2. reemplazar el camino `extract_frames_to_ram -> _write_pngs` por extracción directa a PNG cuando `RIFE` esté activo
3. convertir `RIFE` a proceso observable con `Popen`
4. implementar ventanas contiguas `64-96` frames
5. correr benchmark de `30s`
6. correr benchmark de `5 min`
7. solo después tocar el scheduler completo

## Veredicto

La mejora real ya no depende de "apretar más los numeritos".

Depende de resolver tres cosas:

1. la frontera por PNG de `RIFE`
2. la gobernanza de memoria por bytes
3. el exceso de copias en `ESRGAN`

Si solo afinamos parámetros, el segundo video seguirá siendo demasiado lento.
Si reestructuramos estos tres puntos, sí hay margen real para que el segundo video salga bastante más rápido y con una utilización mucho más seria del hardware.
