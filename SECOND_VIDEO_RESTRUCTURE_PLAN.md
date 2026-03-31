# Plan Maestro de Optimización y Calidad del Segundo Video

Fecha: 2026-03-31

## 1. Objetivo

Entregar el segundo video con estas condiciones simultáneas:

- throughput `>= 0.40x realtime`
- uso útil máximo del hardware disponible
- calidad visual superior al video 1 en rostros
- calidad de audio superior a la cadena actual
- estabilidad total del pipeline en corridas largas

El plan está orientado a dos metas inseparables:

1. reducir al máximo el tiempo de espera usando la PC de forma útil
2. mejorar la calidad final sin introducir artefactos nuevos

## 2. Baseline Actual Medida

### 2.1 Configuración estable actual

- `RIFE` en GPU1
- `ESRGAN + NVENC` en GPU0
- `chunk = 15s`
- `ENHANCE_RIFE_STREAM_WINDOW=192`
- `ENHANCE_RIFE_MIN_WINDOW=64`
- `ENHANCE_MAX_EXTRACT_BYTES_IN_FLIGHT=6 GiB`
- `ENHANCE_MAX_RIFE_READY_BYTES=3 GiB`
- `ENHANCE_MAX_ESRGAN_READY_FRAMES=192`
- `ENHANCE_MAX_NVENC_BUFFERED_FRAMES=8`
- `ENHANCE_RIFE_THREADS=1:8:4`
- `ENHANCE_ESRGAN_PINNED_STAGING=0`

### 2.2 Resultado real del bench largo

Bench validado de `5 min`:

- salida válida: `4480x2520 @ 50 fps`
- `20` chunks procesados
- throughput global: `0.38x realtime`
- promedio por chunk: `39.645s`
- `effective_fps` promedio: `18.922`
- `RIFE` promedio: `39.081s`
- `RIFE fps` promedio: `19.195`
- `ESRGAN` promedio: `29.007s`
- `readback` promedio: `2.069s`
- `encode` promedio: `3.136s`
- `window_avg_frames`: `123.438`
- `window_max_frames`: `188.35`
- `extract_peak_bytes`: `~6.19 GiB`
- `rife_ready_peak_bytes`: `~1.59 GiB`

### 2.3 Brecha exacta contra la meta

Meta:

- `>= 0.40x realtime`

Con chunks de `15s`, eso implica:

- tiempo máximo medio por chunk: `37.5s`

Estado actual:

- tiempo medio observado: `39.645s`

Brecha:

- faltan `2.145s` por chunk
- faltan `~5.41%` de mejora total
- faltan `~1.078 fps` efectivos

## 3. Problemas Pendientes

## 3.1 Rendimiento

El cuello real actual es `RIFE`.

La evidencia es directa:

- `RIFE ~= 39.081s`
- `total ~= 39.645s`

Eso significa que el chunk termina prácticamente cuando termina `RIFE`.

El coste real del camino `RIFE` incluye:

- escritura de entrada a PNG
- spawn del binario `rife-ncnn-vulkan`
- cómputo de interpolación
- polling y drenaje de salida
- lectura de PNGs listos
- cleanup del chunk

## 3.2 Calidad visual

Hallazgos perceptuales ya confirmados en el video 1:

- textos y contornos quedaron mejor
- la fluidez mejoró claramente
- los rostros se sobre suavizan
- la piel queda plástica o maquillada
- se pierde microdetalle en ojos, boca, barba fina y contorno facial
- todavía hay aberraciones puntuales

Hipótesis técnicas principales:

- el modelo actual está sesgado a `anime`
- el `downscale 0.5x` previo a ESRGAN castiga demasiado el detalle facial

## 3.3 Calidad de audio

Hallazgos actuales:

- el audio ya mejora respecto al original
- todavía suena demasiado procesado
- la cadena puede estar aplanando timbre y dinámica

Hipótesis técnica principal:

- `afftdn + loudnorm + dynaudnorm` está priorizando limpieza y control por encima de naturalidad

## 4. Restricciones de Diseño

Estas reglas quedan fijas salvo evidencia nueva muy fuerte:

- no lanzar dos renders largos en paralelo
- no mezclar chunks viejos con arquitectura nueva
- no volver a tomar `ESRGAN` como cuello principal sin evidencia
- no activar `ENHANCE_ESRGAN_PINNED_STAGING=1` como default
- no usar sharpening agresivo como remedio principal
- no sacrificar calidad facial o audio para ganar pocos puntos de throughput

## 5. Requisitos No Negociables

## 5.1 Rendimiento

- uso alto y sostenido de GPU0, GPU1 y CPU sin huecos evitables
- backlog útil, no RAM muerta
- swap estable o bajando
- sin zombies al final

## 5.2 Calidad visual

- mantener la mejora ya lograda en textos, bordes y fluidez
- eliminar el look plástico de rostros
- conservar o mejorar detalle facial real
- no introducir halos notorios

## 5.3 Calidad de audio

- menos ruido
- mayor inteligibilidad
- timbre natural
- dinámica razonable
- evitar sensación de audio demasiado aplastado

## 6. Líneas de Trabajo

## 6.1 Línea A. Calidad base visual y sonora

Objetivo:

- fijar un baseline de calidad aceptable antes del render largo final

Trabajo visual:

1. reevaluar el modelo de upscale para material real
2. reevaluar el `downscale 0.5x` previo a ESRGAN
3. añadir preservación de detalle del original
4. aplicar sharpen solo si sigue siendo necesario y siempre muy suave

Trabajo de audio:

1. comparar la cadena actual con una variante menos aplanadora
2. reevaluar si `dynaudnorm` debe seguir como default
3. ajustar `loudnorm` de forma más conservadora si mejora naturalidad
4. validar con muestras cortas A/B

Criterio de salida de esta línea:

- rostros dejan de verse maquillados
- texto y fluidez no empeoran
- audio más natural que la cadena actual

## 6.2 Línea B. Amortización del coste fijo de `RIFE`

Objetivo:

- quitar la mayor cantidad posible del `~5.41%` faltante sin cambiar backend

Hipótesis principal:

- el tamaño de chunk puede estar penalizando la amortización del binario externo

Experimentos obligatorios:

1. control con `15s`
2. bench con `20s`
3. bench con `30s`

Métricas obligatorias:

- `total_seconds`
- `effective_fps`
- `rife_seconds`
- `extract_peak_bytes`
- swap
- estabilidad de salida

Condición para elegir una variante:

- mejora real en `60s`
- confirmación en `5 min`

## 6.3 Línea C. Afinado disciplinado de `RIFE_THREADS`

Baseline:

- `1:8:4`

Matriz inicial:

1. `1:8:4`
2. `1:10:4`
3. `1:12:4`
4. `1:8:6`
5. `1:10:6`

Regla:

- esta matriz se corre solo con el mejor `chunk size` de la línea B
- no se mezcla con otros cambios

Condición para promover una variante:

- mejora real de `effective_fps`
- confirmación en bench de `5 min`

## 6.4 Línea D. Reducción del overhead del wrapper de `RIFE`

Objetivo:

- sacar `0.5s - 1.5s` por chunk
- reducir tiempos muertos de CPU y filesystem alrededor de `RIFE`

Trabajo:

1. polling incremental en vez de reescanear patrones completos
2. reutilización de directorios temporales por chunk
3. cleanup diferido y más barato
4. pre-creación de estructura temporal

## 6.5 Línea E. Telemetría fina del coste de `RIFE`

Objetivo:

- separar cómputo real de overhead fijo

Métricas nuevas requeridas:

- `rife_spawn_seconds`
- `rife_compute_seconds`
- `rife_drain_seconds`
- `rife_cleanup_seconds`

Esta línea debe permitir responder con evidencia:

- cuánto tarda el modelo
- cuánto tarda el contrato por chunk

## 6.6 Línea F. Ruta de backend nuevo para `RIFE`

Esta línea solo se abre si las líneas B, C, D y E no bastan.

Objetivo:

- eliminar el ABI por PNG y el spawn por chunk

Opciones a explorar:

1. `RIFE` en PyTorch
2. `RIFE` en TensorRT
3. worker persistente de `RIFE`

Condición de apertura:

- no alcanzar `>= 0.40x` tras validar las líneas anteriores

## 7. Secuencia Exacta de Ejecución

### Fase 0. Calidad base

1. generar muestras cortas de rostros y voz
2. ajustar modelo, preservación de detalle y cadena de audio
3. congelar un baseline de calidad

### Fase 1. Chunk size

1. bench `15s`
2. bench `20s`
3. bench `30s`
4. elegir ganador

### Fase 2. Threads de `RIFE`

1. correr matriz sobre el chunk size ganador
2. seleccionar finalista

### Fase 3. Wrapper de `RIFE`

1. añadir telemetría fina
2. optimizar polling
3. optimizar cleanup
4. optimizar directorios temporales

### Fase 4. Validación larga

1. bench `60s`
2. bench `5 min`
3. validar memoria, swap y procesos

### Fase 5. Decisión

1. si cumple, relanzar el segundo video con `--clean`
2. si no cumple, abrir backend nuevo de `RIFE`

## 8. Criterio de Aceptación

El plan se considera exitoso solo si se cumplen todas estas condiciones:

- `effective_fps >= 20.0`
- throughput `>= 0.40x realtime`
- promedio por chunk `<= 37.5s`
- uso alto y sostenido del hardware sin huecos evitables
- swap plana o bajando durante `5 min`
- sin `python3`, `ffmpeg` ni `rife-ncnn-vulkan` zombies al final
- salida válida `4480x2520 @ 50 fps`
- rostros sin sobre difuminado evidente
- audio perceptualmente mejor que la cadena actual

## 9. Qué No Hacer

- no volver a tocar `torch.compile`
- no volver a tocar `TF32`
- no volver a perseguir `ESRGAN` como cuello principal sin evidencia nueva
- no activar por defecto la ruta experimental de `pinned staging`
- no meter helper mode de la `2060` para `ESRGAN`
- no saltar a TensorRT sin agotar primero chunk size, threads y wrapper

## 10. Errores Ya Medidos que No se Deben Repetir

Estos puntos deben conservarse como memoria operativa del plan.

### 10.1 Rendimiento

1. Tratar `ESRGAN` como cuello principal después de la reestructuración.
   - Los benches largos ya mostraron que el tiempo total sigue a `RIFE`.

2. Activar `ENHANCE_ESRGAN_PINNED_STAGING=1` como camino por defecto.
   - La ruta existe, pero ya empeoró el throughput real en esta máquina.

3. Subir `ENHANCE_RIFE_MIN_WINDOW` a `128` como baseline.
   - Ya rindió peor que `64`.

4. Volver a controlar el pipeline por `Queue(maxsize=...)` como mecanismo principal.
   - El control real debe seguir siendo por bytes y frames en vuelo.

5. Lanzar dos renders largos en paralelo.
   - Compite por recursos, degrada estabilidad y contamina la lectura de cuellos reales.

6. Reutilizar chunks viejos del segundo video con la arquitectura nueva.
   - El relanzamiento productivo debe seguir siendo con `--clean`.

7. Abrir múltiples hipótesis en la misma prueba.
   - Cada bench debe mover una sola variable.

### 10.2 Calidad visual

1. Dar por buena la calidad facial del video 1.
   - Ya quedó demostrado que los rostros salen demasiado difuminados.

2. Usar sharpening agresivo como primer remedio.
   - Tiende a crear halos y no recupera microdetalle real.

3. Optimizar solo textos y fluidez ignorando caras.
   - El siguiente render debe mejorar ambas cosas a la vez.

4. Asumir que el modelo actual es automáticamente correcto para rostros humanos.
   - Esa suposición ya quedó cuestionada por el material real.

### 10.3 Audio

1. Dar por cerrada la cadena actual de audio.
   - Ya se observó que sigue siendo mejorable.

2. Priorizar limpieza extrema sobre naturalidad.
   - El audio final no debe quedar artificialmente aplanado.

## 11. Pregunta Operativa Central

Todo el trabajo restante debe responder a esto:

- cómo sacar `~2.15s` por chunk del camino real de `RIFE`
- cómo llenar mejor la PC con trabajo útil
- cómo hacerlo sin volver a degradar rostros ni audio
