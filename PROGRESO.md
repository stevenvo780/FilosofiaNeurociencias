# Progreso del proyecto

Resumen corto y honesto del trabajo hecho para no repetir pruebas que ya quedaron resueltas.

## Resultado final que sí se conserva

El audio final que se considera correcto es:

- `*_audio_mejorado.wav`

Los subtítulos finales son:

- `*.es.srt`
- `*.en.srt`

Y el contenedor final esperado por charla es:

- `*_final.mkv`

## Qué se intentó

### 1. Mejora visual del video

Se dejó armado `enhance.sh` para:

- duplicar resolución,
- duplicar FPS,
- trabajar por chunks,
- repartir la carga entre dos GPU,
- usar Video2X con Real-ESRGAN + RIFE.

Ese flujo sí quedó útil para la parte visual.

### 2. Varias rutas de mejora de audio con IA

Se probaron varias alternativas:

- DeepFilterNet directo,
- chunking paralelo,
- rutas con Resemble,
- ruta multi-etapa sin IA + IA,
- normalizaciones conservadoras después del filtrado.

Varias de esas pruebas fueron técnicamente interesantes, pero no todas dejaron un audio mejor en la práctica.

### 3. Problema importante detectado

En una fase apareció un problema serio de sobresaturación / degradación del audio.

Conclusión práctica:

- no basta con que una IA “limpie más”; si el resultado introduce dureza, bombeo, artefactos o fatiga, no sirve como salida final.

### 4. Decisión final sobre el audio

El flujo que mejor resultado dio fue el conservador:

- `AUDIO_MODE=safe`

Implementado en:

- `scripts/process_charlas_gpu.sh`

Ese flujo genera:

- `*_audio_mejorado.wav`

y fue el que se tomó como referencia final.

## Qué quedó descartado como salida principal

Estos nombres pueden aparecer en logs, scripts o historiales, pero no son la salida canónica final:

- `*_audio_ia_deepfilter.wav`
- `*_audio_ia_multietapa.wav`

No significa que estén “mal programados”; significa que **no fueron los preferidos como resultado final real**.

## Cómo se cerró la parte de subtítulos

La generación de subtítulos quedó resuelta con:

- `whisper.cpp` sobre CUDA para transcripción base,
- detección de idioma,
- traducción según el caso,
- exportación a:
  - `*.es.srt`
  - `*.en.srt`

El flujo principal quedó integrado en:

- `scripts/process_charlas_gpu.sh`

## Estructura final esperada

Por cada charla en `output/charlaN/`:

```text
charlaN.mp4
charlaN_audio_mejorado.wav
charlaN.es.srt
charlaN.en.srt
charlaN_final.mkv
```

Además, para la parte visual general puede existir un máster con doble calidad, por ejemplo:

```text
output/GMT20260320-130023_Recording_2240x1260_4K50.mkv
```

## Decisiones para no perder tiempo después

### Repetir el audio

Si hay que regenerar el audio, empezar por:

- `AUDIO_MODE=safe`

### Repetir el video

Si hay que regenerar la mejora visual, usar:

- `enhance.sh`

### Repetir subtítulos y final

Usar:

- `scripts/process_charlas_gpu.sh`

### No reabrir estas pruebas salvo necesidad real

Evitar volver a invertir tiempo por defecto en:

- multi-etapa IA,
- DeepFilterNet como salida final principal,
- rutas agresivas de denoise/enhancement.

Si se reabren, que sea como investigación aparte, no como flujo base de producción.

## Resumen ejecutivo

La versión corta es esta:

- **video**: `enhance.sh`
- **audio final bueno**: `*_audio_mejorado.wav`
- **subtítulos**: `*.es.srt` y `*.en.srt`
- **script central para charlas**: `scripts/process_charlas_gpu.sh`
- **flujos IA alternativos**: útiles como pruebas, no como salida final por defecto