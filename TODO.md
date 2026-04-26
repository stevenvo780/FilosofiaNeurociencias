# TODO

## Hecho

- [x] Dejar operativo `enhance.sh` para video con resolución ×2 y FPS ×2.
- [x] Generar `*_audio_mejorado.wav` como salida de audio preferida.
- [x] Generar subtítulos `*.es.srt` y `*.en.srt` por charla.
- [x] Generar `*_final.mkv` desde `scripts/process_charlas_gpu.sh`.
- [x] Documentar que el flujo final de audio a conservar es `audio_mejorado` y no multi-etapa.
- [x] Documentar el proceso completo en `README.md` y `PROGRESO.md`.

## Pendiente opcional

- [ ] Regenerar cualquier `*_final.mkv` que haya sido borrado manualmente.
- [ ] Si se necesita, producir una versión por charla con video x2 (`4K50`) + `audio_mejorado` + subtítulos.
- [ ] Limpiar `work/`, `logs/` y salidas históricas sólo después de confirmar respaldo de los archivos finales útiles.
