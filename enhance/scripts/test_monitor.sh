#!/bin/bash
echo "--- PREPARANDO ENTORNO ---"
killall -9 python3 ffmpeg rife-ncnn-vulkan 2>/dev/null
rm -f test_trim.mp4 run_test.log metrics.log
echo "--- CREANDO SLICE DE 15 SEGUNDOS (1 CHUNK FALSO) ---"
ffmpeg -v quiet -y -ss 00:00:00 -t 00:00:15 -i GMT20260320-130023_Recording_2240x1260.mp4 -c copy test_trim.mp4

echo "--- INICIANDO PROCESO PIPELINE ---"
python3 run.py test_trim.mp4 --clean > run_test.log 2>&1 &
PID=$!

echo "--- RECOLECTANDO METRICAS CADA 2 SEGUNDOS ---"
for i in {1..40}; do
  if ! kill -0 $PID 2>/dev/null; then
    break
  fi
  # Header
  echo -e "\n--- [SEGUNDO $((i*2))] ---" >> metrics.log
  
  # CPU & RAM del proceso python
  ps -p $PID -o %cpu,%mem,rss --no-headers >> metrics.log
  
  # GPU Load (5070 Ti y 2060)
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader >> metrics.log
  
  sleep 2
done

wait $PID
echo "--- PROCESO TERMINADO ---"
