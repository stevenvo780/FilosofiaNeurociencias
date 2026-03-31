#!/bin/bash
killall -9 python3 ffmpeg rife-ncnn-vulkan 2>/dev/null
rm -f metrics.log run_test.log test_trim.mp4

ffmpeg -v quiet -y -ss 00:00:00 -t 00:00:15 -i GMT20260320-130023_Recording_2240x1260.mp4 -c copy test_trim.mp4

python3 run.py test_trim.mp4 --clean > run_test.log 2>&1 &
PID=$!

echo "Started test with PID $PID" > metrics.log

for i in {1..30}; do
  if ! kill -0 $PID 2>/dev/null; then
    break
  fi
  echo "--- SEC $((i*2)) ---" >> metrics.log
  ps -p $PID -o %cpu,%mem,rss --no-headers >> metrics.log
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader >> metrics.log
  sleep 2
done

wait $PID
echo "Done" >> metrics.log
