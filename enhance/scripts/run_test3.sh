#!/bin/bash
killall -9 python3 ffmpeg rife-ncnn-vulkan 2>/dev/null
sleep 1
rm -f metrics3.log run_test3.log
rm -rf enhanced/work_test_trim enhanced/test_trim_ai_50fps.mp4

ffmpeg -v quiet -y -ss 0 -t 15 -i GMT20260320-130023_Recording_2240x1260.mp4 -c copy test_trim.mp4

python3 run.py test_trim.mp4 --clean > run_test3.log 2>&1 </dev/null &
PID=$!

echo "PID=$PID" > metrics3.log
for i in $(seq 1 120); do
  if ! kill -0 $PID 2>/dev/null; then
    break
  fi
  echo "--- SEC $i ---" >> metrics3.log
  ps -p $PID -o %cpu --no-headers >> metrics3.log 2>/dev/null
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits >> metrics3.log 2>/dev/null
  sleep 1
done

wait $PID
echo "EXIT=$?" >> metrics3.log
