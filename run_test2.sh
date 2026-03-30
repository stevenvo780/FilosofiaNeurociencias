#!/bin/bash
# Full pipeline test: 15s clip, CPU+GPU metrics every 1s
killall -9 python3 ffmpeg rife-ncnn-vulkan 2>/dev/null
sleep 1
rm -f metrics2.log run_test2.log test_trim.mp4

echo "--- Creating 15s test clip ---"
ffmpeg -v quiet -y -ss 00:00:00 -t 00:00:15 -i GMT20260320-130023_Recording_2240x1260.mp4 -c copy test_trim.mp4

echo "--- Running pipeline (CPU worker DISABLED) ---"
python3 run.py test_trim.mp4 --clean > run_test2.log 2>&1 &
PID=$!

echo "PID=$PID" > metrics2.log
for i in {1..60}; do
  if ! kill -0 $PID 2>/dev/null; then
    break
  fi
  SEC=$((i*1))
  echo "--- SEC $SEC ---" >> metrics2.log
  ps -p $PID -o %cpu,%mem --no-headers >> metrics2.log
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits >> metrics2.log
  sleep 1
done

wait $PID
EXIT=$?
echo "EXIT=$EXIT" >> metrics2.log
echo "--- DONE (exit=$EXIT) ---"
