#!/bin/bash
# test_monitor.sh
echo "--- Starting 2-chunk test ---"
python3 run.py GMT20260320-130023_Recording_2240x1260.mp4 --clean > run_test.log 2>&1 &
PID=$!
echo "Pipeline PID is $PID"

echo "Collecting CPU & GPU metrics every 5 seconds..."
for i in {1..15}; do
  if ! kill -0 $PID 2>/dev/null; then
    break
  fi
  echo "--- CPU % ($PID) ---" >> metrics.log
  ps -p $PID -o %cpu,%mem,rss --no-headers >> metrics.log
  
  echo "--- GPU % ---" >> metrics.log
  nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,pstate --format=csv,noheader >> metrics.log
  sleep 4
done

wait $PID
echo "--- Done! Test logged to run_test.log and metrics.log ---"
