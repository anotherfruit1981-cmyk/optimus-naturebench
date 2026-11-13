#!/bin/bash
# Runs the remaining Full-track gpu_low tasks batch by batch, freeing each batch's data before the next.
NB=~/NatureBench
B=$NB/task-set/batches
cd "$NB" || exit 1
for f in $(ls "$B"/batch_*.txt | sort -V); do
  name=$(basename "$f" .txt)
  [ -f "$B/$name.done" ] && { echo "skip $name (done)"; continue; }
  echo "=== $(date '+%Y-%m-%d %H:%M') starting $name ($(grep -c . "$f") tasks)"
  ~/run_optimus.sh "$f" v11_full_gpu_low 0,1,2,3 || echo "!!! $name: run_optimus returned non-zero"
  while read -r t; do
    d=$NB/data/naturebench_data/tasks/$t
    [ -d "$d" ] && find "$d" -type f ! -name README.md ! -name data_description.md -delete
  done < "$f"
  touch "$B/$name.done"
  echo "=== $(date '+%Y-%m-%d %H:%M') finished $name; disk: $(df -h / | tail -1)"
done
echo "=== all batches finished"
