#!/bin/bash
# Progress check for the OPTIMUS 1.1 NatureBench-25 run.
#   ~/check.sh            (or from your PC: ssh -p <port> root@<ip> '~/check.sh')
R=~/NatureBench/results
python3 ~/status.py v11_nb25_gpu_low
echo
[ -d "$R/v11_nb25_cpu" ] && python3 ~/status.py v11_nb25_cpu && echo
echo "Method families tried per task (OPTIMUS 1.1 steering):"
for L in "$R"/v11_nb25_*/*/workspace/optimus_ledger.md; do
  [ -f "$L" ] || continue
  t=$(basename "$(dirname "$(dirname "$L")")")
  fams=$(tail -n +5 "$L" | awk -F'|' '{gsub(/ /,"",$5); if($5!="") print $5}' | sort -u | tr '\n' ' ')
  n=$(echo "$fams" | wc -w)
  echo "  $t: $n families: $fams"
done
echo
echo "GPU use (index, busy %, memory used / total):"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo
echo "Latest OPTIMUS loop events:"
tail -qn1 "$R"/v11_nb25_*/*/.optimus/loop.log 2>/dev/null | tail -8
