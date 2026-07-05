#!/usr/bin/env bash
# Quick live dashboard for the CIFAR-100 Strategy-2 sweep.
#   bash watch_progress.sh            # one snapshot: progress + current run + GPU
#   bash watch_progress.sh -f         # then follow the ACTIVE run's log live
cd "$(dirname "$0")"
R=results

done=$(find "$R" -name metrics.json 2>/dev/null | grep -v dry | wc -l)
echo "=================== SWEEP PROGRESS: $done / 30 runs ==================="
find "$R" -name metrics.json 2>/dev/null | grep -v dry | sort | sed "s#$R/##;s#/metrics.json##" | \
  while read r; do
    acc=$(python3 -c "import json,sys;d=json.load(open('$R/$r/metrics.json'));b=d['results']['Baseline (k=1)']['Accuracy (%)'];k=d['best_k'];kk=d['results']['k=%d'%k]['Accuracy (%)'];print('base=%.2f  best k=%d -> %.2f'%(b,k,kk))" 2>/dev/null)
    printf "  [done] %-32s %s\n" "$r" "$acc"
  done

# currently running run = newest train.log without a metrics.json sibling
cur=$(grep ">>> \[" "$R/sweep_master.log" 2>/dev/null | tail -1)
echo "-----------------------------------------------------------------------"
echo "  current: $cur"
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
echo "  GPU    : $gpu"
echo "======================================================================="

if [ "${1:-}" = "-f" ]; then
  active=$(ls -t "$R"/*/*/train.log 2>/dev/null | head -1)
  echo ">>> following: $active   (Ctrl-C to stop)"
  tail -n 30 -f "$active" | grep --line-buffered -E "Epoch|Val Loss|Val Acc|New best|Baseline|k=|DONE|RUN_STATUS|Training"
fi
