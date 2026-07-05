#!/usr/bin/env bash
# Autonomous finisher: waits for the current sweep to end, does ONE resume pass to recover
# any failed/missing runs (weights are cached now), then re-aggregates. Fully detached so it
# completes even with no SSH session / client connected.
cd /root/Capstone_KD
source /venv/main/bin/activate 2>/dev/null
log=results/auto_finish.log
echo "AUTO_FINISH start $(date)" >> "$log"
# 1) wait for the first sweep to finish
while pgrep -f "run_experiments.sh" >/dev/null 2>&1; do sleep 60; done
echo "first pass ended $(date)" >> "$log"
# 2) resume pass: retries only runs missing metrics.json (skips completed), then aggregates
bash run_experiments.sh >> results/sweep_master.log 2>&1
echo "resume pass ended $(date)" >> "$log"
# 3) final safety aggregate
python aggregate_results.py >> results/aggregate.log 2>&1
done=$(find results -name metrics.json | grep -v dry | wc -l)
echo "AUTO_FINISH done $(date) — $done/30 runs, summary at results/summary_cifar100.xlsx" >> "$log"
