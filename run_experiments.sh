#!/usr/bin/env bash
#
# run_experiments.sh  (NEW)
# Full sequential sweep for the CIFAR-100 Strategy-2 extension:
#   6 models x 5 seeds = 30 runs, one at a time (avoids GPU OOM), each streamed to its own
#   log via `tee` so progress survives SSH disconnects. Builds the dataset first if missing,
#   then aggregates all runs into paper-ready tables.
#
# Usage:
#   bash run_experiments.sh                # full sweep
#   bash run_experiments.sh --dry-run      # 1 model, 1 seed, 1 epoch, tiny subset (smoke test)
#
# Robust to disconnects — run detached and follow the master log:
#   nohup bash run_experiments.sh > results/sweep_master.log 2>&1 &
#   tail -f results/sweep_master.log
#
set -u
cd "$(dirname "$0")"

# --- activate the python env (non-login shell safe) ---
source /venv/main/bin/activate 2>/dev/null || true

PY=python
DATA_ROOT="$(pwd)/data/cifar100_split"
RESULTS_DIR="$(pwd)/results"
mkdir -p "$RESULTS_DIR"

MODELS=(vit_base_patch16_224 vgg16 resnet101 mobilenet_v2 densenet121 efficientnet_b0)
SEEDS=(1 10 42 100 500)

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

echo "############################################################"
echo "# CIFAR-100 Strategy-2 sweep  |  start: $(date)"
echo "# models=${#MODELS[@]}  seeds=${#SEEDS[@]}  dry_run=$DRY"
echo "############################################################"

# --- Step 1: build dataset if missing ---
if [[ ! -d "$DATA_ROOT/test" ]]; then
  echo ">>> Building CIFAR-100 split..."
  $PY prepare_cifar100.py --data-root "$(pwd)/data" || { echo "DATASET BUILD FAILED"; exit 1; }
fi

if [[ "$DRY" == "1" ]]; then
  echo ">>> DRY-RUN smoke test (mobilenet_v2, seed 42, 1 epoch, subset 512)"
  mkdir -p "$RESULTS_DIR/mobilenet_v2/seed_42"
  $PY run_cifar100.py --model mobilenet_v2 --seed 42 --dry-run --subset 512 --epochs 2 \
      --data-root "$DATA_ROOT" --results-dir "$RESULTS_DIR" \
      2>&1 | tee "$RESULTS_DIR/mobilenet_v2/seed_42/dry_run.log"
  echo ">>> Dry-run finished."
  exit 0
fi

# --- Step 2: sequential full sweep ---
FAILED=()
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_dir="$RESULTS_DIR/$model/seed_$seed"
    mkdir -p "$run_dir"
    # resume-safe: skip runs that already produced metrics.json
    if [[ -f "$run_dir/metrics.json" ]]; then
      echo ">>> SKIP (already done): $model seed=$seed"
      continue
    fi
    echo ""
    echo ">>> [$(date +%H:%M:%S)] START  model=$model  seed=$seed"
    $PY run_cifar100.py --model "$model" --seed "$seed" \
        --data-root "$DATA_ROOT" --results-dir "$RESULTS_DIR" \
        2>&1 | tee "$run_dir/train.log"
    # capture python exit status (not tee's) via PIPESTATUS
    status=${PIPESTATUS[0]}
    if [[ "$status" -ne 0 ]] || [[ ! -f "$run_dir/metrics.json" ]]; then
      echo ">>> FAILED  model=$model seed=$seed (exit=$status)"
      FAILED+=("$model/seed_$seed")
    else
      echo ">>> OK      model=$model seed=$seed"
    fi
  done
done

# --- Step 3: aggregate ---
echo ""
echo ">>> Aggregating results..."
$PY aggregate_results.py --results-dir "$RESULTS_DIR" 2>&1 | tee "$RESULTS_DIR/aggregate.log"

echo ""
echo "############################################################"
echo "# Sweep finished: $(date)"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "# FAILED runs (${#FAILED[@]}): ${FAILED[*]}"
else
  echo "# All 30 runs completed."
fi
echo "############################################################"
