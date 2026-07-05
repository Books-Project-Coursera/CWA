# Experiment Plan — Strategy 2 (Top-k Checkpoint Averaging) on CIFAR-100

## 1. Objective

Extend **Strategy 2** — "Post-Training Top-k Checkpoint Averaging" (`docs/strategy2.pdf`) —
to **CIFAR-100**, across the paper's **6 architectures × 5 seeds**, and report **mean ± std**
so the numbers can be dropped straight into the paper alongside its Tables 4–6.

This directly answers the paper's own **Limitations** section, which states results are from
*single training runs* and calls for *"multiple independent runs with different random seeds
and appropriate statistical analysis."*

## 2. What Strategy 2 is (analysis of the paper)

Train a model **normally**, saving one checkpoint + its validation loss each epoch.

- **Baseline (k=1):** keep the single checkpoint with the lowest validation loss
  `w_baseline = w_{t*}`, `t* = argmin_t ℓ_t`.
- **Strategy 2:** keep the **k checkpoints with the lowest validation loss** and average
  their weights element-wise: `w_avg = (1/k) Σ_{t∈S_k} w_t`, for `k ∈ {2,3,4,5}`.

Properties: **no change** to optimizer / LR schedule / early stopping / epochs; **no extra
inference cost** (still one model). The only overhead is storing epoch checkpoints — which we
therefore **delete after aggregation**. BatchNorm running stats are *not* averaged (kept from
the best checkpoint) and are **recalibrated** on training data after averaging (`update_bn`).

**Models (6, exactly as in the paper):** ViT-B/16, VGG16, ResNet101, MobileNetV2,
DenseNet121, EfficientNetB0.
**Seeds (5, identical across all models):** `1, 10, 42, 100, 500` (42 mandatory).

## 3. Dataset — CIFAR-100, 3-way pre-split (comparable to published work)

CIFAR-100 = 100 classes × 600 = 60,000 images; official split = 50k train / 10k test.
We use the **research-standard validation protocol**: carve a **stratified 5,000-image
validation set (50/class) from the 50k train → 45,000 train / 5,000 val / 10,000 test**,
keeping the **official 10k test set untouched** so accuracy is directly comparable to
published CIFAR-100 numbers (RankingMatch arXiv:2110.04430; AL-eval arXiv:2301.10625;
Ensemble-Distillation-WA arXiv:2206.15047).

The carve uses a **fixed partition seed (42), independent of the 5 training seeds**, so every
model and every training seed sees the **identical** data split (scientific fairness).

Built by `prepare_cifar100.py` into `data/cifar100_split/{train,val,test}/<class>/*.png`
(`ImageFolder`-compatible, 100 human-readable class folders).

## 4. Code changes (all NEW files — the original repo source is never edited)

The base repo (`Dung-04/Capstone_KD` @ `baseline/teacher-student-selection`) already contains
the working primitives; every hyperparameter is read from the `Config` **class**, so the new
driver injects settings by assigning `Config.X` at runtime.

| New file | Role |
|---|---|
| `prepare_cifar100.py` | Download CIFAR-100 → 45k/5k/10k ImageFolder split (idempotent). |
| `dataset_cifar100.py` | **Dataset-loader change:** replaces `dataset.load_dataset()`'s single-folder internal-split logic. `load_cifar100_split(root)` scans the 3 pre-split folders → the same 7-tuple, then reuses the **existing** `dataset.create_dataloaders(...)`. *No edit to `dataset.py`.* |
| `cifar100_configs.py` | Per-model CIFAR-100 hyperparameters + model order + seeds. |
| `run_cifar100.py` | One (model, seed) run: set `Config`, seed RNGs, build loaders, `train_model`, then Baseline + Strategy 2 built from the base primitives (`CheckpointManager.get_top_k_checkpoints`, `evaluate.average_weights`, `evaluate.update_bn`, `evaluate.evaluate_model`, `models.get_model`). Saves metrics/config/weights; deletes epoch pool. |
| `aggregate_results.py` | Multi-seed **mean ± std** tables → `results/summary_cifar100.xlsx` (Mean±Std sheet + one sheet per seed) + `.csv`. |
| `run_experiments.sh` | Sequential 6×5 sweep with `tee` logging + final aggregation; resume-safe. |

### Why `dataset_cifar100.py` instead of editing `dataset.py`
The original `load_dataset()` reads **one** folder and performs its **own** random 70/15/15
split. CIFAR-100 is already split into three folders, so that logic is bypassed: we build the
`(paths, labels)` lists directly from `train/`, `val/`, `test/` (label index = sorted class
name, matching `torchvision.ImageFolder`) and pass them to the untouched `create_dataloaders`.

## 5. Hyperparameters
See `proposed_cifar100_config.csv` (derived from `strategy2.pdf` Tables 1–3 + the old
`config.csv`). Common: Linear-warmup(0.06·epochs)→Cosine(η_min 1e-6); CrossEntropy
(label_smoothing 0.1); no weighted sampler (balanced data); image 224.

**Known deviation:** `train.py` hardcodes `optim.Adam`, so ViT uses **Adam** (prior datasets
used AdamW for ViT). Kept as-is to honor "don't edit source"; consistent across all runs.

## 6. Execution & safety
1. `python prepare_cifar100.py` → verify 45k/5k/10k, 100 classes/split.
2. **Dry-run:** `bash run_experiments.sh --dry-run` (mobilenet_v2, seed 42, 1 epoch, subset
   512) → catch OOM / path / data errors end-to-end.
3. **Full run:** `nohup bash run_experiments.sh > results/sweep_master.log 2>&1 &` — 30
   sequential runs, each `tee`-logged to `results/<model>/seed_<seed>/train.log`
   (SSH-drop safe). Resume-safe (skips runs with an existing `metrics.json`).
4. `python aggregate_results.py` → `results/summary_cifar100.xlsx` (+ `.csv`).

**Disk:** this box has a **16 GB** writable overlay. Mitigations: per-epoch checkpoint pool
capped (`KEEP_TOP_K=5`, `KEEP_LAST_N=1`) and **deleted after aggregation**; only **baseline +
best-k** final weights retained per run (`--keep-weights best`); full metrics/logs always kept.

## 7. Deliverables
- `experiment_plan.md` (this file), `proposed_cifar100_config.csv`, `run_experiments.sh`.
- `results/<Model>/seed_<Seed>/` → `metrics.json`, `run_config.json`, `train.log`, `weights/*.pth`.
- `results/summary_cifar100.xlsx` (Mean±Std + per-seed sheets) and `summary_cifar100.csv`.
