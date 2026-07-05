"""
run_cifar100.py  (NEW — orchestrates ONE (model, seed) run; never edits the base source)

Pipeline for a single run:
  1. Inject per-model + per-seed hyperparameters onto the Config CLASS at runtime.
  2. Seed all RNGs (random / numpy / torch / cuda) + cudnn-deterministic, like main.py.
  3. Build dataloaders from the pre-split CIFAR-100 (dataset_cifar100 + existing create_dataloaders).
  4. Train with the existing train.train_model  ->  CheckpointManager (top-k epoch pool).
  5. Evaluate:
        Baseline (k=1)  = best (lowest val-loss) checkpoint.
        Strategy 2      = element-wise average of the top-k lowest-val-loss checkpoints,
                          for k in {2,3,4,5}, with BatchNorm recalibration (update_bn).
     Built ONLY from the base repo's active primitives:
        CheckpointManager.get_best_checkpoint / get_top_k_checkpoints,
        evaluate.average_weights, evaluate.update_bn, evaluate.evaluate_model, models.get_model.
     (The repo's strategy_* wrapper functions are disabled stubs in this branch, so we
      compose the same maths ourselves from the underlying, still-active functions.)
  6. Save metrics.json + run_config.json + selected weights, then DELETE the per-epoch
     checkpoint pool to keep disk usage bounded (16 GB overlay on this box).

Usage:
  python run_cifar100.py --model resnet101 --seed 42
  python run_cifar100.py --model mobilenet_v2 --seed 42 --dry-run --subset 512 --epochs 1
"""
import argparse
import os

# Must be set BEFORE importing torch (mirrors main.py).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import random
import shutil
import time

import numpy as np
import torch

from config import Config
from dataset import create_dataloaders
from dataset_cifar100 import load_cifar100_split
from train import train_model
from evaluate import average_weights, update_bn, evaluate_model
from models import get_model
from cifar100_configs import CIFAR100_CONFIGS, PRETTY_NAME, TOP_K_VALUES

HERE = os.path.dirname(os.path.abspath(__file__))


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_config(model_name, seed, epochs_override, num_workers):
    """Inject per-model + common hyperparameters onto the Config class (runtime only)."""
    hp = CIFAR100_CONFIGS[model_name]
    Config.RANDOM_SEED = seed
    Config.MODELS = [model_name]
    Config.BATCH_SIZE = hp["batch_size"]
    Config.NUM_EPOCHS = epochs_override if epochs_override else hp["num_epochs"]
    Config.LEARNING_RATE = hp["learning_rate"]
    Config.WEIGHT_DECAY = hp["weight_decay"]
    Config.DROPOUT_RATE = hp["dropout_rate"]
    Config.CLASSIFIER_CONFIG = list(hp["classifier_config"])
    Config.EARLY_STOPPING_PATIENCE = min(hp["early_stopping_patience"], max(1, Config.NUM_EPOCHS - 1))
    # Clamp warmup so T_max = NUM_EPOCHS - WARMUP_EPOCHS >= 1 (avoids CosineAnnealingLR T_max=0).
    Config.WARMUP_EPOCHS = min(max(1, int(Config.NUM_EPOCHS * 0.06)), max(1, Config.NUM_EPOCHS - 1))
    Config.ETA_MIN = 1e-6
    # Common CIFAR-100 settings
    Config.LOSS_FUNCTION = "cross_entropy"
    Config.LABEL_SMOOTHING = 0.1
    Config.USE_WEIGHTED_SAMPLER = False       # CIFAR-100 is balanced
    Config.USE_CROSS_VALIDATION = False
    Config.IMAGE_SIZE = 224
    Config.NUM_WORKERS = num_workers
    # Strategy 2 checkpoint pool: keep exactly the top-5 (for k<=5) + a tiny recency buffer.
    Config.TOP_K_VALUES = list(TOP_K_VALUES)
    Config.KEEP_TOP_K_CHECKPOINTS = 5
    Config.KEEP_LAST_N_CHECKPOINTS = 1
    Config.AUTO_DELETE_CHECKPOINTS = False
    return hp


def subset_split(tuple7, n_train, n_eval, seed):
    """Take a small stratified-ish random subset for the dry-run smoke test."""
    (trp, trl, vap, val, tep, tel, cnames) = tuple7
    rng = random.Random(seed)

    def take(paths, labels, n):
        idx = list(range(len(paths)))
        rng.shuffle(idx)
        idx = idx[:min(n, len(idx))]
        return [paths[i] for i in idx], [labels[i] for i in idx]

    trp, trl = take(trp, trl, n_train)
    vap, val = take(vap, val, n_eval)
    tep, tel = take(tep, tel, n_eval)
    return (trp, trl, vap, val, tep, tel, cnames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CIFAR100_CONFIGS.keys()))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-root", default=os.path.join(HERE, "data", "cifar100_split"))
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--epochs", type=int, default=0, help="override num_epochs (0 = use config)")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="tiny subset smoke test")
    ap.add_argument("--subset", type=int, default=512, help="dry-run train subset size")
    ap.add_argument("--keep-weights", choices=["best", "all", "none"], default="best",
                    help="which strategy weights to retain (disk): best=baseline+best-k")
    args = ap.parse_args()

    model_name = args.model
    seed = args.seed
    pretty = PRETTY_NAME[model_name]
    # dry-run uses 2 epochs (min needed for a valid warmup+cosine schedule) unless overridden
    epochs_override = (args.epochs or 2) if args.dry_run else args.epochs

    hp = apply_config(model_name, seed, epochs_override, args.num_workers)

    run_dir = os.path.join(args.results_dir, model_name, f"seed_{seed}")
    ckpt_dir = os.path.join(run_dir, "ckpt")          # per-epoch pool (deleted at the end)
    weights_dir = os.path.join(run_dir, "weights")    # retained final strategy weights
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    print("=" * 78)
    print(f"RUN  model={model_name} ({pretty})  seed={seed}  dry_run={args.dry_run}")
    print(f"     epochs={Config.NUM_EPOCHS}  batch={Config.BATCH_SIZE}  lr={Config.LEARNING_RATE}  "
          f"wd={Config.WEIGHT_DECAY}  dropout={Config.DROPOUT_RATE}  cls={Config.CLASSIFIER_CONFIG}")
    print("=" * 78)

    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    # ---- Data ----
    data7 = load_cifar100_split(args.data_root)
    if args.dry_run:
        data7 = subset_split(data7, args.subset, max(128, args.subset // 4), seed)
        print(f"[dry-run] subset -> train={len(data7[0])} val={len(data7[2])} test={len(data7[4])}")
    (trp, trl, vap, val_, tep, tel, class_names) = data7
    num_classes = len(class_names)

    train_loader, val_loader, test_loader = create_dataloaders(
        trp, trl, vap, val_, tep, tel, Config.BATCH_SIZE, Config.NUM_WORKERS
    )

    # ---- Train (reuse base train_model) ----
    t0 = time.time()
    checkpoint_manager, history = train_model(
        model_name, train_loader, val_loader, num_classes, device,
        class_names=class_names, save_dir=run_dir, checkpoints_dir=ckpt_dir,
    )
    train_secs = time.time() - t0
    epochs_trained = len(history["train_loss"])

    # ---- Evaluate: Baseline (k=1) + Strategy 2 (k=2..5) ----
    results = {}

    def eval_and_record(tag, model, ckpt_epochs=None):
        res = evaluate_model(model, test_loader, device, num_classes, class_names)
        m = res["metrics"]
        rec = {k: float(v) for k, v in m.items()}
        if ckpt_epochs is not None:
            rec["checkpoint_epochs"] = list(ckpt_epochs)
        results[tag] = rec
        print(f"  [{tag}] Acc={m['Accuracy (%)']:.2f}%  F1={m['F1-Score (%)']:.2f}%  "
              f"AUC={m['AUC (%)']:.2f}%  TestLoss={m['Test Loss']:.4f}")
        return m["Accuracy (%)"]

    # Baseline: best (lowest val-loss) checkpoint
    best_epoch, best_val, best_path = checkpoint_manager.get_best_checkpoint()
    base_model = get_model(model_name, num_classes, freeze_backbone=False)
    base_model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    base_model = base_model.to(device)
    print(f"\nBaseline: epoch={best_epoch}  val_loss={best_val:.4f}")
    eval_and_record("Baseline (k=1)", base_model, ckpt_epochs=[best_epoch])
    results["Baseline (k=1)"]["best_epoch"] = int(best_epoch)
    results["Baseline (k=1)"]["val_loss"] = float(best_val)
    torch.save({"model_state_dict": base_model.state_dict(), "strategy": "Baseline (k=1)",
                "epoch": int(best_epoch), "val_loss": float(best_val)},
               os.path.join(weights_dir, "Strategy_1_best.pth"))
    del base_model
    torch.cuda.empty_cache()

    # Strategy 2: average top-k lowest-val-loss checkpoints
    for k in Config.TOP_K_VALUES:
        top_k = checkpoint_manager.get_top_k_checkpoints(k)
        paths = [p for _, _, p in top_k]
        epochs = [e for e, _, _ in top_k]
        if len(top_k) < k:
            print(f"  [k={k}] WARNING only {len(top_k)} checkpoints available (dry-run/short train)")
        avg_state = average_weights(paths, device)
        m = get_model(model_name, num_classes, freeze_backbone=False)
        m.load_state_dict(avg_state, strict=True)
        m = m.to(device)
        update_bn(m, train_loader, device, num_batches=100)   # BN recalibration
        eval_and_record(f"k={k}", m, ckpt_epochs=epochs)
        torch.save({"model_state_dict": m.state_dict(), "strategy": f"Strategy 2 (k={k})",
                    "k": k, "checkpoint_epochs": epochs},
                   os.path.join(weights_dir, f"Strategy_2_K{k}.pth"))
        del m
        torch.cuda.empty_cache()

    # ---- Determine best k (by test accuracy) & prune weights for disk ----
    k_accs = {k: results[f"k={k}"]["Accuracy (%)"] for k in Config.TOP_K_VALUES}
    best_k = max(k_accs, key=k_accs.get)
    if args.keep_weights == "best":
        keep = {"Strategy_1_best.pth", f"Strategy_2_K{best_k}.pth"}
        for f in os.listdir(weights_dir):
            if f not in keep:
                os.remove(os.path.join(weights_dir, f))
    elif args.keep_weights == "none":
        shutil.rmtree(weights_dir, ignore_errors=True)

    # ---- Persist metrics + config ----
    baseline_acc = results["Baseline (k=1)"]["Accuracy (%)"]
    summary = {
        "model": model_name, "pretty": pretty, "seed": seed,
        "num_classes": num_classes, "dry_run": args.dry_run,
        "epochs_trained": epochs_trained, "train_seconds": round(train_secs, 1),
        "best_k": best_k, "best_k_accuracy": k_accs[best_k],
        "baseline_accuracy": baseline_acc,
        "config": {
            "batch_size": Config.BATCH_SIZE, "num_epochs": Config.NUM_EPOCHS,
            "learning_rate": Config.LEARNING_RATE, "weight_decay": Config.WEIGHT_DECAY,
            "warmup_epochs": Config.WARMUP_EPOCHS, "eta_min": Config.ETA_MIN,
            "early_stopping_patience": Config.EARLY_STOPPING_PATIENCE,
            "dropout_rate": Config.DROPOUT_RATE, "classifier_config": Config.CLASSIFIER_CONFIG,
            "optimizer": "Adam", "scheduler": "LinearWarmup+CosineAnnealing",
            "loss": "cross_entropy(label_smoothing=0.1)", "image_size": Config.IMAGE_SIZE,
            "top_k_values": Config.TOP_K_VALUES,
        },
        "results": results,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(summary["config"] | {"model": model_name, "seed": seed}, f, indent=2)

    # ---- Delete per-epoch checkpoint pool (disk safety) ----
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    print("\n" + "=" * 78)
    print(f"DONE  {model_name} seed={seed}  baseline={baseline_acc:.2f}%  "
          f"best k={best_k} -> {k_accs[best_k]:.2f}%  (epochs={epochs_trained}, {train_secs:.0f}s)")
    print(f"      kept weights: {sorted(os.listdir(weights_dir)) if os.path.isdir(weights_dir) else 'none'}")
    print("RUN_STATUS: SUCCESS")
    print("=" * 78)


if __name__ == "__main__":
    main()
