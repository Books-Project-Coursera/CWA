"""
prepare_cifar100.py  (NEW — does not modify the original source base)

Download CIFAR-100 and materialize it as a *pre-split 3-way ImageFolder* dataset:

    data/cifar100_split/
        train/<class_name>/*.png   (45,000 imgs = 450/class)
        val/<class_name>/*.png     ( 5,000 imgs =  50/class)
        test/<class_name>/*.png    (10,000 imgs = 100/class  — the OFFICIAL test set)

Split convention (research-standard for CIFAR-100, so results are comparable to
published work): the official 50k train / 10k test partition is kept; a *stratified*
5,000-image validation set (50 per class) is carved out of the 50k train, giving
45k / 5k / 10k. The official 10k test set is left completely untouched.
Refs: RankingMatch (arXiv:2110.04430), Active-Learning-eval (arXiv:2301.10625),
Ensemble-Distillation-WA (arXiv:2206.15047).

The carve uses a FIXED split seed (independent of the 5 training seeds) so that every
model and every training seed sees the EXACT same data partition (scientific fairness).

Idempotent: if the split already exists with the right counts, it is left as is.
"""
import argparse
import os
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR100

# Fixed partition seed — deliberately independent of the training seeds {1,10,42,100,500}.
SPLIT_SEED = 42
VAL_PER_CLASS = 50          # 50 * 100 = 5,000 validation images
N_CLASSES = 100


def _write_split(dataset, indices, split_dir, class_names):
    """Write the given dataset indices as PNGs under split_dir/<class_name>/."""
    for cname in class_names:
        os.makedirs(os.path.join(split_dir, cname), exist_ok=True)
    counters = defaultdict(int)
    for idx in indices:
        img, label = dataset[idx]           # img is a PIL.Image (no transform)
        cname = class_names[label]
        counters[label] += 1
        out = os.path.join(split_dir, cname, f"{cname}_{counters[label]:04d}.png")
        img.save(out)
    return sum(counters.values())


def _count_images(split_dir):
    total = 0
    per_class = {}
    if not os.path.isdir(split_dir):
        return 0, {}
    for cname in sorted(os.listdir(split_dir)):
        cdir = os.path.join(split_dir, cname)
        if os.path.isdir(cdir):
            n = len([f for f in os.listdir(cdir) if f.endswith(".png")])
            per_class[cname] = n
            total += n
    return total, per_class


def build(data_root):
    raw_dir = os.path.join(data_root, "cifar100_raw")
    split_root = os.path.join(data_root, "cifar100_split")
    os.makedirs(data_root, exist_ok=True)

    # ---- Idempotency check ----
    expected = {"train": 45000, "val": 5000, "test": 10000}
    if all(_count_images(os.path.join(split_root, s))[0] == n for s, n in expected.items()):
        print(f"✓ CIFAR-100 split already present and correct at: {split_root}")
        return split_root

    print("Downloading CIFAR-100 (torchvision)...")
    train_set = CIFAR100(root=raw_dir, train=True, download=True)
    test_set = CIFAR100(root=raw_dir, train=False, download=True)
    class_names = list(train_set.classes)   # 100 human-readable fine-label names
    assert len(class_names) == N_CLASSES, f"expected 100 classes, got {len(class_names)}"

    train_targets = np.array(train_set.targets)

    # ---- Stratified 45k/5k carve from the 50k train (fixed seed) ----
    rng = np.random.RandomState(SPLIT_SEED)
    val_idx, tr_idx = [], []
    for c in range(N_CLASSES):
        c_indices = np.where(train_targets == c)[0]
        rng.shuffle(c_indices)                       # deterministic per SPLIT_SEED
        val_idx.extend(c_indices[:VAL_PER_CLASS].tolist())
        tr_idx.extend(c_indices[VAL_PER_CLASS:].tolist())
    test_idx = list(range(len(test_set)))            # official 10k test, untouched

    # sanity: train/val disjoint, cover the full 50k
    assert len(set(val_idx) & set(tr_idx)) == 0
    assert len(tr_idx) + len(val_idx) == len(train_set) == 50000

    # ---- Clean any partial previous build, then write ----
    if os.path.isdir(split_root):
        shutil.rmtree(split_root)
    print("Writing train split (45,000)...")
    n_tr = _write_split(train_set, tr_idx, os.path.join(split_root, "train"), class_names)
    print("Writing val split (5,000)...")
    n_va = _write_split(train_set, val_idx, os.path.join(split_root, "val"), class_names)
    print("Writing test split (10,000)...")
    n_te = _write_split(test_set, test_idx, os.path.join(split_root, "test"), class_names)

    # ---- Verify ----
    print("\n=== Verification ===")
    ok = True
    for split, exp in expected.items():
        total, per_class = _count_images(os.path.join(split_root, split))
        n_classes = len(per_class)
        per_vals = set(per_class.values())
        print(f"  {split:5s}: {total:6d} imgs across {n_classes} classes "
              f"(per-class counts: {sorted(per_vals)})")
        ok &= (total == exp and n_classes == N_CLASSES)
    assert ok, "Split verification FAILED — counts do not match expected 45k/5k/10k."
    print(f"\n✓ Built {n_tr}+{n_va}+{n_te} = {n_tr+n_va+n_te} images at: {split_root}")
    print(f"  Split seed (partition) = {SPLIT_SEED}; official 10k test kept intact.")

    # free the raw tar/pickle to save disk (only the PNG split is needed henceforth)
    try:
        shutil.rmtree(raw_dir)
        print(f"  (removed raw download dir {raw_dir} to save disk)")
    except OSError:
        pass
    return split_root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = ap.parse_args()
    build(args.data_root)
