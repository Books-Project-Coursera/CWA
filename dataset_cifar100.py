"""
dataset_cifar100.py  (NEW — does not modify dataset.py)

Reads the pre-split 3-way CIFAR-100 ImageFolder tree produced by prepare_cifar100.py
and returns exactly the 7-tuple that the ORIGINAL dataset.create_dataloaders(...) expects:

    (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names)

This replaces the original `dataset.load_dataset()` (which reads ONE folder and does its
own internal 70/15/15 split) WITHOUT editing it — we simply build the path/label lists from
the already-split folders and hand them to the existing `create_dataloaders`.

Class-index mapping is defined by the sorted class-folder names (identical across the three
splits, since every split contains all 100 classes), matching torchvision.ImageFolder's
convention so labels are consistent and comparable.
"""
import os

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _scan_split(split_dir, class_to_idx):
    paths, labels = [], []
    for cname, cidx in class_to_idx.items():
        cdir = os.path.join(split_dir, cname)
        if not os.path.isdir(cdir):
            continue
        for fname in sorted(os.listdir(cdir)):
            if fname.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(cdir, fname))
                labels.append(cidx)
    return paths, labels


def load_cifar100_split(root):
    """
    Args:
        root: path to the `cifar100_split` directory containing train/ val/ test/.

    Returns:
        train_paths, train_labels, val_paths, val_labels,
        test_paths, test_labels, class_names   (compatible with create_dataloaders)
    """
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")
    for d in (train_dir, val_dir, test_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Expected split folder not found: {d}\n"
                f"Run `python prepare_cifar100.py` first to build the dataset."
            )

    # Sorted class-folder names → deterministic label indices (ImageFolder convention).
    class_names = sorted(
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    )
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    train_paths, train_labels = _scan_split(train_dir, class_to_idx)
    val_paths, val_labels = _scan_split(val_dir, class_to_idx)
    test_paths, test_labels = _scan_split(test_dir, class_to_idx)

    print("CIFAR-100 pre-split loaded:")
    print(f"  classes : {len(class_names)}")
    print(f"  train   : {len(train_paths)}")
    print(f"  val     : {len(val_paths)}")
    print(f"  test    : {len(test_paths)}")

    return (train_paths, train_labels, val_paths, val_labels,
            test_paths, test_labels, class_names)
