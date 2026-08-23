"""Validate that the on-disk CIFAR-100 splits can be read and decoded."""

from collections import Counter

from torchvision.datasets import CIFAR100
from tqdm import tqdm

from config import Config


def check_dataset(data_root=None):
    """Decode every image and report split/class/size consistency."""
    root = data_root or Config.DATA_ROOT
    failures = []

    splits = {
        Config.OFFICIAL_TRAIN_SPLIT: CIFAR100(
            root=root, train=True, download=Config.DOWNLOAD_DATASET
        ),
        Config.OFFICIAL_TEST_SPLIT: CIFAR100(
            root=root, train=False, download=Config.DOWNLOAD_DATASET
        ),
    }

    for split_name, split in splits.items():
        label_counts = Counter(int(label) for label in split.targets)
        image_sizes = Counter()

        for idx in tqdm(range(len(split)), desc=f"Checking {split_name}"):
            try:
                image, _ = split[idx]
                image = image.convert("RGB")
                image.load()
                image_sizes[image.size] += 1
            except Exception as exc:
                failures.append((split_name, idx, str(exc)))

        print(f"\n{split_name}: {len(split):,} images")
        print(
            f"  classes={len(label_counts)}, "
            f"samples/class={min(label_counts.values())}..{max(label_counts.values())}"
        )
        print(f"  image sizes={dict(image_sizes)}")

    print(f"\nDecode failures: {len(failures)}")
    for split_name, idx, error in failures[:20]:
        print(f"  {split_name}[{idx}]: {error}")

    return failures


if __name__ == "__main__":
    Config.validate_config()
    check_dataset(Config.DATA_ROOT)
