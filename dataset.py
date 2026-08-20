"""CIFAR-100 loading, preprocessing, and PyTorch DataLoaders."""

import random

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import InterpolationMode

from config import Config


def worker_init_fn_seed(worker_id):
    """Seed Python's RNG independently in every DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)


class ImageArraySplit:
    """In-memory image/label split backed by a uint8 numpy array.

    Exposes the small read-only API the rest of the pipeline expects from a
    split (``len``, ``split["image"]`` / ``split["label"]`` column access,
    integer indexing, and ``select``) so training, cross-validation and the
    statistics helpers stay unchanged.
    """

    def __init__(self, images, labels):
        self.images = np.asarray(images)
        self.labels = [int(label) for label in labels]
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def _to_pil(self, index):
        return Image.fromarray(self.images[index])

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "image":
                return [self._to_pil(idx) for idx in range(len(self))]
            if key == "label":
                return list(self.labels)
            raise KeyError(f"Unknown column: {key}")

        index = int(key)
        return {"image": self._to_pil(index), "label": self.labels[index]}

    def select(self, indices):
        """Return a new split containing only ``indices`` (used by K-Fold CV)."""
        indices = [int(idx) for idx in indices]
        return ImageArraySplit(
            self.images[indices],
            [self.labels[idx] for idx in indices],
        )


def concatenate_datasets(splits):
    """Concatenate ``ImageArraySplit`` objects into a single split."""
    splits = list(splits)
    if not splits:
        raise ValueError("concatenate_datasets requires at least one split")

    images = np.concatenate([split.images for split in splits], axis=0)
    labels = [label for split in splits for label in split.labels]
    return ImageArraySplit(images, labels)


class CachedImageDataset(Dataset):
    def __init__(self, split, transform=None):
        self.dataset = split
        self.transform = transform
        self.labels = [int(label) for label in split["label"]]
        # Cache ảnh dạng PIL vào RAM ngay khi init
        print("Caching images to RAM...")
        all_images = split["image"]
        self._cache = [img.convert("RGB") for img in all_images]
    def __len__(self):
        return len(self._cache)
    def __getitem__(self, idx):
        image = self._cache[idx]  # O(1), không cần đọc file
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(split="train"):
    """Build full-image 224px transforms for ImageNet-pretrained models.

    CIFAR-100 images are 32x32, so the resize upsamples them to the resolution
    the pretrained backbones expect.
    """
    common = [
        transforms.Resize(
            (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
    ]

    if split == "train":
        return transforms.Compose(
            common
            + [
                transforms.RandomHorizontalFlip(p=Config.HORIZONTAL_FLIP_PROB),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=Config.IMAGE_MEAN,
                    std=Config.IMAGE_STD,
                ),
                transforms.RandomErasing(
                    p=Config.RANDOM_ERASING_PROB,
                    scale=Config.RANDOM_ERASING_SCALE,
                    ratio=Config.RANDOM_ERASING_RATIO,
                    value=Config.RANDOM_ERASING_VALUE,
                ),
            ]
        )

    return transforms.Compose(
        common
        + [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=Config.IMAGE_MEAN,
                std=Config.IMAGE_STD,
            ),
        ]
    )


def load_dataset(dataset_name, val_ratio=0.1, random_seed=42, data_root=None):
    """Load CIFAR-100 and reserve the official test split for testing.

    torchvision ships 50,000 labelled training images and 10,000 labelled test
    images. We stratify the official training split into 45,000 training and
    5,000 validation samples, while leaving the official test split untouched
    as the final test set. The dataset is read from disk only
    (``download=False``): torchvision expects ``<data_root>/cifar-100-python``.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")

    normalized_name = dataset_name.lower().replace("-", "").replace("_", "")
    if normalized_name != "cifar100":
        raise ValueError(
            f"This pipeline is configured for CIFAR-100, got '{dataset_name}'"
        )

    root = data_root or Config.DATA_ROOT
    print(f"\nLoading torchvision CIFAR-100 from: {root}")
    official_train = CIFAR100(
        root=root, train=True, download=Config.DOWNLOAD_DATASET
    )
    official_test = CIFAR100(
        root=root, train=False, download=Config.DOWNLOAD_DATASET
    )

    class_names = list(official_train.classes)
    official_train_labels = [int(label) for label in official_train.targets]
    test_labels = [int(label) for label in official_test.targets]

    # Fixed, stratified holdout so every seed/model sees the same 45k/5k split
    # for a given random_seed.
    train_idx, val_idx = train_test_split(
        np.arange(len(official_train_labels)),
        test_size=val_ratio,
        stratify=official_train_labels,
        random_state=random_seed,
        shuffle=True,
    )
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    train_data = ImageArraySplit(
        official_train.data[train_idx],
        [official_train_labels[idx] for idx in train_idx],
    )
    val_data = ImageArraySplit(
        official_train.data[val_idx],
        [official_train_labels[idx] for idx in val_idx],
    )
    test_data = ImageArraySplit(official_test.data, test_labels)

    train_labels = list(train_data.labels)
    val_labels = list(val_data.labels)

    print("\nDataset split (stratified):")
    print(f"  Train: {len(train_data):,} images")
    print(f"  Val:   {len(val_data):,} images (from official train)")
    print(f"  Test:  {len(test_data):,} images (official test, held out)")
    print(f"  Classes: {len(class_names)}")
    print(f"  Random seed: {random_seed}")

    return (
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
        class_names,
    )


def create_dataloaders(
    train_data,
    train_labels,
    val_data,
    val_labels,
    test_data,
    test_labels,
    batch_size,
    num_workers=4,
):
    """Create reproducible PyTorch DataLoaders from the CIFAR-100 splits."""
    train_dataset = CachedImageDataset(train_data, transform=get_transforms("train"))
    val_dataset = CachedImageDataset(val_data, transform=get_transforms("val"))
    test_dataset = CachedImageDataset(test_data, transform=get_transforms("test"))

    sampler = None
    use_shuffle = True
    if Config.USE_WEIGHTED_SAMPLER:
        class_sample_counts = torch.bincount(torch.tensor(train_labels))
        if torch.any(class_sample_counts == 0):
            raise ValueError("Weighted sampler cannot handle a class with zero samples")
        class_weights = 1.0 / class_sample_counts.float()
        sample_weights = class_weights[torch.tensor(train_labels)]
        sampler = WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        use_shuffle = False
        print("  WeightedRandomSampler: ENABLED")
    else:
        print("  WeightedRandomSampler: DISABLED (dataset is balanced)")

    use_cuda = torch.cuda.is_available()
    use_persistent = Config.PERSISTENT_WORKERS and num_workers > 0
    generator = torch.Generator().manual_seed(Config.RANDOM_SEED)

    common_loader_args = {
        "num_workers": num_workers,
        "pin_memory": Config.PIN_MEMORY and use_cuda,
        "persistent_workers": use_persistent,
    }
    if num_workers > 0:
        common_loader_args["prefetch_factor"] = Config.PREFETCH_FACTOR

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=use_shuffle,
        drop_last=Config.TRAIN_DROP_LAST,
        worker_init_fn=worker_init_fn_seed,
        generator=generator,
        **common_loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_loader_args,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_loader_args,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    Config.validate_config()
    loaded = load_dataset(
        Config.DATASET_NAME,
        Config.VALIDATION_RATIO,
        Config.RANDOM_SEED,
    )
    train_data, train_labels, val_data, val_labels, test_data, test_labels, _ = loaded
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
        Config.BATCH_SIZE,
        Config.NUM_WORKERS,
    )
    images, labels = next(iter(train_loader))
    print("\nDataset loaded successfully")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  First batch: images={tuple(images.shape)}, labels={tuple(labels.shape)}")
