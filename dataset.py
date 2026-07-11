"""Tiny ImageNet loading, preprocessing, and PyTorch DataLoaders."""

import random

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset as load_hf_dataset
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import Config


def worker_init_fn_seed(worker_id):
    """Seed Python's RNG independently in every DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)


class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.labels = [int(label) for label in hf_dataset["label"]]
        # Cache ảnh dạng PIL vào RAM ngay khi init
        print("Caching images to RAM...")
        # Tốt hơn nhiều - batch access
        all_images = hf_dataset["image"]  # Arrow đọc toàn bộ column một lần
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
    """Build full-image 224px transforms for ImageNet-pretrained models."""
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


def load_dataset(dataset_name, val_ratio=0.1, random_seed=42):
    """Load Tiny ImageNet and reserve the official validation split for testing.

    The Hugging Face dataset contains 100,000 labelled training images and 10,000
    labelled validation images. We stratify the official training split into
    90,000 training and 10,000 validation samples, while leaving the official
    validation split untouched as the final test set.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")

    print(f"\nLoading Hugging Face dataset: {dataset_name}")
    dataset = load_hf_dataset(dataset_name)

    required_splits = {Config.HF_TRAIN_SPLIT, Config.HF_TEST_SPLIT}
    missing_splits = required_splits.difference(dataset.keys())
    if missing_splits:
        raise ValueError(
            f"Dataset is missing required split(s): {sorted(missing_splits)}"
        )

    official_train = dataset[Config.HF_TRAIN_SPLIT]
    official_test = dataset[Config.HF_TEST_SPLIT]
    required_columns = {"image", "label"}
    for split_name, split_dataset in (
        (Config.HF_TRAIN_SPLIT, official_train),
        (Config.HF_TEST_SPLIT, official_test),
    ):
        missing_columns = required_columns.difference(split_dataset.column_names)
        if missing_columns:
            raise ValueError(
                f"Split '{split_name}' is missing column(s): {sorted(missing_columns)}"
            )

    split = official_train.train_test_split(
        test_size=val_ratio,
        stratify_by_column="label",
        seed=random_seed,
    )
    train_data = split["train"]
    val_data = split["test"]
    test_data = official_test

    label_feature = official_train.features["label"]
    class_names = list(label_feature.names)
    train_labels = [int(label) for label in train_data["label"]]
    val_labels = [int(label) for label in val_data["label"]]
    test_labels = [int(label) for label in test_data["label"]]

    print("\nDataset split (stratified):")
    print(f"  Train: {len(train_data):,} images")
    print(f"  Val:   {len(val_data):,} images (from official train)")
    print(f"  Test:  {len(test_data):,} images (official valid, held out)")
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
    """Create reproducible PyTorch DataLoaders from Hugging Face splits."""
    train_dataset = HFImageDataset(train_data, transform=get_transforms("train"))
    val_dataset = HFImageDataset(val_data, transform=get_transforms("val"))
    test_dataset = HFImageDataset(test_data, transform=get_transforms("test"))

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
