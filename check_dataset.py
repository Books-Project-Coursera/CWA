"""Validate that all cached Hugging Face Tiny ImageNet images can be decoded."""

from collections import Counter

from datasets import load_dataset as load_hf_dataset
from tqdm import tqdm

from config import Config


def check_dataset(dataset_name):
    """Decode every image and report split/class/size consistency."""
    dataset = load_hf_dataset(dataset_name)
    failures = []

    for split_name in (Config.HF_TRAIN_SPLIT, Config.HF_TEST_SPLIT):
        split = dataset[split_name]
        label_counts = Counter(int(label) for label in split["label"])
        image_sizes = Counter()

        for idx in tqdm(range(len(split)), desc=f"Checking {split_name}"):
            try:
                image = split[idx]["image"].convert("RGB")
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
    check_dataset(Config.DATASET_NAME)
