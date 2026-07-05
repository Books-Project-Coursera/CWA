"""
cifar100_configs.py  (NEW)

Per-model hyperparameters proposed for CIFAR-100, derived from the Strategy-2 paper
(strategy2.pdf, Tables 1-3) and the old per-dataset `config.csv`. Applied at RUNTIME by
setting attributes on the imported `Config` class in run_cifar100.py — the original
config.py file is never edited.

Common to all models (set in run_cifar100.py):
  - Scheduler : Linear Warmup (0.06 * epochs) -> Cosine Annealing (eta_min = 1e-6)
  - Loss      : CrossEntropy, label_smoothing = 0.1
  - Sampler   : USE_WEIGHTED_SAMPLER = False   (CIFAR-100 is class-balanced)
  - Image size: 224 (32 -> 224 upscale for ImageNet-pretrained backbones)
  - Strategy 2: TOP_K_VALUES = [2, 3, 4, 5], KEEP_TOP_K_CHECKPOINTS = 5

The 6 evaluated architectures match the paper exactly:
  ViT-B/16, VGG16, ResNet101, MobileNetV2, DenseNet121, EfficientNetB0.
"""

# model_name (repo key) -> hyperparameters
CIFAR100_CONFIGS = {
    "vit_base_patch16_224": dict(
        batch_size=64,  num_epochs=40, learning_rate=2e-5, weight_decay=1e-7,
        dropout_rate=0.5, classifier_config=[256, 128], early_stopping_patience=8,
    ),
    "vgg16": dict(
        # LR raised 5e-5 -> 5e-4 (paper's VGG16 value on Burmese/Potato) and cap 60 -> 100,
        # patience 12 -> 15: the 5e-5/60ep run never converged (hit cap, best epoch 56-59),
        # so VGG16 was re-run to proper convergence for a fair comparison.
        batch_size=64,  num_epochs=100, learning_rate=5e-4, weight_decay=1e-6,
        dropout_rate=0.5, classifier_config=[256, 128], early_stopping_patience=15,
    ),
    "resnet101": dict(
        batch_size=128, num_epochs=60, learning_rate=3e-4, weight_decay=1e-6,
        dropout_rate=0.5, classifier_config=[256, 128], early_stopping_patience=12,
    ),
    "mobilenet_v2": dict(
        batch_size=128, num_epochs=60, learning_rate=3e-4, weight_decay=1e-6,
        dropout_rate=0.5, classifier_config=[256, 128], early_stopping_patience=12,
    ),
    "densenet121": dict(
        batch_size=128, num_epochs=60, learning_rate=3e-4, weight_decay=1e-6,
        dropout_rate=0.5, classifier_config=[256, 128], early_stopping_patience=12,
    ),
    "efficientnet_b0": dict(
        batch_size=128, num_epochs=60, learning_rate=5e-4, weight_decay=1e-6,
        dropout_rate=0.5, classifier_config=[256, 128], early_stopping_patience=12,
    ),
}

# Canonical order used everywhere (matches paper Table column order).
MODEL_ORDER = [
    "vit_base_patch16_224",
    "vgg16",
    "resnet101",
    "mobilenet_v2",
    "densenet121",
    "efficientnet_b0",
]

# Pretty names for tables / reports (paper column headers).
PRETTY_NAME = {
    "vit_base_patch16_224": "ViT",
    "vgg16": "VGG16",
    "resnet101": "ResNet101",
    "mobilenet_v2": "MobileNetV2",
    "densenet121": "DenseNet121",
    "efficientnet_b0": "EfficientNetB0",
}

# The 5 training seeds — identical across all models (Seed 42 mandatory).
SEEDS = [1, 10, 42, 100, 500]

# Strategy 2 k-values reported (Baseline = k=1).
TOP_K_VALUES = [2, 3, 4, 5]
