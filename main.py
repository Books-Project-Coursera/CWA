"""
Main script to run complete baseline research pipeline
"""
import os
import argparse
import re
import shutil
import subprocess
import sys


def _preparse_environment_args():
    """Apply CUDA-related CLI args before torch is imported."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", "--cuda-visible-devices", dest="cuda_visible_devices")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cublas-workspace-config", default=":4096:8")
    args, _ = parser.parse_known_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", args.cublas_workspace_config)


_preparse_environment_args()
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

from config import Config
from dataset import load_dataset, create_dataloaders, concatenate_datasets
from train import train_model, CheckpointManager
from evaluate import evaluate_all_strategies, export_results_to_excel, create_performance_charts, save_confusion_matrices
from visualization import print_dataset_statistics


SUPPORTED_MODELS = [
    "vgg16",
    "resnet18",
    "resnet101",
    "mobilenet_v2",
    "densenet121",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "convit_tiny",
]

MODEL_ALIASES = {
    "vit_base": "vit_base_patch16_224",
    "vit_b16": "vit_base_patch16_224",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "mobilenetv2": "mobilenet_v2",
    "mobilenet_v2": "mobilenet_v2",
}


def sanitize_run_name(value):
    """Return a filesystem-safe run name."""
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._-") or "run"


def parse_model_list(values):
    """Parse repeated, space-separated, or comma-separated model args."""
    if not values:
        return None

    models = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                models.append(MODEL_ALIASES.get(item, item))

    if any(model.lower() == "all" for model in models):
        return SUPPORTED_MODELS.copy()

    invalid = [model for model in models if model not in SUPPORTED_MODELS]
    if invalid:
        raise ValueError(
            "Unsupported model(s): "
            + ", ".join(invalid)
            + ". Supported models: "
            + ", ".join(SUPPORTED_MODELS)
        )

    return models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate pretrained image classification models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "--models",
        dest="models",
        nargs="+",
        help="Model(s) to run. Use a space-separated list, comma-separated list, or 'all'.",
    )
    parser.add_argument(
        "--data-root",
        "--dataset-path",
        dest="data_root",
        help=(
            "Folder that contains 'cifar-100-python'. Overrides Config.DATA_ROOT. "
            "The dataset is never downloaded."
        ),
    )
    parser.add_argument("--results-dir", "--output-dir", dest="results_dir", help="Base directory for run outputs.")
    parser.add_argument(
        "--checkpoints-dir",
        help="Optional base directory for training checkpoints. A per-run subfolder is still created.",
    )
    parser.add_argument("--run-name", help="Optional readable name for this run folder.")
    parser.add_argument(
        "--gpu",
        "--cuda-visible-devices",
        dest="cuda_visible_devices",
        help="CUDA_VISIBLE_DEVICES value, e.g. 0, 1, or 0,1. Set before torch import.",
    )
    parser.add_argument("--batch-size", type=int, help="Override Config.BATCH_SIZE.")
    parser.add_argument("--epochs", type=int, help="Override Config.NUM_EPOCHS.")
    parser.add_argument("--warmup-epochs", type=int, help="Override Config.WARMUP_EPOCHS.")
    parser.add_argument("--eta-min", type=float, help="Override Config.ETA_MIN for CosineAnnealingLR.")
    parser.add_argument("--early-stopping", type=int, help="Override Config.EARLY_STOPPING_PATIENCE.")
    parser.add_argument(
        "--fc-layers",
        nargs="+",
        type=int,
        help="Hidden layer sizes for the classifier head, e.g. --fc-layers 256 128.",
    )
    parser.add_argument("--dropout", type=float, help="Override Config.DROPOUT_RATE.")
    parser.add_argument("--num-workers", type=int, help="Override Config.NUM_WORKERS.")
    parser.add_argument(
        "--profile-batches",
        type=int,
        help="Print DataLoader and compute timing for the first N training batches.",
    )
    parser.add_argument(
        "--dataset-stats",
        action="store_true",
        help="Print image-size dataset statistics before training. This opens up to 1000 images per run.",
    )
    parser.add_argument("--_method-leg", dest="_method_leg", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--methods",
        "--method",
        dest="methods",
        nargs="+",
        metavar="M",
        help=(
            "Method weight-averaging can chay: top-k | last-n | ema | swa "
            "(hoac 'all'/'none'). Cho phep to hop va dau phay: "
            "--methods top-k ema swa | --methods EMA,SWA. "
            "Strategy 1 (best checkpoint) LUON duoc eval lam baseline. "
            "'swa' doi LR schedule o 25%% cuoi nen tu dong duoc tach thanh mot "
            "experiment RIENG (run folder co hau to '_swa')."
        ),
    )
    parser.add_argument(
        "--ema-window-epochs",
        dest="ema_window_epochs",
        type=float,
        help="Override Config.EMA_WINDOW_EPOCHS - cua so trung binh cua EMA tinh theo epoch.",
    )
    parser.add_argument(
        "--ema-decay",
        dest="ema_decays",
        type=float,
        nargs="+",
        help="Override Config.EMA_DECAYS (dat tay). Bo qua de tu suy tu --ema-window-epochs.",
    )
    parser.add_argument(
        "--swa-lr",
        dest="swa_lr",
        type=float,
        help="Override Config.SWA_LR. Bo qua de tu tinh (LEARNING_RATE + ETA_MIN) / 2.",
    )
    parser.add_argument(
        "--swa-lr-start",
        dest="swa_lr_start_frac",
        type=float,
        help="Override Config.SWA_LR_START_FRAC - moc chuyen sang constant LR (mac dinh 0.75).",
    )
    parser.add_argument(
        "--swa-lr-schedule",
        dest="swa_lr_schedule",
        choices=["truncate", "inherit"],
        help=(
            "'truncate' (mac dinh): 75%% dau chay scheduler chuan, 25%% sau giu LR hang so. "
            "'inherit': SWA average ngay tren cosine, khong doi LR (cung trajectory)."
        ),
    )
    parser.add_argument("--seed", type=int, help="Override Config.RANDOM_SEED.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Run these seeds sequentially. If omitted and --seed is not set, Config.SEEDS is used.",
    )
    parser.add_argument("--lr", type=float, help="Override Config.LEARNING_RATE.")
    parser.add_argument("--weight-decay", type=float, help="Override Config.WEIGHT_DECAY.")
    parser.add_argument(
        "--optimizer",
        choices=Config.SUPPORTED_OPTIMIZERS,
        help="Override Config.OPTIMIZER.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        help="Override Config.SGD_MOMENTUM (used by sgd and rmsprop).",
    )
    parser.add_argument(
        "--no-nesterov",
        action="store_true",
        help="Disable Nesterov momentum for --optimizer sgd.",
    )
    parser.add_argument(
        "--no-fused-optimizer",
        action="store_true",
        help="Disable fused optimizer kernels even on CUDA.",
    )
    parser.add_argument("--cv", action="store_true", help="Enable cross-validation.")
    parser.add_argument("--no-cv", action="store_true", help="Disable cross-validation.")
    parser.add_argument("--cv-splits", type=int, help="Override Config.CV_N_SPLITS.")
    parser.add_argument("--weighted-sampler", action="store_true", help="Enable WeightedRandomSampler.")
    parser.add_argument("--no-weighted-sampler", action="store_true", help="Disable WeightedRandomSampler.")
    parser.add_argument("--auto-delete-checkpoints", action="store_true", help="Delete training checkpoints after evaluation.")
    parser.add_argument("--keep-checkpoints", action="store_true", help="Keep training checkpoints after evaluation.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CUDA behavior. Slower, but more reproducible.",
    )
    parser.add_argument(
        "--cublas-workspace-config",
        default=":4096:8",
        help="CUBLAS_WORKSPACE_CONFIG used only with --deterministic.",
    )
    return parser.parse_args()


def apply_cli_overrides(args):
    if getattr(args, "methods", None):
        Config.normalize_methods(args.methods)
    for attr, value in (
        ("EMA_WINDOW_EPOCHS", getattr(args, "ema_window_epochs", None)),
        ("EMA_DECAYS", getattr(args, "ema_decays", None)),
        ("SWA_LR", getattr(args, "swa_lr", None)),
        ("SWA_LR_START_FRAC", getattr(args, "swa_lr_start_frac", None)),
        ("SWA_LR_SCHEDULE", getattr(args, "swa_lr_schedule", None)),
    ):
        if value is not None:
            setattr(Config, attr, value)
    models = parse_model_list(args.models)
    if models:
        Config.MODELS = models

    overrides = [
        ("DATA_ROOT", args.data_root),
        ("RESULTS_DIR", args.results_dir),
        ("CHECKPOINTS_DIR", args.checkpoints_dir),
        ("BATCH_SIZE", args.batch_size),
        ("NUM_EPOCHS", args.epochs),
        ("WARMUP_EPOCHS", args.warmup_epochs),
        ("ETA_MIN", args.eta_min),
        ("EARLY_STOPPING_PATIENCE", args.early_stopping),
        ("DROPOUT_RATE", args.dropout),
        ("NUM_WORKERS", args.num_workers),
        ("PROFILE_BATCHES", args.profile_batches),
        ("RANDOM_SEED", args.seed),
        ("LEARNING_RATE", args.lr),
        ("WEIGHT_DECAY", args.weight_decay),
        ("OPTIMIZER", args.optimizer),
        ("SGD_MOMENTUM", args.momentum),
        ("CV_N_SPLITS", args.cv_splits),
    ]
    for attr, value in overrides:
        if value is not None:
            setattr(Config, attr, value)

    if args.fc_layers is not None:
        Config.CLASSIFIER_CONFIG = args.fc_layers

    if args.no_nesterov:
        Config.SGD_NESTEROV = False

    if args.no_fused_optimizer:
        Config.USE_FUSED_OPTIMIZER = False

    if args.dataset_stats:
        Config.PRINT_DATASET_STATS = True

    if args.epochs is not None and args.warmup_epochs is None:
        Config.WARMUP_EPOCHS = min(5, max(1, int(Config.NUM_EPOCHS * 0.1)))

    if args.cv and args.no_cv:
        raise ValueError("Use only one of --cv or --no-cv.")
    if args.cv:
        Config.USE_CROSS_VALIDATION = True
    if args.no_cv:
        Config.USE_CROSS_VALIDATION = False

    if args.weighted_sampler and args.no_weighted_sampler:
        raise ValueError("Use only one of --weighted-sampler or --no-weighted-sampler.")
    if args.weighted_sampler:
        Config.USE_WEIGHTED_SAMPLER = True
    if args.no_weighted_sampler:
        Config.USE_WEIGHTED_SAMPLER = False

    if args.auto_delete_checkpoints and args.keep_checkpoints:
        raise ValueError("Use only one of --auto-delete-checkpoints or --keep-checkpoints.")
    if args.auto_delete_checkpoints:
        Config.AUTO_DELETE_CHECKPOINTS = True
    if args.keep_checkpoints:
        Config.AUTO_DELETE_CHECKPOINTS = False


def run_method_legs_if_needed(args):
    """
    Tach Config.METHODS thanh cac "leg" co LR SCHEDULE KHAC NHAU roi chay tuan tu.

    Vi sao phai tach:

    - top-k / last-n / ema chi QUAN SAT weights, KHONG doi LR schedule => chung
      mot run, cung mot trajectory => so sanh PAIRED theo seed.
    - swa o mode 'truncate' giu LR HANG SO o 25% cuoi => DOI trajectory => bat
      buoc run rieng. Neu de chung, so lieu top-k/last-n/ema se duoc tinh tren
      mot LR schedule KHONG PHAI cua chung.

    Moi leg la mot process rieng (giong run_seed_jobs_if_needed) de Config va
    scheduler khong bi dinh trang thai cua leg truoc.

    Returns:
        True neu da spawn cac leg con (caller nen return ngay).
    """
    if getattr(args, "_method_leg", False):
        return False
    if not Config.swa_changes_schedule():
        return False

    shared = [m for m in Config.METHODS if m != "swa"]
    if not shared:
        return False

    base_args = [a for a in sys.argv[1:]]
    # Bo --methods/--method va --run-name cu: moi leg tu dat gia tri rieng, de
    # lai se thanh co trung lap trong argv cua process con.
    strip_flags = ("--methods", "--method", "--run-name")
    cleaned = []
    skip = False
    for token in base_args:
        if token in strip_flags or any(token.startswith(f + "=") for f in strip_flags):
            skip = not token.count("=")
            continue
        if skip:
            if token.startswith("-"):
                skip = False
            else:
                continue
        cleaned.append(token)

    base_name = sanitize_run_name(args.run_name) if args.run_name else sanitize_run_name(
        "_".join(Config.MODELS) if Config.MODELS else "models"
    )

    legs = [
        {"methods": shared, "run_name": base_name},
        {"methods": ["swa"], "run_name": base_name + "_swa"},
    ]

    print("" + chr(10) + "=" * 70)
    print(" %d EXPERIMENT RIENG BIET (LR schedule khac nhau)" % len(legs))
    for idx, leg in enumerate(legs, 1):
        sched = "scheduler chuan" if "swa" not in leg["methods"] else (
            "constant LR %g tu %.0f%% budget" % (Config.resolved_swa_lr(),
                                                 float(Config.SWA_LR_START_FRAC) * 100))
        print("   %d. %-28s methods=%s  |  %s" % (idx, leg["run_name"], leg["methods"], sched))
    print("=" * 70)

    for idx, leg in enumerate(legs, 1):
        child = cleaned + ["--methods", *leg["methods"], "--run-name", leg["run_name"],
                           "--_method-leg"]
        print("" + chr(10) + "#" * 70)
        print("#  EXPERIMENT %d/%d: methods=%s | run_name=%s"
              % (idx, len(legs), leg["methods"], leg["run_name"]))
        print("#" * 70)
        subprocess.run([sys.executable, __file__, *child], check=True)

    print("" + chr(10) + "=" * 70)
    print(" METHOD LEGS COMPLETED")
    print("=" * 70)
    return True


def run_seed_jobs_if_needed(args):
    """Run each configured seed as a separate process with its own output folder."""
    if args.seed is not None:
        return False

    seeds = args.seeds if args.seeds else getattr(Config, "SEEDS", None)
    if not seeds:
        return False

    seeds = [int(seed) for seed in seeds]
    if len(seeds) <= 1:
        Config.RANDOM_SEED = seeds[0]
        return False

    base_args = sys.argv[1:]
    original_run_name = args.run_name

    print("\n" + "=" * 70)
    print(f" MULTI-SEED RUN: {seeds}")
    print("=" * 70)

    for seed in seeds:
        child_args = base_args + ["--seed", str(seed)]
        if original_run_name:
            child_args += ["--run-name", f"{sanitize_run_name(original_run_name)}_seed{seed}"]
        else:
            model_part = "_".join(Config.MODELS) if Config.MODELS else "models"
            child_args += ["--run-name", f"{sanitize_run_name(model_part)}_seed{seed}"]

        print(f"\n[Seed {seed}] Starting: {sys.executable} {os.path.basename(__file__)} {' '.join(child_args)}")
        subprocess.run([sys.executable, __file__, *child_args], check=True)

    print("\n" + "=" * 70)
    print(" MULTI-SEED RUN COMPLETED")
    print("=" * 70)
    return True


def get_next_run_folder(base_results_dir, run_name=None):
    """
    Tạo folder mới cho mỗi lần chạy
    Tự động tăng số thứ tự: results/1/, results/2/, results/3/, ...
    
    Args:
        base_results_dir: Thư mục results gốc
    
    Returns:
        run_folder: Đường dẫn đến folder cho lần chạy này
        run_number: Số thứ tự lần chạy
    """
    os.makedirs(base_results_dir, exist_ok=True)

    if run_name:
        safe_name = sanitize_run_name(run_name)
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        candidates = [
            safe_name,
            f"{safe_name}_{suffix}_{pid}",
        ]

        for candidate in candidates:
            run_folder = os.path.join(base_results_dir, candidate)
            try:
                os.mkdir(run_folder)
                return run_folder, candidate
            except FileExistsError:
                continue

        counter = 1
        while True:
            run_id = f"{safe_name}_{suffix}_{pid}_{counter}"
            run_folder = os.path.join(base_results_dir, run_id)
            try:
                os.mkdir(run_folder)
                return run_folder, run_id
            except FileExistsError:
                counter += 1
    
    # Tìm tất cả các folder có dạng số
    existing_runs = []
    for item in os.listdir(base_results_dir):
        item_path = os.path.join(base_results_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            existing_runs.append(int(item))
    
    # Tìm số tiếp theo
    next_run = max(existing_runs) + 1 if existing_runs else 1
    
    # Tạo folder mới
    while True:
        run_folder = os.path.join(base_results_dir, str(next_run))
        try:
            os.mkdir(run_folder)
            return run_folder, next_run
        except FileExistsError:
            next_run += 1


def save_model_results(model_name, results, output_dir):
    """
    Save individual model results to Excel (2 sheets: macro + per-class)

    Args:
        model_name: Name of the model
        results: Dictionary with strategy results
        output_dir: Directory to save results
    """
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Sheet 1: Macro-averaged metrics
    macro_rows = []
    for strategy_name, result in results.items():
        row = {
            'Model': model_name,
            'Strategy': strategy_name,
            **result['metrics'],
            'Total Time (s)': round(
                float(result.get('timing', {}).get('total_seconds', 0.0)), 1
            ),
        }
        macro_rows.append(row)

    df_macro = pd.DataFrame(macro_rows)

    # Reorder columns (including Test Loss)
    column_order = ['Model', 'Strategy', 'Test Loss', 'Accuracy (%)', 'Precision (%)',
                   'Recall (%)', 'F1-Score (%)', 'AUC (%)', 'Total Time (s)']
    df_macro = df_macro[column_order]

    # Sheet 2: Per-class metrics
    per_class_rows = []
    for strategy_name, result in results.items():
        for cls_name, cls_metrics in result['per_class'].items():
            pc_row = {
                'Model': model_name,
                'Strategy': strategy_name,
                'Class': cls_name,
                **cls_metrics
            }
            per_class_rows.append(pc_row)

    df_per_class = pd.DataFrame(per_class_rows)
    pc_column_order = ['Model', 'Strategy', 'Class', 'Precision (%)', 'Recall (%)',
                       'F1-Score (%)', 'Specificity (%)', 'AUC (%)', 'Support']
    df_per_class = df_per_class[pc_column_order]

    # Save both sheets to Excel
    excel_path = os.path.join(model_dir, f'{model_name}_results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_macro.to_excel(writer, sheet_name='Macro Results', index=False)
        df_per_class.to_excel(writer, sheet_name='Per-Class Results', index=False)
    print(f"  ✓ Results saved to: {excel_path}")

    return df_macro


def delete_model_checkpoints(model_name, checkpoints_dir):
    """
    Delete all checkpoints for a specific model
    
    Args:
        model_name: Name of the model
        checkpoints_dir: Base checkpoints directory
    """
    model_checkpoint_dir = os.path.join(checkpoints_dir, model_name)
    
    if os.path.exists(model_checkpoint_dir):
        try:
            shutil.rmtree(model_checkpoint_dir)
            print(f"  ✓ Deleted checkpoints: {model_checkpoint_dir}")
        except Exception as e:
            print(f"  ✗ Error deleting checkpoints: {str(e)}")
    else:
        print(f"  ⚠ No checkpoints found at: {model_checkpoint_dir}")


def export_run_config(run_folder, num_classes=None, class_names=None, 
                      train_count=0, val_count=0, test_count=0):
    """
    Xuất toàn bộ config của lần chạy ra file Excel (run_config.xlsx).
    Mỗi nhóm config = 1 sheet riêng, dễ đọc và so sánh giữa các lần chạy.
    Nếu tắt focal loss → các param focal tự set 0/None.
    Nếu tắt WRS → ghi rõ DISABLED.
    
    Args:
        run_folder: Folder lưu kết quả của lần chạy
        num_classes: Số lượng class
        class_names: Danh sách tên class
        train_count, val_count, test_count: Số lượng ảnh mỗi split
    """
    import platform
    import timm
    import torchvision

    is_focal = Config.LOSS_FUNCTION == 'poly_focal'
    cuda_available = torch.cuda.is_available()
    gpu_names = (
        ", ".join(
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        )
        if cuda_available else "CPU only"
    )
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = "N/A"
        git_dirty = "N/A"

    dataset_rows = [
        ("dataset_name", Config.DATASET_NAME),
        ("data_root", Config.DATA_ROOT),
        ("dataset_source", "torchvision.datasets.CIFAR100 (download=False)"),
        ("official_train_split", Config.OFFICIAL_TRAIN_SPLIT),
        ("official_test_split", Config.OFFICIAL_TEST_SPLIT),
        ("num_classes", num_classes),
        ("class_names", ", ".join(class_names) if class_names else ""),
        ("validation_ratio_from_official_train", Config.VALIDATION_RATIO),
        ("train_samples", train_count),
        ("val_samples", val_count),
        ("test_samples", test_count),
        ("image_size", Config.IMAGE_SIZE),
        ("resize_interpolation", Config.RESIZE_INTERPOLATION),
        ("normalization_mean", str(Config.IMAGE_MEAN)),
        ("normalization_std", str(Config.IMAGE_STD)),
        ("random_seed", Config.RANDOM_SEED),
        ("all_configured_seeds", str(Config.SEEDS)),
    ]

    model_rows = [
        ("models", ", ".join(Config.MODELS)),
        ("vit_pretrained_model_id", Config.VIT_PRETRAINED_MODEL_ID),
        ("pretrained", Config.PRETRAINED),
        ("classifier_config", str(Config.CLASSIFIER_CONFIG)),
        ("classifier_dropout_rate", Config.DROPOUT_RATE),
        ("model_drop_rate", Config.MODEL_DROP_RATE),
        ("attention_drop_rate", Config.MODEL_ATTN_DROP_RATE),
        ("drop_path_rate", Config.MODEL_DROP_PATH_RATE),
    ]

    training_rows = [
        ("batch_size_train_val_test", Config.BATCH_SIZE),
        ("effective_global_batch_size", Config.BATCH_SIZE),
        ("num_epochs", Config.NUM_EPOCHS),
        ("early_stopping_patience", Config.EARLY_STOPPING_PATIENCE),
        ("gradient_clip_norm", Config.GRAD_CLIP_NORM),
    ]

    is_adam_family = Config.OPTIMIZER.lower() in ("adam", "adamw")
    optimizer_scheduler_rows = [
        ("optimizer", Config.OPTIMIZER),
        ("optimizer_betas", str(Config.OPTIMIZER_BETAS) if is_adam_family else "N/A"),
        ("optimizer_epsilon", Config.OPTIMIZER_EPS if Config.OPTIMIZER.lower() != "sgd" else "N/A"),
        ("sgd_momentum", Config.SGD_MOMENTUM if not is_adam_family else "N/A"),
        ("sgd_nesterov", Config.SGD_NESTEROV if Config.OPTIMIZER.lower() == "sgd" else "N/A"),
        ("rmsprop_alpha", Config.RMSPROP_ALPHA if Config.OPTIMIZER.lower() == "rmsprop" else "N/A"),
        ("weight_decay_mode", "decoupled (AdamW)" if Config.OPTIMIZER.lower() == "adamw" else "coupled L2"),
        ("fused_optimizer_requested", Config.USE_FUSED_OPTIMIZER),
        ("fused_optimizer_effective", Config.USE_FUSED_OPTIMIZER and cuda_available),
        ("learning_rate", Config.LEARNING_RATE),
        ("weight_decay", Config.WEIGHT_DECAY),
        ("weight_decay_exclusions", "bias, 1D/norm params, model no_weight_decay set"),
        ("scheduler", Config.SCHEDULER),
        ("warmup_epochs", Config.WARMUP_EPOCHS),
        ("warmup_start_factor", Config.WARMUP_START_FACTOR),
        ("cosine_eta_min", Config.ETA_MIN),
    ]

    loss_rows = [
        ("loss_function", Config.LOSS_FUNCTION),
        ("label_smoothing", Config.LABEL_SMOOTHING if not is_focal else 0),
        ("focal_gamma", Config.FOCAL_GAMMA if is_focal else 0),
        ("poly_epsilon", Config.POLY_EPSILON if is_focal else 0),
        ("class_weight_method", Config.CLASS_WEIGHT_METHOD if is_focal else "N/A"),
    ]

    augmentation_rows = [
        ("horizontal_flip_probability", Config.HORIZONTAL_FLIP_PROB),
        ("use_mixup_cutmix", Config.USE_MIXUP_CUTMIX),
        ("mixup_alpha", Config.MIXUP_ALPHA if Config.USE_MIXUP_CUTMIX else 0),
        ("cutmix_alpha", Config.CUTMIX_ALPHA if Config.USE_MIXUP_CUTMIX else 0),
        ("mixup_or_cutmix_probability", Config.MIXUP_PROB if Config.USE_MIXUP_CUTMIX else 0),
        ("cutmix_switch_probability", Config.MIXUP_SWITCH_PROB if Config.USE_MIXUP_CUTMIX else 0),
        ("mixup_mode", Config.MIXUP_MODE if Config.USE_MIXUP_CUTMIX else "disabled"),
        ("random_erasing_probability", Config.RANDOM_ERASING_PROB),
        ("random_erasing_scale", str(Config.RANDOM_ERASING_SCALE)),
        ("random_erasing_ratio", str(Config.RANDOM_ERASING_RATIO)),
        ("random_erasing_value", Config.RANDOM_ERASING_VALUE),
        ("validation_augmentation", "resize + normalize only"),
        ("test_augmentation", "resize + normalize only"),
    ]

    precision_dataloader_rows = [
        ("amp_requested", Config.USE_AMP),
        ("amp_dtype", Config.AMP_DTYPE if Config.USE_AMP else "float32"),
        ("amp_effective", Config.USE_AMP and cuda_available),
        ("float32_matmul_precision", Config.FLOAT32_MATMUL_PRECISION),
        ("torch_compile_requested", Config.USE_TORCH_COMPILE),
        ("torch_compile_mode", Config.TORCH_COMPILE_MODE),
        ("torch_compile_effective", Config.USE_TORCH_COMPILE and cuda_available),
        ("num_workers", Config.NUM_WORKERS),
        ("prefetch_factor", Config.PREFETCH_FACTOR),
        ("persistent_workers", Config.PERSISTENT_WORKERS),
        ("pin_memory", Config.PIN_MEMORY),
        ("train_drop_last", Config.TRAIN_DROP_LAST),
    ]

    sampler_cv_rows = [
        ("use_weighted_random_sampler", Config.USE_WEIGHTED_SAMPLER),
        ("use_cross_validation", Config.USE_CROSS_VALIDATION),
        ("cv_n_splits", Config.CV_N_SPLITS if Config.USE_CROSS_VALIDATION else 0),
        ("cv_pool", "official train only" if Config.USE_CROSS_VALIDATION else "N/A"),
        ("external_test", f"official {Config.OFFICIAL_TEST_SPLIT}"),
    ]

    eval_rows = [
        ("top_k_values", str(Config.TOP_K_VALUES)),
        ("last_n_epochs", Config.LAST_N_EPOCHS),
        ("keep_last_n_checkpoints", Config.KEEP_LAST_N_CHECKPOINTS),
        ("keep_top_k_checkpoints", Config.KEEP_TOP_K_CHECKPOINTS),
        ("auto_delete_checkpoints", Config.AUTO_DELETE_CHECKPOINTS),
        ("save_strategy_checkpoints", Config.SAVE_STRATEGY_CHECKPOINTS),
        ("experiment_name", Config.EXPERIMENT_NAME),
    ]

    runtime_rows = [
        ("run_timestamp", datetime.now().isoformat(timespec="seconds")),
        ("command", " ".join([sys.executable, *sys.argv])),
        ("hostname", platform.node()),
        ("platform", platform.platform()),
        ("git_commit", git_commit),
        ("git_worktree_dirty", git_dirty),
        ("python_version", platform.python_version()),
        ("torch_version", torch.__version__),
        ("torchvision_version", torchvision.__version__),
        ("timm_version", timm.__version__),
        ("cuda_available", cuda_available),
        ("cuda_runtime_version", torch.version.cuda or "N/A"),
        ("cudnn_version", torch.backends.cudnn.version() or "N/A"),
        ("visible_gpu_count", torch.cuda.device_count()),
        ("gpu_names", gpu_names),
        ("bf16_supported", torch.cuda.is_bf16_supported() if cuda_available else False),
    ]

    notes_rows = [
        ("WRS + Focal Loss", 
         "BOTH ACTIVE - WRS handles imbalance at data level, Focal Loss at loss level. "
         "May double-correct for imbalance." 
         if (Config.USE_WEIGHTED_SAMPLER and is_focal) else "No conflict"),
        ("Metrics Averaging", 
         "All metrics (Precision, Recall, F1, AUC) use MACRO averaging. "
            "Accuracy is computed as overall (correct/total)."),
        ("Batch 1024 LR", "Use learning_rate=1e-4 when scaling train batch from 512 to 1024."),
        ("Train accuracy with Mixup", "Expected correctness under soft target distributions."),
    ]
    
    config_path = os.path.join(run_folder, "run_config.xlsx")
    with pd.ExcelWriter(config_path, engine='openpyxl') as writer:
        for sheet_name, rows in [
            ("Dataset & Splitting", dataset_rows),
            ("Model", model_rows),
            ("Training Hyperparams", training_rows),
            ("Optimizer & Scheduler", optimizer_scheduler_rows),
            ("Loss Function", loss_rows),
            ("Augmentation", augmentation_rows),
            ("Precision & DataLoader", precision_dataloader_rows),
            ("Sampler & CV", sampler_cv_rows),
            ("Evaluation & Output", eval_rows),
            ("Runtime Environment", runtime_rows),
            ("Notes", notes_rows),
        ]:
            df = pd.DataFrame(rows, columns=["Parameter", "Value"])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"  [OK] Full run config exported to: {config_path}")
    return config_path


def main():
    """
    Main pipeline (Process each model sequentially to save disk space):
    1. Validate configuration
    2. Load and split dataset
    3. FOR EACH MODEL:
       - Train model
       - Evaluate with 3 strategies
       - Save individual results
       - Delete checkpoints
    4. Combine all results to Excel
    5. Generate combined performance charts
    """
    args = parse_args()
    apply_cli_overrides(args)
    torch.set_float32_matmul_precision(Config.FLOAT32_MATMUL_PRECISION)
    if run_method_legs_if_needed(args):
        return
    if run_seed_jobs_if_needed(args):
        return

    print("\n" + "="*70)
    print(" BASELINE RESEARCH - PRETRAINED MODELS EVALUATION")
    print("="*70)
    
    # Set random seeds for reproducibility across ALL models
    print(f"\n🔒 Setting random seeds for reproducibility (seed={Config.RANDOM_SEED})...")
    import random
    import numpy as np
    random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.RANDOM_SEED)
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)
        if args.deterministic:
            # Reproducibility mode is slower and requires CUBLAS_WORKSPACE_CONFIG.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    print("✓ Random seeds set successfully")
    
    # Step 1: Validate configuration
    print("\n[Step 1/6] Validating configuration...")
    Config.validate_config()
    
    # Tạo folder riêng cho lần chạy này
    run_folder, run_number = get_next_run_folder(Config.RESULTS_DIR, args.run_name)
    print(f"\n📁 Lần chạy thứ: {run_number}")
    print(f"📁 Kết quả sẽ được lưu tại: {run_folder}")
    
    # Keep training checkpoints isolated per run to allow concurrent terminals.
    if args.checkpoints_dir:
        run_checkpoints_dir = os.path.join(Config.CHECKPOINTS_DIR, sanitize_run_name(str(run_number)))
    else:
        run_checkpoints_dir = os.path.join(run_folder, "training_checkpoints")
    os.makedirs(run_checkpoints_dir, exist_ok=True)
    print(f"Training checkpoints: {run_checkpoints_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        if Config.USE_AMP and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "USE_AMP=True requires a CUDA GPU with BF16 support for this pipeline"
            )
        print(
            f"H100 profile: batch={Config.BATCH_SIZE}, "
            f"precision={Config.AMP_DTYPE if Config.USE_AMP else 'float32'}, "
            f"workers={Config.NUM_WORKERS}"
        )
    else:
        print("⚠ Running in CPU mode. Training will be slower but uses less memory.")
        print("  To enable GPU: Increase Windows virtual memory (paging file) to 16-32GB")
    
    # Step 2: Load dataset
    print("\n[Step 2/6] Loading and splitting dataset...")
    train_data, train_labels, val_data, val_labels, test_data, test_labels, class_names = load_dataset(
        Config.DATASET_NAME,
        Config.VALIDATION_RATIO,
        Config.RANDOM_SEED
    )
    
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # Export full run config
    print("\n[Config] Exporting run configuration...")
    export_run_config(
        run_folder, 
        num_classes=num_classes, 
        class_names=class_names,
        train_count=len(train_data),
        val_count=len(val_data),
        test_count=len(test_data)
    )
    
    # Warn if both WRS and Focal Loss are active
    if Config.USE_WEIGHTED_SAMPLER and Config.LOSS_FUNCTION == 'poly_focal':
        print("\n⚠ WARNING: Cả WeightedRandomSampler và PolyFocalLoss đều đang BẬT!")
        print("  → WRS xử lý imbalance ở data level (oversampling minority class)")
        print("  → Focal Loss xử lý imbalance ở loss level (focus on hard examples)")
        print("  → Có thể gây double-correction. Hãy cân nhắc tắt 1 trong 2 nếu kết quả không tốt.")
    
    # ===================== CROSS-VALIDATION MODE (Pure K-Fold) =====================
    if Config.USE_CROSS_VALIDATION:
        print(f"\n{'='*70}")
        print(f" PURE CROSS-VALIDATION ({Config.CV_N_SPLITS}-Fold Stratified)")
        print(f"{'='*70}")
        
        # Fold only the official training pool; keep official valid untouched.
        all_data = concatenate_datasets([train_data, val_data])
        all_labels = train_labels + val_labels
        
        print(f"  CV pool: {len(all_data)} official-train images")
        print(f"  External test: {len(test_data)} official-test images")
        
        skf = StratifiedKFold(
            n_splits=Config.CV_N_SPLITS, 
            shuffle=True, 
            random_state=Config.RANDOM_SEED
        )
        
        all_fold_results = {}  # {model_name: [fold_results]}
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_data, all_labels), 1):
            print(f"\n{'='*70}")
            print(f" FOLD {fold_idx}/{Config.CV_N_SPLITS}")
            print(f"{'='*70}")
            
            fold_train_data = all_data.select(train_idx.tolist())
            fold_train_labels = [all_labels[i] for i in train_idx]
            fold_val_data = all_data.select(val_idx.tolist())
            fold_val_labels = [all_labels[i] for i in val_idx]
            
            print(
                f"  Train: {len(fold_train_data)} | "
                f"Val (fold): {len(fold_val_data)} | "
                f"Test (external): {len(test_data)}"
            )
            
            # Create dataloaders for this fold
            fold_train_loader, fold_val_loader, fold_test_loader = create_dataloaders(
                fold_train_data, fold_train_labels,
                fold_val_data, fold_val_labels,
                test_data, test_labels,
                Config.BATCH_SIZE,
                Config.NUM_WORKERS
            )
            
            fold_folder = os.path.join(run_folder, f"fold_{fold_idx}")
            os.makedirs(fold_folder, exist_ok=True)
            
            for model_name in Config.MODELS:
                print(f"\n  [Fold {fold_idx}] Training {model_name}...")
                try:
                    fold_ckpt_dir = os.path.join(fold_folder, model_name, 'training_checkpoints')
                    checkpoint_manager, history, train_extra = train_model(
                        model_name,
                        fold_train_loader,
                        fold_val_loader,
                        num_classes,
                        device,
                        class_names=class_names,
                        train_labels=fold_train_labels,
                        save_dir=fold_folder,
                        checkpoints_dir=fold_ckpt_dir
                    )
                    
                    # Evaluate trên fold test set (phần data luân phiên làm test)
                    strategy_ckpt_dir = (
                        os.path.join(fold_folder, model_name, 'checkpoints')
                        if Config.SAVE_STRATEGY_CHECKPOINTS else None
                    )
                    results = evaluate_all_strategies(
                        model_name, checkpoint_manager, fold_test_loader, fold_train_loader,
                        num_classes, device, class_names=class_names, save_dir=strategy_ckpt_dir,
                        shadows=train_extra.get('shadows'),
                        train_seconds=train_extra.get('train_seconds', 0.0),
                    )
                    
                    if model_name not in all_fold_results:
                        all_fold_results[model_name] = []
                    all_fold_results[model_name].append(results)

                    # In kết quả tất cả strategies của fold này
                    print(f"\n  📊 Fold {fold_idx} - {model_name}:")
                    for strategy_name, result in results.items():
                        m = result['metrics']
                        print(f"     {strategy_name:<30} Acc: {m['Accuracy (%)']:.2f}% | F1: {m['F1-Score (%)']:.2f}% | AUC: {m['AUC (%)']:.2f}%")

                    save_model_results(model_name, results, fold_folder)

                    # === Chỉ giữ lại Strategy 1 checkpoint (best val_loss) của fold này ===
                    if strategy_ckpt_dir and os.path.exists(strategy_ckpt_dir):
                        pth_files = [f for f in os.listdir(strategy_ckpt_dir) if f.endswith('.pth')]
                        strategy1_src = os.path.join(strategy_ckpt_dir, 'Strategy_1_best.pth')
                        strategy1_dst = os.path.join(strategy_ckpt_dir, f'strategy1_fold{fold_idx}_checkpoint.pth')

                        # Rename Strategy 1 checkpoint to a stable name first (avoid collision)
                        if os.path.exists(strategy1_src):
                            os.rename(strategy1_src, strategy1_dst)

                        # Delete all other strategy checkpoints
                        for fname in pth_files:
                            if fname == 'Strategy_1_best.pth':
                                continue  # already renamed above
                            fpath = os.path.join(strategy_ckpt_dir, fname)
                            try:
                                os.remove(fpath)
                            except Exception as e:
                                print(f"    ⚠ Could not delete {fname}: {e}")

                        if os.path.exists(strategy1_dst):
                            print(f"    ✓ Kept: Strategy 1 (best val_loss) → strategy1_fold{fold_idx}_checkpoint.pth")
                        else:
                            print(f"    ⚠ Strategy 1 checkpoint not found in {strategy_ckpt_dir}")

                    # Xóa training checkpoints (epoch_*.pth, best_checkpoint.pth, etc.)
                    if os.path.exists(fold_ckpt_dir):
                        try:
                            shutil.rmtree(fold_ckpt_dir)
                            print(f"    🧹 Deleted training checkpoints: {fold_ckpt_dir}")
                        except Exception as e:
                            print(f"    ⚠ Could not delete training checkpoints: {e}")
                    
                    print(f"  ✅ Fold {fold_idx} - {model_name} completed")
                    
                except Exception as e:
                    print(f"  ✗ Fold {fold_idx} - {model_name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # ===================== Tổng hợp kết quả CV =====================
        print(f"\n{'='*70}")
        print(f" CROSS-VALIDATION SUMMARY ({Config.CV_N_SPLITS}-Fold)")
        print(f"{'='*70}")

        metric_keys = ['Test Loss', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'AUC (%)']

        # === Sheet 1: CV Summary — Mean ± Std per strategy per model ===
        cv_summary_rows = []
        for model_name, fold_results_list in all_fold_results.items():
            strategy_names = list(fold_results_list[0].keys())
            for strategy_name in strategy_names:
                fold_metrics = []
                for fold_res in fold_results_list:
                    if strategy_name in fold_res:
                        fold_metrics.append(fold_res[strategy_name]['metrics'])

                if fold_metrics:
                    row = {'Model': model_name, 'Strategy': strategy_name}
                    for key in metric_keys:
                        values = [m[key] for m in fold_metrics if key in m]
                        if values:
                            mean_v = np.mean(values)
                            std_v  = np.std(values)
                            fmt = '.4f' if key == 'Test Loss' else '.2f'
                            row[f"{key} (mean)"] = round(mean_v, 4 if key == 'Test Loss' else 2)
                            row[f"{key} (std)"]  = round(std_v,  4 if key == 'Test Loss' else 2)
                            row[f"{key} (mean ± std)"] = f"{mean_v:{fmt}} ± {std_v:{fmt}}"
                    cv_summary_rows.append(row)

        cv_summary_df = pd.DataFrame(cv_summary_rows)

        # === Sheet 2: Fold Details — raw values per fold per model per strategy ===
        cv_detail_rows = []
        for model_name, fold_results_list in all_fold_results.items():
            for fold_idx_0, fold_res in enumerate(fold_results_list):
                for strategy_name, result in fold_res.items():
                    m = result['metrics']
                    detail_row = {'Model': model_name, 'Fold': fold_idx_0 + 1, 'Strategy': strategy_name}
                    for key in metric_keys:
                        if key in m:
                            detail_row[key] = round(m[key], 4 if key == 'Test Loss' else 2)
                    cv_detail_rows.append(detail_row)

        cv_detail_df = pd.DataFrame(cv_detail_rows)
        if not cv_detail_df.empty:
            cv_detail_df = cv_detail_df.sort_values(['Model', 'Fold', 'Strategy']).reset_index(drop=True)

        # === Xuất Excel ===
        cv_excel_path = os.path.join(run_folder, 'cv_summary_results.xlsx')
        with pd.ExcelWriter(cv_excel_path, engine='openpyxl') as writer:
            cv_summary_df.to_excel(writer, sheet_name='CV Summary', index=False)
            cv_detail_df.to_excel(writer, sheet_name='Fold Details', index=False)

        print(f"\n✓ CV Summary saved to: {cv_excel_path}")
        print(f"  - Sheet 'CV Summary': Mean ± Std per strategy across {Config.CV_N_SPLITS} folds")
        print(f"  - Sheet 'Fold Details': Raw values per fold per strategy")
        
        # In summary ra console
        print(f"\n{'='*70}")
        print(f" CV SUMMARY (Mean ± Std per Strategy):")
        print(f"{'='*70}")
        for _, row in cv_summary_df.iterrows():
            print(f"  {row['Model']} - {row['Strategy']}:")
            for key in metric_keys:
                col = f"{key} (mean ± std)"
                if col in row:
                    print(f"    {key}: {row[col]}")
        
        print(f"\n{'='*70}")
        print(f" CROSS-VALIDATION COMPLETED!")
        print(f"{'='*70}")
        return
    
    # ===================== NORMAL MODE (no CV) =====================
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, train_labels,
        val_data, val_labels,
        test_data, test_labels,
        Config.BATCH_SIZE,
        Config.NUM_WORKERS
    )
    
    # Step 3: Train and evaluate each model (one at a time to save disk space)
    print(f"\n[Step 3/6] Training and evaluating {len(Config.MODELS)} models...")
    print("  Strategy: Train → Evaluate → Save Results → Delete Checkpoints")
    
    # Optional: opens up to 1000 images, so keep disabled for high-compute runs.
    if Config.PRINT_DATASET_STATS:
        print_dataset_statistics(concatenate_datasets([train_data, val_data, test_data]),
                                 train_labels + val_labels + test_labels,
                                 class_names)
    
    all_model_results = {}
    successfully_processed = []
    
    for idx, model_name in enumerate(Config.MODELS, 1):
        print(f"\n{'='*70}")
        print(f"[Model {idx}/{len(Config.MODELS)}] Processing: {model_name}")
        print(f"{'='*70}")
        
        try:
            # 3.1: Train model
            print(f"\n  [3.1] Training {model_name}...")
            checkpoint_manager, history, train_extra = train_model(
                model_name,
                train_loader,
                val_loader,
                num_classes,
                device,
                class_names=class_names,
                train_labels=train_labels,
                save_dir=run_folder,
                checkpoints_dir=run_checkpoints_dir
            )
            print(f"  ✓ Training completed for {model_name}")
            
            # 3.2: Evaluate with 3 strategies
            print(f"\n  [3.2] Evaluating {model_name} with 3 strategies...")
            # Tạo folder lưu checkpoint cho các strategy
            strategy_checkpoint_dir = (
                os.path.join(run_folder, model_name, 'checkpoints')
                if Config.SAVE_STRATEGY_CHECKPOINTS else None
            )
            results = evaluate_all_strategies(
                model_name,
                checkpoint_manager,
                test_loader,
                train_loader,  # CRITICAL: Pass train_loader for BatchNorm update
                num_classes,
                device,
                class_names=class_names,
                save_dir=strategy_checkpoint_dir,
                shadows=train_extra.get('shadows'),
                train_seconds=train_extra.get('train_seconds', 0.0),
            )
            all_model_results[model_name] = results
            print(f"  ✓ Evaluation completed for {model_name}")
            
            # 3.3: Save individual model results
            print(f"\n  [3.3] Saving results for {model_name}...")
            save_model_results(model_name, results, run_folder)

            # 3.3.1: Save confusion matrices for this model
            model_dir = os.path.join(run_folder, model_name)
            save_confusion_matrices(
                {model_name: results}, model_dir, class_names=class_names
            )
            
            # 3.4: Delete checkpoints to free disk space (conditional)
            if Config.AUTO_DELETE_CHECKPOINTS:
                print(f"\n  [3.4] Cleaning up checkpoints for {model_name}...")
                delete_model_checkpoints(model_name, run_checkpoints_dir)
            else:
                print(f"\n  [3.4] Keeping checkpoints for {model_name} (AUTO_DELETE_CHECKPOINTS=False)")
            
            successfully_processed.append(model_name)
            print(f"\n  ✅ {model_name} completed successfully!")
            
        except Exception as e:
            print(f"\n  ✗ Error processing {model_name}: {str(e)}")
            print(f"  Skipping {model_name} and continuing with next model...")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_model_results:
        print("\n✗ No models processed successfully. Exiting...")
        return
    
    print(f"\n{'='*70}")
    print(f"✓ Successfully processed {len(successfully_processed)}/{len(Config.MODELS)} models")
    print(f"  Models: {', '.join(successfully_processed)}")
    print(f"{'='*70}")
    
    # Step 4: Combine all results to single Excel
    print(f"\n[Step 4/6] Combining all results to Excel...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(run_folder, f'all_models_results.xlsx')
    
    df = export_results_to_excel(all_model_results, excel_path)
    
    # Display summary
    print("\n" + "="*70)
    print("COMBINED RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    
    # Step 5: Generate combined performance charts
    print(f"\n[Step 5/6] Generating combined performance chart...")
    create_performance_charts(df, run_folder)
    
    # Step 6: Experiment info already saved via export_run_config
    print(f"\n[Step 6/6] Run config already exported at start.")
    
    # Final summary
    print("\n" + "="*70)
    print(" BASELINE RESEARCH COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Lần chạy #{run_number}:")
    print(f"  - Folder: {run_folder}")
    print(f"  - Combined Excel: {excel_path}")
    print(f"  - Combined Chart: {os.path.join(run_folder, 'performance_comparison.png')}")
    print(f"  - Run Config: {os.path.join(run_folder, 'run_config.xlsx')}")
    print(f"  - Individual Results: {run_folder}/<model_name>/")
    print(f"\n💾 Disk Space Optimization:")
    if Config.AUTO_DELETE_CHECKPOINTS:
        print(f"  - Training checkpoints were deleted after evaluation")
    else:
        print(f"  - Training checkpoints kept at: {run_checkpoints_dir}")
    if Config.SAVE_STRATEGY_CHECKPOINTS:
        print(f"  - Strategy checkpoints/results kept under: {run_folder}/<model_name>/")
    else:
        print(f"  - Strategy checkpoints were not saved")
    
    # Find best model (based on Strategy 1 F1-Score)
    strategy_1_df = df[df['Strategy'] == 'Strategy 1']
    best_idx = strategy_1_df['F1-Score (%)'].idxmax()
    best_model = strategy_1_df.loc[best_idx, 'Model']
    best_f1 = strategy_1_df.loc[best_idx, 'F1-Score (%)']
    best_acc = strategy_1_df.loc[best_idx, 'Accuracy (%)']
    
    print(f"\n🏆 Best Model (Strategy 1):")
    print(f"  Model: {best_model}")
    print(f"  Accuracy: {best_acc:.2f}%")
    print(f"  F1-Score: {best_f1:.2f}%")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
