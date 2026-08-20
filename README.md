# Baseline Research - Pretrained Models Evaluation

Pipeline tự động train & evaluate pretrained models cho bài toán image classification.

**Dataset: CIFAR-100** (`torchvision.datasets.CIFAR100`, `download=False`).
Official train 50,000 ảnh được chia stratified thành **45,000 train / 5,000 validation**;
official test **10,000 ảnh giữ nguyên**, chỉ dùng làm test set cuối cùng.

## Cấu trúc Project

```
├── config.py          # Toàn bộ cấu hình (dataset, training, loss, sampler, CV, ...)
├── main.py            # Main pipeline - chạy file này
├── train.py           # Training loop, early stopping, checkpoint management
├── evaluate.py        # 3 chiến thuật đánh giá + export Excel
├── models.py          # Pretrained models với custom classifier head
├── dataset.py         # Data loading, augmentation, WeightedRandomSampler
├── losses.py          # PolyFocalLoss + class weight computation
├── visualization.py   # Training curves, dataset statistics
├── requirements.txt   # Dependencies
└── results/           # Kết quả tự động lưu theo từng lần chạy (results/1/, results/2/, ...)
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách chạy

```bash
python main.py
```

CLI overrides are available, so you do not need to edit `config.py` for every server run:

```bash
python main.py --model resnet18 --run-name resnet18
python main.py --model resnet18 --seed 100 --run-name resnet18_seed100
python main.py --model vit_base --batch-size 64 --epochs 50 --lr 2e-5 --fc-layers 256 128 --dropout 0.5
python main.py --model resnet18,densenet121 --results-dir /scratch/$USER/potato_results
```

`argparse` is part of the Python standard library, so no extra package is needed in `requirements.txt`.

For concurrent terminals, every run creates an isolated folder under `results/` (or `--results-dir`). Training checkpoints are also isolated per run, so two terminals running the same model will not overwrite each other.

By default, `python main.py --model <name>` runs the configured seeds `1, 10, 100, 500` sequentially. Use `--seed <value>` to run only one seed in a terminal.

Pipeline tự động: Validate config → Load dataset → Train từng model → Evaluate 3 strategies → Export Excel + Charts.

Kết quả mỗi lần chạy lưu riêng tại `results/<run_number>/` gồm:
- `run_config.xlsx` — toàn bộ config của lần chạy
- `all_models_results.xlsx` — bảng so sánh tất cả model
- `<model_name>/` — kết quả chi tiết, confusion matrix, training curves

## Cấu hình (`config.py`)

Mở `config.py`, chỉnh các biến cần thiết:

| Nhóm | Biến quan trọng | Mô tả |
|------|-----------------|-------|
| **Dataset** | `DATA_ROOT` | Thư mục chứa `cifar-100-python`, mặc định `/lustre/fsmisc/dataset` |
| | `DOWNLOAD_DATASET` | `False` — dataset đã có sẵn trên server, không tải lại |
| | `VALIDATION_RATIO` | Tỉ lệ validation lấy stratified từ official train (mặc định 0.1 → 45,000 train / 5,000 val) |
| **Model** | `MODELS` | List model cần train (comment/uncomment để chọn) |
| | `CLASSIFIER_CONFIG` | Hidden layers của classifier head, VD: `[512]` |
| | `DROPOUT_RATE` | Dropout rate cho classifier |
| **Training** | `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE` | Hyperparameters cơ bản |
| | `WEIGHT_DECAY` | L2 regularization |
| | `EARLY_STOPPING_PATIENCE` | Dừng sớm nếu val_loss không giảm sau N epochs |
| **Loss** | `LOSS_FUNCTION` | `'cross_entropy'` hoặc `'poly_focal'` |
| | `label_smoothing` | Label smoothing (chỉ cho CrossEntropy) |
| | `FOCAL_GAMMA`, `POLY_EPSILON` | Params cho PolyFocalLoss |
| **Sampler** | `USE_WEIGHTED_SAMPLER` | `True/False` — bật WeightedRandomSampler xử lý class imbalance |
| **Cross-Val** | `USE_CROSS_VALIDATION` | `True/False` — bật Stratified K-Fold CV |
| | `CV_N_SPLITS` | Số fold (mặc định 5) |
| **Output** | `AUTO_DELETE_CHECKPOINTS` | Tự xóa checkpoints sau evaluate để tiết kiệm disk |

## Models hỗ trợ

Uncomment trong `Config.MODELS`:

```python
MODELS = [
    'vgg16',
    'resnet18',
    'resnet101',
    'mobilenet_v2',
    'densenet121',
    'efficientnet_b0',
    'vit_base_patch16_224',
]
```

## 3 Chiến thuật Đánh giá

| Strategy | Mô tả |
|----------|-------|
| **Best Checkpoint** | Checkpoint có val_loss thấp nhất |
| **Top-K Average** | Trung bình weights của K checkpoint tốt nhất (K = 2,3,4,5) |
| **Last-N Average** | Trung bình weights của N epoch cuối cùng |

## Cross-Validation

Khi `USE_CROSS_VALIDATION = True`:
- Data `train + val` gộp thành CV pool
- `test` giữ nguyên làm hold-out
- Dùng `StratifiedKFold` (sklearn) chia K fold, giữ tỉ lệ class
- Kết quả cuối: **mean ± std** qua K fold → lưu vào `cv_summary_results.xlsx`

## Metrics

Tất cả metrics dùng **Macro averaging** (trung bình đều giữa các class):

| Metric | Cách tính |
|--------|-----------|
| Accuracy | Overall correct / total |
| Precision | Macro average |
| Recall | Macro average |
| F1-Score | Macro average |
| AUC | Macro average, one-vs-rest |

Kết quả bao gồm cả **per-class breakdown** (Precision, Recall, F1, Specificity, AUC, Support).

## Reproduce kết quả

1. Set `RANDOM_SEED = 42` (mặc định) — đảm bảo cùng data split, cùng weight init
2. Kiểm tra `DATA_ROOT` (torchvision đọc `<DATA_ROOT>/cifar-100-python`, `download=False`)
3. Chọn model trong `MODELS`
4. Chạy `python main.py`

Seed cố định cho: `random`, `numpy`, `torch`, `CUDA`. Thêm `--deterministic` nếu cần bật deterministic CUDA/cuBLAS.

## Ghi chú

- **WRS + Focal Loss đồng thời**: Không lỗi code, nhưng có thể double-correct class imbalance. Cân nhắc chỉ bật 1 trong 2.
- **LR Scheduler**: Linear Warmup → Cosine Annealing
- **Data Augmentation** (chỉ train): resize bicubic 224, horizontal flip,
  Mixup/CutMix kiểu DeiT và Random Erasing

## 💾 Checkpoints

Training checkpoints are temporary by default. They are written during training/evaluation, then deleted after evaluation because `AUTO_DELETE_CHECKPOINTS=True`. Strategy checkpoints are not saved because `SAVE_STRATEGY_CHECKPOINTS=False`.

```
results/<run_number>/
├── training_checkpoints/
│   └── <model_name>/
│       ├── epoch_001_val_loss_0.xxxx.pth
│       ├── best_checkpoint.pth
│       └── checkpoint_info.json
└── <model_name>/
    ├── checkpoints/
    ├── training_curves/
    └── <model_name>_results.xlsx
```

If you pass `--checkpoints-dir /scratch/...`, the code still creates a per-run subfolder inside that directory. Use `--keep-checkpoints` only when you explicitly need checkpoint files for later debugging.

## 🔧 Tùy chỉnh

### Thay đổi learning rate decay:

Trong `config.py`:

```python
LR_DECAY_PATIENCE = 5  # Giảm LR sau 5 epochs val_loss không cải thiện
LR_DECAY_FACTOR = 0.5  # Nhân LR với 0.5
```

### Thay đổi custom classifier:

Trong `config.py`:

```python
CLASSIFIER_CONFIG = [256, 128, 64]  # 3 hidden layers
DROPOUT_RATE = 0.5
```

### Thay đổi data augmentation:

Trong `dataset.py`, function `get_transforms()`:

```python
transform = transforms.Compose([
    transforms.Resize(
        (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        interpolation=InterpolationMode.BICUBIC,
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25),
])
```

### Thêm/bớt models:

Trong `config.py`:

```python
MODELS = [
    'vgg16',
    'resnet101',
    # Thêm/bớt models ở đây
]
```

## High-Compute Server Notes

Run one model per terminal or one model per scheduler job:

```bash
python main.py --model resnet18 --run-name resnet18
python main.py --model densenet121 --run-name densenet121
python main.py --model vit_base_patch16_224 --run-name vit_b16
```

On SLURM-style systems, prefer the scheduler GPU assignment:

```bash
srun --gres=gpu:1 --cpus-per-task=8 python main.py --model resnet18 --num-workers 8 --results-dir $SCRATCH/potato_results
```

Use `--deterministic` only when exact reproducibility is more important than speed. That flag sets `CUBLAS_WORKSPACE_CONFIG`; without it, the code uses faster cuDNN benchmarking. There is no separate `culabs` package to install.

## H100 profile

The default profile targets one H100:

- batch 512 for training, validation, and testing;
- BF16 autocast, fused AdamW, and `torch.compile(mode="max-autotune")`;
- 16 DataLoader workers with pinned memory and persistent workers;
- learning rate 5e-5 at batch 512, automatically scaled with batch size;
- linear warmup for 5 epochs followed by cosine decay to 1e-6.

Run the default batch-512 profile:

```bash
python main.py --model vit_base_patch16_224 --seed 1
```

Run batch 1024; LR is automatically scaled to 1e-4 unless `--lr` is given:

```bash
python main.py --model vit_base_patch16_224 --seed 1 --batch-size 1024
```

Every run writes dataset, model, optimizer, scheduler, augmentation, precision,
DataLoader, evaluation, and runtime environment settings to `run_config.xlsx`.

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (recommended) hoặc CPU
- RAM: >= 8GB
- GPU: >= 6GB VRAM (recommended)

## 🎓 Sử dụng cho Research

Code này được thiết kế để:
- Dễ dàng thay đổi dataset
- Tự động hóa toàn bộ pipeline
- Export kết quả professional
- Tái sử dụng cho nhiều experiments

Chỉ cần kiểm tra `DATA_ROOT` trong `config.py` và chạy `python main.py`!

## 📝 Citation

Nếu sử dụng code này cho research, vui lòng ghi nguồn phù hợp.

## 🐛 Troubleshooting

### Lỗi out of memory:
- Giảm `BATCH_SIZE` trong `config.py`
- Giảm `NUM_WORKERS`

### Lỗi không tìm thấy dataset:
- Kiểm tra `DATA_ROOT` phải là thư mục **cha** của `cifar-100-python`
- Có thể override khi chạy: `python main.py --data-root /lustre/fsmisc/dataset`

### Model không train:
- Kiểm tra GPU/CUDA availability
- Kiểm tra dependencies đã cài đủ chưa

## 📧 Support

Nếu có vấn đề, vui lòng mở issue hoặc liên hệ.
