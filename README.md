# Strategy 2 — Object Detection (Ultralytics YOLO + Pascal VOC)

Nhánh này **chỉ làm object detection với các model YOLO**, dùng **toàn bộ
Ultralytics API** (train / val / export — không tự viết training loop).
Kế thừa convention của repo: config-driven (`config.py`), CLI override,
export metrics ra Excel bằng `pandas` + `openpyxl`.

Code classification (Tiny ImageNet) nằm ở nhánh `Strategy2_TinyImageNet`.

## Data split: val' độc lập tách từ train (điều kiện của Strategy 2)

`VOC.yaml` gốc của Ultralytics ([docs](https://docs.ultralytics.com/datasets/detect/voc))
có vấn đề: **split `val` và `test` trùng nhau** (cùng trỏ `images/test2007`) —
không có validation set độc lập, nên không thể chọn/average checkpoint theo val
mà không leakage. Pipeline này tự xử lý (xem `dataset.py`):

| Split | Nội dung | Số ảnh |
|---|---|---|
| `train'` | trainval VOC2007+2012 **trừ** phần tách ra làm val' | ~14,896 (VAL_RATIO=0.1) |
| `val'` | **tách ngẫu nhiên từ train** theo `VAL_RATIO` (seed cố định) | ~1,655 |
| `test` | VOC2007 test — giữ nguyên làm hold-out | 4,952 |

- **val'** chỉ dùng để: chọn `best.pt`, early stopping, rank Top-K checkpoint.
- **test** chỉ dùng cho **báo cáo cuối** (`EVAL_SPLIT = "test"` mặc định).
- Split idempotent theo `(seed, ratio)` — chạy lại không tách lại, reproduce được.
- File sinh ra nằm ở dataset root: `holdout_seed{S}_val{R}.yaml` + 2 file `.txt`
  danh sách ảnh (Ultralytics hỗ trợ split dạng txt list).
- Đặt `VAL_RATIO = 0` sẽ dùng nguyên data.yaml gốc → **val ≡ test, Strategy 2
  bị leakage** — pipeline sẽ in cảnh báo; chỉ dùng khi data custom của bạn đã
  có val độc lập sẵn.

## 2 Strategy

| Strategy | Mô tả |
|---|---|
| **Strategy 1** | `best.pt` — checkpoint có fitness cao nhất trên val' (Ultralytics tự chọn) |
| **Strategy 2** | **Average weights của Top-K checkpoint** tốt nhất trên val' (K = 2, 3, 4, 5) — giống Top-K Average của nhánh classification |

Cơ chế checkpoint cho Strategy 2 (xem `train.py`):
- Train với `save_period=1` → Ultralytics lưu checkpoint mỗi epoch;
- `TopKCheckpointManager` (callback `on_model_save`) **prune ngay** checkpoint
  ngoài Top-K theo fitness val' → **disk chỉ giữ đúng K checkpoint cần thiết**
  (+ `best.pt`/`last.pt`), không lưu tất cả epoch;
- Ranking ghi vào `weights/strategy2_checkpoints.json`;
- Sau train: average EMA weights của Top-K (`strategy2_top{K}_avg.pt`) —
  **skip BN running stats** (giữ từ ckpt tốt nhất, không average) rồi
  **`update_bn_stats`** re-estimate BN trên train (forward pass
  `BN_UPDATE_BATCHES=100`) — y hệt cơ chế `average_weights` + `update_bn`
  của nhánh `Strategy2_TinyImageNet`. Cuối cùng `model.val()` trên test —
  so sánh trực tiếp với Strategy 1.
- Fitness = `0.1*mAP50 + 0.9*mAP50-95` (định nghĩa của Ultralytics, không tự chế).

## Cấu trúc

```
├── config.py       # TOÀN BỘ config (model, data, val ratio, strategy 2, export...)
├── main.py         # Entrypoint: train / strategies / eval / export / export-model
├── dataset.py      # Tách val' độc lập từ train, sinh holdout data.yaml
├── train.py        # model.train() + TopKCheckpointManager + Edge AI hook
├── evaluate.py     # model.val(), Strategy 1/2, metrics console, Excel 2 sheet
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## 1. Set model (bắt buộc — KHÔNG hardcode trong code)

Mở `config.py`, sửa đúng 1 dòng:

```python
MODEL = "yolov8n.pt"   # hoặc "yolo11n.pt", "yolov5nu.pt", "path/to/custom.pt", "yolov8n.yaml"
```

Hoặc override lúc chạy: `python main.py train --model yolo11n.pt`.
`MODEL` nhận mọi giá trị mà `ultralytics.YOLO()` nhận — đổi version YOLO =
sửa 1 dòng, không đụng code.

## 2. Train

```bash
python main.py train --model yolov8n.pt --device 0 --name yolov8n_voc
# hoặc chỉnh hết trong config.py rồi:
python main.py train
```

Pipeline tự động: tách val' từ train (nếu chưa có) → train (val mỗi epoch trên
val', giữ Top-K checkpoint) → in metrics val cuối → **đánh giá Strategy 1 vs
Strategy 2 (Top-2/3/4/5) trên test** → export Excel.

Output tại `results/detection/<name>/`:

```
├── weights/
│   ├── best.pt                     # Strategy 1
│   ├── last.pt
│   ├── epoch{N}.pt × K             # Top-K checkpoint (đã prune, chỉ giữ K file)
│   ├── strategy2_checkpoints.json  # ranking fitness
│   └── strategy2_top{K}_avg.pt     # weights đã average (Strategy 2)
├── results.csv                     # per-epoch (nguồn sheet PerEpoch)
├── detection_results.xlsx          # Excel 2 sheet
└── args.yaml, plots...             # Ultralytics tự sinh
```

## 3. Metrics & Excel

Console in cho **từng strategy**: Precision, Recall, mAP@0.5, mAP@0.75,
mAP@0.5:0.95, Fitness + **AP per class** (đọc từ `results.box.*` của
Ultralytics — không tự tính lại mAP), kèm bảng so sánh strategy và best
strategy theo mAP@0.5:0.95.

File `detection_results.xlsx` gồm 2 sheet:

| Sheet | Nội dung |
|---|---|
| `Summary` | Run info (model, data, val_ratio, imgsz, epochs, batch, seed, ngày chạy, note data split) + **bảng overall metrics theo strategy** + **AP per class theo strategy** |
| `PerEpoch` | Mỗi epoch 1 row từ `results.csv`: `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `val/box_loss`, `val/cls_loss`, `val/dfl_loss`, `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, lr... |

## 4. Các lệnh khác

```bash
# Đánh giá lại Strategy 1 + 2 trên run đã train (không cần train lại)
python main.py strategies --run-dir results/detection/yolov8n_voc

# Eval 1 file weights bất kỳ trên split tùy chọn
python main.py eval --weights results/detection/yolov8n_voc/weights/best.pt --split test

# Export Excel offline từ results.csv (không cần GPU/dataset)
python main.py export --run-dir results/detection/yolov8n_voc --output bao_cao.xlsx

# Edge AI: xuất ONNX / TensorRT
python main.py export-model --weights .../weights/best.pt --format onnx
python main.py export-model --weights .../weights/strategy2_top5_avg.pt --format engine --half
```

## 5. (Optional) Edge AI export sau train

Tắt mặc định. Bật trong `config.py` (`EXPORT_ENABLED = True`, chọn
`EXPORT_FORMAT`/`EXPORT_HALF`...) hoặc thêm cờ `--export-after-train` khi train.
Dùng `model.export()` của Ultralytics; TensorRT (`engine`) cần GPU + TensorRT.

## Chạy trên Vast.ai (gợi ý workflow)

```bash
git clone https://github.com/Dung-04/Capstone_KD && cd Capstone_KD
git checkout claude/strategy2-object-detection-yolo-ttjplw
pip install -r requirements.txt

# (tuỳ chọn) trỏ chỗ chứa dataset về volume lớn của instance
yolo settings datasets_dir=/workspace/datasets

# train — VOC tự download lần đầu (~2.8 GB), val' tự tách từ train
python main.py train --model yolov8n.pt --device 0 --name yolov8n_voc

# chạy nền + giữ log khi rớt SSH
nohup python main.py train --model yolov8n.pt --device 0 --name yolov8n_voc \
    > train_voc.log 2>&1 &
tail -f train_voc.log

# lấy về máy: results/detection/yolov8n_voc/detection_results.xlsx
```

## Config chính (`config.py`)

| Nhóm | Biến | Mô tả |
|---|---|---|
| **Model** | `MODEL` | Model YOLO — bạn tự set (placeholder `None`), đổi version = 1 dòng |
| **Dataset** | `DATA` | `VOC.yaml` (auto-download) hoặc data.yaml custom |
| | `VAL_RATIO` | Tỉ lệ tách val' từ train (mặc định 0.1; 0 = không tách — leakage!) |
| **Training** | `EPOCHS`, `IMGSZ`, `BATCH`, `DEVICE`, `LR0`, `PATIENCE`... | Hyperparameter Ultralytics |
| | `EXTRA_TRAIN_ARGS` | Dict truyền thêm train-arg Ultralytics bất kỳ |
| **Strategy 2** | `USE_STRATEGY2` | Bật/tắt Top-K averaging |
| | `TOP_K_VALUES` | Các K cần so sánh (mặc định `[2,3,4,5]`) |
| | `KEEP_TOP_K_CHECKPOINTS` | Số checkpoint giữ trên disk (≥ max K) |
| | `USE_BN_UPDATE`, `BN_UPDATE_BATCHES` | Re-estimate BN sau average (bắt buộc để mAP không tụt) |
| **Optimizer/LR** | `LR0`, `LRF`, `MOMENTUM`, `WEIGHT_DECAY`, `WARMUP_EPOCHS`, `WARMUP_MOMENTUM`, `WARMUP_BIAS_LR`, `COS_LR` | Tương đương nhóm Optimizer/Scheduler của repo gốc |
| **Loss** | `BOX_GAIN`, `CLS_GAIN`, `DFL_GAIN`, `LABEL_SMOOTHING`, `DROPOUT`, `NBS`, `CLOSE_MOSAIC` | Loss gains YOLO + regularization |
| **Aug** | `HSV_H/S/V`, `DEGREES`, `TRANSLATE`, `SCALE`, `SHEAR`, `PERSPECTIVE`, `FLIPUD`, `FLIPLR`, `MOSAIC`, `MIXUP`, `CUTMIX`, `COPY_PASTE`, `ERASING`, `AUTO_AUGMENT` | Toàn bộ augmentation Ultralytics |
| **Precision** | `AMP`, `MULTI_SCALE`, `RECT`, `SINGLE_CLS`, `FREEZE`, `DETERMINISTIC` | AMP, multi-scale, freeze N layer đầu |
| **Eval** | `EVAL_SPLIT` | Split báo cáo cuối (`"test"` mặc định) |
| **Output** | `PROJECT`, `NAME`, `EXCEL_OUTPUT` | Thư mục run + file Excel |
| **Edge AI** | `EXPORT_ENABLED`, `EXPORT_FORMAT`, `EXPORT_HALF`... | Export sau train (tắt mặc định) |

## Ghi chú / Assumptions

1. **Config bằng `config.py`** (class `Config`, UPPERCASE) theo đúng convention
   repo gốc; CLI override qua `main.py` như cũ.
2. **Averaging dùng EMA weights** trong mỗi checkpoint (phần Ultralytics thực sự
   deploy); tensor int (BN `num_batches_tracked`) lấy từ checkpoint tốt nhất.
   Không chạy lại BN-update sau khi average (khác classification — YOLO val
   trực tiếp cho kết quả hợp lệ, có thể thêm sau nếu cần).
3. **Thư mục run do Ultralytics quản lý** (`PROJECT/NAME`, tự đánh số) — vì
   `results.csv`, weights, plots đều do Ultralytics ghi; gốc `results/detection`
   giữ theo pattern `results/` của repo.
4. Seed mặc định `RANDOM_SEED = 1` — dùng cho cả tách val' và
   `model.train(seed=...)`; cùng seed + ratio → cùng split (reproduce được).
5. Checkpoint mỗi epoch chứa cả optimizer state nên hơi nặng, nhưng chỉ tồn tại
   tối đa `KEEP_TOP_K_CHECKPOINTS` file tại mọi thời điểm; file
   `strategy2_top{K}_avg.pt` đã strip optimizer nên nhẹ.

## Troubleshooting

- **Out of memory**: giảm `BATCH` (hoặc `--batch -1` để auto theo VRAM), giảm `IMGSZ`.
- **Dataset tải chậm/hết disk**: `yolo settings datasets_dir=<path>` trỏ về volume lớn.
- **`strategies` báo không có checkpoint**: run đó train với `USE_STRATEGY2=False`
  (hoặc `--no-strategy2`) nên không lưu checkpoint epoch — chỉ eval được Strategy 1.
- **Số class lệch (80 vs 20)**: bạn đang eval weights COCO thô chưa fine-tune
  trên VOC — hãy train trước rồi eval bằng `best.pt`.
