# Strategy 2 — Instance Segmentation (Ultralytics YOLO-seg + Carparts)

Pipeline này chỉ dùng model YOLO segmentation (`-seg`) qua Ultralytics API.
Mọi cấu hình chính nằm trong `config.py`; `main.py` cung cấp CLI để train,
đánh giá Strategy 1/2, xuất Excel và export model.

## Data split

Mặc định dùng `carparts-seg.yaml` của Ultralytics:

| Split | Số ảnh | Mục đích |
|---|---:|---|
| `train` | 3156 | Huấn luyện và BN recalibration |
| `val` | 401 | Fitness, `best.pt`, early stopping, xếp hạng Top-K |
| `test` | 276 | Báo cáo cuối |

`VAL_RATIO = 0` giữ nguyên ba split độc lập trên. Với dataset custom, phải bảo
đảm `val` không trùng `test`. Nếu cần tự tách validation từ train thì đặt
`VAL_RATIO > 0`; `dataset.py` sẽ tạo split cố định theo seed.

Pipeline dùng `carparts-seg.yaml` được pin ngay trong project thay vì YAML cũ
đóng gói trong `ultralytics==8.3.152`.

## Hai strategy

| Strategy | Mô tả |
|---|---|
| Strategy 1 | `best.pt` tạm — raw checkpoint có raw-model fitness validation cao nhất |
| Strategy 2 | Uniform element-wise average raw weights của Top-K raw checkpoint |

Ultralytics segmentation fitness được dùng thống nhất cho `best.pt`, early
stopping và Top-K:

```text
fitness =
    0.1 × mAP50(B) + 0.9 × mAP50-95(B)
  + 0.1 × mAP50(M) + 0.9 × mAP50-95(M)
```

`B` là bounding box, `M` là mask. Precision và Recall không đóng góp vào
fitness.

Với tập checkpoint được chọn là `S_K`, Strategy 2 thực hiện:

```text
w_avg = (1 / K) × Σ w_t^raw,  t ∈ S_K
```

Đây là uniform element-wise averaging, không có parameter mới được học trong
quá trình averaging.

## Raw checkpoint averaging

Ultralytics mặc định duy trì EMA, dùng EMA cho validation và chỉ lưu EMA vào
checkpoint. Pipeline này chủ động tắt EMA khi bắt đầu train để phương pháp khớp
raw checkpoint averaging trong paper:

1. Optimizer cập nhật raw `trainer.model`.
2. Cuối mỗi epoch, raw `state_dict` được copy 1:1 sang một module validation
   riêng; validation và fitness dùng đúng snapshot raw đó.
3. Early stopping và `best.pt` dùng raw-model fitness.
4. Mỗi `epochN.pt` lưu raw FP32 model trong field `model`, với `ema=None`.
5. Sau training, chọn Top-K raw checkpoints và uniform-average raw parameters
   ở FP32; averaged checkpoint tạm cũng dùng FP32.
6. Sau khi đã lấy đủ metrics/Excel, tất cả checkpoint tạm được xóa.

Module validation riêng chỉ để cô lập `torch.inference_mode()` khỏi training
model; nó không thực hiện EMA smoothing. Vì vậy không có EMA smoothing trước
phép Top-K averaging.

## Checkpoint và BatchNorm

Khi `USE_STRATEGY2=True`:

- `save_period=1` lưu một `epochN.pt` mỗi epoch.
- Callback tắt EMA và lấy raw-model `trainer.fitness`.
- Chỉ giữ `KEEP_TOP_K_CHECKPOINTS` checkpoint có fitness cao nhất.
- Ranking được ghi vào `weights/strategy2_checkpoints.json`.
- Raw FP32 model parameters được uniform-average.
- BN `running_mean`, `running_var`, `num_batches_tracked` không được average.
- Sau averaging, `update_bn_stats()` re-estimate BN statistics bằng ảnh train
  với task segmentation trước khi đánh giá.
- Mỗi `strategy2_topK_avg.pt` được xóa ngay sau `model.val()`.
- Cuối mỗi seed, toàn bộ thư mục `weights/` (`best.pt`, `last.pt`, raw
  `epochN.pt`, ranking JSON còn lại) được xóa trong khối `finally`.

Run cũ dùng EMA hoặc JSON xếp hạng cũ sẽ bị từ chối; cần train lại vì raw
checkpoints tương ứng không tồn tại trong run cũ.

## Cài đặt

```bash
pip install -r requirements.txt
```

Phiên bản Ultralytics được khóa trong `requirements.txt` vì pipeline sử dụng
callback, checkpoint format và `SegmentMetrics` nội bộ.

## Chạy

Chỉnh `MODEL`, `DATA` và các hyperparameter trong `config.py`, hoặc override qua
CLI:

```bash
# Train tất cả seed trong Config.RANDOM_SEED; tên output cố định, dễ đọc
python main.py train --exp_name carparts_yolov8s_raw_topk_run01

# Đánh giá lại Strategy 1 và Strategy 2 của một run
python main.py strategies --run-dir results/segmentation/<experiment>/seed_42

# Đánh giá một weights segmentation
python main.py eval --weights <run>/weights/best.pt --split test

# Xuất Excel offline từ results.csv
python main.py export --run-dir <run> --output segmentation_results.xlsx

# Export ONNX
python main.py export-model --weights <run>/weights/best.pt --format onnx
```

Pipeline từ chối model detection thông thường như `yolov8s.pt`; hãy dùng model
segmentation như `yolov8s-seg.pt`.

Trên server có thể sửa tên experiment trong `server_train_commands.txt`, rồi
chạy:

```bash
bash server_train_commands.txt
```

`--exp-name` và `--exp_name` tương đương. Khi truyền tham số này, output là
`results/segmentation/<exp_name>/`; pipeline không ghép timestamp. Một tên đã
tồn tại và không rỗng sẽ bị từ chối để tránh trộn hai thí nghiệm.

## Output

```text
results/segmentation/<exp_name>/
├── experiment_config.json
├── multi_seed_summary.xlsx
└── seed_<N>/
    ├── results.csv
    └── segmentation_results_seed<N>.xlsx
```

Mặc định không còn `weights/` sau khi một seed hoàn tất:
`DELETE_CHECKPOINTS_AFTER_RUN=True`. Checkpoint chỉ tồn tại tạm thời trong lúc
rank, averaging, BN update, evaluation và export tùy chọn. Vì vậy muốn giữ hoặc
deploy model `.pt`, cần chủ động đổi policy này trước khi train; mặc định hiện
tại ưu tiên không lưu checkpoint theo yêu cầu thí nghiệm.

Excel gồm:

- `Summary`: mask Precision, Recall, mAP50, mAP75, mAP50-95, combined
  box+mask fitness và mask AP per class.
- `PerEpoch`: toàn bộ cột train/validation do Ultralytics ghi trong
  `results.csv`, gồm cả box và mask metrics.

Khi chỉ export offline từ `results.csv`, bảng overall sử dụng các cột mask
`metrics/*(M)`.

## Config quan trọng

| Nhóm | Biến |
|---|---|
| Model/data | `MODEL`, `DATA`, `VAL_RATIO` |
| Training | `EPOCHS`, `IMGSZ`, `BATCH`, `DEVICE`, `PATIENCE`, `RANDOM_SEED` |
| Optimizer/LR | `OPTIMIZER`, `LR0`, `LRF`, `WARMUP_EPOCHS`, `COS_LR` |
| Augmentation | `MIXUP`, `COPY_PASTE`, `EXTRA_TRAIN_ARGS` |
| Strategy 2 | `TOP_K_VALUES`, `KEEP_TOP_K_CHECKPOINTS`, `USE_BN_UPDATE`, `BN_UPDATE_BATCHES` |
| Evaluation | `EVAL_SPLIT`, `CONF`, `IOU` |
| Output | `PROJECT`, `EXP_NAME`, `DELETE_CHECKPOINTS_AFTER_RUN` |

`PATIENCE` đếm số epoch không có fitness cải thiện nghiêm ngặt; early stopping
không dựa trên validation loss.
