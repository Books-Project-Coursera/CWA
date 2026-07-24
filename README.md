# Strategy 2 — Object Detection (Ultralytics YOLO + Pascal VOC)

Pipeline huấn luyện object detection bằng Ultralytics, chọn Top-K checkpoint
trên validation rồi uniform average raw learnable parameters. Phiên bản thư
viện được khóa tại `ultralytics==8.3.152`.

## Data split và chống test leakage

`VOC.yaml` gốc có `val` và `test` cùng trỏ tới VOC2007 test. Vì vậy
`dataset.py` tách validation độc lập từ tập train khi `VAL_RATIO=0.1`:

| Split | Vai trò |
|---|---|
| `train'` | Train model và re-estimate BatchNorm sau averaging |
| `val'` | Tính fitness, early stopping, chọn `best.pt`, rank Top-K |
| `test` | Chỉ báo cáo hiệu năng cuối cho các strategy/K đã định trước |

Không được nhìn kết quả test để chọn K. Code chỉ in tất cả K trong
`TOP_K_VALUES`; không còn dòng tự chọn “best strategy” từ test.

## Raw checkpoint và phép averaging

Khi bật Strategy 2:

1. EMA smoothing của Ultralytics được tắt.
2. Sau mỗi epoch, raw training weights được copy chính xác sang một module
   validation riêng. Validation, fitness, early stopping và `best.pt` đều theo
   raw weights.
3. Callback lưu raw model ở FP32 vào `epochN.pt`, gắn
   `weight_source="raw"` và chỉ giữ `KEEP_TOP_K_CHECKPOINTS` checkpoint có
   validation fitness cao nhất.
4. Với từng K đã định trước, code thực hiện:

   \[
   \mathbf{w}_{avg}=\frac{1}{K}\sum_{t\in\mathcal{S}_K}\mathbf{w}_t
   \]

   Đây là uniform element-wise average của tất cả learnable parameters:
   convolution/linear weights, bias và affine parameters của BatchNorm.
5. Không average state không learnable như `running_mean`, `running_var`,
   `num_batches_tracked`, anchors, stride hoặc cache. Những state này không nằm
   trong \(\mathbf{w}\). BatchNorm statistics được re-estimate bằng train split,
   tuyệt đối không dùng validation/test.
6. Average và checkpoint kết quả giữ FP32. FP32 có sai số làm tròn thấp hơn
   FP16 khi cộng và chia K tensor.

Detection fitness trong Ultralytics 8.3.152 là:

\[
\text{fitness}=0.1\,\mathrm{mAP}_{50}+0.9\,\mathrm{mAP}_{50:95}
\]

`trainer.fitness` chính là criterion được dùng để rank checkpoint và điều
khiển early stopping.

## Checkpoint lifecycle

Checkpoint chỉ là artifact tạm:

- Trong train: giữ tối đa Top-K raw epoch checkpoint, cộng `best.pt`/`last.pt`
  do trainer cần.
- Mỗi averaged checkpoint bị xóa ngay sau khi `model.val()` hoàn tất.
- Sau khi đã export metrics/Excel (và model deploy nếu bật), toàn bộ thư mục
  `weights/` của seed bị xóa, kể cả khi evaluation gặp lỗi.
- `results.csv`, Excel, config snapshot, args và plots vẫn được giữ.

Vì vậy lệnh `strategies`/`eval` từ checkpoint cũ chỉ dùng được nếu
`DELETE_CHECKPOINTS_AFTER_RUN=False` trước khi train.

## Cài đặt và chạy

```bash
pip install -r requirements.txt
python main.py train --exp_name voc_yolov8s_raw_topk_run01
```

`--exp-name` và `--exp_name` đều hợp lệ. Tên được dùng nguyên văn để tạo:

```text
results/detection/voc_yolov8s_raw_topk_run01/
├── experiment_config.json
├── seed_1/
│   ├── results.csv
│   ├── detection_results_seed1.xlsx
│   ├── args.yaml
│   └── plots...
└── multi_seed_summary.xlsx
```

Tên experiment đã tồn tại và không rỗng sẽ bị từ chối để tránh trộn kết quả.
Lệnh mẫu cho server nằm trong `server_train_commands.txt`.

Các lệnh khác:

```bash
# Tắt Strategy 2
python main.py train --exp_name baseline_run01 --no-strategy2

# Đánh giá một checkpoint được giữ lại có chủ ý
python main.py eval --weights path/to/model.pt --split test

# Export model deploy; cần giữ checkpoint hoặc bật export-after-train
python main.py train --exp_name deploy_run01 --export-after-train
```

## Config quan trọng

| Biến | Ý nghĩa |
|---|---|
| `MODEL` | Detection model, ví dụ `yolov8s.pt` |
| `VAL_RATIO` | Tỉ lệ tách validation độc lập từ train; VOC mặc định là `0.1` |
| `PATIENCE` | Early stopping theo raw validation fitness |
| `TOP_K_VALUES` | Các K được định trước để báo cáo |
| `KEEP_TOP_K_CHECKPOINTS` | Phải ≥ `max(TOP_K_VALUES)` |
| `USE_BN_UPDATE` | Re-estimate BN buffers trên train sau averaging |
| `DELETE_CHECKPOINTS_AFTER_RUN` | Xóa checkpoint tạm sau mỗi seed |
| `EVAL_SPLIT` | Split báo cáo cuối, mặc định `test` |
| `EXP_NAME` | Tên experiment ổn định trên server |

## Output metrics

Excel chứa:

- `Summary`: Precision, Recall, mAP@0.5, mAP@0.75, mAP@0.5:0.95, fitness và
  per-class AP cho từng strategy/K.
- `PerEpoch`: dữ liệu nguyên gốc từ `results.csv`.
- `multi_seed_summary.xlsx`: kết quả từng seed và mean ± std.

Troubleshooting:

- OOM: giảm `BATCH`/`IMGSZ` hoặc dùng `--batch -1`.
- Không đủ Top-K: số epoch thực tế trước early stopping nhỏ hơn K.
- Run cũ báo không phải raw checkpoint: phải train lại bằng code hiện tại;
  code chủ động không trộn EMA checkpoint với raw averaging.
