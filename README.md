# Strategy 2 — Object Detection (Ultralytics YOLO + Pascal VOC)

Pipeline huấn luyện object detection bằng Ultralytics, chọn Top-K checkpoint
trên validation rồi uniform average raw learnable parameters. Phiên bản thư
viện được khóa tại `ultralytics==8.3.152`.

## Data split và chống test leakage

`VOC.yaml` gốc của Ultralytics khai báo:

| Split | Ảnh | Nguồn |
|---|---|---|
| train | 16,551 | VOC2007 trainval (5,011) + VOC2012 trainval (11,540) |
| val | 4,952 | VOC2007 test |
| test | 4,952 | **cùng ảnh VOC2007 test** — không có holdout riêng |

Nghĩa là `val ≡ test`. Method dựa trên Top-K checkpoint chọn theo validation
nên nếu dùng nguyên bộ này thì checkpoint được chọn trên chính tập báo cáo →
leakage. `dataset.py` vì thế tách validation độc lập ra khỏi train với
`VAL_RATIO=0.1`:

| Split | Ảnh | Vai trò |
|---|---|---|
| `train'` | 14,896 | Train model và re-estimate BatchNorm sau averaging |
| `val'` | 1,655 | Tính fitness, early stopping, chọn best checkpoint, rank Top-K |
| `test` | 4,952 | VOC2007 test, giữ nguyên; chỉ dùng báo cáo cuối |

Việc tách dùng chính seed của run (`random.Random(seed).shuffle`), nên mỗi seed
có một split `train'/val'` khác nhau — biến thiên của data split được tính vào
std giữa các seed. File split được cache theo `(seed, ratio)` tại dataset root
(`holdout_seed<S>_val10.{yaml,txt}`) nên chạy lại cùng seed sẽ tái lập y hệt.

Không được nhìn kết quả test để chọn K. Code chỉ in tất cả K trong
`TOP_K_VALUES`; không có dòng nào tự chọn “best strategy” từ test.

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
   convolution/linear weights, bias và affine parameters (γ, β) của BatchNorm.
   **Không phải EMA**: không có hệ số decay, không phụ thuộc thứ tự, mỗi
   checkpoint đóng góp đúng `1/K` bất kể rank hay epoch.
5. Không average state không learnable như `running_mean`, `running_var`,
   `num_batches_tracked`, anchors, stride hoặc cache. Những state này không nằm
   trong \(\mathbf{w}\); chúng được giữ từ checkpoint tốt nhất rồi ước lượng
   lại (xem mục BatchNorm bên dưới).
6. Average và checkpoint kết quả giữ FP32; phép cộng thực hiện ở float64 rồi
   mới hạ về dtype gốc, nên kết quả không phụ thuộc thứ tự cộng và không tích
   lũy sai số làm tròn theo K.
7. Trước khi average, code kiểm tra mọi checkpoint có cùng tập key và cùng
   shape; lệch là dừng chứ không im lặng bỏ qua.

## BatchNorm recalibration

Sau khi average, weights của các layer trước BN đã đổi nhưng `running_mean` /
`running_var` vẫn là của checkpoint cũ → phân phối activation lệch thống kê.
`evaluate.update_bn_stats()` ước lượng lại (tương đương
`torch.optim.swa_utils.update_bn`):

1. `reset_running_stats()` và đặt `momentum = None` → BN dùng **cumulative
   moving average**, tức trung bình chính xác trên toàn bộ batch đã đi qua,
   không phụ thuộc thứ tự batch, không có quán tính.
2. **Lặp qua training data ở chế độ forward-only**:
   - toàn bộ nằm trong `torch.no_grad()` → không dựng graph, không backward;
   - không tạo optimizer, không có `optimizer.step()`;
   - `model.requires_grad_(False)` cho toàn mạng;
   - `model.eval()` cho toàn mạng, riêng các module BN gọi `.train()` để chúng
     tích lũy mean/var;
   - kết thúc vòng lặp code **assert** mọi `parameter.grad is None`, sai là
     raise — đây là bằng chứng chạy được rằng không có cập nhật gradient nào.
3. Chỉ `running_mean` / `running_var` / `num_batches_tracked` thay đổi;
   learnable parameters giữ nguyên đúng giá trị vừa average.

Dữ liệu dùng là **split `train'`**, không bao giờ chạm `val'` hay `test`.
Dataloader dùng `mode='train'` (`BN_UPDATE_AUGMENT=True`) để phân phối ảnh khớp
đúng phân phối mà BN stats gốc được tích lũy trên đó lúc train.

Vì Strategy 2 được BN-recalibrate còn baseline thì không, `BN_UPDATE_CONTROL=True`
thêm một dòng **“Strategy 1 + BN recal”** — baseline đã BN-recalibrate — để tách
phần cải thiện do *averaging* khỏi phần do *hiệu chỉnh BN*. Nếu không có control
này thì không thể quy kết cải thiện cho method.

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
- Sau đó `tidy_run_dir()` rút gọn thư mục seed xuống còn **Excel + `charts/`**:
  ảnh/plot dồn vào `charts/training/`, `results.csv` và `args.yaml` bị xóa
  (nội dung đã nằm nguyên trong sheet `PerEpoch` và `RunInfo`), ảnh debug
  `train_batch*`/`val_batch*` bị xóa. File lạ không nhận diện được thì **giữ
  nguyên**, không xóa mù.

Chạy `--keep-checkpoints` để tắt cả hai bước trên khi cần debug. Lệnh
`strategies`/`eval` từ run cũ chỉ dùng được nếu run đó đã train với
`--keep-checkpoints` (hoặc `DELETE_CHECKPOINTS_AFTER_RUN=False`).

## Cài đặt và chạy

```bash
pip install -r requirements.txt
python main.py train --exp-name yolov8s_exp1
```

`--exp-name` và `--exp_name` đều hợp lệ. Tên được dùng nguyên văn cho thư mục
kết quả; một lệnh chạy trọn 5 seed trong `RANDOM_SEED` và sinh:

```text
results/detection/yolov8s_exp1/
├── SUMMARY.xlsx              ★ mean ± std của cả 5 seed × mọi strategy
├── experiment_config.json    snapshot config lúc chạy
├── charts/
│   ├── 01_topk_curve_mAP50-95.png    mAP theo K + dải ±std + mức baseline
│   ├── 02_strategy_mAP50-95.png      dot plot mean ± std mọi strategy
│   ├── 03_strategy_mAP50.png
│   ├── 04_delta_vs_baseline.png      Δ vs baseline, kèm chấm từng seed
│   └── 05_per_seed_paired.png        mỗi seed một đường (paired)
└── seeds/
    ├── seed_1/
    │   ├── results_seed_1.xlsx
    │   └── charts/
    │       ├── training/             curve + confusion matrix trên val'
    │       ├── test_best/            PR/F1 curve + confusion matrix trên test
    │       ├── test_best_bn/
    │       ├── test_top_2/ … test_top_5/
    ├── seed_10/ …
```

Không có file `.pt` nào được giữ lại. Tên experiment đã tồn tại và không rỗng
sẽ bị từ chối để tránh trộn kết quả giữa các lần chạy.

Một seed lỗi (OOM, hỏng data…) **không** làm mất kết quả của các seed đã xong:
code ghi nhận lỗi, chạy tiếp seed sau, và vẫn xuất `SUMMARY.xlsx` từ các seed
thành công cùng danh sách seed thất bại ở cuối log.

Các lệnh khác:

```bash
# Tắt Strategy 2
python main.py train --exp-name baseline_run01 --no-strategy2

# Đổi tập K cần báo cáo
python main.py train --exp-name k_sweep01 --top-k 2 3 5 8

# Ablation: tắt BN recalibration
python main.py train --exp-name no_bn01 --no-bn-update

# Giữ checkpoint + run dir nguyên vẹn để debug
python main.py train --exp-name debug01 --keep-checkpoints

# Đánh giá một checkpoint được giữ lại có chủ ý
python main.py eval --weights path/to/model.pt --split test

# Export model deploy
python main.py train --exp-name deploy_run01 --export-after-train
```

## Config quan trọng

| Biến | Ý nghĩa |
|---|---|
| `MODEL` | Detection model, ví dụ `yolov8s.pt` |
| `VAL_RATIO` | Tỉ lệ tách validation độc lập từ train; VOC mặc định là `0.1` |
| `PATIENCE` | Early stopping theo raw validation fitness |
| `RANDOM_SEED` | Int hoặc list seed; list = chạy lần lượt rồi tổng hợp mean ± std |
| `TOP_K_VALUES` | Các K được định trước để báo cáo |
| `KEEP_TOP_K_CHECKPOINTS` | Phải ≥ `max(TOP_K_VALUES)` |
| `STRATEGY1_FROM_RAW_TOPK` | Baseline lấy từ rank #1 raw FP32 thay vì `best.pt` FP16 |
| `USE_BN_UPDATE` | Re-estimate BN buffers trên `train'` sau averaging |
| `BN_UPDATE_BATCHES` | Số batch forward khi recalibrate BN |
| `BN_UPDATE_AUGMENT` | Dataloader `mode='train'` (khớp phân phối lúc train) |
| `BN_UPDATE_CONTROL` | Thêm dòng ablation “Strategy 1 + BN recal” |
| `DELETE_CHECKPOINTS_AFTER_RUN` | Xóa checkpoint tạm sau mỗi seed |
| `TIDY_RUN_DIR` | Rút gọn thư mục seed xuống còn Excel + charts |
| `MAKE_CHARTS` | Vẽ chart tổng hợp cuối experiment |
| `EVAL_SPLIT` | Split báo cáo cuối, mặc định `test` |
| `EXP_NAME` | Tên experiment ổn định trên server |

Tại sao `STRATEGY1_FROM_RAW_TOPK=True`: `best.pt` được Ultralytics
`strip_optimizer()` lưu ở FP16, trong khi checkpoint average là FP32. Rank #1
trong bảng ranking là *đúng cùng epoch, cùng weights* với `best.pt` nhưng giữ
FP32, nên dùng nó làm baseline thì hai nhánh đi qua cùng precision và cùng
đường load/eval. Tie-break của ranking (`fitness` giảm dần, hòa thì epoch sớm
hơn) được đặt khớp đúng ngữ nghĩa `best_fitness` của Ultralytics để rank #1
luôn trùng epoch với `best.pt`.

## Output metrics

`seeds/seed_<N>/results_seed_<N>.xlsx` — 5 sheet:

| Sheet | Nội dung |
|---|---|
| `RunInfo` | Toàn bộ hyperparameter + note về data split |
| `Overall` | 1 dòng/strategy: P, R, mAP@0.5, mAP@0.75, mAP@0.5:0.95, fitness, epoch nguồn |
| `PerClass` | AP từng class × strategy |
| `TopK_Checkpoints` | Epoch nào lọt Top-K, fitness `val'`, dùng cho K nào |
| `PerEpoch` | Nguyên `results.csv` của Ultralytics |

`SUMMARY.xlsx` — 7 sheet:

| Sheet | Nội dung |
|---|---|
| `Mean_Std` | ★ 1 dòng/strategy, có sẵn cột `"0.5150 ± 0.0047"` copy thẳng vào paper |
| `PerSeed` | Số liệu thô, 1 dòng/(seed × strategy) |
| `Delta_vs_Baseline` | Δ paired vs baseline: mean ± std, số seed thắng, t-statistic, p-value |
| `PerClass_Mean_Std` | AP từng class, mean ± std trên các seed |
| `PerClass_PerSeed` | AP từng class, số liệu thô |
| `TopK_Checkpoints` | Epoch lọt Top-K ở từng seed |
| `Config` | Snapshot config lúc chạy |

`Delta_vs_Baseline` dùng **paired two-sided t-test**: cùng seed nghĩa là cùng
data split và cùng quá trình train, nên so sánh theo cặp mới đúng thiết kế thí
nghiệm (unpaired sẽ bị nuốt bởi biến thiên giữa các seed, vốn lớn hơn hiệu ứng
nhiều lần). `p` tính bằng regularized incomplete beta, không cần scipy.

Troubleshooting:

- OOM: giảm `BATCH`/`IMGSZ` hoặc dùng `--batch -1`.
- Không đủ Top-K: số epoch thực tế trước early stopping nhỏ hơn K.
- Run cũ báo không phải raw checkpoint: phải train lại bằng code hiện tại;
  code chủ động không trộn EMA checkpoint với raw averaging.
