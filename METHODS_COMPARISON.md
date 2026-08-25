# Quy trình so sánh Top-K vs EMA vs SWA — và bảng kiểm "chuẩn chỉnh"

File này áp dụng cho **cả hai nhánh** (`CWA_Detection` / VOC và `CWA_Segmentation` /
Carparts). Code: [`averaging.py`](averaging.py) · [`evaluate.py`](evaluate.py) ·
[`config.py`](config.py).

EMA và SWA đều dựng trên **`torch.optim.swa_utils` của PyTorch** — không tự viết
lại phép average.

---

## 1. Quy trình — mỗi seed đúng MỘT lần train

```
                  ┌─ SGD nesterov (mom .937, wd 1e-3, lr0 5e-3, cosine → lrf .01, warmup 5ep) ─┐
   pretrained ────┤                    MỘT trajectory duy nhất                                 ├──► w_final
                  └───── EMA/SWA chỉ QUAN SÁT weights, không đụng optimization ────────────────┘
                        │                    │                      │
   mỗi optimizer step ──┤        mỗi epoch ──┤          mỗi epoch ──┤
                        ▼                    ▼                      ▼
                   EMA shadow            SWA shadow          raw ckpt epoch_i
                   (×N decay)           (từ 75% budget)     → prune giữ Top-K
                        │                    │                      │
              snapshot = EPOCH CUỐI   snapshot = EPOCH CUỐI   rank theo fitness val
                        └────────────┬───────────────────────────────┘
                                     ▼
                   BN recalibration (GIỐNG NHAU cho mọi bản average)
                                     ▼
                     model.val() trên split test → Excel + chart
```

Thứ tự trong một epoch: batch loop → `on_train_epoch_end` (SWA snapshot) →
Ultralytics validate raw model → `best.pt` + early stopping → `on_model_save`
(raw ckpt FP32, prune Top-K) → `on_fit_epoch_end`.

**Mặc định không có val shadow ⇒ EMA/SWA không tốn thêm một giây train nào.**

---

## 2. Ba toán tử average

| | Top-K (method của bạn) | EMA | SWA |
|---|---|---|---|
| Công thức | `w = (1/K)·Σ w_i` | `w ← d·w + (1−d)·w_t` | `w ← (w·n + w_t)/(n+1)` |
| Implement | `evaluate.average_checkpoints` | `AveragedModel(multi_avg_fn=get_ema_multi_avg_fn(d), use_buffers=True)` | `AveragedModel(multi_avg_fn=get_swa_multi_avg_fn(), use_buffers=False)` |
| Nhịp | post-hoc, K ckpt tốt nhất | mỗi optimizer step | mỗi epoch, từ `start·budget` |
| Nguồn | Strategy 2 | Morales-Brotons TMLR 2024 | Izmailov UAI 2018, Alg. 1 |
| BN buffer | giữ từ ckpt tốt nhất | được EMA cùng | đồng bộ từ model mới nhất |
| Snapshot báo cáo | chọn theo fitness val | **epoch cuối** | **epoch cuối** |
| Hyperparam | `K ∈ {2,3,4,5}` pre-register | `decay` — MỘT giá trị | `start = 0.75` |

Cả ba đều đi qua **cùng một** `update_bn_stats()` sau khi average.

### 2.1 EMA — không warmup, và ràng buộc `decay^N`

Dùng đúng công thức lũy thừa thuần của PyTorch, **không có ramp**. Hệ quả: phần
weights **khởi tạo** còn sót lại trong EMA cuối là `decay^N` (N = tổng optimizer
step). Đây là con số quyết định decay nào dùng được:

| decay | cửa sổ (Det / Seg) | residue Det (N=11 700) | residue Seg (N=2 500) |
|---|---|---|---|
| 0.9 | 0.09 / 0.4 ep | 0 % | 0 % |
| 0.99 | 0.85 / 4 ep | 0 % | 0 % |
| 0.998 | 4.3 / 20 ep | 0 % | 0.7 % |
| 0.999 | 8.5 / 40 ep | 0 % | **8.2 %** |
| 0.9995 | 17 / 80 ep | 0.3 % | **28.6 %** |
| 0.9999 | 85 / 400 ep | **31 %** | **78 %** |

Ngưỡng an toàn (residue < 1 %): Detection `decay ≤ 0.9996`, Segmentation
`decay ≤ 0.9982`.

**Mặc định dùng ĐÚNG MỘT decay mỗi nhánh** ⇒ đúng một dòng "EMA" trong bảng, không
phải chọn decay nào để báo cáo (tránh cherry-pick trên test). Giá trị được chọn
sao cho **cửa sổ trung bình ~9–10 epoch** — cùng thang với cửa sổ SWA (25 epoch
cuối) — và residue gần 0:

| nhánh | N (step) | `EMA_DECAYS` | cửa sổ | residue |
|---|---|---|---|---|
| Detection | 11 700 | `[0.999]` | 8.6 epoch | 8.3e-06 |
| Segmentation | 2 500 | `[0.996]` | 10.0 epoch | 4.5e-05 |

Hai nhánh **bắt buộc decay khác nhau** vì số step/epoch chêch nhau ~4.7 lần.
Thêm giá trị vào list = thêm dòng kết quả; khi sweep nên bật `--shadow-val` để
chọn decay trên val. `averaging.py` in cửa sổ + residue ra log lúc
`on_train_start` kèm cảnh báo nếu > 1 %, nên sai cấu hình là thấy ngay.

Cửa sổ và residue được tính theo **optimizer step** (đã chia `trainer.accumulate`),
không phải số batch.

### 2.2 SWA — ba chế độ LR (`SWA_LR_SCHEDULE`)

Constant LR là **một phần thuật toán của SWA**, không phải của Top-K/EMA — nên
không tồn tại một run duy nhất mà cả hai cùng được chạy đúng setup của mình.

| mode | Pha 1 | Pha 2 | Budget | Trajectory |
|---|---|---|---|---|
| `inherit` (mặc định) | cosine trọn `EPOCHS` | — | 1.00 | **giống hệt** Top-K/EMA ⇒ paired |
| `extend` | cosine **trọn** `EPOCHS` | +`SWA_EXTRA_BUDGET`×`EPOCHS` epoch @ `SWA_LR` | 1.25 | `EPOCHS` epoch đầu **trùng khít** run chuẩn |
| `truncate` | cosine **cắt** ở `SWA_LR_START_FRAC` | phần còn lại @ `SWA_LR` | 1.00 | khác từ mốc cắt |

Cả hai biến thể constant đều có trong Izmailov et al.:

- `extend` — Shake-Shake / PyramidNet / ImageNet: *"we use a **full budget** to get an
  initialization for the procedure, and then train ... for 0.25 and 0.5 budgets"* (§4.1).
- `truncate` — VGG / WRN / PreResNet: *"we first run standard SGD training for ≈75 % of
  the training budget ... we just stop the training early **without modifying the
  learning rate schedule**"* (§3.2).

**`extend` là lựa chọn tốt nhất cho so sánh này**: `EPOCHS` epoch đầu trùng khít
(đã kiểm chứng **bit-exact**) run mà Top-K/EMA dùng, nên SWA chỉ là phần **nối thêm**
— không làm lệch method của bạn một chút nào. Đánh đổi: SWA được nhiều epoch hơn
⇒ phải ghi rõ **"1.25 budget"** trong paper (paper gốc cũng báo cáo đúng kiểu vậy).

Cài đặt chỉ đổi **đúng một thứ**: lambda của `LambdaLR` có sẵn (không dựng
scheduler mới ⇒ không đụng `initial_lr`). Ở `extend`, cosine được **dựng lại trên
`Config.EPOCHS`** chứ không phải tổng epoch, vì `_setup_scheduler` của Ultralytics
mặc định trải cosine trên toàn bộ `trainer.epochs`. LR thực tế vẫn ghi vào
`results.csv` (`lr/pg0..2`) nên audit lại được bằng mắt.

Cửa sổ average của SWA tự bám đúng pha constant LR, và code **cảnh báo** nếu
`PATIENCE > 0` hoặc nếu top-k/ema cũng bật trong run constant.

**Chọn `SWA_LR`** — với `lr0 = 5e-3`:

| ứng viên | giá trị | ×lr0 | căn cứ |
|---|---|---|---|
| trung bình nhân | 5.00e-4 | 0.10 | cận dưới khoảng Izmailov dùng thực tế |
| **mặc định — `SWA_LR = None` ⇒ tự tính `(lr0 + lr0·lrf)/2`** | **2.525e-3** | **0.505** | Izmailov §4.3: "intermediate value between the largest and the smallest learning rate used in the annealing scheme" |

`SWA_LR` tự tính theo `LR0`/`LRF` **hiện hành** nên `--lr0` trên CLI cũng được tôn
trọng; đặt số cụ thể (hoặc `--swa-lr`) để ghi đè.


---

## 3. Bảng kiểm chuẩn chỉnh

| # | Tiêu chí | Trạng thái | Cơ chế / bằng chứng |
|---|---|---|---|
| 1 | Cùng optimizer + LR schedule | ✅ | Cùng MỘT `model.train()`. Chế độ `constant` là ngoại lệ CÓ CHỦ Ý, được cảnh báo và ghi vào config snapshot |
| 2 | Cùng seed, data split, budget epoch | ✅ | Cùng run |
| 3 | EMA/SWA không đổi optimization | ✅ | Chỉ đọc weights; `trainer.ema.enabled=False` nên val/`best.pt`/early stopping luôn dùng raw weights ở MỌI tổ hợp `--methods` |
| 4 | Phép average là implementation chuẩn | ✅ | `torch.optim.swa_utils` — unit-test khớp công thức tay tới `1e-6` |
| 5 | Không method nào nhìn split `test` | ✅ | `test` chỉ dùng ở bước eval cuối |
| 6 | Cùng xử lý BatchNorm | ✅ | Cả 3 qua `update_bn_stats()` giống hệt (reset stats, `momentum=None`, forward-only trên split train, assert không có `.grad`) |
| 7 | Tách "do averaging" khỏi "do BN" | ✅ | Dòng control: Det `Strategy 1 + BN recal` · Seg `BN recal control (K=1)` |
| 8 | Baseline cùng precision với bản average | ✅ | Det: Strategy 1 = rank #1 raw FP32 · Seg: thêm dòng `Top-1 (best raw ckpt)` |
| 9 | So sánh paired + kiểm định thống kê | ✅ | Paired two-sided t-test trên 5 seed, cả hai nhánh; khớp `scipy.stats.ttest_1samp` tới `1e-15` |
| 10 | EMA decay không bị đặt sai | ✅ | Residue `decay^N` in ra log + cảnh báo tự động; grid mặc định đã theo ngưỡng của từng dataset |
| 11 | Audit lại được sau khi chạy | ✅ | `averaging_shadows.json` (giữ params, epoch, số update) + `experiment_config.json` + cột `lr/pg*` trong `results.csv` |
| 12 | Tín hiệu chọn snapshot | ⚠️ | Top-K chọn K checkpoint theo fitness val; **EMA/SWA không chọn gì cả** (lấy epoch cuối). Xem §5.1 |
| 13 | SWA có LR đủ cao để "explore" | ⚠️→✅ | `inherit`: không (LR trung bình 3.1e-4 ở cửa sổ 75–99). `constant`: có. Xem §5.2 |

---

## 4. Đã kiểm chứng bằng gì

- **Unit test EMA**: khớp công thức `d·w + (1−d)·w_t` tay tới `1e-6`, kể cả
  `EMA_UPDATE_PERIOD=2`; BN buffer được EMA đúng (`use_buffers=True`).
- **Unit test SWA**: khớp trung bình cộng chính xác; BN buffer đồng bộ từ model
  mới nhất (`use_buffers=False`), đúng Alg. 1.
- **Unit test constant-LR**: 75 epoch đầu khớp cosine gốc **bit-exact**; từ epoch
  75 chỉ còn đúng một giá trị LR = `SWA_LR`; không rò rỉ sang epoch 74.
- **Đếm step**: cửa sổ/residue tính theo optimizer step thật (đã chia
  `trainer.accumulate`), verify với `batch=2, nbs=64 → accumulate=32`.
- **Chạy thật end-to-end** cả hai nhánh trên `coco8` / `coco8-seg` với
  `--methods all`, `--methods EMA,SWA`, và cả ba mode `--swa-lr-schedule`.
- **Đối chiếu t-test** với `scipy.stats.ttest_1samp` — khớp `1e-15`.
- **Bug đã tìm ra và sửa**: `final_eval()` của Ultralytics gọi lại
  `on_fit_epoch_end`, khiến val shadow ghi đè `validator.metrics` — dòng "FINAL
  VALIDATION on val' (best ckpt)" in ra số của SWA thay vì `best.pt`. Đã chặn
  bằng cờ `validator.training` + guard trùng epoch.

---

## 5. Hai điểm ⚠ — ghi vào Limitations

### 5.1 Top-K được chọn theo validation, EMA/SWA thì không

Đây là **lựa chọn có chủ ý**: mỗi baseline chạy đúng giao thức chuẩn của nó.
EMA/SWA lấy trạng thái ở **epoch cuối**, gộp toàn bộ trajectory kể cả các epoch
trong đuôi `patience` sau best checkpoint (best = epoch 30, patience = 5, dừng ở
35 ⇒ báo cáo `x_EMA^35`). Đó là cách cả hai paper và Ultralytics ship EMA.

Nếu reviewer hỏi ngược lại, có sẵn đường lui **không cần train lại**: bật
`--shadow-val --shadow-select best_val` để EMA/SWA cũng được chọn epoch theo
fitness val (cùng tập, cùng đại lượng với Top-K). Giá: mỗi shadow tốn thêm một
lượt val mỗi epoch (VOC ~7 %, Carparts ~4 % thời gian một epoch).

### 5.2 SWA ở chế độ `inherit` không có LR cao

Cửa sổ 75–99 của cosine chỉ chạy ở LR trung bình `3.11e-4` (6 % lr0) — thấp hơn
mức Izmailov khuyến nghị. Hai lớp phòng thủ:

1. **Precedent**: paper EMA (TMLR 04/2024) App. F Table 22 chỉ có MỘT cấu hình
   optimizer (SGD Nesterov + cosine annealing), không có run constant-LR riêng
   cho SWA, mà SWA trong Table 1 của họ vẫn best/second-best ở 3/5 dòng.
2. **Có luôn run đúng thuật toán**: `--swa-lr-schedule extend` cho ra biến thể
   gốc. Report nó thành experiment riêng.

Các mục còn lại: `close_mosaic=10` cắt ngang cửa sổ SWA (mosaic tắt từ 90 % budget);
`BN_UPDATE_CLOSE_MOSAIC` của nhánh Segmentation quyết định một lần theo epoch của
checkpoint rank #1 rồi dùng chung cho cả ba method (đối xứng, không tối ưu riêng);
resume giữa chừng không tái tạo được shadow (code cảnh báo).

---

## 6. Lệnh chạy

```bash
# Bảng chính: cả ba method, một lần train, cùng trajectory
python main.py train --exp-name voc_compare_run01 --methods top-k ema swa
```

```bash
# SWA đúng thuật toán gốc — experiment RIÊNG, CHỈ SWA.
# cosine chạy trọn 100 epoch (trùng khít run trên) rồi nối 25 epoch constant LR.
python main.py train --exp-name voc_swa_const --methods swa   --swa-lr-schedule extend --swa-extra-budget 0.25 --patience 0
```

⚠ **Không lấy số Top-K/EMA từ run thứ hai** — constant LR là thuật toán của SWA,
không phải của chúng. Code cảnh báo nếu bạn vô tình bật kèm.

Bảng chính lấy mỗi method từ run có schedule NATIVE của nó:

| Dòng | Lấy từ | Schedule |
|---|---|---|
| Strategy 1 / +BN recal | run 1 | cosine |
| **Top-K (K=2..5)** | run 1 | cosine ← native của bạn |
| **EMA** | run 1 | cosine ← native (EMA không đòi schedule riêng) |
| SWA (budget-matched) | run 1 | cosine — paired với Top-K/EMA |
| **SWA (1.25 budget, const LR)** | run 2 | constant ← native của SWA |

`--methods` (hoặc `--method`) nhận `top-k | ema | swa | all | none`, không phân
biệt hoa thường, chấp nhận dấu phẩy: `--methods EMA,SWA`.

Cờ khác: `--ema-decay`, `--ema-update-period`, `--swa-start`, `--swa-period`,
`--swa-lr`, `--swa-lr-schedule`, `--swa-lr-start`, `--swa-extra-budget`, `--shadow-val`,
`--shadow-select`, `--shadow-val-period`, `--top-k`.

**Số dòng kết quả** (mặc định): Detection 8 (1 baseline + 1 BN control + 4 Top-K +
1 EMA + 1 SWA) · Segmentation 9 (thêm dòng `Top-1 raw`).

---

## 7. Cách viết trong paper

Experimental Setup nói rõ 4 câu: *tất cả method weight-averaging được tính từ
cùng một training run trên mỗi seed; không method nào can thiệp vào optimization;
EMA và SWA dùng implementation chuẩn `torch.optim.swa_utils` và lấy trạng thái ở
epoch cuối theo giao thức của chính paper gốc, trong khi K của chúng tôi được
pre-register và báo cáo đầy đủ; mọi model đã average đều được BN recalibrate như
nhau.* Bảng chính báo cáo mean ± std trên 5 seed; Δ và p-value paired lấy từ
sheet `Delta_vs_Baseline`. Run `--swa-lr-schedule extend` báo cáo thành bảng
ablation riêng, ghi rõ nó dùng LR schedule khác.
