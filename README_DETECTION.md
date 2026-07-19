# Strategy 2 — Object Detection (Ultralytics YOLO + Pascal VOC)

Mở rộng Strategy 2 từ classification (Tiny ImageNet) sang **object detection**,
dùng **toàn bộ Ultralytics API** (train / val / export) — không tự viết training
loop. Code kế thừa convention của nhánh `Strategy2_TinyImageNet`: config-driven,
CLI override, export metrics ra Excel bằng `pandas` + `openpyxl`.

## ⚠️ LƯU Ý QUAN TRỌNG TRƯỚC KHI ĐỌC SỐ LIỆU: `val` ≡ `test`

Dataset VOC built-in của Ultralytics ([docs](https://docs.ultralytics.com/datasets/detect/voc))
thực chất chỉ có **train / test**:

| Split trong `VOC.yaml` | Trỏ tới | Số ảnh |
|---|---|---|
| `train` | trainval VOC2007 + VOC2012 | 16,551 |
| `val` | **test VOC2007** | 4,952 |
| `test` | **test VOC2007** (trùng với `val`) | 4,952 |

Hệ quả:
- **Metrics "val" in ra trong lúc train thực chất đo trên VOC2007 test.**
- **KHÔNG có validation set độc lập** tách khỏi test → best.pt được chọn theo
  fitness trên chính tập test. Khi báo cáo phải ghi rõ điều này để người đọc
  không hiểu nhầm rằng đã có held-out val riêng (và không kết luận nhầm về
  data leakage / overfitting khi thấy số "val" và "test" giống nhau).
- Chọn `--split val` hay `--split test` khi eval đều cho **cùng một kết quả**.
- Muốn có val độc lập thật sự: tự tách một phần train ra và viết `data.yaml`
  custom rồi trỏ `data:` tới file đó (ngoài scope mặc định của nhánh này).

Note này cũng được ghi thẳng vào sheet `Summary` của file Excel output
(`NOTE val/test`) và comment trong `configs/detection_voc.yaml`.

## Cấu trúc

```
├── configs/
│   └── detection_voc.yaml     # Config detection (model, data, epochs, imgsz, batch, ...)
├── detection/
│   ├── config.py              # Load YAML + CLI override + validate (pattern main.py gốc)
│   ├── train.py               # model.train() + val cuối + Excel + Edge AI hook
│   ├── eval.py                # model.val() với weights đã train
│   ├── metrics_utils.py       # Đọc metrics từ DetMetrics (results.box.*) + in console
│   └── export_metrics.py      # Excel 2 sheet: Summary + PerEpoch (parse results.csv)
├── main_detection.py          # Entrypoint: train / eval / export / export-model
└── README_DETECTION.md        # File này
```

Các file classification của nhánh gốc (`main.py`, `train.py`, `evaluate.py`, ...)
giữ nguyên, không bị ảnh hưởng.

## Cài đặt

```bash
pip install -r requirements.txt   # đã gồm ultralytics, pandas, openpyxl
```

## 1. Set model path (bắt buộc, KHÔNG hardcode trong code)

Mở `configs/detection_voc.yaml`, sửa đúng 1 dòng:

```yaml
model: yolov8n.pt      # hoặc yolo11n.pt, yolov5nu.pt, path/to/custom.pt, yolov8n.yaml
```

Hoặc không sửa file, override lúc chạy:

```bash
python main_detection.py train --model yolo11n.pt
```

`model` nhận mọi giá trị mà `ultralytics.YOLO()` nhận: tên pretrained weights
(tự download), đường dẫn `.pt` custom, hoặc `.yaml` kiến trúc để train from
scratch. Đổi version YOLO = đổi 1 dòng config, không đụng code.

## 2. Train

```bash
# Dùng toàn bộ giá trị trong config
python main_detection.py train --config configs/detection_voc.yaml

# Override nhanh qua CLI (pattern giống main.py của nhánh classification)
python main_detection.py train --model yolov8n.pt --epochs 50 --batch 32 --imgsz 640 --device 0 --name yolov8n_voc_e50
```

- `data: VOC.yaml` → Ultralytics **tự động download** Pascal VOC lần chạy đầu
  (~2.8 GB, về thư mục `datasets_dir` trong settings của Ultralytics; xem/đổi
  bằng `yolo settings`). Dùng data đã tải sẵn: sửa `data:` trỏ tới `data.yaml`
  của bạn.
- Output: `results/detection/<name>/` gồm `weights/best.pt`, `weights/last.pt`,
  `results.csv` (per-epoch), plots, và `detection_results.xlsx` (tự export
  ngay sau train).
- Sau train, pipeline tự chạy val cuối trên `best.pt`, **in ra console**:
  Precision, Recall, mAP@0.5, mAP@0.75, mAP@0.5:0.95, Fitness và **AP per class**
  (đọc từ `results.box.*` của Ultralytics, không tự tính lại).

## 3. Eval

```bash
python main_detection.py eval --weights results/detection/train/weights/best.pt

# Chọn split rõ ràng (nhớ: với VOC built-in thì val ≡ test)
python main_detection.py eval --weights results/detection/train/weights/best.pt --split test

# Chỉ in metrics, không xuất Excel
python main_detection.py eval --weights results/detection/train/weights/best.pt --no-excel
```

Mặc định `--split` không set → dùng đúng split mặc định của `data.yaml`
(`val`). Cần weights **đã train trên VOC** — weights COCO thô (80 class) sẽ
lệch số class với VOC (20 class).

## 4. Export Excel (2 sheet)

Train và eval đã tự export Excel. Chạy lại thủ công từ một run dir bất kỳ:

```bash
# Offline — chỉ parse results.csv (Summary lấy từ epoch cuối, KHÔNG có per-class AP)
python main_detection.py export --run-dir results/detection/train

# Đầy đủ — chạy thêm model.val() để Summary có per-class AP
python main_detection.py export --run-dir results/detection/train --weights results/detection/train/weights/best.pt

# Đổi chỗ lưu
python main_detection.py export --run-dir results/detection/train --output bao_cao_voc.xlsx
```

Nội dung file `.xlsx`:

| Sheet | Nội dung | Nguồn |
|---|---|---|
| `Summary` | Run info (model, data, imgsz, epochs, batch, ngày chạy, note val≡test) + overall metrics (P, R, mAP@0.5, mAP@0.75, mAP@0.5:0.95, Fitness) + bảng **AP per class** | `model.val()` → `DetMetrics` |
| `PerEpoch` | Mỗi epoch 1 row: `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `val/box_loss`, `val/cls_loss`, `val/dfl_loss`, `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, lr... | `results.csv` do Ultralytics tự sinh |

## 5. (Optional) Edge AI export — ONNX / TensorRT / FP16

Tắt mặc định. Bật trong config:

```yaml
export:
  enabled: true
  format: onnx     # hoặc engine (TensorRT), openvino, tflite...
  half: true       # FP16
```

Hoặc qua CLI:

```bash
# Export ngay sau train
python main_detection.py train --export-after-train

# Export weights đã có sẵn, không cần train lại
python main_detection.py export-model --weights results/detection/train/weights/best.pt --format onnx
python main_detection.py export-model --weights results/detection/train/weights/best.pt --format engine --half
```

Dùng `model.export()` của Ultralytics; `format: engine` (TensorRT) cần GPU +
TensorRT cài sẵn trên máy.

## Chạy trên Vast.ai (gợi ý workflow)

```bash
git clone <repo-url> && cd Capstone_KD
git checkout claude/strategy2-object-detection-yolo-ttjplw
pip install -r requirements.txt

# (tuỳ chọn) đổi chỗ chứa dataset về volume lớn của instance
yolo settings datasets_dir=/workspace/datasets

# train — VOC tự download lần đầu (~2.8 GB)
python main_detection.py train --config configs/detection_voc.yaml \
    --model yolov8n.pt --device 0 --name yolov8n_voc

# chạy nền + giữ log khi rớt SSH
nohup python main_detection.py train --model yolov8n.pt --device 0 --name yolov8n_voc \
    > train_voc.log 2>&1 &
tail -f train_voc.log

# lấy kết quả về máy: results/detection/yolov8n_voc/detection_results.xlsx
```

## Assumptions (convention chưa rõ trong repo → hướng đã chọn)

1. **Config YAML thay vì class `Config` Python**: nhánh gốc dùng `config.py`
   (class), nhưng yêu cầu đặt ra là config yaml/CLI + train-args của
   Ultralytics vốn là key-value → dùng YAML (`configs/detection_voc.yaml`),
   giữ nguyên pattern CLI-override và `validate_config()` của nhánh gốc.
2. **Code detection nằm trong package `detection/`** thay vì đè lên
   `train.py`/`evaluate.py` gốc (tránh phá pipeline classification cùng nhánh);
   entrypoint riêng `main_detection.py` theo pattern "1 entrypoint" của repo.
3. **Thư mục run do Ultralytics quản lý** (`project/name`, tự đánh số
   `train`, `train2`, ...) thay vì `results/<số>` như nhánh gốc — vì
   `results.csv`, weights, plots đều do Ultralytics ghi vào đó; chỉ đổi gốc
   `project: results/detection` cho khớp pattern `results/` của repo.
4. **Seed mặc định = 1** (`Config.SEEDS[0]` của nhánh gốc), truyền thẳng vào
   `model.train(seed=...)`; không tự set seed thủ công ngoài Ultralytics.
5. `requirements.txt` giữ nguyên các dependency classification (torch, timm...)
   vì nhánh này vẫn chứa cả pipeline classification; chỉ thêm `ultralytics`.
