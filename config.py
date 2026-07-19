"""
Configuration file for Strategy 2 - Object Detection (Ultralytics YOLO + Pascal VOC)

Nhánh này CHỈ làm object detection với các model YOLO (qua Ultralytics API).
Mọi tham số chỉnh ở đây; các giá trị hay dùng đều override được qua CLI (main.py).
"""
import os


class Config:

    # ===================== Model Configuration =====================
    # KHÔNG chốt cứng version YOLO nào làm mặc định — bạn TỰ SET giá trị này
    # (hoặc truyền --model khi chạy). Nhận mọi giá trị mà ultralytics.YOLO()
    # nhận, đổi version = sửa đúng 1 dòng:
    #   - Pretrained weights : "yolov8n.pt" | "yolo11n.pt" | "yolov5nu.pt" | ...
    #   - Custom weights     : "path/to/your_best.pt"
    #   - Train from scratch : "yolov8n.yaml" (hoặc .yaml kiến trúc custom)
    MODEL = None  # <-- ĐẶT MODEL CỦA BẠN Ở ĐÂY, ví dụ: "yolov8n.pt"

    # ===================== Dataset Configuration =====================
    # "VOC.yaml" = Pascal VOC built-in của Ultralytics, TỰ ĐỘNG DOWNLOAD lần đầu
    # (https://docs.ultralytics.com/datasets/detect/voc). Data đã tải sẵn /
    # data custom: trỏ tới file data.yaml của bạn.
    #
    # ⚠️ VOC.yaml GỐC: split `val` và `test` TRÙNG NHAU (đều là VOC2007 test,
    # 4952 ảnh) — KHÔNG có validation set độc lập. Strategy 2 cần chọn/average
    # checkpoint theo fitness trên val ĐỘC LẬP với test, nên pipeline tự tách
    # VAL_RATIO từ train làm validation riêng (xem dataset.py):
    #   train (16551 ảnh) → train' (1 - VAL_RATIO) + val' (VAL_RATIO, holdout)
    #   test  = VOC2007 test (4952 ảnh), giữ nguyên làm hold-out báo cáo cuối
    DATA = "VOC.yaml"
    VAL_RATIO = 0.1  # tỉ lệ tách val' từ train (giống VALIDATION_RATIO nhánh classification)
                     # = 0 → dùng nguyên data.yaml gốc: val ≡ test, Strategy 2
                     #   sẽ chọn checkpoint trên chính tập test (leakage) — tránh!

    # ===================== Training Configuration =====================
    EPOCHS = 100
    IMGSZ = 640
    BATCH = 16        # -1 = auto-batch theo VRAM (chỉ áp dụng khi train)
    DEVICE = None     # None = auto (GPU nếu có); "0" | "0,1" | "cpu"
    WORKERS = 8
    OPTIMIZER = "auto"  # auto | SGD | Adam | AdamW | ...
    LR0 = 0.01
    LRF = 0.01
    PATIENCE = 100    # early stopping của Ultralytics (epoch không cải thiện fitness val)
    PRETRAINED = True
    CACHE = False
    RESUME = False

    # Truyền thêm train-arg Ultralytics bất kỳ mà không cần sửa code
    # (https://docs.ultralytics.com/modes/train/#train-settings)
    EXTRA_TRAIN_ARGS = {}  # ví dụ: {"cos_lr": True, "close_mosaic": 10}

    # ===================== Strategy Configuration =====================
    # Strategy 1: best.pt — checkpoint có fitness cao nhất trên val' (Ultralytics tự chọn)
    # Strategy 2: average weights của Top-K checkpoint tốt nhất trên val'
    #             (giống nhánh classification; fitness = 0.1*mAP50 + 0.9*mAP50-95)
    USE_STRATEGY2 = True
    TOP_K_VALUES = [2, 3, 4, 5]
    # Chỉ giữ đúng K checkpoint tốt nhất trên disk: checkpoint mỗi epoch được
    # Ultralytics lưu (save_period=1) rồi TopKCheckpointManager prune NGAY nếu
    # ngoài Top-K — không lưu tất cả epoch (xem train.py)
    KEEP_TOP_K_CHECKPOINTS = 5  # nên = max(TOP_K_VALUES)

    # ===================== Evaluation Configuration =====================
    # Split dùng cho báo cáo cuối (Strategy 1 vs Strategy 2):
    #   "test" (mặc định — VOC2007 test) | "val" (val' holdout) | None (theo data.yaml)
    EVAL_SPLIT = "test"
    CONF = None  # confidence threshold; None = mặc định Ultralytics khi val (0.001)
    IOU = None   # NMS IoU threshold; None = mặc định Ultralytics

    # ===================== Output Configuration =====================
    PROJECT = os.path.join("results", "detection")  # thư mục output gốc
    NAME = None          # None = Ultralytics tự đánh số (train, train2, ...)
    EXIST_OK = False
    EXCEL_OUTPUT = None  # None = <run_dir>/detection_results.xlsx

    # Random seed: dùng cho cả tách val' (dataset.py) và model.train(seed=...)
    RANDOM_SEED = 1

    # ===================== Edge AI Export (optional) =====================
    # Hook xuất model sau train bằng model.export() — TẮT MẶC ĐỊNH.
    # Bật EXPORT_ENABLED=True (hoặc --export-after-train) khi cần deploy edge.
    EXPORT_ENABLED = False
    EXPORT_FORMAT = "onnx"   # onnx | engine (TensorRT) | openvino | tflite | ...
    EXPORT_HALF = False      # FP16 (hữu ích cho TensorRT/edge)
    EXPORT_IMGSZ = None      # None = dùng IMGSZ phía trên
    EXPORT_DYNAMIC = False
    EXPORT_SIMPLIFY = True
    EXPORT_DEVICE = None     # export TensorRT cần GPU → "0" nếu format engine

    VALID_EVAL_SPLITS = (None, "val", "test", "train")

    @classmethod
    def validate_config(cls, require_model=True):
        """Validate configuration (giữ pattern validate_config của repo gốc)."""
        if require_model and not cls.MODEL:
            raise ValueError(
                "Chưa set model. Đặt Config.MODEL trong config.py "
                "(ví dụ: 'yolov8n.pt', 'yolo11n.pt', 'yolov5nu.pt', .pt/.yaml custom) "
                "hoặc truyền --model khi chạy CLI."
            )

        if not cls.DATA:
            raise ValueError("DATA must not be empty (VOC.yaml hoặc path tới data.yaml custom)")

        if not 0.0 <= float(cls.VAL_RATIO) < 1.0:
            raise ValueError("VAL_RATIO must be in [0, 1)")

        if int(cls.EPOCHS) <= 0:
            raise ValueError("EPOCHS must be positive")

        if int(cls.IMGSZ) <= 0:
            raise ValueError("IMGSZ must be positive")

        # BATCH = -1 hợp lệ (auto-batch của Ultralytics); chỉ cấm 0
        if int(cls.BATCH) == 0:
            raise ValueError("BATCH must be positive, or -1 for Ultralytics auto-batch")

        if int(cls.WORKERS) < 0:
            raise ValueError("WORKERS must be non-negative")

        if cls.EVAL_SPLIT not in cls.VALID_EVAL_SPLITS:
            raise ValueError(f"EVAL_SPLIT must be one of {cls.VALID_EVAL_SPLITS}")

        if cls.USE_STRATEGY2:
            if not cls.TOP_K_VALUES or any(int(k) <= 1 for k in cls.TOP_K_VALUES):
                raise ValueError("TOP_K_VALUES must be a non-empty list of ints > 1")
            if int(cls.KEEP_TOP_K_CHECKPOINTS) < max(cls.TOP_K_VALUES):
                raise ValueError(
                    "KEEP_TOP_K_CHECKPOINTS must be >= max(TOP_K_VALUES) "
                    "để đủ checkpoint cho mọi giá trị K"
                )
            if float(cls.VAL_RATIO) == 0.0:
                print(
                    "⚠ WARNING: USE_STRATEGY2=True nhưng VAL_RATIO=0 → không có val "
                    "độc lập, checkpoint sẽ được chọn trên chính tập test (leakage)!"
                )

        if cls.EXPORT_ENABLED and not cls.EXPORT_FORMAT:
            raise ValueError("EXPORT_ENABLED=True requires EXPORT_FORMAT (e.g. onnx, engine)")

        print("[OK] Config validated successfully")
        print(f"  Model : {cls.MODEL or '(chưa set — bắt buộc khi train)'}")
        print(f"  Data  : {cls.DATA} | VAL_RATIO: {cls.VAL_RATIO}")
        print(f"  Epochs: {cls.EPOCHS} | imgsz: {cls.IMGSZ} | batch: {cls.BATCH}")
        print(f"  Strategy 2: {'ON — Top-K ' + str(cls.TOP_K_VALUES) if cls.USE_STRATEGY2 else 'OFF'}")
        print(f"  Eval split: {cls.EVAL_SPLIT or 'mặc định theo data.yaml'}")
