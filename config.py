"""
Configuration file for Strategy 2 - Instance Segmentation (Ultralytics YOLO + Carparts)

Nhánh này CHỈ làm instance segmentation với các model YOLO (qua Ultralytics API).
Mọi tham số chỉnh ở đây; các giá trị hay dùng đều override được qua CLI (main.py).
"""
import os


class Config:

    # ===================== Model Configuration =====================
    # KHÔNG chốt cứng version YOLO nào làm mặc định — bạn TỰ SET giá trị này
    # (hoặc truyền --model khi chạy). Nhận mọi giá trị mà ultralytics.YOLO()
    # nhận, đổi version = sửa đúng 1 dòng:
    #   - Pretrained weights : "yolov8s-seg.pt" | "yolo11s-seg.pt" (SEGMENTATION models)
    #   - Custom weights     : "path/to/your_best.pt"
    #   - Train from scratch : "yolov8s-seg.yaml" (hoặc .yaml kiến trúc custom)
    # ⚠️ PHẢI dùng segmentation model (-seg) cho instance segmentation!
    MODEL = "yolov8s-seg.pt"  # <-- ĐẶT MODEL CỦA BẠN Ở ĐÂY, VD: "yolov8n-seg.pt"

    # ===================== Dataset Configuration =====================
    # "carparts-seg.yaml" = Carparts Segmentation built-in của Ultralytics
    # TỰ ĐỘNG DOWNLOAD lần đầu (~133 MB)
    # (https://docs.ultralytics.com/datasets/segment/carparts-seg)
    # Data đã tải sẵn / data custom: trỏ tới file data.yaml của bạn.
    #
    # ✓ carparts-seg.yaml: train (3156), val (401), test (276) ảnh
    # ✓ Đã có validation set độc lập → không cần tách val' từ train
    #
    # Nếu dùng VOC cho detection: set VAL_RATIO > 0 để tách holdout val'
    # Nếu dùng dataset custom đã có val độc lập: set VAL_RATIO = 0
    DATA = "carparts-seg.yaml"
    VAL_RATIO = 0  # Carparts đã có val độc lập → không cần tách
                   # = 0 → dùng nguyên data.yaml gốc (train, val, test rõ ràng)

    # ===================== Training Configuration =====================
    EPOCHS = 100
    IMGSZ = 640
    BATCH = 128     # -1 = auto-batch theo VRAM (chỉ áp dụng khi train)
    DEVICE = None     # None = auto (GPU nếu có); "0" | "0,1" | "cpu"
    WORKERS = 24
    PATIENCE = 10   # Ultralytics early stopping (epoch không cải thiện fitness val)
    PRETRAINED = True
    CACHE = False     # False | "ram" | "disk" — cache dataset
    RESUME = False
    DETERMINISTIC = True  # Ultralytics đặt torch.deterministic + seed reproducible

    # ===================== Optimizer & LR Schedule (Overridden Only) =====================
    OPTIMIZER = "auto"  # Bộ tối ưu (auto, SGD, Adam, AdamW, RMSprop, ...)
    LR0 = 5e-3       # LR ban đầu (mặc định Ultralytics là 0.01)
    LRF = 0.01        # Hệ số LR cuối cùng (final learning rate factor = lr0 * lrf)
    WARMUP_EPOCHS = 5  # Số epoch warmup (mặc định Ultralytics là 3.0)
    COS_LR = True      # Dùng cosine learning rate decay (mặc định Ultralytics là False)

    # ===================== Loss Function =====================
    # Focal loss là cải tiến cho detection classification loss (BCE).
    # Instance segmentation dùng mask loss (không focal) → config này IGNORED.
    # Giữ lại structure code để compatibility, nhưng không ảnh hưởng training.
    LOSS_FUNCTION = "bce"  # Segmentation: không dùng focal loss
    FOCAL_GAMMA = 1.5      # (Ignored for segmentation)
    FOCAL_ALPHA = 0.25     # (Ignored for segmentation)

    # ===================== Augmentation (Overridden Only) =====================
    MIXUP = 0.0            # mixup không tương thích well với instance segmentation
    COPY_PASTE = 0.1       # copy-paste hữu ích cho segmentation (đặc biệt parts)

    # ===================== Strategy Configuration =====================
    # Strategy 1: best.pt — checkpoint có fitness cao nhất trên val' (Ultralytics tự chọn)
    # Strategy 2: average weights của Top-K checkpoint tốt nhất trên val'
    #             (giống nhánh Strategy2_TinyImageNet; fitness = 0.1*mAP50 + 0.9*mAP50-95)
    USE_STRATEGY2 = True
    TOP_K_VALUES = [2, 3, 4, 5]
    # Chỉ giữ đúng K checkpoint tốt nhất trên disk: checkpoint mỗi epoch được
    # Ultralytics lưu (save_period=1) rồi TopKCheckpointManager prune NGAY nếu
    # ngoài Top-K — không lưu tất cả epoch (xem train.py)
    KEEP_TOP_K_CHECKPOINTS = 5  # nên = max(TOP_K_VALUES)

    # BatchNorm update sau khi average — bắt chước update_bn() của repo gốc:
    # sau average, running_mean/running_var giữ nguyên từ ckpt tốt nhất (không
    # average vì đây là population stats) nhưng weights đổi → phải chạy
    # forward pass trên train để re-estimate BN stats khớp weights mới,
    # nếu không mAP của Strategy 2 sẽ tụt do BN lệch phân phối.
    USE_BN_UPDATE = True
    BN_UPDATE_BATCHES = 100  # giống num_batches=100 của repo gốc

    # ===================== Evaluation Configuration =====================
    # Split dùng cho báo cáo cuối (Strategy 1 vs Strategy 2):
    #   "test" (mặc định — hold-out test set) | "val" (validation set)
    EVAL_SPLIT = "test"
    CONF = None  # confidence threshold; None = mặc định Ultralytics khi val (0.001)
    IOU = None   # NMS IoU threshold; None = mặc định Ultralytics

    # ===================== Output Configuration =====================
    PROJECT = os.path.join("results", "segmentation")  # thư mục output gốc (thay đổi từ detection)
    NAME = None          # None = Ultralytics tự đánh số (train, train2, ...)
    EXIST_OK = False
    EXCEL_OUTPUT = None  # None = <run_dir>/segmentation_results.xlsx

    # Random seed: dùng cho cả tách val' (dataset.py) và model.train(seed=...).
    # Hỗ trợ số nguyên đơn lẻ (ví dụ: 42) hoặc danh sách các seed (ví dụ: [42, 100, 2026]).
    RANDOM_SEED = [1, 10, 42, 100, 500]

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
    EXTRA_TRAIN_ARGS = {}

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

        if not cls.OPTIMIZER:
            raise ValueError("OPTIMIZER must not be empty")

        if float(cls.WARMUP_EPOCHS) < 0:
            raise ValueError("WARMUP_EPOCHS must be non-negative")

        if not 0.0 <= float(cls.LRF) <= 1.0:
            raise ValueError("LRF must be in [0, 1]")

        for prob in ("MIXUP", "COPY_PASTE"):
            if not 0.0 <= float(getattr(cls, prob)) <= 1.0:
                raise ValueError(f"{prob} must be in [0, 1]")

        if str(cls.LOSS_FUNCTION).lower() not in ("bce", "focal"):
            raise ValueError(
                f"LOSS_FUNCTION={cls.LOSS_FUNCTION!r} không hỗ trợ. "
                "Chọn: 'bce' (Ultralytics default) hoặc 'focal' (detection only)."
            )
        if float(cls.FOCAL_GAMMA) < 0:
            raise ValueError("FOCAL_GAMMA must be non-negative")
        if not 0.0 <= float(cls.FOCAL_ALPHA) <= 1.0:
            raise ValueError("FOCAL_ALPHA must be in [0, 1]")
        # Note: Focal loss chỉ áp dụng cho detection, segmentation bỏ qua

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
        print(f"  Task  : Instance Segmentation")
        print(f"  Strategy 2: {'ON — Top-K ' + str(cls.TOP_K_VALUES) if cls.USE_STRATEGY2 else 'OFF'}")
        print(f"  Eval split: {cls.EVAL_SPLIT or 'mặc định theo data.yaml'}")