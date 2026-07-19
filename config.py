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
    PATIENCE = 100    # Ultralytics early stopping (epoch không cải thiện fitness val)
    PRETRAINED = True
    CACHE = False     # False | "ram" | "disk" — cache dataset
    RESUME = False
    DETERMINISTIC = True  # Ultralytics đặt torch.deterministic + seed reproducible

    # ===================== Optimizer & LR Schedule =====================
    # Tương ứng nhóm "Optimizer / Scheduler" của Strategy2_TinyImageNet
    # (OPTIMIZER, LR/WD, WARMUP_*, SCHEDULER=linear_warmup_cosine, ETA_MIN...).
    # Ultralytics SGD-momentum với warmup + cos_lr là tương đương gần nhất.
    OPTIMIZER = "auto"  # auto | SGD | Adam | AdamW | NAdam | RAdam | RMSProp
    LR0 = 0.01          # LR ban đầu
    LRF = 0.01          # LR cuối = LR0 * LRF (Ultralytics dùng linear/cosine tới đây)
    MOMENTUM = 0.937    # SGD momentum / Adam β1
    WEIGHT_DECAY = 5e-4
    WARMUP_EPOCHS = 3.0
    WARMUP_MOMENTUM = 0.8  # momentum khởi động warmup (tăng dần tới MOMENTUM)
    WARMUP_BIAS_LR = 0.1   # bias LR khi warmup (giảm dần về LR0)
    COS_LR = False         # True = cosine LR (giống SCHEDULER='linear_warmup_cosine' của repo gốc)

    # ===================== Loss Function =====================
    # Chọn cls loss cho detection — song song LOSS_FUNCTION của nhánh
    # Strategy2_TinyImageNet ('cross_entropy' | 'poly_focal').
    # - "bce"  : nn.BCEWithLogitsLoss(reduction="none") — mặc định Ultralytics
    #            (v8DetectionLoss dùng BCE, không phải CE — vì detection cls
    #            là multi-label per anchor, không phải single-label softmax).
    # - "focal": thay bằng FocalBCE (losses.py) — BCE * (1-p_t)^γ * α_factor,
    #            công thức Focal Loss chuẩn, có hook install_cls_loss.
    LOSS_FUNCTION = "bce"
    FOCAL_GAMMA = 1.5  # focusing param — tăng γ dồn học vào hard examples
    FOCAL_ALPHA = 0.25  # balancing param — 0 tắt

    # ===================== Loss Gains =====================
    # Trọng số 3 thành phần loss của Ultralytics YOLO (box regression + class
    # + distribution focal loss). Không có tương ứng trực tiếp trong repo gốc
    # (classification chỉ có 1 loss), nhưng đây là hyperparam then chốt của
    # detection theo tuning guide.
    BOX_GAIN = 7.5
    CLS_GAIN = 0.5
    DFL_GAIN = 1.5
    LABEL_SMOOTHING = 0.0  # giống LABEL_SMOOTHING của repo gốc (chỉ khác 0 nếu cần)
    DROPOUT = 0.0          # dropout ở detection head (tương đương DROPOUT_RATE repo gốc)
    NBS = 64               # nominal batch size — Ultralytics scale WD theo BATCH/NBS
    CLOSE_MOSAIC = 10      # tắt mosaic ở N epoch cuối (theo YOLOv8 tuning)

    # ===================== Augmentation =====================
    # Nhóm augmentation của Ultralytics — tương ứng "USE_MIXUP_CUTMIX,
    # MIXUP_ALPHA, HORIZONTAL_FLIP_PROB, RANDOM_ERASING_*" của repo gốc, có
    # thêm augmentation dành riêng cho detection (HSV, geometric, mosaic).
    HSV_H = 0.015     # hue jitter (0-1)
    HSV_S = 0.7       # saturation jitter (0-1)
    HSV_V = 0.4       # value/brightness jitter (0-1)
    DEGREES = 0.0     # random rotation ±deg
    TRANSLATE = 0.1   # random translation (fraction of image)
    SCALE = 0.5       # random scale (±)
    SHEAR = 0.0       # random shear deg
    PERSPECTIVE = 0.0 # random perspective (0-0.001)
    FLIPUD = 0.0      # xác suất flip trục dọc
    FLIPLR = 0.5      # xác suất flip trục ngang (tương đương HORIZONTAL_FLIP_PROB)
    BGR = 0.0         # xác suất đổi channel order BGR
    MOSAIC = 1.0      # xác suất mosaic 4-image
    MIXUP = 0.0       # xác suất mixup (repo gốc bật, YOLO detection thường 0)
    CUTMIX = 0.0      # xác suất cutmix (repo gốc bật, YOLO detection thường 0)
    COPY_PASTE = 0.0  # copy-paste augmentation cho detection
    AUTO_AUGMENT = "randaugment"  # randaugment | autoaugment | augmix (classification only cho backbone-pretrain)
    ERASING = 0.4     # random erasing prob (repo gốc RANDOM_ERASING_PROB=0.25)

    # ===================== Precision & Runtime =====================
    # Tương ứng "USE_AMP, AMP_DTYPE, MULTI_SCALE..." của repo gốc.
    AMP = True         # Ultralytics auto-mixed precision (FP16/BF16 theo GPU)
    MULTI_SCALE = 0.0  # >0 = random rescale ảnh mỗi batch (fraction ±); 0 = tắt
    RECT = False       # rectangular training (nhóm ảnh cùng aspect ratio → nhanh hơn)
    SINGLE_CLS = False # gộp mọi class thành 1 (chỉ để debug)
    FREEZE = None      # freeze N layer đầu (backbone); None = không freeze

    # Truyền thêm train-arg Ultralytics bất kỳ mà không cần sửa code, và ghi
    # đè MỌI key ở trên nếu cần (https://docs.ultralytics.com/modes/train/)
    EXTRA_TRAIN_ARGS = {}

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

        # Sanity check các hyperparam mới
        if not 0.0 <= float(cls.MOMENTUM) < 1.0:
            raise ValueError("MOMENTUM must be in [0, 1)")
        if not 0.0 <= float(cls.WEIGHT_DECAY) < 1.0:
            raise ValueError("WEIGHT_DECAY must be in [0, 1)")
        if float(cls.WARMUP_EPOCHS) < 0:
            raise ValueError("WARMUP_EPOCHS must be non-negative")
        if not 0.0 <= float(cls.LABEL_SMOOTHING) < 1.0:
            raise ValueError("LABEL_SMOOTHING must be in [0, 1)")
        for prob in ("FLIPUD", "FLIPLR", "BGR", "MOSAIC", "MIXUP", "CUTMIX",
                     "COPY_PASTE", "ERASING"):
            if not 0.0 <= float(getattr(cls, prob)) <= 1.0:
                raise ValueError(f"{prob} must be in [0, 1]")

        if str(cls.LOSS_FUNCTION).lower() not in ("bce", "focal"):
            raise ValueError(
                f"LOSS_FUNCTION={cls.LOSS_FUNCTION!r} không hỗ trợ. "
                "Chọn: 'bce' (Ultralytics default) hoặc 'focal'."
            )
        if float(cls.FOCAL_GAMMA) < 0:
            raise ValueError("FOCAL_GAMMA must be non-negative")
        if not 0.0 <= float(cls.FOCAL_ALPHA) <= 1.0:
            raise ValueError("FOCAL_ALPHA must be in [0, 1]")

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
        print(f"  Loss  : {cls.LOSS_FUNCTION}"
              + (f" (γ={cls.FOCAL_GAMMA}, α={cls.FOCAL_ALPHA})" if cls.LOSS_FUNCTION == 'focal' else ""))
        print(f"  Strategy 2: {'ON — Top-K ' + str(cls.TOP_K_VALUES) if cls.USE_STRATEGY2 else 'OFF'}")
        print(f"  Eval split: {cls.EVAL_SPLIT or 'mặc định theo data.yaml'}")
