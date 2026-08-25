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
    MODEL = "yolov8s.pt"  # <-- ĐẶT MODEL CỦA BẠN Ở ĐÂY, ví dụ: "yolov8n.pt"

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
    BATCH = 128     # -1 = auto-batch theo VRAM (chỉ áp dụng khi train)
    DEVICE = None     # None = auto (GPU nếu có); "0" | "0,1" | "cpu"
    WORKERS = 8
    # Số worker cho dataloader của các lượt SAU train (model.val của từng
    # strategy + BN recalibration). Tách khỏi WORKERS vì RAM, không vì tốc độ:
    # mỗi worker giữ prefetch_factor(=2) batch trong hàng đợi, tức
    #     workers × 2 × batch × 3 × imgsz² byte
    # ≈ 7 GB với WORKERS=24, batch=128, imgsz=640 cho MỘT dataloader. Lúc train
    # chỉ có 1 loader nên chịu được, nhưng mỗi seed còn chạy 6 lượt val + 5 lượt
    # BN recalibration nối nhau. None = min(WORKERS, 8).
    EVAL_WORKERS = 8
    # Tự hạ WORKERS/EVAL_WORKERS theo số CPU mà SLURM thực sự cấp cho job.
    # Cần vì Ultralytics chặn worker bằng os.cpu_count() (CPU của CẢ NODE) còn
    # PyTorch cảnh báo theo os.sched_getaffinity (CPU của job) → mặc kệ thì log
    # đầy "This DataLoader will create N worker processes in total".
    AUTO_LIMIT_WORKERS = True
    PATIENCE = 10   # Ultralytics early stopping (epoch không cải thiện fitness val)
    PRETRAINED = True
    CACHE = False     # False | "ram" | "disk" — cache dataset
    RESUME = False
    DETERMINISTIC = True  # Ultralytics đặt torch.deterministic + seed reproducible

    # ===================== Optimizer & LR Schedule (Overridden Only) =====================
    OPTIMIZER = "SGD"  # Bộ tối ưu (auto, SGD, Adam, AdamW, RMSprop, ...)
    LR0 = 5e-3       # LR ban đầu (mặc định Ultralytics là 0.01)
    LRF = 0.01        # Hệ số LR cuối cùng (final learning rate factor = lr0 * lrf)
    WARMUP_EPOCHS = 5  # Số epoch warmup (mặc định Ultralytics là 3.0)
    COS_LR = True      # Dùng cosine learning rate decay (mặc định Ultralytics là False)

    # ===================== Loss Function =====================
    # Chọn cls loss cho detection — song song LOSS_FUNCTION của nhánh
    # Strategy2_TinyImageNet ('cross_entropy' | 'poly_focal').
    # - "bce"  : nn.BCEWithLogitsLoss(reduction="none") — mặc định Ultralytics
    # - "focal": thay bằng FocalBCE (losses.py) — BCE * (1-p_t)^γ * α_factor,
    #            công thức Focal Loss chuẩn, có hook install_cls_loss.
    LOSS_FUNCTION = "focal"
    FOCAL_GAMMA = 1.5  # focusing param — tăng γ dồn học vào hard examples
    FOCAL_ALPHA = 0.25  # balancing param — 0 tắt

    # ===================== Augmentation (Overridden Only) =====================
    MIXUP = 0.15       # Bật nhẹ mixup cho detection (mặc định Ultralytics là 0.0)
    # ⚠ COPY_PASTE CHỈ CÓ TÁC DỤNG VỚI LABEL DẠNG SEGMENTATION.
    # ultralytics.data.augment.CopyPaste.__call__ return luôn khi
    # len(labels["instances"].segments) == 0. Label VOC do Ultralytics convert là
    # bbox-only → giá trị này là NO-OP trên VOC, đặt bao nhiêu cũng không đổi gì.
    # Giữ lại để dùng khi chuyển sang dataset có segment; đặt 0.0 nếu muốn config
    # phản ánh đúng những gì thực sự chạy.
    COPY_PASTE = 0.15

    # ===================== Method Comparison =====================
    # Method weight-averaging cần chạy:
    #   "top-k" : Strategy 2 của bạn — uniform average Top-K checkpoint tốt nhất.
    #   "ema"   : Exponential Moving Average of weights (Morales-Brotons, TMLR 2024).
    #   "swa"   : Stochastic Weight Averaging (Izmailov, UAI 2018) + constant LR.
    #
    # ⭐ TỰ TÁCH RUN THEO LR SCHEDULE (xem train.run_experiments):
    #   - top-k và ema chỉ QUAN SÁT weights, KHÔNG đổi LR schedule ⇒ chung MỘT run,
    #     cùng trajectory ⇒ so sánh PAIRED theo seed.
    #   - swa (mode "truncate"/"extend") giữ LR HẰNG SỐ ở pha cuối ⇒ ĐỔI trajectory
    #     ⇒ tự động tách thành EXPERIMENT RIÊNG, kèm PATIENCE=0.
    #
    # Vậy `python main.py` (mặc định dưới đây) sẽ chạy NỐI NHAU 2 experiment:
    #     <exp>       methods=[top-k, ema]  cosine chuẩn,  patience=10
    #     <exp>_swa   methods=[swa]         constant LR,    patience=0
    # Tổng 2 run × 5 seed = 10 lần train, mỗi lần 100 epoch.
    #
    # CLI: --methods swa,ema | --methods top-k ema swa | --methods all | --methods swa
    METHODS = ["top-k", "ema", "swa"]
    VALID_METHODS = ("top-k", "ema", "swa")

    # ---- EMA (baseline 1) ----
    # Implement bằng torch.optim.swa_utils.AveragedModel + get_ema_multi_avg_fn:
    #     w_ema ← decay · w_ema + (1 − decay) · w_t      (mỗi optimizer step)
    # KHÔNG warmup/ramp — công thức lũy thừa thuần của thư viện.
    #
    # MỘT giá trị decay duy nhất ⇒ đúng MỘT dòng "EMA" trong bảng kết quả,
    # không phải chọn decay nào để báo cáo (tránh cherry-pick trên test).
    #
    # Vì sao là 0.999 — hai ràng buộc đồng thời, với N ≈ 11 700 optimizer step
    # (14 896 ảnh train', batch 128, 100 epoch ⇒ 117 step/epoch):
    #   (a) CỬA SỔ trung bình 1/(1−decay) = 1 000 step = 8.6 epoch — cùng thang
    #       với cửa sổ SWA (25 epoch cuối) và Top-K (K = 2..5 checkpoint).
    #   (b) RESIDUE decay^N = 8.3e-06 ⇒ weights khởi tạo đã bị quên hẳn. Không có
    #       warmup nên đây là ràng buộc cứng: decay ≤ 0.9996 (residue < 1%).
    #       0.9999 để lại 31% là model pretrained ⇒ EMA HỎNG.
    # Thêm giá trị vào list nếu muốn sweep (mỗi giá trị = thêm một dòng kết quả);
    # khi đó nên bật --shadow-val để chọn decay trên val chứ không phải trên test.
    # train.py in cửa sổ + residue của decay ra log ngay khi bắt đầu train.
    EMA_DECAYS = [0.999]
    EMA_UPDATE_PERIOD = 1   # cập nhật mỗi N optimizer step (paper EMA dùng T=16)

    # ---- SWA (baseline 2) ----
    # Implement bằng torch.optim.swa_utils.AveragedModel + get_swa_multi_avg_fn:
    #     w_swa ← (w_swa · n + w_t) / (n + 1)      (mỗi epoch, cycle length c = 1)
    # use_buffers=False ⇒ PyTorch average đúng learnable params còn BN running
    # stats được ĐỒNG BỘ từ model nguồn — khớp Algorithm 1 của Izmailov, nơi BN
    # được tính lại bằng một lượt forward trên train sau khi average.
    SWA_START_FRACS = [0.75]   # mốc bắt đầu average, theo tỉ lệ budget
    SWA_PERIOD = 1             # average mỗi N epoch

    # ---- LR schedule cho pha SWA ----
    # ⭐ CHỐT CHO PAPER: "truncate" — 75% đầu chạy ĐÚNG cosine schedule của bạn,
    #    25% sau giữ LR HẰNG SỐ = SWA_LR. TỔNG BUDGET GIỮ NGUYÊN 100 epoch ⇒
    #    công bằng tuyệt đối về số epoch với Top-K/EMA.
    #    Đây là biến thể "1 budget" của Izmailov et al. cho VGG / WRN / PreResNet:
    #    "we first run standard SGD training for ≈75% of the training budget ...
    #     we just stop the training early WITHOUT MODIFYING the learning rate
    #     schedule" (§3.2). 75 epoch đầu trùng khít run chuẩn (bit-exact).
    #
    # ⚠ constant LR ĐỔI TRAJECTORY ⇒ run bật SWA là MỘT EXPERIMENT RIÊNG.
    #   Top-K/EMA PHẢI lấy số từ run khác (METHODS mặc định dưới đây). Code
    #   RAISE lỗi nếu bạn bật swa chung với top-k/ema ở chế độ constant.
    #
    # Hai mode còn lại giữ trong code nhưng KHÔNG dùng cho paper:
    #   "inherit" : SWA average trên chính cosine (không constant) — cùng trajectory
    #               với Top-K/EMA. LR ở cửa sổ 75–99 chỉ còn ~6% lr0.
    #   "extend"  : cosine trọn 100 epoch RỒI nối thêm epoch constant (1.25 budget)
    #               — biến thể Shake-Shake/ImageNet (§4.1), SWA được nhiều epoch hơn.
    SWA_LR_SCHEDULE = "truncate"  # "truncate" (chốt) | "inherit" | "extend"

    # Mốc cắt cosine để chuyển sang constant LR (mode "truncate").
    # 0.75 ⇒ epoch 0..74 chạy cosine y nguyên, epoch 75..99 chạy constant, và SWA
    # average đúng 25 epoch đó.
    SWA_LR_START_FRAC = 0.75

    # Chỉ dùng khi mode = "extend": số epoch constant-LR nối thêm, theo tỉ lệ EPOCHS.
    SWA_EXTRA_BUDGET = 0.25

    SWA_LR = None                 # constant LR của pha SWA.
                                  # None = TỰ TÍNH giá trị TRUNG GIAN giữa LR lớn nhất và
                                  # nhỏ nhất của annealing schedule, đúng khuyến nghị
                                  # Izmailov et al. §4.3:
                                  #     SWA_LR = (lr0 + lr0·lrf) / 2
                                  # Với lr0=5e-3, lrf=0.01 ⇒ (5e-3 + 5e-5)/2 = 2.525e-3
                                  # (0.505 × lr0 — đúng cận trên khoảng 0.1×–0.5× lr0
                                  #  mà Izmailov dùng thực tế, ví dụ CIFAR-100 WRN).
                                  # Tự tính theo lr0/lrf HIỆN HÀNH nên --lr0 trên CLI cũng
                                  # được tôn trọng. Đặt số cụ thể để ghi đè (hoặc --swa-lr).

    # ---- Snapshot báo cáo của EMA/SWA ----
    # "final"   (MẶC ĐỌNH): lấy trạng thái shadow ở EPOCH CUỐI của run — đúng
    #           cách dùng chuẩn của cả hai paper. Mọi epoch đều đã được tính vào
    #           trung bình lũy thừa, kể cả các epoch trong đuôi `patience` sau
    #           best checkpoint (ví dụ best = epoch 30, patience = 5, dừng ở 35
    #           ⇒ EMA báo cáo là x_EMA^35, đã nuốt cả epoch 31–35).
    #           KHÔNG cần nhìn validation ⇒ không tốn thêm thời gian train.
    # "best_val": chọn epoch có fitness val cao nhất (cần SHADOW_VAL_ENABLED=True).
    SHADOW_SELECT = "final"       # "final" | "best_val"
    # Bật để có thêm đường cong fitness val của shadow theo epoch (ghi vào
    # averaging_shadows.json, hữu ích cho phụ lục). Cái giá: mỗi shadow tốn thêm
    # MỘT lượt val trên tập val mỗi SHADOW_VAL_PERIOD epoch.
    SHADOW_VAL_ENABLED = False
    SHADOW_VAL_PERIOD = 1

    # ===================== Strategy Configuration =====================
    # Strategy 1: best.pt — RAW checkpoint có fitness cao nhất trên val'
    # Strategy 2: uniform element-wise average tất cả RAW learnable parameters
    #             của Top-K checkpoint theo raw-model fitness validation.
    #             EMA smoothing được tắt trong train.py để ranking nhất quán.
    USE_STRATEGY2 = True   # tự đồng bộ theo METHODS trong normalize_methods()
    TOP_K_VALUES = [2, 3, 4, 5]
    # Chỉ giữ đúng K checkpoint tốt nhất trên disk: checkpoint mỗi epoch được
    # Ultralytics lưu (save_period=1) rồi TopKCheckpointManager prune NGAY nếu
    # ngoài Top-K — không lưu tất cả epoch (xem train.py)
    KEEP_TOP_K_CHECKPOINTS = 5  # nên = max(TOP_K_VALUES)

    # Strategy 1 lấy từ rank #1 của chính bảng ranking Top-K (raw FP32) thay vì
    # best.pt. Cùng epoch, cùng weights — nhưng best.pt được Ultralytics lưu ở
    # FP16 nên nếu so trực tiếp với bản average FP32 thì hai nhánh lệch
    # precision. True = so sánh apples-to-apples. False = dùng best.pt.
    STRATEGY1_FROM_RAW_TOPK = True

    # BatchNorm recalibration sau khi average — tương đương update_bn() của
    # torch.optim.swa_utils: sau average, running_mean/running_var giữ nguyên từ
    # ckpt tốt nhất (không average vì đây là population stats) nhưng weights đã
    # đổi → phải LẶP QUA TRAINING DATA ở chế độ forward-only (no_grad, không
    # optimizer) để ước lượng lại BN stats khớp weights mới. Không làm thì mAP
    # của Strategy 2 tụt do BN lệch phân phối.
    USE_BN_UPDATE = True
    BN_UPDATE_BATCHES = 100   # số batch forward (giống num_batches=100 repo gốc)
    BN_UPDATE_AUGMENT = True  # True = dataloader mode='train' (đúng phân phối đã
                              # augment mà BN stats gốc được tích lũy trên đó)
    BN_UPDATE_AMP = True      # autocast trên CUDA cho khớp precision lúc train
    # Control quan trọng cho paper: Strategy 2 được BN-recalibrate còn Strategy 1
    # thì không → không thể biết cải thiện đến từ AVERAGING hay từ BN. Bật cờ này
    # để eval thêm dòng "Strategy 1 + BN recal" (baseline đã BN-recalibrate) làm
    # ablation. Tốn thêm 1 lần BN update + 1 lần eval mỗi seed.
    BN_UPDATE_CONTROL = True

    # ===================== Evaluation Configuration =====================
    # Split dùng cho báo cáo cuối (Strategy 1 vs Strategy 2):
    #   "test" (mặc định — VOC2007 test) | "val" (val' holdout) | None (theo data.yaml)
    EVAL_SPLIT = "test"
    CONF = None  # confidence threshold; None = mặc định Ultralytics khi val (0.001)
    IOU = None   # NMS IoU threshold; None = mặc định Ultralytics

    # ===================== Output Configuration =====================
    # Cây thư mục kết quả của `python main.py train --exp-name <NAME>`:
    #
    #   results/detection/<NAME>/
    #   ├── SUMMARY.xlsx            ★ mean ± std của TẤT CẢ seed × strategy
    #   ├── experiment_config.json  snapshot config lúc chạy
    #   ├── charts/                 chart tổng hợp cả experiment (5 hình)
    #   └── seeds/seed_<N>/
    #       ├── results_seed_<N>.xlsx
    #       └── charts/{training, test_<strategy>}/
    #
    # Không giữ lại bất kỳ checkpoint .pt nào.
    PROJECT = os.path.join("results", "detection")  # thư mục output gốc
    # Truyền qua CLI: --exp-name voc_yolov8s_raw_topk_run01
    # Có EXP_NAME thì pipeline dùng đúng tên này, không ghép timestamp/model.
    EXP_NAME = None
    NAME = None          # prefix legacy khi không truyền EXP_NAME
    EXIST_OK = False
    EXCEL_OUTPUT = None  # chỉ dùng cho `main.py eval/export`; train luôn tự đặt tên
    # Checkpoint chỉ là file tạm để rank/average/eval; Excel/chart được giữ.
    DELETE_CHECKPOINTS_AFTER_RUN = True
    # Dọn run dir mỗi seed xuống còn đúng Excel + charts/
    TIDY_RUN_DIR = True
    KEEP_RESULTS_CSV = False    # nội dung đã nằm nguyên trong sheet PerEpoch
    KEEP_SAMPLE_IMAGES = False  # train_batch*.jpg / val_batch*.jpg (ảnh debug, nặng)
    MAKE_CHARTS = True          # vẽ chart tổng hợp mean ± std ở cuối experiment

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

    # Alias người dùng hay gõ → tên chuẩn trong VALID_METHODS
    METHOD_ALIASES = {
        "topk": "top-k", "top_k": "top-k", "top-k": "top-k", "strategy2": "top-k",
        "s2": "top-k", "cwa": "top-k",
        "ema": "ema",
        "swa": "swa",
    }

    @classmethod
    def normalize_methods(cls, methods=None):
        """
        Chuẩn hóa cls.METHODS: parse alias, tách dấu phẩy, bỏ trùng, giữ thứ tự.

        Chấp nhận: ["top-k", "ema"], "EMA,SWA", ["all"], ["none"].
        Đồng bộ luôn USE_STRATEGY2 = ("top-k" in METHODS) để phần code cũ
        (build_train_args, run_strategy_evaluation, Excel...) không phải đổi.
        """
        raw = cls.METHODS if methods is None else methods
        if raw is None:
            raw = []
        if isinstance(raw, str):
            raw = [raw]

        tokens = []
        for item in raw:
            tokens.extend(str(item).replace(";", ",").split(","))

        resolved = []
        for token in tokens:
            name = token.strip().lower()
            if not name:
                continue
            if name == "all":
                resolved.extend(cls.VALID_METHODS)
                continue
            if name in ("none", "off"):
                resolved = []
                continue
            canonical = cls.METHOD_ALIASES.get(name)
            if canonical is None:
                raise ValueError(
                    f"--method {token!r} không hợp lệ. Chọn trong "
                    f"{list(cls.VALID_METHODS)} (hoặc 'all'/'none'), "
                    "ví dụ: --method top-k ema swa | --method EMA,SWA"
                )
            resolved.append(canonical)

        # Giữ thứ tự chuẩn để bảng kết quả luôn nhất quán giữa các lần chạy.
        cls.METHODS = [m for m in cls.VALID_METHODS if m in set(resolved)]
        cls.USE_STRATEGY2 = "top-k" in cls.METHODS
        return cls.METHODS

    @classmethod
    def swa_lr_mode(cls):
        """'inherit' | 'extend' | 'truncate' — chuẩn hoá, chấp nhận alias 'constant'."""
        mode = str(cls.SWA_LR_SCHEDULE).strip().lower()
        if mode == "constant":
            return "extend"
        if mode not in ("inherit", "extend", "truncate"):
            raise ValueError(
                f"SWA_LR_SCHEDULE={cls.SWA_LR_SCHEDULE!r} không hợp lệ. "
                "Chọn 'inherit' | 'extend' | 'truncate'."
            )
        return mode

    @classmethod
    def swa_extra_epochs(cls):
        """Số epoch constant-LR nối thêm (chỉ > 0 ở mode 'extend')."""
        if not cls.method_enabled("swa") or cls.swa_lr_mode() != "extend":
            return 0
        return max(1, int(round(float(cls.SWA_EXTRA_BUDGET) * int(cls.EPOCHS))))

    @classmethod
    def total_epochs(cls):
        """
        Số epoch THẬT truyền cho model.train().

        Mode 'extend' nối thêm pha constant-LR SAU khi cosine chạy trọn EPOCHS,
        nên tổng = EPOCHS + swa_extra_epochs(). Các mode khác giữ nguyên EPOCHS.
        """
        return int(cls.EPOCHS) + cls.swa_extra_epochs()

    @classmethod
    def resolved_swa_lr(cls):
        """
        Giá trị constant LR thực tế của pha SWA.

        ``SWA_LR = None`` ⇒ trung bị́nh cộng của LR lớn nhất và nhỏ nhất trong
        annealing schedule: ``(lr0 + lr0·lrf) / 2`` — "intermediate value between
        the largest and the smallest learning rate used in the annealing scheme"
        (Izmailov et al. §4.3). Tính theo LR0/LRF hiện hành nên CLI override vẫn đúng.
        """
        if cls.SWA_LR is None:
            return (float(cls.LR0) + float(cls.LR0) * float(cls.LRF)) / 2.0
        return float(cls.SWA_LR)

    @classmethod
    def method_enabled(cls, name):
        return str(name).lower() in cls.METHODS

    @classmethod
    def any_method(cls):
        """Có ít nhất một method cần RAW checkpoint / raw-weight validation."""
        return bool(cls.METHODS)

    @classmethod
    def validate_config(cls, require_model=True):
        """Validate configuration (giữ pattern validate_config của repo gốc)."""
        cls.normalize_methods()
        if require_model and not cls.MODEL:
            raise ValueError(
                "Chưa set model. Đặt Config.MODEL trong config.py "
                "(ví dụ: 'yolov8n.pt', 'yolo11n.pt', 'yolov5nu.pt', .pt/.yaml custom) "
                "hoặc truyền --model khi chạy CLI."
            )

        if not cls.DATA:
            raise ValueError("DATA must not be empty (VOC.yaml hoặc path tới data.yaml custom)")

        if cls.EXP_NAME:
            exp_name = str(cls.EXP_NAME).strip()
            if exp_name in (".", "..") or any(separator in exp_name for separator in ("/", "\\")):
                raise ValueError(
                    "EXP_NAME phải là một tên thư mục đơn, không chứa '/' hoặc '\\'. "
                    "Ví dụ: voc_yolov8s_raw_topk_run01."
                )

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

        if cls.EVAL_WORKERS is not None and int(cls.EVAL_WORKERS) < 0:
            raise ValueError("EVAL_WORKERS must be non-negative, or None to use min(WORKERS, 8)")

        # Cảnh báo RAM HOST (không phải VRAM): mỗi worker của dataloader giữ
        # prefetch_factor(=2) batch ảnh uint8 trong hàng đợi. Đây là thứ khiến
        # job bị OOM killer giết ("Killed" / slurmstepd: oom_kill) chứ không
        # phải CUDA out of memory.
        if int(cls.BATCH) > 0:
            from memory import cpu_quota, training_prefetch_bytes

            workers = int(cls.WORKERS)
            if cls.AUTO_LIMIT_WORKERS:
                workers = min(workers, max(1, cpu_quota() // 2))
            queue_bytes = training_prefetch_bytes(workers, int(cls.BATCH), int(cls.IMGSZ))
            print(
                f"  RAM prefetch (ước lượng): ~{queue_bytes / 1024 ** 3:.1f} GB — "
                f"train loader {workers} worker × batch {cls.BATCH} + "
                f"val loader {workers * 2} worker × batch {cls.BATCH * 2} "
                "(Ultralytics tự nhân đôi cả hai cho val)"
            )
            if queue_bytes > 8 * 1024 ** 3:
                print(
                    "⚠ WARNING: hàng đợi prefetch có thể chiếm "
                    f"~{queue_bytes / 1024 ** 3:.1f} GB RAM host. Xin đủ --mem cho job SLURM, "
                    "hoặc giảm WORKERS / BATCH nếu bị OOM killer."
                )

        if not cls.OPTIMIZER:
            raise ValueError("OPTIMIZER must not be empty")

        if float(cls.WARMUP_EPOCHS) < 0:
            raise ValueError("WARMUP_EPOCHS must be non-negative")

        if not 0.0 <= float(cls.LRF) <= 1.0:
            raise ValueError("LRF must be in [0, 1]")

        for prob in ("MIXUP", "COPY_PASTE"):
            if not 0.0 <= float(getattr(cls, prob)) <= 1.0:
                raise ValueError(f"{prob} must be in [0, 1]")

        if float(cls.COPY_PASTE) > 0 and str(cls.DATA).endswith("VOC.yaml"):
            print(
                f"⚠ WARNING: COPY_PASTE={cls.COPY_PASTE} nhưng label VOC là bbox-only. "
                "Ultralytics chỉ áp dụng copy-paste khi có segmentation mask "
                "(CopyPaste.__call__ return ngay nếu instances.segments rỗng) → "
                "tham số này KHÔNG có tác dụng gì trên VOC."
            )

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

        seeds = cls.RANDOM_SEED if isinstance(cls.RANDOM_SEED, (list, tuple)) else [cls.RANDOM_SEED]
        if not seeds:
            raise ValueError("RANDOM_SEED must be an int or a non-empty list of ints")
        if len(set(int(s) for s in seeds)) != len(seeds):
            raise ValueError(f"RANDOM_SEED có seed trùng nhau: {list(seeds)}")

        if cls.USE_STRATEGY2:
            if not cls.TOP_K_VALUES or any(int(k) <= 1 for k in cls.TOP_K_VALUES):
                raise ValueError("TOP_K_VALUES must be a non-empty list of ints > 1")
            if int(cls.KEEP_TOP_K_CHECKPOINTS) < max(cls.TOP_K_VALUES):
                raise ValueError(
                    "KEEP_TOP_K_CHECKPOINTS must be >= max(TOP_K_VALUES) "
                    "để đủ checkpoint cho mọi giá trị K"
                )
            if int(cls.EPOCHS) < max(cls.TOP_K_VALUES):
                raise ValueError(
                    f"EPOCHS={cls.EPOCHS} < max(TOP_K_VALUES)={max(cls.TOP_K_VALUES)} "
                    "→ không đủ checkpoint để average"
                )
        if cls.any_method():
            if cls.USE_BN_UPDATE and int(cls.BN_UPDATE_BATCHES) <= 0:
                raise ValueError("BN_UPDATE_BATCHES must be positive khi USE_BN_UPDATE=True")
            if float(cls.VAL_RATIO) == 0.0 and str(cls.DATA).endswith("VOC.yaml"):
                print(
                    "⚠ WARNING: có method averaging nhưng VAL_RATIO=0 → không có val "
                    "độc lập, checkpoint/snapshot sẽ được chọn trên chính tập test (leakage)!"
                )
            if not cls.USE_BN_UPDATE:
                print(
                    "⚠ WARNING: USE_BN_UPDATE=False → sau khi average, BN running stats "
                    "không khớp weights mới. Cả Top-K, EMA lẫn SWA đều bị tụt oan "
                    "(Izmailov et al. nêu rõ BN phải được tính lại sau khi average)."
                )

        # ---- EMA / SWA hyper-parameters ----
        if cls.method_enabled("ema"):
            if not cls.EMA_DECAYS:
                raise ValueError("EMA_DECAYS must be a non-empty list")
            for decay in cls.EMA_DECAYS:
                if not 0.0 < float(decay) < 1.0:
                    raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
            if int(cls.EMA_UPDATE_PERIOD) < 1:
                raise ValueError("EMA_UPDATE_PERIOD must be >= 1")

        if cls.method_enabled("swa"):
            if not cls.SWA_START_FRACS:
                raise ValueError("SWA_START_FRACS must be a non-empty list")
            for frac in cls.SWA_START_FRACS:
                if not 0.0 <= float(frac) < 1.0:
                    raise ValueError(f"SWA start fraction must be in [0, 1), got {frac}")
            if int(cls.SWA_PERIOD) < 1:
                raise ValueError("SWA_PERIOD must be >= 1")
            mode = cls.swa_lr_mode()
            if mode != "inherit":
                if cls.SWA_LR is not None and not 0.0 < float(cls.SWA_LR):
                    raise ValueError(
                        "SWA_LR must be positive, or None to auto-compute (lr0 + lr0*lrf)/2"
                    )
                if mode == "extend" and not 0.0 < float(cls.SWA_EXTRA_BUDGET):
                    raise ValueError("SWA_EXTRA_BUDGET must be positive when SWA_LR_SCHEDULE='extend'")
                if mode == "truncate" and not 0.0 <= float(cls.SWA_LR_START_FRAC) < 1.0:
                    raise ValueError("SWA_LR_START_FRAC must be in [0, 1)")
                extra = cls.swa_extra_epochs()
                total = cls.total_epochs()
                if mode == "extend":
                    detail = (f"cosine chạy trọn {cls.EPOCHS} epoch (đúng run chuẩn) rồi nối "
                              f"{extra} epoch constant ⇒ tổng {total} epoch "
                              f"({total / int(cls.EPOCHS):.2f} budget)")
                else:
                    detail = (f"cắt cosine ở {float(cls.SWA_LR_START_FRAC):.0%} rồi chạy nốt "
                              f"bằng constant ⇒ tổng vẫn {total} epoch (1.00 budget)")
                print(
                    f"⚠ SWA_LR_SCHEDULE={mode!r}: LR giữ hằng {cls.resolved_swa_lr():g} "
                    f"({cls.resolved_swa_lr() / float(cls.LR0):.3f} × lr0"
                    + (" — tự tính (lr0+lr0·lrf)/2" if cls.SWA_LR is None else "") + f"); {detail}. "
                    "Đây là TRAJECTORY KHÁC với run chuẩn ⇒ phải báo cáo như một experiment riêng."
                )
                # LỖI CỨNG, không phải cảnh báo: constant LR đổi trajectory từ mốc cắt
                # trở đi. Nếu top-k/ema cùng bật thì chúng bị tính trên một LR
                # schedule KHÔNG PHẢI của chúng ⇒ số liệu vô giá trị cho paper.
                # Chặn hẳn để không thể vô tình trộn hai trajectory vào một bảng.
                # KHÔNG raise: run_experiments() trong train.py tự TÁCH METHODS
                # thành các leg có LR schedule khác nhau rồi chạy tuần tự — leg
                # top-k/ema dùng cosine chuẩn, leg swa dùng constant LR + patience=0.
                # Ở đây chỉ báo cho biết sẽ có mấy experiment.
                shared = [m for m in ("top-k", "ema") if cls.method_enabled(m)]
                if shared:
                    print(
                        f"ℹ {shared} không đổi LR schedule còn 'swa' thì có ⇒ sẽ chạy "
                        f"2 EXPERIMENT RIÊNG BIỆT nối nhau: leg 1 = {shared} (cosine chuẩn), "
                        "leg 2 = ['swa'] (constant LR, patience=0). Xem run_experiments()."
                    )
                if int(cls.PATIENCE) > 0:
                    print(
                        f"ℹ Leg SWA sẽ tự dùng PATIENCE=0 (thay vì {cls.PATIENCE}): constant LR "
                        "làm val fitness đi ngang nên early stopping sẽ cắt mất pha SWA. "
                        "Các leg khác giữ nguyên PATIENCE."
                    )

        if str(cls.SHADOW_SELECT).lower() not in ("best_val", "final"):
            raise ValueError("SHADOW_SELECT must be 'best_val' or 'final'")
        if int(cls.SHADOW_VAL_PERIOD) < 1:
            raise ValueError("SHADOW_VAL_PERIOD must be >= 1")
        if (cls.method_enabled("ema") or cls.method_enabled("swa")) \
                and not cls.SHADOW_VAL_ENABLED and str(cls.SHADOW_SELECT).lower() == "best_val":
            print(
                "ℹ SHADOW_VAL_ENABLED=False → SHADOW_SELECT bị hạ về 'final': "
                "EMA/SWA lấy snapshot ở epoch cuối thay vì epoch tốt nhất trên val'."
            )

        if cls.EXPORT_ENABLED and not cls.EXPORT_FORMAT:
            raise ValueError("EXPORT_ENABLED=True requires EXPORT_FORMAT (e.g. onnx, engine)")

        print("[OK] Config validated successfully")
        print(f"  Model : {cls.MODEL or '(chưa set — bắt buộc khi train)'}")
        print(f"  Data  : {cls.DATA} | VAL_RATIO: {cls.VAL_RATIO} (val' tách từ train)")
        extra = cls.swa_extra_epochs()
        print(f"  Epochs: {cls.EPOCHS}"
              + (f" (+{extra} constant-LR cho SWA = {cls.total_epochs()})" if extra else "")
              + f" | imgsz: {cls.IMGSZ} | batch: {cls.BATCH}")
        from memory import cpu_quota

        train_workers = int(cls.WORKERS)
        eval_workers = (int(cls.EVAL_WORKERS) if cls.EVAL_WORKERS is not None
                        else min(int(cls.WORKERS), 8))
        limited = ""
        if cls.AUTO_LIMIT_WORKERS:
            quota = cpu_quota()
            capped_train = min(train_workers, max(1, quota // 2))
            capped_eval = min(eval_workers, quota)
            if (capped_train, capped_eval) != (train_workers, eval_workers):
                limited = (f"  (đã hạ từ {train_workers}/{eval_workers} "
                           f"theo {quota} CPU job được cấp)")
            train_workers, eval_workers = capped_train, capped_eval
        print(
            f"  Workers: {train_workers} khi train (Ultralytics dùng "
            f"{train_workers * 2} cho val loader) | {eval_workers} khi val/BN recal{limited}"
        )
        print(f"  Seeds : {list(seeds)}")
        print(f"  Task  : Object Detection")
        print(f"  Experiment name: {cls.EXP_NAME or '(auto timestamp)'}")
        print(f"  Loss  : {cls.LOSS_FUNCTION}"
              + (f" (γ={cls.FOCAL_GAMMA}, α={cls.FOCAL_ALPHA})" if cls.LOSS_FUNCTION == 'focal' else ""))
        print(f"  Methods: {', '.join(cls.METHODS) if cls.METHODS else '(none — chỉ train + Strategy 1)'}")
        if cls.USE_STRATEGY2:
            print(f"    top-k      : K ∈ {cls.TOP_K_VALUES} — uniform element-wise mean của "
                  "learnable params (KHÔNG EMA)")
        if cls.method_enabled("ema"):
            print(f"    ema        : decay ∈ {list(cls.EMA_DECAYS)} (không warmup), "
                  f"update mỗi {cls.EMA_UPDATE_PERIOD} optimizer step")
        if cls.method_enabled("swa"):
            print(f"    swa        : start ∈ {[f'{float(f):.0%}' for f in cls.SWA_START_FRACS]} "
                  f"budget, average mỗi {cls.SWA_PERIOD} epoch")
            mode = cls.swa_lr_mode()
            if mode == "inherit":
                print("    swa LR     : theo cosine schedule chung (inherit) — cùng trajectory")
            else:
                print(f"    swa LR     : HẰNG SỐ {cls.resolved_swa_lr():g} "
                      f"({cls.resolved_swa_lr() / float(cls.LR0):.3f} × lr0"
                      + (" — auto" if cls.SWA_LR is None else "") + f") — mode '{mode}'")
                if mode == "extend":
                    print(f"                 cosine trọn {cls.EPOCHS} epoch + "
                          f"{cls.swa_extra_epochs()} epoch constant = {cls.total_epochs()} epoch "
                          f"({cls.total_epochs() / int(cls.EPOCHS):.2f} budget) — TRAJECTORY RIÊNG")
                else:
                    print(f"                 cắt cosine ở {float(cls.SWA_LR_START_FRAC):.0%} budget, "
                          f"tổng vẫn {cls.total_epochs()} epoch — TRAJECTORY RIÊNG")
        if cls.method_enabled("ema") or cls.method_enabled("swa"):
            print(
                "    snapshot   : "
                + (f"best trên val' (val shadow mỗi {cls.SHADOW_VAL_PERIOD} epoch)"
                   if cls.SHADOW_VAL_ENABLED and str(cls.SHADOW_SELECT).lower() == "best_val"
                   else "epoch cuối (không val shadow)")
            )
        if cls.any_method():
            print(
                "    BN recal   : "
                + (f"ON — {cls.BN_UPDATE_BATCHES} batch trên split train, forward-only "
                   "(áp dụng GIỐNG NHAU cho top-k / ema / swa)"
                   if cls.USE_BN_UPDATE else "OFF")
            )
            print(f"    BN control : {'ON (thêm dòng Strategy 1 + BN recal)' if cls.BN_UPDATE_CONTROL else 'OFF'}")
            print(f"    Strategy 1 : {'rank #1 raw FP32' if cls.STRATEGY1_FROM_RAW_TOPK else 'best.pt (FP16)'}")
        print(
            "  Checkpoints: "
            + ("temporary → delete after evaluation" if cls.DELETE_CHECKPOINTS_AFTER_RUN else "keep")
        )
        print(f"  Eval split: {cls.EVAL_SPLIT or 'mặc định theo data.yaml'}")
