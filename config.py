"""
Configuration file for baseline research
"""
import os

class Config:

    # torchvision CIFAR-100. The official `test` split (10,000 images) is kept
    # untouched as the final test set; validation comes from the official train.
    DATASET_NAME = "cifar100"
    DATA_ROOT = "/lustre/fsmisc/dataset"  # torchvision looks for <root>/cifar-100-python
    DOWNLOAD_DATASET = False              # Dataset is pre-staged on the server
    OFFICIAL_TRAIN_SPLIT = "train"        # 50,000 images
    OFFICIAL_TEST_SPLIT = "test"          # 10,000 images, never split
    VALIDATION_RATIO = 0.1  # Stratified holdout from official train: 45,000 / 5,000
        
    # ===================== Training Configuration =====================
    BATCH_SIZE = 128               # THAY ĐỔI: 512 → 1200 (H100 có đủ VRAM)
    NUM_EPOCHS = 60
    LEARNING_RATE = 1e-4            # THAY ĐỔI: 1e-4 → 2e-4
                                    # Linear scaling rule: LR tỉ lệ với batch size
                                    # 1e-4 × (1200/512) ≈ 2.34e-4, làm tròn xuống 2e-4
                                    # (conservative hơn vì ViT nhạy cảm với LR lớn)
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 6            # THAY ĐỔI: 10 → 12
                                    # Batch lớn hơn → ít steps/epoch hơn (90k/1200 = 75 steps)
                                    # so với trước (90k/512 = 175 steps), cần thêm epoch warmup
                                    # để đủ số warmup steps bảo vệ backbone
                                    # LƯU Ý (CIFAR-100): train pool chỉ 45,000 ảnh →
                                    # 45k/1024 ≈ 43 steps/epoch, ít hơn Tiny ImageNet.
                                    # Cân nhắc tăng WARMUP_EPOCHS nếu backbone bị lệch sớm.
    WARMUP_START_FACTOR = 0.01      # Bắt đầu từ LR=2e-6, gentle với pretrained backbone
    ETA_MIN = 1e-6
    SCHEDULER = "linear_warmup_cosine"

    # Optimizer
    # Chọn 1 trong SUPPORTED_OPTIMIZERS, hoặc override khi chạy: --optimizer adam
    # Lưu ý: chỉ AdamW dùng decoupled weight decay; adam/sgd/rmsprop dùng L2
    # cổ điển nên cùng WEIGHT_DECAY sẽ KHÔNG cho hiệu quả tương đương.
    SUPPORTED_OPTIMIZERS = ("adamw", "adam", "sgd", "rmsprop")  # khớp với OPTIMIZER_CLASSES trong train.py
    OPTIMIZER = "adam"
    OPTIMIZER_BETAS = (0.9, 0.999)  # adam / adamw
    OPTIMIZER_EPS = 1e-8           # adam/adamw/rmsprop. KHONG dat 0.0: Adam eps=0 gay
                                   # 0/0 = NaN o cac param co second-moment = 0 (thay o
                                   # seed1/seed100 NaN tu epoch 1). run_config.xlsx ghi 0.0
                                   # la LATENT BUG, 1e-8 la default PyTorch.
    SGD_MOMENTUM = 0.9              # sgd / rmsprop
    SGD_NESTEROV = True             # sgd
    RMSPROP_ALPHA = 0.99            # rmsprop
    USE_FUSED_OPTIMIZER = True      # tự bỏ qua nếu optimizer không có fused kernel
    GRAD_CLIP_NORM = 1.0

    # H100 execution settings
    USE_AMP = True
    AMP_DTYPE = "bfloat16"
    FLOAT32_MATMUL_PRECISION = "high"
    USE_TORCH_COMPILE = False
    TORCH_COMPILE_MODE = "reduce-overhead"

    # DataLoader settings for a strong server CPU
    NUM_WORKERS = 16
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True
    PIN_MEMORY = True
    TRAIN_DROP_LAST = True
    PROFILE_BATCHES = 0
    PRINT_DATASET_STATS = False
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 12
    
    # Learning Rate Decay
    LR_DECAY_PATIENCE = 5
    LR_DECAY_FACTOR = 0.5
    
    # ===================== Sampler Configuration =====================
    USE_WEIGHTED_SAMPLER = False
    
    # ===================== Cross-Validation Configuration =====================
    USE_CROSS_VALIDATION = False
    CV_N_SPLITS = 5
    
    # ===================== Loss Function Configuration =====================
    LOSS_FUNCTION = 'cross_entropy'
    LABEL_SMOOTHING = 0.05
    FOCAL_GAMMA = 2.0
    POLY_EPSILON = 1.0
    CLASS_WEIGHT_METHOD = 'inverse_freq'

    # ===================== Model Configuration =====================
    MODELS = [
    'efficientnet_b0']
    PRETRAINED = True
    VIT_PRETRAINED_MODEL_ID = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
    
    CLASSIFIER_CONFIG = [512,256]       # THAY ĐỔI: [512, 256] → [256]
                                    # Head [512, 256] quá lớn cho CIFAR-100 100 classes.
                                    # Head phức tạp → gradient lớn → destabilize backbone.
                                    # [256] đủ capacity mà ít noise hơn khi fine-tune
    DROPOUT_RATE = 0.3
    MODEL_DROP_RATE = 0.0
    MODEL_ATTN_DROP_RATE = 0.0
    MODEL_DROP_PATH_RATE = 0.1
    
    # ===================== Image Configuration =====================
    IMAGE_SIZE = 224
    RESIZE_INTERPOLATION = "bicubic"
    IMAGE_MEAN = (0.5, 0.5, 0.5)
    IMAGE_STD = (0.5, 0.5, 0.5)

    USE_MIXUP_CUTMIX = True
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 1.0
    MIXUP_SWITCH_PROB = 0.5
    MIXUP_MODE = "batch"
    HORIZONTAL_FLIP_PROB = 0.5
    RANDOM_ERASING_PROB = 0.25
    RANDOM_ERASING_SCALE = (0.02, 0.33)
    RANDOM_ERASING_RATIO = (0.3, 3.3)
    RANDOM_ERASING_VALUE = "random"
    
    # ===================== Method Comparison (EMA / SWA baselines) =====================
    # Method weight-averaging cần chạy. Strategy 1 (best checkpoint) LUÔN được
    # eval làm baseline, không cần khai ở đây.
    #   "top-k"  : Strategy 2 của bạn — uniform average Top-K checkpoint tốt nhất.
    #   "last-n" : Strategy 3 — uniform average N epoch cuối.
    #   "ema"    : Exponential Moving Average of weights (Morales-Brotons, TMLR 2024).
    #   "swa"    : Stochastic Weight Averaging (Izmailov, UAI 2018) + constant LR.
    #
    # ⭐ TỰ TÁCH RUN THEO LR SCHEDULE (xem main.run_method_legs_if_needed):
    #   - top-k / last-n / ema chỉ QUAN SÁT weights, KHÔNG đổi LR schedule ⇒ chung
    #     MỘT run, cùng trajectory ⇒ so sánh PAIRED theo seed.
    #   - swa giữ LR HẰNG SỐ ở 25% cuối ⇒ ĐỔI trajectory ⇒ tự động tách thành
    #     EXPERIMENT RIÊNG (run folder riêng, hậu tố "_swa").
    #
    # Vậy `python main.py` sẽ chạy NỐI NHAU 2 experiment:
    #     <run>       methods=[top-k, last-n, ema]  cosine chuẩn
    #     <run>_swa   methods=[swa]                 constant LR từ 75% budget
    # CLI: --methods swa,ema | --methods top-k ema swa | --methods all | --methods swa
    METHODS = ["top-k", "last-n", "ema", "swa"]
    VALID_METHODS = ("top-k", "last-n", "ema", "swa")

    # ---- EMA ----
    # torch.optim.swa_utils.AveragedModel + get_ema_multi_avg_fn:
    #     w_ema ← decay · w_ema + (1 − decay) · w_t      (mỗi optimizer step)
    # KHÔNG warmup/ramp — công thức lũy thừa thuần của thư viện.
    #
    # EMA_DECAYS = None ⇒ TỰ SUY decay từ cửa sổ mục tiêu:
    #     decay = 1 − 1 / (EMA_WINDOW_EPOCHS × steps_per_epoch)
    # Vì sao phải tự suy: cửa sổ của EMA tính theo OPTIMIZER STEP, mà số step/epoch
    # chênh nhau hàng lần giữa ba bộ dữ liệu (CIFAR-100 ~351, Tiny ImageNet ~87).
    # Một hằng số decay dùng chung sẽ cho ba cửa sổ hoàn toàn khác nhau ⇒ không so
    # sánh được. Tự suy theo cửa sổ thì mọi dataset đều có EMA "nhớ" đúng 10 epoch.
    #
    # ⚠ RÀNG BUỘC decay khi không warmup: phần weights KHỞ I TẠO còn sót lại trong
    # EMA cuối = decay^N. Với cách tự suy ở trên, residue rút gọn thành e^(−E/W)
    # — KHÔNG phụ thuộc dataset: E=60, W=10 ⇒ 0.25%. An toàn (ngưỡng 1%).
    # train.py in cửa sổ + residue ra log ngay khi bắt đầu train.
    EMA_WINDOW_EPOCHS = 10
    EMA_DECAYS = None       # None = tự suy; hoặc đặt list, ví dụ [0.999]

    # ---- SWA ----
    # torch.optim.swa_utils.AveragedModel + get_swa_multi_avg_fn:
    #     w_swa ← (w_swa · n + w_t) / (n + 1)      (mỗi epoch, cycle length c = 1)
    # use_buffers=False ⇒ PyTorch average đúng learnable params còn BN running stats
    # được ĐỒNG BỘ từ model nguồn — khớp Algorithm 1, nơi BN được tính lại bằng
    # một lượt forward trên train sau khi average (evaluate.update_bn).
    #
    # ⭐ CHỐT CHO PAPER: "truncate" — 75% đầu chạy ĐÚNG scheduler của bạn
    #    (LinearLR warmup + CosineAnnealingLR), 25% sau giữ LR HẰNG SỐ = SWA_LR.
    #    TỔNG BUDGET GIỮ NGUYÊN ⇒ công bằng tuyệt đối về số epoch với các method khác.
    #    Biến thể "1 budget" của Izmailov et al. §3.2: "we first run standard SGD
    #    training for ≈75% of the training budget ... we just stop the training
    #    early WITHOUT MODIFYING the learning rate schedule".
    #    "inherit" = không đổi LR (SWA average ngay trên cosine) — giữ để tham khảo.
    SWA_LR_SCHEDULE = "truncate"   # "truncate" (chốt) | "inherit"
    SWA_LR_START_FRAC = 0.75       # mốc chuyển sang constant LR + bắt đầu average

    SWA_LR = 1e-5                  # ĐẶT TAY = 1e-5. Ban dau None = tu tinh (LEARNING_RATE + ETA_MIN)/2
                                   # và nhỏ nhất của annealing schedule, đúng khuyến
                                   # nghị Izmailov et al. §4.3 ("intermediate value
                                   # between the largest and the smallest learning
                                   # rate used in the annealing scheme"):
                                   #     SWA_LR = (LEARNING_RATE + ETA_MIN) / 2
                                   # Tự tính theo LEARNING_RATE/ETA_MIN HIỆN HÀNH nên
                                   # --lr trên CLI cũng được tôn trọng.

    # ===================== Evaluation Configuration =====================
    TOP_K_VALUES = [2, 3, 4, 5]
    LAST_N_EPOCHS = 10
    KEEP_LAST_N_CHECKPOINTS = 10
    KEEP_TOP_K_CHECKPOINTS = 5
    
    # ===================== Output Configuration =====================
    if os.path.exists('/kaggle'):
        CHECKPOINTS_DIR = "/kaggle/working/checkpoints"
        RESULTS_DIR = "/kaggle/working/results"
    else:
        CHECKPOINTS_DIR = "checkpoints"
        RESULTS_DIR = "results"
    
    AUTO_DELETE_CHECKPOINTS = True
    SAVE_STRATEGY_CHECKPOINTS = False
    KEEP_RESULTS = True
    
    # Random seed for reproducibility
    SEEDS = [1, 10, 42, 100, 500]
    RANDOM_SEED = SEEDS[0]
        
    # ===================== W&B Configuration =====================
    USE_WANDB = False
    WANDB_API_KEY = "8ad789629890d812ecffc9f0fce138a75f63f992"
    WANDB_PROJECT = "BurmeseGrape-Capstone"
    WANDB_ENTITY = None
    EXPERIMENT_NAME = "baseline_exp1"
    
    # Alias nguoi dung hay go -> ten chuan trong VALID_METHODS
    METHOD_ALIASES = {
        "topk": "top-k", "top_k": "top-k", "top-k": "top-k",
        "strategy2": "top-k", "s2": "top-k", "cwa": "top-k",
        "lastn": "last-n", "last_n": "last-n", "last-n": "last-n",
        "strategy3": "last-n", "s3": "last-n",
        "ema": "ema",
        "swa": "swa",
    }

    @classmethod
    def normalize_methods(cls, methods=None):
        """
        Chuan hoa cls.METHODS: parse alias, tach dau phay, bo trung, giu thu tu.

        Chap nhan: ["top-k", "ema"], "EMA,SWA", ["all"], ["none"].
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
                    f"--methods {token!r} khong hop le. Chon trong "
                    f"{list(cls.VALID_METHODS)} (hoac 'all'/'none'), "
                    "vi du: --methods top-k ema swa | --methods EMA,SWA"
                )
            resolved.append(canonical)

        # Giu thu tu chuan de bang ket qua luon nhat quan giua cac lan chay.
        cls.METHODS = [m for m in cls.VALID_METHODS if m in set(resolved)]
        return cls.METHODS

    @classmethod
    def method_enabled(cls, name):
        return str(name).lower() in cls.METHODS

    @classmethod
    def swa_lr_mode(cls):
        """'inherit' | 'truncate' - chuan hoa, chap nhan alias 'constant'."""
        mode = str(cls.SWA_LR_SCHEDULE).strip().lower()
        if mode == "constant":
            return "truncate"
        if mode not in ("inherit", "truncate"):
            raise ValueError(
                f"SWA_LR_SCHEDULE={cls.SWA_LR_SCHEDULE!r} khong hop le. "
                "Chon 'truncate' hoac 'inherit'."
            )
        return mode

    @classmethod
    def swa_changes_schedule(cls):
        """SWA co doi LR schedule khong => co phai tach run rieng khong."""
        return cls.method_enabled("swa") and cls.swa_lr_mode() != "inherit"

    @classmethod
    def resolved_swa_lr(cls):
        """
        Gia tri constant LR thuc te cua pha SWA.

        SWA_LR = None => trung binh cong cua LR lon nhat va nho nhat trong
        annealing schedule: (LEARNING_RATE + ETA_MIN) / 2 - "intermediate value
        between the largest and the smallest learning rate used in the annealing
        scheme" (Izmailov et al. 4.3).
        """
        if cls.SWA_LR is None:
            return (float(cls.LEARNING_RATE) + float(cls.ETA_MIN)) / 2.0
        return float(cls.SWA_LR)

    @classmethod
    def validate_methods(cls):
        """Kiem tra cau hinh EMA/SWA; goi tu validate_config()."""
        cls.normalize_methods()

        if cls.method_enabled("ema"):
            if cls.EMA_DECAYS is not None:
                if not cls.EMA_DECAYS:
                    raise ValueError("EMA_DECAYS phai la None (tu suy) hoac list khong rong")
                for decay in cls.EMA_DECAYS:
                    if not 0.0 < float(decay) < 1.0:
                        raise ValueError(f"EMA decay phai trong khoang (0, 1), nhan {decay}")
            elif float(cls.EMA_WINDOW_EPOCHS) <= 0:
                raise ValueError("EMA_WINDOW_EPOCHS phai duong")

        if cls.method_enabled("swa"):
            mode = cls.swa_lr_mode()
            if not 0.0 <= float(cls.SWA_LR_START_FRAC) < 1.0:
                raise ValueError("SWA_LR_START_FRAC phai trong [0, 1)")
            if cls.SWA_LR is not None and float(cls.SWA_LR) <= 0.0:
                raise ValueError("SWA_LR phai duong, hoac None de tu tinh (lr + eta_min)/2")
            if mode != "inherit":
                start = int(float(cls.SWA_LR_START_FRAC) * int(cls.NUM_EPOCHS))
                print(
                    "⚠ SWA_LR_SCHEDULE='%s': LR giu hang %g tu epoch %d/%d (%.0f%% budget). "
                    "Day la TRAJECTORY KHAC voi run chuan => phai bao cao nhu mot experiment rieng."
                    % (mode, cls.resolved_swa_lr(), start + 1, cls.NUM_EPOCHS,
                       float(cls.SWA_LR_START_FRAC) * 100)
                )
                # BAT BUOC tat early stopping khi SWA doi schedule: SWA bat dau
                # average tu epoch floor(start_frac*E)+1 (75% => epoch 46/60).
                # Neu early stopping cat truoc moc do, SWA gom 0 snapshot => KHONG
                # co dong SWA nao trong ket qua (da thay o seed42 dung epoch 40,
                # seed10 dung epoch 33). Dat patience = NUM_EPOCHS-1 => khong bao
                # gio trigger, dam bao chay du budget de SWA co du epoch.
                start = int(float(cls.SWA_LR_START_FRAC) * int(cls.NUM_EPOCHS)) + 1
                if int(cls.EARLY_STOPPING_PATIENCE) != 0:
                    print(
                        "⚠ SWA can chay du %d epoch (average tu epoch %d) nhung "
                        "EARLY_STOPPING_PATIENCE=%d co the cat truoc do => TU DONG dat "
                        "EARLY_STOPPING_PATIENCE = 0 (TAT early stopping) cho leg SWA."
                        % (int(cls.NUM_EPOCHS), start, int(cls.EARLY_STOPPING_PATIENCE))
                    )
                    cls.EARLY_STOPPING_PATIENCE = 0

                others = [m for m in ("top-k", "last-n", "ema") if cls.method_enabled(m)]
                if others:
                    print(
                        "ℹ %s khong doi LR schedule con 'swa' thi co => se chay 2 EXPERIMENT "
                        "RIENG BIET noi nhau: leg 1 = %s (scheduler chuan), leg 2 = ['swa'] "
                        "(constant LR). Xem main.run_method_legs_if_needed()." % (others, others)
                    )

    @classmethod
    def print_methods(cls):
        print("  Methods: %s" % (", ".join(cls.METHODS) if cls.METHODS else "(none)"))
        if cls.method_enabled("top-k"):
            print("    top-k  : K in %s (Strategy 2)" % list(cls.TOP_K_VALUES))
        if cls.method_enabled("last-n"):
            print("    last-n : N = %s (Strategy 3)" % cls.LAST_N_EPOCHS)
        if cls.method_enabled("ema"):
            detail = ("dat tay %s" % list(cls.EMA_DECAYS)) if cls.EMA_DECAYS else (
                "tu suy cho cua so %s epoch" % cls.EMA_WINDOW_EPOCHS)
            print("    ema    : decay %s, khong warmup" % detail)
        if cls.method_enabled("swa"):
            mode = cls.swa_lr_mode()
            tail = ("LR HANG SO %g - TRAJECTORY RIENG" % cls.resolved_swa_lr()
                    if mode != "inherit" else "LR theo scheduler chung (inherit)")
            print("    swa    : average tu %.0f%% budget, %s"
                  % (float(cls.SWA_LR_START_FRAC) * 100, tail))

    @classmethod
    def get_num_classes(cls):
        """Return the known number of CIFAR-100 (fine label) classes."""
        return 100
    
    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        if not cls.DATASET_NAME:
            raise ValueError("DATASET_NAME must not be empty")

        if not cls.DATA_ROOT:
            raise ValueError("DATA_ROOT must not be empty")

        if not cls.DOWNLOAD_DATASET and not os.path.isdir(
            os.path.join(cls.DATA_ROOT, "cifar-100-python")
        ):
            raise ValueError(
                f"CIFAR-100 not found at {os.path.join(cls.DATA_ROOT, 'cifar-100-python')}. "
                "Set DATA_ROOT to the folder that contains 'cifar-100-python'."
            )

        if not 0.0 < cls.VALIDATION_RATIO < 1.0:
            raise ValueError("VALIDATION_RATIO must be strictly between 0 and 1")

        for name in ("MIXUP_PROB", "MIXUP_SWITCH_PROB", "RANDOM_ERASING_PROB"):
            value = getattr(cls, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if cls.USE_MIXUP_CUTMIX and cls.LOSS_FUNCTION != "cross_entropy":
            raise ValueError(
                "Mixup/CutMix currently requires LOSS_FUNCTION='cross_entropy' "
                "because PolyFocalLoss only accepts hard labels"
            )
        
        if cls.EARLY_STOPPING_PATIENCE >= cls.NUM_EPOCHS:
            raise ValueError("Early stopping patience should be less than num_epochs")

        if cls.WARMUP_EPOCHS >= cls.NUM_EPOCHS:
            raise ValueError("Warmup epochs should be less than num_epochs")

        if cls.ETA_MIN >= cls.LEARNING_RATE:
            raise ValueError("ETA_MIN must be smaller than LEARNING_RATE")

        if cls.BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be positive")

        if cls.NUM_WORKERS < 0:
            raise ValueError("NUM_WORKERS must be non-negative")

        if cls.PREFETCH_FACTOR <= 0:
            raise ValueError("PREFETCH_FACTOR must be positive")

        if cls.AMP_DTYPE != "bfloat16":
            raise ValueError("This H100 pipeline currently supports AMP_DTYPE='bfloat16'")

        if cls.OPTIMIZER.lower() not in cls.SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported OPTIMIZER '{cls.OPTIMIZER}'. "
                f"Choose one of: {', '.join(cls.SUPPORTED_OPTIMIZERS)}"
            )

        if cls.OPTIMIZER.lower() == "sgd" and cls.SGD_NESTEROV and cls.SGD_MOMENTUM <= 0:
            raise ValueError("SGD_NESTEROV=True requires SGD_MOMENTUM > 0")
        
        cls.validate_methods()

        print("[OK] Config validated successfully")
        print(f"  Dataset: {cls.DATASET_NAME} (root={cls.DATA_ROOT})")
        print(f"  Number of classes: {cls.get_num_classes()}")
        print(f"  Optimizer: {cls.OPTIMIZER}")
        print(f"  Models to train: {len(cls.MODELS)}")
        cls.print_methods()
