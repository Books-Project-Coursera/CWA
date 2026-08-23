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
    OPTIMIZER_EPS = 1e-8            # adam / adamw / rmsprop
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
    'vgg16']
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
    
    # ===================== Evaluation Configuration =====================
    TOP_K_VALUES = [2, 3, 4, 5]
    # Patience dùng RIÊNG để giới hạn candidate pool của Strategy 2 (post-hoc).
    # Pool chỉ nhận epoch phá kỷ lục val_loss; sau khi có STRATEGY2_POOL_PATIENCE
    # epoch liên tiếp không cải thiện thì pool bị khóa, dù training vẫn chạy tiếp.
    # Tách biệt hoàn toàn với EARLY_STOPPING_PATIENCE (điều khiển dừng training thật).
    STRATEGY2_POOL_PATIENCE = 10
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

        if cls.STRATEGY2_POOL_PATIENCE <= 0:
            raise ValueError("STRATEGY2_POOL_PATIENCE must be positive")

        if not cls.TOP_K_VALUES:
            raise ValueError("TOP_K_VALUES must not be empty")

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
        
        print("[OK] Config validated successfully")
        print(f"  Dataset: {cls.DATASET_NAME} (root={cls.DATA_ROOT})")
        print(f"  Number of classes: {cls.get_num_classes()}")
        print(f"  Optimizer: {cls.OPTIMIZER}")
        print(f"  Models to train: {len(cls.MODELS)}")
