"""
Configuration file for baseline research
"""
import os

class Config:

    # Hugging Face Tiny ImageNet. The official `valid` split is kept as test.
    DATASET_NAME = "zh-plus/tiny-imagenet"
    HF_TRAIN_SPLIT = "train"
    HF_TEST_SPLIT = "valid"
    VALIDATION_RATIO = 0.1  # Stratified holdout from the official train split
        
    # ===================== Training Configuration =====================
    BATCH_SIZE = 1200               # THAY ĐỔI: 512 → 1200 (H100 có đủ VRAM)
    NUM_EPOCHS = 60
    LEARNING_RATE = 2e-4            # THAY ĐỔI: 1e-4 → 2e-4
                                    # Linear scaling rule: LR tỉ lệ với batch size
                                    # 1e-4 × (1200/512) ≈ 2.34e-4, làm tròn xuống 2e-4
                                    # (conservative hơn vì ViT nhạy cảm với LR lớn)
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 12              # THAY ĐỔI: 10 → 12
                                    # Batch lớn hơn → ít steps/epoch hơn (90k/1200 = 75 steps)
                                    # so với trước (90k/512 = 175 steps), cần thêm epoch warmup
                                    # để đủ số warmup steps bảo vệ backbone
    WARMUP_START_FACTOR = 0.01      # Bắt đầu từ LR=2e-6, gentle với pretrained backbone
    ETA_MIN = 1e-6
    SCHEDULER = "linear_warmup_cosine"

    # Optimizer
    OPTIMIZER = "adamw"
    OPTIMIZER_BETAS = (0.9, 0.999)
    OPTIMIZER_EPS = 1e-8
    USE_FUSED_OPTIMIZER = True
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
    EARLY_STOPPING_PATIENCE = 20
    
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
    LABEL_SMOOTHING = 0.1
    FOCAL_GAMMA = 2.0
    POLY_EPSILON = 1.0
    CLASS_WEIGHT_METHOD = 'inverse_freq'

    # ===================== Model Configuration =====================
    MODELS = [
        'vit_base_patch16_224'
    ]
    PRETRAINED = True
    VIT_PRETRAINED_MODEL_ID = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
    
    CLASSIFIER_CONFIG = [256]       # THAY ĐỔI: [512, 256] → [256]
                                    # Head [512, 256] quá lớn cho TinyImageNet 200 classes.
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
    SEEDS = [1, 10, 100, 500]
    RANDOM_SEED = SEEDS[0]
        
    # ===================== W&B Configuration =====================
    USE_WANDB = False
    WANDB_API_KEY = "8ad789629890d812ecffc9f0fce138a75f63f992"
    WANDB_PROJECT = "BurmeseGrape-Capstone"
    WANDB_ENTITY = None
    EXPERIMENT_NAME = "baseline_exp1"
    
    @classmethod
    def get_num_classes(cls):
        """Return the known number of Tiny ImageNet classes."""
        return 200
    
    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        if not cls.DATASET_NAME:
            raise ValueError("DATASET_NAME must not be empty")

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

        if cls.OPTIMIZER.lower() != "adamw":
            raise ValueError("This training pipeline currently supports OPTIMIZER='adamw'")
        
        print("[OK] Config validated successfully")
        print(f"  Dataset: {cls.DATASET_NAME}")
        print(f"  Number of classes: {cls.get_num_classes()}")
        print(f"  Models to train: {len(cls.MODELS)}")
