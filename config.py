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
    BATCH_SIZE = 512
    NUM_EPOCHS = 100
    LR_REFERENCE_BATCH_SIZE = 512
    BASE_LEARNING_RATE = 5e-5
    AUTO_SCALE_LEARNING_RATE = True
    LEARNING_RATE = BASE_LEARNING_RATE * (BATCH_SIZE / LR_REFERENCE_BATCH_SIZE)
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 5
    WARMUP_START_FACTOR = 0.1
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
    USE_TORCH_COMPILE = True
    TORCH_COMPILE_MODE = "max-autotune"

    # DataLoader settings for a strong server CPU
    NUM_WORKERS = 16
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True
    PIN_MEMORY = True
    TRAIN_DROP_LAST = True
    PROFILE_BATCHES = 0  # Set >0 to print DataLoader vs GPU compute timing for first N train batches
    PRINT_DATASET_STATS = False  # Opens up to 1000 images before training; keep off for server runs
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 20
    
    # Learning Rate Decay
    LR_DECAY_PATIENCE = 5  # Reduce LR if val_loss doesn't improve for 5 epochs
    LR_DECAY_FACTOR = 0.5  # Multiply LR by this factor when decaying
    
    # ===================== Sampler Configuration =====================
    USE_WEIGHTED_SAMPLER = False  # Bật/tắt WeightedRandomSampler (xử lý class imbalance ở data level)
    
    # ===================== Cross-Validation Configuration =====================
    USE_CROSS_VALIDATION = False  # Bật/tắt Cross-Validation (dùng sklearn StratifiedKFold)
    CV_N_SPLITS = 5              # Số fold cho Cross-Validation
    
    # ===================== Loss Function Configuration =====================
    # Loss function: 'cross_entropy' or 'poly_focal'
    LOSS_FUNCTION = 'cross_entropy'  # Thay đổi thành 'poly_focal' để sử dụng PolyFocalLoss
    LABEL_SMOOTHING = 0.1  # DeiT-style label smoothing
    # PolyFocalLoss parameters (only used when LOSS_FUNCTION = 'poly_focal')
    FOCAL_GAMMA = 2.0       # Focusing parameter: higher = more focus on hard examples
    POLY_EPSILON = 1.0      # Poly coefficient: boosts gradient for ambiguous samples
    CLASS_WEIGHT_METHOD = 'inverse_freq'  # 'inverse_freq' or 'effective_num'

    # ===================== Model Configuration =====================
    MODELS = [
        # 'vgg16',  
        # 'resnet18',
        # 'resnet101',
        # 'mobilenet_v2'
        # 'densenet121'
        # 'efficientnet_b0',
        'vit_base_patch16_224'
    ]
    PRETRAINED = True
    VIT_PRETRAINED_MODEL_ID = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
    
    # Custom classifier configuration
    # Định nghĩa các lớp fully connected tùy chỉnh
    # Format: [hidden_dim1, hidden_dim2, ..., num_classes]
    # Đơn giản hóa cho dataset nhỏ (~10k ảnh) để tránh overfitting
    CLASSIFIER_CONFIG = [512, 256]  # User-selected custom MLP head
    DROPOUT_RATE = 0.4
    MODEL_DROP_RATE = 0.0
    MODEL_ATTN_DROP_RATE = 0.0
    MODEL_DROP_PATH_RATE = 0.1
    
    # ===================== Image Configuration =====================
    IMAGE_SIZE = 224
    RESIZE_INTERPOLATION = "bicubic"
    IMAGE_MEAN = (0.5, 0.5, 0.5)
    IMAGE_STD = (0.5, 0.5, 0.5)

    # DeiT-style augmentation. MIXUP_ALPHA and CUTMIX_ALPHA are beta
    # distribution parameters; MIXUP_PROB is the probability of applying
    # batch mixing, and MIXUP_SWITCH_PROB chooses CutMix instead of Mixup.
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
    # Strategy 2: Top-K checkpoints to average
    TOP_K_VALUES = [2, 3, 4, 5]
    
    # Strategy 3: Number of last epochs to average
    LAST_N_EPOCHS = 10
    
    # Checkpoint management - Memory optimization
    KEEP_LAST_N_CHECKPOINTS = 10  # Keep last N epoch checkpoints
    KEEP_TOP_K_CHECKPOINTS = 5    # Keep top K best val_loss checkpoints
    
    # ===================== Output Configuration =====================
    # Kaggle output được lưu tại /kaggle/working
    if os.path.exists('/kaggle'):
        CHECKPOINTS_DIR = "/kaggle/working/checkpoints"
        RESULTS_DIR = "/kaggle/working/results"
    else:
        CHECKPOINTS_DIR = "checkpoints"
        RESULTS_DIR = "results"
    
    # Checkpoint management - TỰ ĐỘNG XÓA SAU KHI EVALUATE
    AUTO_DELETE_CHECKPOINTS = True  # Set True để xóa checkpoints sau khi evaluate, False để giữ lại
    SAVE_STRATEGY_CHECKPOINTS = False  # Set False để không lưu checkpoint của Strategy 1/2/3 sau evaluate
    KEEP_RESULTS = True             # Luôn giữ results (Excel, charts)
    
    # Random seed for reproducibility
    SEEDS = [1, 10, 100, 500]
    RANDOM_SEED = SEEDS[0]
        
    # ===================== W&B Configuration =====================
    # W&B tracking
    USE_WANDB = False  # Set to False to disable wandb
    WANDB_API_KEY = "8ad789629890d812ecffc9f0fce138a75f63f992"  # Your wandb API key
    WANDB_PROJECT = "BurmeseGrape-Capstone"  # Tên project trên wandb
    WANDB_ENTITY = None  # Tên team/user wandb (None = default user)
    # EXPERIMENT_NAME sẽ được set động khi chạy (ví dụ: "experiment_1", "experiment_2")
    EXPERIMENT_NAME = "baseline_exp1"  # ⚠️ THAY ĐỔI CHO MỖI EXPERIMENT
    
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
