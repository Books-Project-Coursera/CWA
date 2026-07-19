"""
Train Ultralytics YOLO cho object detection (Pascal VOC).

Toàn bộ training loop giao cho Ultralytics (model.train): LR schedule,
augmentation, early stopping, best.pt/last.pt, results.csv đều do Ultralytics
quản lý trong run dir. Phần thêm vào cho Strategy 2:

- save_period=1 để Ultralytics lưu checkpoint mỗi epoch, NHƯNG
- TopKCheckpointManager (callback on_model_save) prune NGAY checkpoint ngoài
  Top-K theo fitness trên val' → disk chỉ giữ đúng K checkpoint cần thiết
  (+ best.pt/last.pt), không lưu tất cả epoch.
"""
import json
from pathlib import Path

from config import Config
from dataset import prepare_dataset
from evaluate import RANKING_FILE, print_detection_metrics, run_strategy_evaluation


class TopKCheckpointManager:
    """
    Callback 'on_model_save' của Ultralytics: sau mỗi lần trainer lưu
    checkpoint epoch (save_period=1), ghi nhận fitness trên val' của epoch đó
    và xóa ngay checkpoint tệ nhất nếu vượt quá KEEP_TOP_K_CHECKPOINTS.

    Ranking được ghi ra <run_dir>/weights/strategy2_checkpoints.json để
    evaluate.rank_checkpoints() dùng lại khi average Top-K (Strategy 2).
    Không đụng tới best.pt / last.pt của Ultralytics.
    """

    def __init__(self, keep_top_k):
        self.keep_top_k = int(keep_top_k)
        self.records = {}  # filename -> {"epoch": int, "fitness": float}

    def on_model_save(self, trainer):
        weights_dir = Path(trainer.save_dir) / "weights"
        # trainer.fitness = fitness epoch hiện tại trên val' (set trong validate())
        fitness = float(trainer.fitness) if trainer.fitness is not None else float("-inf")

        # Checkpoint epoch mới xuất hiện (epoch*.pt chưa ghi nhận) thuộc epoch này
        for ckpt in weights_dir.glob("epoch*.pt"):
            if ckpt.name not in self.records:
                self.records[ckpt.name] = {"epoch": int(trainer.epoch), "fitness": fitness}

        # Prune: chỉ giữ Top-K theo fitness (tie-break: giữ epoch mới hơn)
        while len(self.records) > self.keep_top_k:
            worst = min(
                self.records,
                key=lambda name: (self.records[name]["fitness"], self.records[name]["epoch"]),
            )
            worst_path = weights_dir / worst
            if worst_path.exists():
                worst_path.unlink()
            del self.records[worst]

        (weights_dir / RANKING_FILE).write_text(json.dumps(self.records, indent=2))


def build_train_args(data_yaml):
    """
    Map Config → kwargs của model.train().
    Group hyperparam theo tuning guide của Ultralytics
    (https://docs.ultralytics.com/guides/hyperparameter-tuning).
    EXTRA_TRAIN_ARGS được update SAU CÙNG nên có thể ghi đè mọi key chuẩn.
    """
    train_args = {
        # ---- Training core ----
        "data": str(data_yaml),
        "epochs": int(Config.EPOCHS),
        "imgsz": int(Config.IMGSZ),
        "batch": int(Config.BATCH),
        "workers": int(Config.WORKERS),
        "seed": int(Config.RANDOM_SEED),
        "patience": int(Config.PATIENCE),
        "pretrained": bool(Config.PRETRAINED),
        "cache": Config.CACHE,
        "resume": bool(Config.RESUME),
        "deterministic": bool(Config.DETERMINISTIC),
        "project": Config.PROJECT,
        "exist_ok": bool(Config.EXIST_OK),

        # ---- Optimizer & LR schedule ----
        "optimizer": Config.OPTIMIZER,
        "lr0": float(Config.LR0),
        "lrf": float(Config.LRF),
        "momentum": float(Config.MOMENTUM),
        "weight_decay": float(Config.WEIGHT_DECAY),
        "warmup_epochs": float(Config.WARMUP_EPOCHS),
        "warmup_momentum": float(Config.WARMUP_MOMENTUM),
        "warmup_bias_lr": float(Config.WARMUP_BIAS_LR),
        "cos_lr": bool(Config.COS_LR),

        # ---- Loss gains + regularization ----
        "box": float(Config.BOX_GAIN),
        "cls": float(Config.CLS_GAIN),
        "dfl": float(Config.DFL_GAIN),
        "dropout": float(Config.DROPOUT),
        "nbs": int(Config.NBS),
        "close_mosaic": int(Config.CLOSE_MOSAIC),

        # ---- Augmentation ----
        "hsv_h": float(Config.HSV_H),
        "hsv_s": float(Config.HSV_S),
        "hsv_v": float(Config.HSV_V),
        "degrees": float(Config.DEGREES),
        "translate": float(Config.TRANSLATE),
        "scale": float(Config.SCALE),
        "shear": float(Config.SHEAR),
        "perspective": float(Config.PERSPECTIVE),
        "flipud": float(Config.FLIPUD),
        "fliplr": float(Config.FLIPLR),
        "bgr": float(Config.BGR),
        "mosaic": float(Config.MOSAIC),
        "mixup": float(Config.MIXUP),
        "cutmix": float(Config.CUTMIX),
        "copy_paste": float(Config.COPY_PASTE),
        "auto_augment": Config.AUTO_AUGMENT,
        "erasing": float(Config.ERASING),

        # ---- Precision & runtime ----
        "amp": bool(Config.AMP),
        "multi_scale": float(Config.MULTI_SCALE),
        "rect": bool(Config.RECT),
        "single_cls": bool(Config.SINGLE_CLS),
    }

    # label_smoothing: Ultralytics 8.4+ có thể đã bỏ key này — chỉ truyền nếu
    # còn hỗ trợ để không vỡ khi bản mới hơn strip đi
    if float(Config.LABEL_SMOOTHING) > 0:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "label_smoothing" in DEFAULT_CFG_DICT:
            train_args["label_smoothing"] = float(Config.LABEL_SMOOTHING)

    if Config.FREEZE is not None:
        train_args["freeze"] = Config.FREEZE

    if Config.USE_STRATEGY2:
        # Lưu ckpt mỗi epoch để có nguồn chọn Top-K; TopKCheckpointManager
        # prune ngay nên disk không phình theo số epoch
        train_args["save_period"] = 1

    # DEVICE=None → để Ultralytics tự chọn, không truyền key
    if Config.DEVICE is not None:
        train_args["device"] = Config.DEVICE
    if Config.NAME:
        train_args["name"] = Config.NAME
    train_args.update(Config.EXTRA_TRAIN_ARGS or {})
    return train_args


def train_detector():
    """
    Pipeline train hoàn chỉnh:
    1. Chuẩn bị data (tách val' độc lập từ train nếu VAL_RATIO > 0)
    2. model.train() với TopKCheckpointManager (nếu USE_STRATEGY2)
    3. In metrics val cuối trên best.pt
    4. Strategy evaluation (Strategy 1 vs Strategy 2 Top-K) trên EVAL_SPLIT
       + export Excel 2 sheet
    5. (Optional) export model cho Edge AI

    Returns:
        dict đường dẫn các artifact chính của run.
    """
    # Import trễ để validate config / --help không cần ultralytics
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print(" STRATEGY 2 - OBJECT DETECTION TRAINING (Ultralytics YOLO)")
    print("=" * 70)
    print(f"  Model: {Config.MODEL}")
    print(f"  Data : {Config.DATA} (VOC.yaml built-in sẽ tự download lần đầu)")

    # Step 1: data với val' độc lập (điều kiện tiên quyết của Strategy 2)
    data_yaml = prepare_dataset()

    # Step 2: train
    model = YOLO(Config.MODEL)
    if Config.USE_STRATEGY2:
        manager = TopKCheckpointManager(Config.KEEP_TOP_K_CHECKPOINTS)
        model.add_callback("on_model_save", manager.on_model_save)
        print(f"  Strategy 2: giữ Top-{Config.KEEP_TOP_K_CHECKPOINTS} checkpoint theo fitness val'")

    # model.train() trả về DetMetrics của lượt val CUỐI trên best.pt (split val')
    metrics = model.train(**build_train_args(data_yaml))

    run_dir = Path(model.trainer.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    print(f"\n  ✓ Training completed. Run dir: {run_dir}")
    print(f"  ✓ Best weights: {best_weights}")

    # Step 3: metrics trên val' (holdout) — KHÔNG phải số liệu báo cáo cuối
    if metrics is not None:
        print_detection_metrics(metrics, header="FINAL VALIDATION on val' (best.pt)")

    # Step 4: báo cáo cuối trên EVAL_SPLIT (mặc định test = VOC2007 test)
    # gồm Strategy 1 (best.pt) và Strategy 2 (Top-K average) + Excel
    strategy_results = run_strategy_evaluation(run_dir, data=data_yaml)

    # Step 5: Edge AI hook (tắt mặc định)
    exported_model = export_model(best_weights) if Config.EXPORT_ENABLED else None

    return {
        "run_dir": str(run_dir),
        "best_weights": str(best_weights),
        "strategies": list(strategy_results),
        "exported_model": exported_model,
    }


def export_model(weights):
    """
    Xuất weights đã train sang format deploy (ONNX/TensorRT/...) bằng
    model.export() của Ultralytics — phục vụ hướng Edge AI.
    """
    from ultralytics import YOLO

    export_args = {
        "format": Config.EXPORT_FORMAT,
        "half": bool(Config.EXPORT_HALF),
        "dynamic": bool(Config.EXPORT_DYNAMIC),
        "simplify": bool(Config.EXPORT_SIMPLIFY),
        "imgsz": int(Config.EXPORT_IMGSZ or Config.IMGSZ),
    }
    if Config.EXPORT_DEVICE is not None:
        export_args["device"] = Config.EXPORT_DEVICE

    print(f"\n  [Edge AI] Exporting {weights} → format='{export_args['format']}' ...")
    model = YOLO(str(weights))
    exported_path = model.export(**export_args)
    print(f"  ✓ Exported model: {exported_path}")
    return exported_path
