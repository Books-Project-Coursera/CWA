"""
Train Ultralytics YOLO cho instance segmentation (Carparts).

Toàn bộ training loop giao cho Ultralytics (model.train): LR schedule,
augmentation, early stopping, best.pt/last.pt, results.csv đều do Ultralytics
quản lý trong run dir. Phần thêm vào cho Strategy 2:

- save_period=1 để Ultralytics lưu checkpoint mỗi epoch, NHƯNG
- TopKCheckpointManager (callback on_model_save) prune NGAY checkpoint ngoài
  Top-K theo VAL_LOSS trên val' → disk chỉ giữ đúng K checkpoint cần thiết
  (+ best.pt/last.pt), không lưu tất cả epoch.
  
NOTE: Dùng val_loss (thay vì fitness) để ranking checkpoint, giống như early
stopping của Ultralytics cũng dựa trên val_loss.
"""
import json
from datetime import datetime
from pathlib import Path

from config import Config
from dataset import prepare_dataset
from evaluate import RANKING_FILE, export_multi_seed_summary, print_detection_metrics, run_strategy_evaluation
from losses import install_cls_loss


class TopKCheckpointManager:
    """
    Callback 'on_model_save' của Ultralytics: sau mỗi lần trainer lưu
    checkpoint epoch (save_period=1), ghi nhận val_loss trên val' của epoch đó
    và xóa ngay checkpoint tệ nhất (val_loss cao) nếu vượt quá KEEP_TOP_K_CHECKPOINTS.

    Ranking được ghi ra <run_dir>/weights/strategy2_checkpoints.json để
    evaluate.rank_checkpoints() dùng lại khi average Top-K (Strategy 2).
    Không đụng tới best.pt / last.pt của Ultralytics.
    
    NOTE: Dùng val_loss (nhỏ = tốt) thay vì fitness (cao = tốt), giống early
    stopping của Ultralytics.
    """

    def __init__(self, keep_top_k):
        self.keep_top_k = int(keep_top_k)
        self.records = {}  # filename -> {"epoch": int, "val_loss": float}

    def on_model_save(self, trainer):
        weights_dir = Path(trainer.save_dir) / "weights"
        
        # Lấy val_loss từ results.csv (được Ultralytics update mỗi epoch)
        results_csv = Path(trainer.save_dir) / "results.csv"
        val_loss = float("inf")  # Default = infinity nếu không tìm thấy
        
        if results_csv.exists():
            try:
                import csv
                with open(results_csv, 'r') as f:
                    reader = list(csv.DictReader(f))
                    if reader:
                        last_row = reader[-1]  # Dòng cuối = epoch hiện tại
                        # Ultralytics ghi cột "val/loss" hoặc tương tự
                        if "val/loss" in last_row:
                            val_loss = float(last_row["val/loss"])
                        elif "val_loss" in last_row:
                            val_loss = float(last_row["val_loss"])
            except Exception as e:
                print(f"    ⚠ Lỗi khi đọc val_loss từ results.csv: {e}")

        # Checkpoint epoch mới xuất hiện (epoch*.pt chưa ghi nhận) thuộc epoch này
        for ckpt in weights_dir.glob("epoch*.pt"):
            if ckpt.name not in self.records:
                self.records[ckpt.name] = {"epoch": int(trainer.epoch), "val_loss": val_loss}

        # Prune: chỉ giữ Top-K theo val_loss (nhỏ nhất = tốt nhất; tie-break: giữ epoch mới hơn)
        while len(self.records) > self.keep_top_k:
            worst = max(
                self.records,
                key=lambda name: (self.records[name]["val_loss"], -self.records[name]["epoch"]),
            )
            worst_path = weights_dir / worst
            if worst_path.exists():
                worst_path.unlink()
            del self.records[worst]

        (weights_dir / RANKING_FILE).write_text(json.dumps(self.records, indent=2))


def build_train_args(data_yaml):
    """
    Map Config → kwargs của model.train().
    Chỉ truyền các tham số override, còn lại để mặc định của Ultralytics.
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

        # ---- Optimizer & LR schedule (Overridden) ----
        "optimizer": Config.OPTIMIZER,
        "lr0": float(Config.LR0),
        "lrf": float(Config.LRF),
        "warmup_epochs": float(Config.WARMUP_EPOCHS),
        "cos_lr": bool(Config.COS_LR),

        # ---- Augmentation (Overridden) ----
        "mixup": float(Config.MIXUP),
        "copy_paste": float(Config.COPY_PASTE),
    }

    if Config.USE_STRATEGY2:
        # Lưu ckpt mỗi epoch để có nguồn chọn Top-K; TopKCheckpointManager
        # prune ngay nên disk không phình theo số epoch
        train_args["save_period"] = 1
    else:
        # Không dùng Strategy 2 → không cần lưu checkpoint
        train_args["save"] = False

    if Config.DEVICE is not None:
        train_args["device"] = Config.DEVICE
    if Config.NAME:
        train_args["name"] = Config.NAME
    train_args.update(Config.EXTRA_TRAIN_ARGS or {})
    return train_args


def print_multi_seed_summary(all_runs_results):
    """In bảng tổng hợp kết quả của nhiều seeds chạy thử nghiệm độc lập."""
    from evaluate import extract_overall_metrics

    print("\n" + "=" * 80)
    print(" MULTI-SEED RUNS SUMMARY")
    print("=" * 80)
    
    # Gom metrics của các seed để hiển thị
    strategies = list(all_runs_results[0]["strategy_results"].keys())
    
    header = f"  {'Seed':<6} |"
    for strat in strategies:
        header += f" {strat:<18} (mAP50 / mAP50-95) |"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for run in all_runs_results:
        seed = run["seed"]
        row_str = f"  {seed:<6} |"
        for strat in strategies:
            metrics = run["strategy_results"].get(strat)
            if metrics is not None:
                m = extract_overall_metrics(metrics)
                row_str += f" {m['mAP@0.5']:>6.4f} / {m['mAP@0.5:0.95']:>10.4f}      |"
            else:
                row_str += f" {'N/A':<18} |"
        print(row_str)

    print("=" * 80)


def train_detector():
    """
    Pipeline train hoàn chỉnh cho instance segmentation:
    1. Tạo thư mục experiment group duy nhất (timestamp + tên model) cho lần chạy này
    2. Chuẩn bị data (tách val' nếu cần theo VAL_RATIO, hỗ trợ seed dạng list/int)
    3. Loop qua tất cả các seed → mỗi seed = 1 subfolder riêng bên trong experiment group
    4. Huấn luyện model.train() với TopKCheckpointManager (nếu USE_STRATEGY2)
    5. Báo cáo đánh giá Strategy 1 (best.pt) và Strategy 2 (Top-K average) trên split test
    6. Xuất Excel và Edge AI export nếu được bật
    7. Lưu snapshot config (experiment_config.json) + bảng tổng kết đa seed vào experiment group
    """
    # Import trễ để validate config / --help không cần ultralytics
    from ultralytics import YOLO

    # ── Lưu cấu hình gốc ──────────────────────────────────────────────────────
    orig_seeds   = Config.RANDOM_SEED
    orig_project = Config.PROJECT          # thư mục root gốc (results/detection)
    orig_name    = Config.NAME             # None hoặc custom name người dùng đặt

    # Chuẩn hóa seeds thành list
    if isinstance(orig_seeds, (list, tuple)):
        seeds = [int(s) for s in orig_seeds]
    else:
        seeds = [int(orig_seeds)]

    # ── Tạo thư mục experiment group (DUY NHẤT cho lần bấm chạy này) ──────────
    # Định dạng: exp_YYYYMMDD_HHMMSS_<model_stem>
    # Ví dụ:     exp_20260720_214200_yolov8s
    # → Mọi seed của thí nghiệm này được gom vào đây, không bao giờ trùng với
    #   lần chạy khác dù dùng cùng model / cùng seed.
    model_stem = Path(str(Config.MODEL)).stem          # "yolov8s" từ "yolov8s.pt"
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name  = orig_name if orig_name else "exp"     # tôn trọng --name nếu người dùng đặt
    exp_group  = f"{base_name}_{timestamp}_{model_stem}"
    exp_dir    = Path(orig_project) / exp_group
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Redirect PROJECT → bên trong experiment group;
    # mỗi seed sẽ tạo subfolder seed_<N> trong đây
    Config.PROJECT = str(exp_dir)

    print("\n" + "=" * 70)
    print(" STRATEGY 2 - INSTANCE SEGMENTATION TRAINING (Ultralytics YOLO-seg)")
    print("=" * 70)
    print(f"  Model      : {Config.MODEL}")
    print(f"  Data       : {Config.DATA}")
    print(f"  Seeds      : {seeds}")
    print(f"  Experiment : {exp_dir}")
    print("=" * 70)

    # ── Lưu snapshot config tại thời điểm chạy ────────────────────────────────
    config_snapshot = {
        "experiment_group" : exp_group,
        "timestamp"        : timestamp,
        "model"            : Config.MODEL,
        "data"             : Config.DATA,
        "val_ratio"        : Config.VAL_RATIO,
        "epochs"           : Config.EPOCHS,
        "imgsz"            : Config.IMGSZ,
        "batch"            : Config.BATCH,
        "seeds"            : seeds,
        "optimizer"        : Config.OPTIMIZER,
        "lr0"              : Config.LR0,
        "lrf"              : Config.LRF,
        "warmup_epochs"    : Config.WARMUP_EPOCHS,
        "cos_lr"           : Config.COS_LR,
        "loss_function"    : Config.LOSS_FUNCTION,
        "focal_gamma"      : Config.FOCAL_GAMMA,
        "focal_alpha"      : Config.FOCAL_ALPHA,
        "use_strategy2"    : Config.USE_STRATEGY2,
        "top_k_values"     : Config.TOP_K_VALUES,
        "eval_split"       : Config.EVAL_SPLIT,
    }
    config_path = exp_dir / "experiment_config.json"
    config_path.write_text(json.dumps(config_snapshot, indent=2, ensure_ascii=False))
    print(f"  ✓ Config snapshot saved → {config_path.name}")

    all_runs_results = []

    for idx, seed in enumerate(seeds):
        print(f"\n>>>> [Seed {idx+1}/{len(seeds)}] Bắt đầu train với RANDOM_SEED = {seed} <<<<")

        # Ghi đè seed, đặt tên subfolder = seed_<N>
        Config.RANDOM_SEED = seed
        Config.NAME        = f"seed_{seed}"

        # Step 1: data với val' độc lập (tách tương ứng theo seed hiện tại)
        data_yaml = prepare_dataset()

        # Step 2: train
        model = YOLO(Config.MODEL)
        # Swap cls loss NẾU Config.LOSS_FUNCTION != 'bce'
        install_cls_loss(model)

        if Config.USE_STRATEGY2:
            manager = TopKCheckpointManager(Config.KEEP_TOP_K_CHECKPOINTS)
            model.add_callback("on_model_save", manager.on_model_save)
            print(f"  Strategy 2: giữ Top-{Config.KEEP_TOP_K_CHECKPOINTS} checkpoint theo val_loss nhỏ nhất")

        # model.train() trả về DetMetrics của lượt val CUỐI trên best.pt (split val')
        metrics = model.train(**build_train_args(data_yaml))

        run_dir      = Path(model.trainer.save_dir)
        best_weights = run_dir / "weights" / "best.pt"
        print(f"\n  ✓ Training completed for seed {seed}. Run dir: {run_dir}")
        print(f"  ✓ Best weights: {best_weights}")

        # Step 3: metrics trên val' (holdout) — KHÔNG phải số liệu báo cáo cuối
        val_metrics = {}
        if metrics is not None:
            print_detection_metrics(metrics, header=f"FINAL VALIDATION on val' (best.pt) | Seed {seed}")
            from evaluate import extract_overall_metrics
            val_metrics = extract_overall_metrics(metrics)

        # Step 4: báo cáo cuối trên EVAL_SPLIT (mặc định test = VOC2007 test)
        # gồm Strategy 1 (best.pt) và Strategy 2 (Top-K average) + Excel
        strategy_results = run_strategy_evaluation(run_dir, data=data_yaml, seed=seed)

        # Step 5: Edge AI hook (tắt mặc định)
        exported_model = export_model(best_weights) if Config.EXPORT_ENABLED else None

        all_runs_results.append({
            "seed"           : seed,
            "run_dir"        : str(run_dir),
            "best_weights"   : str(best_weights),
            "val_metrics"    : val_metrics,
            "strategy_results": strategy_results,
            "exported_model" : exported_model,
        })

    # ── Khôi phục cấu hình gốc ────────────────────────────────────────────────
    Config.RANDOM_SEED = orig_seeds
    Config.PROJECT     = orig_project
    Config.NAME        = orig_name

    # In bảng tổng kết nếu chạy nhiều seed
    if len(seeds) > 1:
        print_multi_seed_summary(all_runs_results)

    # Export multi-seed summary vào THƯ MỤC EXPERIMENT GROUP (không phải project root)
    # → mỗi lần bấm chạy sẽ có 1 file tổng hợp riêng, không bị ghi đè
    export_multi_seed_summary(all_runs_results, str(exp_dir))

    print("\n" + "=" * 70)
    print(f"  ✓ EXPERIMENT COMPLETE")
    print(f"  ✓ All results saved to: {exp_dir}")
    print("=" * 70)

    return all_runs_results



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
