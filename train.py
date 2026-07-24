"""
Train Ultralytics YOLO cho instance segmentation (Carparts).

Toàn bộ training loop giao cho Ultralytics (model.train): LR schedule,
augmentation, early stopping, best.pt/last.pt, results.csv đều do Ultralytics
quản lý trong run dir. Phần thêm vào cho Strategy 2:

- save_period=1 để Ultralytics lưu checkpoint mỗi epoch, NHƯNG
- TopKCheckpointManager (callback on_model_save) prune NGAY checkpoint ngoài
  Top-K theo FITNESS trên val' → disk chỉ giữ đúng K checkpoint cần thiết
  (+ best.pt/last.pt), không lưu tất cả epoch.

Fitness được lấy trực tiếp từ ``trainer.fitness`` — đúng cùng đại lượng mà
Ultralytics dùng cho best.pt và early stopping. EMA được tắt để validation,
fitness, best.pt và các checkpoint epoch đều dùng raw model weights.
"""
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from config import Config
from dataset import prepare_dataset
from evaluate import RANKING_FILE, export_multi_seed_summary, print_detection_metrics, run_strategy_evaluation
from losses import install_cls_loss


def unwrap_ultralytics_model(model):
    """
    Lấy base model tương thích cả Ultralytics cũ và mới.

    - Bản mới dùng ``unwrap_model`` (hỗ trợ torch.compile + DP/DDP).
    - ultralytics==8.3.152 dùng tên cũ ``de_parallel``.
    - Fallback cuối giữ đúng logic unwrap chính thức để code không phụ thuộc
      cứng vào tên helper nội bộ của Ultralytics.
    """
    try:
        from ultralytics.utils.torch_utils import unwrap_model
    except ImportError:
        try:
            from ultralytics.utils.torch_utils import de_parallel
        except ImportError:
            from torch import nn

            while True:
                if hasattr(model, "_orig_mod") and isinstance(model._orig_mod, nn.Module):
                    model = model._orig_mod
                elif hasattr(model, "module") and isinstance(model.module, nn.Module):
                    model = model.module
                else:
                    return model
        else:
            return de_parallel(model)
    else:
        return unwrap_model(model)


class TopKCheckpointManager:
    """
    Quản lý raw-model checkpoint cho Strategy 2.

    ``on_train_start`` tắt EMA smoothing. ``on_train_epoch_end`` copy 1:1 raw
    state sang module validation riêng, nhờ đó fitness, early stopping và
    best.pt dùng raw weights mà không chạy inference trên training module.

    ``on_model_save`` lưu raw FP32 model của epoch hiện tại, ghi nhận fitness
    trên val và xóa checkpoint tệ nhất nếu vượt KEEP_TOP_K_CHECKPOINTS.

    Ranking được ghi ra <run_dir>/weights/strategy2_checkpoints.json để
    evaluate.rank_checkpoints() dùng lại khi average Top-K (Strategy 2).
    Không đụng tới best.pt / last.pt của Ultralytics.
    """

    def __init__(self, keep_top_k):
        self.keep_top_k = int(keep_top_k)
        self.records = {}  # filename -> {"epoch": int, "fitness": float, "weight_source": "raw"}

    @staticmethod
    def on_train_start(trainer):
        """
        Tắt EMA smoothing nhưng giữ model validation là một module riêng.

        Không được trỏ ``ema.ema`` trực tiếp vào ``trainer.model``: validator
        chạy dưới torch.inference_mode() và YOLO head có thể cache inference
        tensors vào module, khiến backward của epoch sau bị RuntimeError.
        """
        if getattr(trainer, "ema", None) is None:
            raise RuntimeError("Ultralytics trainer chưa khởi tạo ModelEMA.")

        # ModelEMA.update() trở thành no-op. ema.ema vẫn là deepcopy riêng được
        # Ultralytics tạo lúc setup; ta chỉ đồng bộ raw state_dict trước mỗi val.
        trainer.ema.enabled = False
        TopKCheckpointManager.sync_raw_validation_model(trainer)
        print(
            "  Strategy 2: EMA smoothing disabled — validation dùng bản sao "
            "đồng bộ chính xác RAW weights"
        )

    @staticmethod
    def sync_raw_validation_model(trainer):
        """
        Copy chính xác raw weights/buffers sang model validation riêng.

        Đây là phép copy 1:1 tại cùng epoch, không phải exponential moving
        average và không tạo learnable parameter mới.
        """
        import torch

        raw_model = unwrap_ultralytics_model(trainer.model)
        validation_model = unwrap_ultralytics_model(trainer.ema.ema)

        # Phòng trường hợp một phiên bản/custom trainer đã alias hai object.
        if validation_model is raw_model:
            validation_model = deepcopy(raw_model).eval()
            validation_model.requires_grad_(False)
            trainer.ema.ema = validation_model

        with torch.no_grad():
            validation_model.load_state_dict(raw_model.state_dict(), strict=True)
        validation_model.eval()

    @staticmethod
    def on_train_epoch_end(trainer):
        """Đồng bộ raw snapshot sau optimizer step cuối, ngay trước validation."""
        TopKCheckpointManager.sync_raw_validation_model(trainer)

    def on_model_save(self, trainer):
        weights_dir = Path(trainer.save_dir) / "weights"
        ranking_path = weights_dir / RANKING_FILE

        # Resume-safe: chỉ nhận ranking raw mới; run cũ dùng EMA/val_loss phải
        # train lại vì checkpoint phù hợp có thể đã bị prune.
        if not self.records and ranking_path.exists():
            saved = json.loads(ranking_path.read_text(encoding="utf-8"))
            if any(
                "fitness" not in info or info.get("weight_source") != "raw"
                for info in saved.values()
            ):
                raise RuntimeError(
                    f"{ranking_path} không phải ranking raw-weight hiện tại. "
                    "Hãy dùng run mới để tạo Top-K raw checkpoints."
                )
            self.records = {
                name: {
                    "epoch": int(info["epoch"]),
                    "fitness": float(info["fitness"]),
                    "weight_source": "raw",
                }
                for name, info in saved.items()
                if (weights_dir / name).exists()
            }

        # Ultralytics đã validate trước callback này; trainer.fitness chính là
        # criterion dùng cho best.pt và EarlyStopping.
        fitness = getattr(trainer, "fitness", None)
        try:
            fitness = float(fitness)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Ultralytics không cung cấp trainer.fitness hợp lệ.") from exc
        if not math.isfinite(fitness):
            raise RuntimeError(f"trainer.fitness không hữu hạn tại epoch {trainer.epoch}: {fitness}")

        # save_period=1 tạo đúng epoch{trainer.epoch}.pt trước on_model_save.
        ckpt = weights_dir / f"epoch{int(trainer.epoch)}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint vừa lưu không tồn tại: {ckpt}")

        # Ghi đè epoch checkpoint mặc định (field 'ema') bằng raw FP32 model
        # rõ ràng trong field 'model'. Không lưu optimizer vì file này chỉ dùng
        # cho post-training averaging, không dùng resume.
        import torch
        import ultralytics

        raw_model = deepcopy(unwrap_ultralytics_model(trainer.model)).float().cpu()
        raw_ckpt = {
            "epoch": int(trainer.epoch),
            "best_fitness": float(getattr(trainer, "best_fitness", fitness)),
            "model": raw_model,
            "ema": None,
            "updates": 0,
            "optimizer": None,
            "train_args": vars(trainer.args),
            "train_metrics": {**getattr(trainer, "metrics", {}), "fitness": fitness},
            "date": datetime.now().isoformat(timespec="seconds"),
            "version": ultralytics.__version__,
            "weight_source": "raw",
        }
        temp_ckpt = ckpt.with_suffix(".tmp")
        torch.save(raw_ckpt, temp_ckpt)
        temp_ckpt.replace(ckpt)

        self.records[ckpt.name] = {
            "epoch": int(trainer.epoch),
            "fitness": fitness,
            "weight_source": "raw",
        }

        # Prune: fitness thấp nhất tệ nhất; nếu hòa thì loại epoch cũ hơn.
        while len(self.records) > self.keep_top_k:
            worst = min(
                self.records,
                key=lambda name: (self.records[name]["fitness"], self.records[name]["epoch"]),
            )
            worst_path = weights_dir / worst
            if worst_path.exists():
                worst_path.unlink()
            del self.records[worst]

        ranking_path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def delete_checkpoint_artifacts(run_dir):
    """
    Xóa toàn bộ checkpoint tạm trong ``<run_dir>/weights``.

    Hàm này chỉ chạy sau khi đã hoàn tất average, evaluation, Excel và export
    tùy chọn. Các kết quả không phải checkpoint như results.csv, plots, args.yaml
    và Excel được giữ nguyên.
    """
    weights_dir = Path(run_dir) / "weights"
    if not weights_dir.exists():
        return 0

    files = [path for path in weights_dir.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    for path in files:
        path.unlink()

    # Xóa các thư mục con rỗng rồi xóa luôn weights/.
    directories = [path for path in weights_dir.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        directory.rmdir()
    weights_dir.rmdir()

    print(
        f"  ✓ Đã xóa {len(files)} checkpoint/file ranking tạm "
        f"({total_bytes / (1024 ** 2):.1f} MB): {weights_dir}"
    )
    return len(files)


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

    # Invariants của Strategy 2: phải validate và lưu checkpoint ở mọi epoch
    # để mỗi epoch có fitness + epochN.pt tương ứng. Không cho EXTRA_TRAIN_ARGS
    # vô tình phá vỡ hai điều kiện này.
    if Config.USE_STRATEGY2:
        train_args["val"] = True
        train_args["save"] = True
        train_args["save_period"] = 1
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
    1. Tạo experiment group theo --exp-name; fallback timestamp nếu không truyền
    2. Chuẩn bị data (tách val' nếu cần theo VAL_RATIO, hỗ trợ seed dạng list/int)
    3. Loop qua tất cả các seed → mỗi seed = 1 subfolder riêng bên trong experiment group
    4. Huấn luyện model.train() với TopKCheckpointManager (nếu USE_STRATEGY2)
    5. Báo cáo đánh giá Strategy 1 (best.pt) và Strategy 2 (Top-K average) trên split test
    6. Xuất Excel và Edge AI export nếu được bật
    7. Xóa toàn bộ checkpoint tạm, chỉ giữ metrics/CSV/Excel/config/plots
    """
    # Import trễ để validate config / --help không cần ultralytics
    from ultralytics import YOLO

    # ── Lưu cấu hình gốc ──────────────────────────────────────────────────────
    orig_seeds   = Config.RANDOM_SEED
    orig_project = Config.PROJECT          # thư mục root gốc (results/segmentation)
    orig_exp_name = Config.EXP_NAME
    orig_name    = Config.NAME             # None hoặc custom name người dùng đặt

    # Chuẩn hóa seeds thành list
    if isinstance(orig_seeds, (list, tuple)):
        seeds = [int(s) for s in orig_seeds]
    else:
        seeds = [int(orig_seeds)]

    # ── Tạo thư mục experiment group ─────────────────────────────────────────
    # Có --exp-name: dùng ĐÚNG tên người dùng đặt, dễ quản lý trên server.
    # Không có: fallback tên tự động timestamp + model để tương thích code cũ.
    model_stem = Path(str(Config.MODEL)).stem          # "yolov8s" từ "yolov8s.pt"
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    if orig_exp_name:
        exp_group = str(orig_exp_name).strip()
    else:
        base_name = orig_name if orig_name else "exp"
        exp_group = f"{base_name}_{timestamp}_{model_stem}"
    exp_dir    = Path(orig_project) / exp_group

    # Tên explicit phải trỏ tới một experiment mới để không trộn nhiều lần chạy.
    if orig_exp_name and exp_dir.exists() and any(exp_dir.iterdir()):
        raise FileExistsError(
            f"Experiment đã tồn tại và không rỗng: {exp_dir}\n"
            "Hãy chọn --exp-name khác để kết quả các lần chạy không bị trộn."
        )
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
        "explicit_exp_name": orig_exp_name,
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
        "delete_checkpoints_after_run": Config.DELETE_CHECKPOINTS_AFTER_RUN,
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
        if getattr(model, "task", None) != "segment":
            raise ValueError(
                "Pipeline này chỉ hỗ trợ instance segmentation. "
                f"Model {Config.MODEL!r} có task={getattr(model, 'task', None)!r}; "
                "hãy dùng weights/config segmentation (ví dụ yolov8s-seg.pt)."
            )
        # Swap cls loss NẾU Config.LOSS_FUNCTION != 'bce'
        install_cls_loss(model)

        if Config.USE_STRATEGY2:
            manager = TopKCheckpointManager(Config.KEEP_TOP_K_CHECKPOINTS)
            model.add_callback("on_train_start", manager.on_train_start)
            model.add_callback("on_train_epoch_end", manager.on_train_epoch_end)
            model.add_callback("on_model_save", manager.on_model_save)
            print(
                f"  Strategy 2: giữ Top-{Config.KEEP_TOP_K_CHECKPOINTS} RAW checkpoint "
                "theo raw-model fitness cao nhất"
            )

        run_dir = None
        try:
            # model.train() trả về SegmentMetrics của lượt val CUỐI trên best.pt.
            metrics = model.train(**build_train_args(data_yaml))

            run_dir      = Path(model.trainer.save_dir)
            best_weights = run_dir / "weights" / "best.pt"
            print(f"\n  ✓ Training completed for seed {seed}. Run dir: {run_dir}")
            print(f"  ✓ Best weights tạm thời: {best_weights}")

            # Step 3: metrics trên val' (holdout) — KHÔNG phải số liệu báo cáo cuối
            val_metrics = {}
            if metrics is not None:
                print_detection_metrics(
                    metrics,
                    header=f"FINAL VALIDATION on val' (best.pt) | Seed {seed}",
                )
                from evaluate import extract_overall_metrics
                val_metrics = extract_overall_metrics(metrics)

            # Step 4: báo cáo cuối trên EVAL_SPLIT (mặc định test độc lập)
            # gồm Strategy 1 (best.pt) và Strategy 2 (Top-K average) + Excel.
            strategy_results = run_strategy_evaluation(run_dir, data=data_yaml, seed=seed)

            # Step 5: Edge AI hook (tắt mặc định). Export phải chạy trước cleanup.
            exported_model = export_model(best_weights) if Config.EXPORT_ENABLED else None

            all_runs_results.append({
                "seed"              : seed,
                "run_dir"           : str(run_dir),
                "best_weights"      : None,
                "checkpoint_policy" : "temporary_then_deleted",
                "val_metrics"       : val_metrics,
                "strategy_results"  : strategy_results,
                "exported_model"    : exported_model,
            })
        finally:
            # Kể cả evaluation/export lỗi, không để checkpoint tạm nằm lại trên server.
            if Config.DELETE_CHECKPOINTS_AFTER_RUN:
                cleanup_dir = run_dir
                if cleanup_dir is None:
                    trainer = getattr(model, "trainer", None)
                    save_dir = getattr(trainer, "save_dir", None)
                    cleanup_dir = Path(save_dir) if save_dir else None
                if cleanup_dir is not None:
                    delete_checkpoint_artifacts(cleanup_dir)

    # ── Khôi phục cấu hình gốc ────────────────────────────────────────────────
    Config.RANDOM_SEED = orig_seeds
    Config.PROJECT     = orig_project
    Config.EXP_NAME    = orig_exp_name
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
