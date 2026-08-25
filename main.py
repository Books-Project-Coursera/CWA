"""
Main entrypoint — Strategy 2 Instance Segmentation (Ultralytics YOLO-seg).

Nhánh này CHỈ dùng model YOLO. Mọi config chỉnh trong config.py; các giá trị
hay dùng override được qua CLI (pattern như repo gốc).

    # Train (model lấy từ Config.MODEL, đặt tên experiment rõ ràng trên server)
    python main.py train --exp-name yolov8s_exp1
    #   → results/segmentation/yolov8s_exp1/
    #       README.md, experiment_config.json,
    #       summary/{yolov8s_exp1_summary.xlsx, charts/},
    #       seeds/seed_<N>/{seed_<N>_results.xlsx, charts/, logs/}

    # So sánh method: Top-K (của bạn) vs EMA vs SWA — MỘT lần train cho cả ba,
    # cùng seed / cùng config / cùng trajectory ⇒ so sánh paired theo seed.
    python main.py train --exp-name carparts_compare_run01 --method top-k ema swa
    python main.py train --exp-name carparts_ema_swa       --method EMA,SWA
    python main.py train --exp-name carparts_topk_only     --method top-k

    # Đánh giá lại Strategy 1 + Strategy 2 trên một run đã train
    python main.py strategies --run-dir results/segmentation/<experiment>/seeds/seed_42

    # Eval một file weights bất kỳ (in mAP/P/R/per-class AP + Excel)
    python main.py eval --weights <run>/weights/best.pt --split test

    # Export Excel từ run dir (offline, chỉ cần results.csv)
    python main.py export --run-dir <run>

    # Edge AI: xuất weights sang ONNX/TensorRT
    python main.py export-model --weights <run>/weights/best.pt

Chi tiết về fitness, EMA, averaging và data split: README.md
"""
import argparse
import sys

from config import Config

EVAL_SPLIT_CHOICES = ["val", "test", "train"]


def configure_console_encoding():
    """Cho phép help/log tiếng Việt chạy ổn định trên Windows console."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                # Một số IDE/notebook bọc stream và không cho reconfigure.
                pass


def add_common_args(parser):
    """Args chung cho mọi subcommand — mọi giá trị đều override Config."""
    parser.add_argument(
        "--model",
        help="Model YOLO-seg: yolov8s-seg.pt, yolo11s-seg.pt, .pt/.yaml segmentation custom... "
             "Override Config.MODEL.",
    )
    parser.add_argument("--data", help="carparts-seg.yaml hoặc path data.yaml segmentation custom.")
    parser.add_argument("--val-ratio", type=float, help="Tỉ lệ tách val' từ train (0 = không tách).")
    parser.add_argument("--epochs", type=int, help="Override Config.EPOCHS.")
    parser.add_argument("--imgsz", type=int, help="Override Config.IMGSZ.")
    parser.add_argument("--batch", type=int, help="Override Config.BATCH (-1 = auto-batch).")
    parser.add_argument("--device", help="Device: 0 | 0,1 | cpu. Mặc định auto.")
    parser.add_argument("--workers", type=int, help="Override Config.WORKERS.")
    parser.add_argument("--seed", type=int, nargs="+", help="Override Config.RANDOM_SEED (chấp nhận 1 hoặc nhiều seed, ví dụ: --seed 42 100).")
    parser.add_argument("--optimizer", help="Override Config.OPTIMIZER (auto/SGD/AdamW/...).")
    parser.add_argument("--lr0", type=float, help="Override Config.LR0.")
    parser.add_argument("--lrf", type=float, help="Override Config.LRF.")
    parser.add_argument("--patience", type=int, help="Override Config.PATIENCE.")
    parser.add_argument(
        "--loss",
        choices=["bce", "focal"],
        help="Override Config.LOSS_FUNCTION: 'bce' (mặc định Ultralytics) hoặc 'focal' (FocalBCE).",
    )
    parser.add_argument("--focal-gamma", type=float, help="Override Config.FOCAL_GAMMA.")
    parser.add_argument("--focal-alpha", type=float, help="Override Config.FOCAL_ALPHA.")
    parser.add_argument("--project", help="Override Config.PROJECT (thư mục output gốc).")
    parser.add_argument(
        "--exp-name",
        "--exp_name",
        dest="exp_name",
        help="Tên chính xác của thư mục experiment trên server, không tự ghép timestamp.",
    )
    parser.add_argument(
        "--name",
        help="Prefix legacy cho tên tự động khi không truyền --exp-name.",
    )
    # ---- Method comparison (Top-K vs EMA vs SWA) ----
    parser.add_argument(
        "--method", "--methods", nargs="+", dest="methods", metavar="M",
        help="Method weight-averaging cần chạy: top-k | ema | swa (hoặc 'all'/'none'). "
             "Cho phép tổ hợp và dấu phẩy: --method top-k ema swa | --method EMA,SWA. "
             "Tất cả method được tính trên CÙNG một trajectory nên chạy một lần là "
             "đủ để so sánh paired theo seed. Override Config.METHODS.",
    )
    parser.add_argument(
        "--ema-decay", "--ema_decay", dest="ema_decays", type=float, nargs="+",
        help="Override Config.EMA_DECAYS (decay theo optimizer step), ví dụ: "
             "--ema-decay 0.9 0.99 0.999.",
    )
    parser.add_argument(
        "--ema-update-period", "--ema_update_period", dest="ema_update_period", type=int,
        help="Override Config.EMA_UPDATE_PERIOD (cập nhật EMA mỗi N optimizer step).",
    )
    parser.add_argument(
        "--swa-start", "--swa_start", dest="swa_start_fracs", type=float, nargs="+",
        help="Override Config.SWA_START_FRACS — mốc bắt đầu SWA theo tỉ lệ budget, "
             "ví dụ: --swa-start 0.75 (mặc định, đúng Izmailov et al.).",
    )
    parser.add_argument("--swa-period", "--swa_period", dest="swa_period", type=int,
                        help="Override Config.SWA_PERIOD (average mỗi N epoch).")
    parser.add_argument(
        "--swa-lr-schedule", "--swa_lr_schedule", dest="swa_lr_schedule",
        choices=["inherit", "extend", "truncate", "constant"],
        help="LR cho pha SWA. 'inherit' (MẶC ĐỊNH): giữ cosine chung — cùng trajectory "
             "với Top-K/EMA. 'extend' (= alias 'constant'): chạy TRỌN cosine của bạn "
             "(đúng bằng run Top-K) rồi NỐI THÊM --swa-extra-budget × EPOCHS epoch "
             "constant LR. 'truncate': cắt cosine ở --swa-lr-start rồi chạy nốt bằng "
             "constant, tổng budget giữ nguyên. Hai mode sau ĐỔI TRAJECTORY ⇒ là "
             "experiment riêng, nên chạy kèm --methods swa --patience 0.",
    )
    parser.add_argument(
        "--swa-extra-budget", "--swa_extra_budget", dest="swa_extra_budget", type=float,
        help="Chỉ cho mode 'extend': số epoch constant-LR nối thêm, theo tỉ lệ EPOCHS "
             "(0.25 ⇒ 100+25=125 epoch = 1.25 budget). Override Config.SWA_EXTRA_BUDGET.",
    )
    parser.add_argument("--swa-lr", "--swa_lr", dest="swa_lr", type=float,
                        help="Override Config.SWA_LR — giá trị constant LR của pha SWA.")
    parser.add_argument(
        "--swa-lr-start", "--swa_lr_start", dest="swa_lr_start_frac", type=float,
        help="Override Config.SWA_LR_START_FRAC — mốc chuyển sang constant LR "
             "theo tỉ lệ budget (mặc định 0.75).",
    )
    parser.add_argument(
        "--shadow-select", "--shadow_select", dest="shadow_select",
        choices=["best_val", "final"],
        help="Chọn snapshot EMA/SWA: 'final' (trạng thái ở epoch cuối — MẶC ĐỊNH, "
             "đúng cách dùng chuẩn của cả hai paper) hoặc 'best_val' (epoch tốt nhất trên "
             "val, cần --shadow-val). Override Config.SHADOW_SELECT.",
    )
    parser.add_argument(
        "--shadow-val-period", "--shadow_val_period", dest="shadow_val_period", type=int,
        help="Khi đã bật --shadow-val: val shadow mỗi N epoch (mặc định 1).",
    )
    parser.add_argument(
        "--shadow-val", action="store_true",
        help="Bật val shadow EMA/SWA trên tập val mỗi epoch — sinh đường cong fitness "
             "cho phụ lục (ghi vào averaging_shadows.json) và là điều kiện để dùng "
             "--shadow-select best_val. Mặc định TẮT → không tốn thêm thời gian train.",
    )
    parser.add_argument(
        "--no-shadow-val", action="store_true",
        help="Tắt hẳn việc val shadow EMA/SWA trên val → snapshot lấy ở epoch cuối.",
    )
    parser.add_argument(
        "--top-k", type=int, nargs="+", dest="top_k_values",
        help="Override Config.TOP_K_VALUES, ví dụ: --top-k 2 3 5.",
    )
    parser.add_argument(
        "--split",
        choices=EVAL_SPLIT_CHOICES,
        help="Split cho báo cáo cuối (Config.EVAL_SPLIT). Mặc định 'test'.",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strategy 2 - Instance Segmentation pipeline (Ultralytics YOLO-seg).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # required=False: gõ trống `python main.py` thì mặc định là `train` (xem cuối hàm).
    subparsers = parser.add_subparsers(dest="command")

    p_train = subparsers.add_parser(
        "train",
        help="Train YOLO (val' tách riêng từ train) → so sánh Strategy 1 vs "
             "Strategy 2 trên test → export Excel.",
    )
    add_common_args(p_train)
    p_train.add_argument(
        "--no-strategy2", action="store_true",
        help="Tắt Strategy 2 (không lưu/average Top-K checkpoint).",
    )
    p_train.add_argument(
        "--export-after-train", action="store_true",
        help="Bật Config.EXPORT_ENABLED: xuất model (ONNX/... theo config) sau train.",
    )

    p_strategies = subparsers.add_parser(
        "strategies",
        help="Đánh giá lại Strategy 1 + Strategy 2 trên một run đã train + export Excel.",
    )
    add_common_args(p_strategies)
    p_strategies.add_argument("--run-dir", required=True, help="Run dir Ultralytics đã train.")

    p_eval = subparsers.add_parser(
        "eval", help="Eval một file weights bằng model.val(), in metrics + export Excel."
    )
    add_common_args(p_eval)
    p_eval.add_argument("--weights", help="Weights đã train (best.pt). Mặc định dùng Config.MODEL.")
    p_eval.add_argument("--run-dir", help="Run dir chứa results.csv để ghép sheet PerEpoch.")
    p_eval.add_argument("--no-excel", action="store_true", help="Chỉ in metrics, không export Excel.")

    p_export = subparsers.add_parser(
        "export", help="Export Excel (Summary + PerEpoch) từ run dir đã train (offline)."
    )
    add_common_args(p_export)
    p_export.add_argument("--run-dir", required=True, help="Run dir Ultralytics (chứa results.csv).")
    p_export.add_argument("--output", dest="excel_output", help="Path file .xlsx output.")

    p_export_model = subparsers.add_parser(
        "export-model", help="Edge AI: xuất weights đã train sang ONNX/TensorRT (model.export)."
    )
    add_common_args(p_export_model)
    p_export_model.add_argument("--weights", required=True, help="Weights đã train cần export.")
    p_export_model.add_argument("--format", dest="export_format", help="Override Config.EXPORT_FORMAT.")
    p_export_model.add_argument("--half", action="store_true", help="Export FP16 (Config.EXPORT_HALF).")

    # `python main.py` (không subcommand) ⇒ hiểu là `python main.py train`, để
    # chạy thẳng bằng toàn bộ default trong config.py.
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        print("[main] Không có subcommand → mặc định chạy 'train' với config.py hiện tại.")
        argv = ["train", *argv]
    return parser.parse_args(argv)


def apply_cli_overrides(args):
    """CLI override lên Config — pattern giữ nguyên từ repo gốc."""
    overrides = [
        ("MODEL", getattr(args, "model", None)),
        ("DATA", getattr(args, "data", None)),
        ("VAL_RATIO", getattr(args, "val_ratio", None)),
        ("EPOCHS", getattr(args, "epochs", None)),
        ("IMGSZ", getattr(args, "imgsz", None)),
        ("BATCH", getattr(args, "batch", None)),
        ("DEVICE", getattr(args, "device", None)),
        ("WORKERS", getattr(args, "workers", None)),
        ("RANDOM_SEED", getattr(args, "seed", None)),
        ("OPTIMIZER", getattr(args, "optimizer", None)),
        ("LR0", getattr(args, "lr0", None)),
        ("LRF", getattr(args, "lrf", None)),
        ("PATIENCE", getattr(args, "patience", None)),
        ("LOSS_FUNCTION", getattr(args, "loss", None)),
        ("FOCAL_GAMMA", getattr(args, "focal_gamma", None)),
        ("FOCAL_ALPHA", getattr(args, "focal_alpha", None)),
        ("PROJECT", getattr(args, "project", None)),
        ("EXP_NAME", getattr(args, "exp_name", None)),
        ("NAME", getattr(args, "name", None)),
        ("EVAL_SPLIT", getattr(args, "split", None)),
        ("EXCEL_OUTPUT", getattr(args, "excel_output", None)),
        ("EXPORT_FORMAT", getattr(args, "export_format", None)),
        ("TOP_K_VALUES", getattr(args, "top_k_values", None)),
        ("EMA_DECAYS", getattr(args, "ema_decays", None)),
        ("EMA_UPDATE_PERIOD", getattr(args, "ema_update_period", None)),
        ("SWA_START_FRACS", getattr(args, "swa_start_fracs", None)),
        ("SWA_PERIOD", getattr(args, "swa_period", None)),
        ("SWA_LR_SCHEDULE", getattr(args, "swa_lr_schedule", None)),
        ("SWA_LR", getattr(args, "swa_lr", None)),
        ("SWA_EXTRA_BUDGET", getattr(args, "swa_extra_budget", None)),
        ("SWA_LR_START_FRAC", getattr(args, "swa_lr_start_frac", None)),
        ("SHADOW_SELECT", getattr(args, "shadow_select", None)),
        ("SHADOW_VAL_PERIOD", getattr(args, "shadow_val_period", None)),
    ]
    for attr, value in overrides:
        if value is not None:
            setattr(Config, attr, value)

    if getattr(args, "methods", None):
        Config.normalize_methods(args.methods)
    if getattr(args, "shadow_val", False):
        Config.SHADOW_VAL_ENABLED = True
    if getattr(args, "no_shadow_val", False):
        Config.SHADOW_VAL_ENABLED = False
    if getattr(args, "no_strategy2", False):
        # Giữ tương thích ngược: tắt Top-K nhưng không đụng EMA/SWA.
        Config.normalize_methods([m for m in Config.METHODS if m != "top-k"])
    if getattr(args, "export_after_train", False):
        Config.EXPORT_ENABLED = True
    if getattr(args, "half", False):
        Config.EXPORT_HALF = True


def main():
    args = parse_args()
    apply_cli_overrides(args)

    if args.command == "train":
        Config.validate_config(require_model=True)
        from train import run_experiments

        run_experiments()

    elif args.command == "strategies":
        Config.validate_config(require_model=False)
        from evaluate import run_strategy_evaluation

        # data ưu tiên: --data > args.yaml của run > Config.DATA
        data = args.data or None
        run_strategy_evaluation(args.run_dir, data=data)

    elif args.command == "eval":
        Config.validate_config(require_model=False)
        from evaluate import (
            evaluate_weights,
            export_to_excel,
            infer_run_dir,
            resolve_data_from_run,
        )

        weights = args.weights or Config.MODEL
        run_dir = args.run_dir or infer_run_dir(weights)
        # data ưu tiên: --data > args.yaml của run (giữ đúng holdout split) > Config.DATA
        data = args.data or (resolve_data_from_run(run_dir) if run_dir else None) or Config.DATA
        metrics = evaluate_weights(weights, data, split=Config.EVAL_SPLIT)
        if not args.no_excel:
            target_dir = run_dir if run_dir else getattr(metrics, "save_dir", ".")
            export_to_excel(
                target_dir,
                strategy_results={"Eval": metrics},
                data=data,
                split=Config.EVAL_SPLIT,
            )

    elif args.command == "export":
        Config.validate_config(require_model=False)
        from evaluate import export_to_excel

        export_to_excel(args.run_dir, strategy_results=None)

    elif args.command == "export-model":
        Config.validate_config(require_model=False)
        from train import export_model

        export_model(args.weights)


if __name__ == "__main__":
    configure_console_encoding()
    main()
