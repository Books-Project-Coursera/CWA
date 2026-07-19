"""
Entrypoint cho pipeline Object Detection (Strategy 2 mở rộng) —
Ultralytics YOLO + Pascal VOC. Giữ pattern "1 entrypoint" như main.py của
nhánh classification, nhưng chia subcommand theo detection flow:

    # Train (model lấy từ config, hoặc override --model)
    python main_detection.py train --config configs/detection_voc.yaml
    python main_detection.py train --model yolov8n.pt --epochs 50 --batch 32

    # Eval weights đã train (in mAP/P/R/per-class AP + export Excel)
    python main_detection.py eval --weights results/detection/train/weights/best.pt --split test

    # Export Excel 2 sheet từ run dir (offline, chỉ cần results.csv);
    # thêm --weights để chạy val lấy per-class AP cho sheet Summary
    python main_detection.py export --run-dir results/detection/train
    python main_detection.py export --run-dir results/detection/train --weights results/detection/train/weights/best.pt

    # Edge AI: xuất weights đã train sang ONNX/TensorRT (model.export)
    python main_detection.py export-model --weights results/detection/train/weights/best.pt

Chi tiết + note quan trọng (VOC val ≡ test): README_DETECTION.md
"""
import argparse

from detection.config import load_config, apply_cli_overrides, validate_config
from detection.eval import evaluate_detector
from detection.export_metrics import export_run_to_excel
from detection.train import train_detector, export_model

DEFAULT_CONFIG_PATH = "configs/detection_voc.yaml"
EVAL_SPLIT_CHOICES = ["val", "test", "train"]


def add_common_args(parser):
    """Args chung cho mọi subcommand — mọi giá trị đều override config YAML."""
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path tới file config YAML của detection pipeline.",
    )
    parser.add_argument(
        "--model",
        help="Model YOLO: yolov8n.pt, yolo11n.pt, yolov5nu.pt, .pt/.yaml custom... "
             "Override key `model` trong config.",
    )
    parser.add_argument("--data", help="VOC.yaml (auto-download) hoặc path data.yaml custom.")
    parser.add_argument("--epochs", type=int, help="Override số epoch.")
    parser.add_argument("--imgsz", type=int, help="Override image size.")
    parser.add_argument("--batch", type=int, help="Override batch size (-1 = auto-batch).")
    parser.add_argument("--device", help="Device: 0 | 0,1 | cpu. Mặc định auto.")
    parser.add_argument("--workers", type=int, help="Override số DataLoader workers.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("--optimizer", help="Override optimizer (auto/SGD/AdamW/...).")
    parser.add_argument("--lr0", type=float, help="Override learning rate ban đầu.")
    parser.add_argument("--lrf", type=float, help="Override final LR fraction.")
    parser.add_argument("--patience", type=int, help="Override early-stopping patience.")
    parser.add_argument("--project", help="Override thư mục output gốc (project).")
    parser.add_argument("--name", help="Override tên run (folder con trong project).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strategy 2 - Object Detection pipeline (Ultralytics YOLO + Pascal VOC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser(
        "train", help="Train YOLO trên VOC, in metrics val cuối, export Excel 2 sheet."
    )
    add_common_args(p_train)
    p_train.add_argument(
        "--export-after-train",
        action="store_true",
        help="Bật export.enabled: xuất model (ONNX/... theo config) ngay sau train.",
    )

    p_eval = subparsers.add_parser(
        "eval", help="Eval weights đã train bằng model.val(), in metrics + export Excel."
    )
    add_common_args(p_eval)
    p_eval.add_argument("--weights", help="Weights đã train (best.pt). Mặc định dùng `model` trong config.")
    p_eval.add_argument(
        "--split",
        choices=EVAL_SPLIT_CHOICES,
        help="Split đánh giá; mặc định theo data.yaml. LƯU Ý VOC built-in: val ≡ test (VOC2007).",
    )
    p_eval.add_argument("--run-dir", help="Run dir train (chứa results.csv) để ghép sheet PerEpoch.")
    p_eval.add_argument("--no-excel", action="store_true", help="Chỉ in metrics, không export Excel.")

    p_export = subparsers.add_parser(
        "export", help="Export Excel (Summary + PerEpoch) từ run dir đã train."
    )
    add_common_args(p_export)
    p_export.add_argument("--run-dir", required=True, help="Run dir Ultralytics (chứa results.csv).")
    p_export.add_argument(
        "--weights",
        help="Optional: chạy model.val() với weights này để sheet Summary có per-class AP.",
    )
    p_export.add_argument("--split", choices=EVAL_SPLIT_CHOICES, help="Split cho lượt val (khi có --weights).")
    p_export.add_argument("--output", dest="excel_output", help="Path file .xlsx output.")

    p_export_model = subparsers.add_parser(
        "export-model", help="Edge AI: xuất weights đã train sang ONNX/TensorRT (model.export)."
    )
    add_common_args(p_export_model)
    p_export_model.add_argument("--weights", required=True, help="Weights đã train cần export.")
    p_export_model.add_argument("--format", dest="export_format", help="Override export.format (onnx/engine/...).")
    p_export_model.add_argument("--half", action="store_true", help="Export FP16 (override export.half).")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    if args.command == "train":
        validate_config(cfg, require_model=True)
        train_detector(cfg)

    elif args.command == "eval":
        validate_config(cfg, require_model=False)
        weights = args.weights or cfg.get("model")
        evaluate_detector(
            cfg, weights, export_excel=not args.no_excel, run_dir=args.run_dir
        )

    elif args.command == "export":
        validate_config(cfg, require_model=False)
        if args.weights:
            # Có weights → val để lấy per-class AP, rồi export Excel vào run dir
            evaluate_detector(cfg, args.weights, export_excel=True, run_dir=args.run_dir)
        else:
            # Offline: chỉ parse results.csv (Summary lấy từ epoch cuối)
            export_run_to_excel(args.run_dir, metrics=None, cfg=cfg)

    elif args.command == "export-model":
        if args.export_format:
            cfg["export"]["format"] = args.export_format
        if args.half:
            cfg["export"]["half"] = True
        validate_config(cfg, require_model=False)
        export_model(cfg, args.weights)


if __name__ == "__main__":
    main()
