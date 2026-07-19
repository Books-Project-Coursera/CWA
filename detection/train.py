"""
Train Ultralytics YOLO cho object detection (Pascal VOC).

Khác với nhánh classification (tự viết training loop trong train.py gốc),
toàn bộ train/val ở đây giao cho Ultralytics (model.train / model.val):
checkpoint (best.pt/last.pt), early stopping (patience), LR schedule,
augmentation, results.csv đều do Ultralytics quản lý trong thư mục run.
"""
from pathlib import Path

from detection.export_metrics import export_run_to_excel
from detection.metrics_utils import print_detection_metrics


def build_train_args(cfg):
    """
    Map config YAML → kwargs của model.train().
    extra_train_args được update SAU CÙNG nên có thể ghi đè mọi key chuẩn.
    """
    train_args = {
        "data": cfg["data"],
        "epochs": int(cfg["epochs"]),
        "imgsz": int(cfg["imgsz"]),
        "batch": int(cfg["batch"]),
        "workers": int(cfg["workers"]),
        "seed": int(cfg["seed"]),
        "optimizer": cfg["optimizer"],
        "lr0": float(cfg["lr0"]),
        "lrf": float(cfg["lrf"]),
        "patience": int(cfg["patience"]),
        "pretrained": bool(cfg["pretrained"]),
        "cache": cfg["cache"],
        "resume": bool(cfg["resume"]),
        "project": cfg["project"],
        "exist_ok": bool(cfg["exist_ok"]),
    }
    # device=None → để Ultralytics tự chọn, không truyền key
    if cfg.get("device") is not None:
        train_args["device"] = cfg["device"]
    if cfg.get("name"):
        train_args["name"] = cfg["name"]
    train_args.update(cfg.get("extra_train_args") or {})
    return train_args


def train_detector(cfg):
    """
    Train → in metrics val cuối (trên best.pt) → export Excel → (optional)
    export model cho Edge AI. Trả về dict đường dẫn các artifact chính.
    """
    # Import trễ để validate config / --help không cần cài ultralytics
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print(" STRATEGY 2 - OBJECT DETECTION TRAINING (Ultralytics YOLO)")
    print("=" * 70)
    print(f"  Model: {cfg['model']}")
    print(f"  Data : {cfg['data']} (VOC.yaml built-in sẽ tự động download lần đầu)")

    model = YOLO(cfg["model"])
    train_args = build_train_args(cfg)
    # model.train() trả về DetMetrics của lượt val CUỐI trên best.pt.
    # Với VOC.yaml built-in: lượt val này đo trên VOC2007 test (val ≡ test).
    metrics = model.train(**train_args)

    run_dir = Path(model.trainer.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    print(f"\n  ✓ Training completed. Run dir: {run_dir}")
    print(f"  ✓ Best weights: {best_weights}")

    if metrics is not None:
        print_detection_metrics(metrics, header="FINAL VALIDATION (best.pt)")

    # Excel 2 sheet: Summary (từ DetMetrics) + PerEpoch (từ results.csv)
    excel_path = export_run_to_excel(run_dir, metrics=metrics, cfg=cfg)

    exported_model = export_model_if_enabled(cfg, best_weights)

    return {
        "run_dir": str(run_dir),
        "best_weights": str(best_weights),
        "excel": excel_path,
        "exported_model": exported_model,
    }


def export_model_if_enabled(cfg, weights):
    """Edge AI hook sau train: chỉ chạy khi export.enabled=true (tắt mặc định)."""
    export_cfg = cfg.get("export") or {}
    if not export_cfg.get("enabled", False):
        return None
    return export_model(cfg, weights)


def export_model(cfg, weights):
    """
    Xuất weights đã train sang format deploy (ONNX/TensorRT/...) bằng
    model.export() của Ultralytics — phục vụ hướng Edge AI.
    """
    from ultralytics import YOLO

    export_cfg = cfg.get("export") or {}
    export_args = {
        "format": export_cfg.get("format", "onnx"),
        "half": bool(export_cfg.get("half", False)),
        "dynamic": bool(export_cfg.get("dynamic", False)),
        "simplify": bool(export_cfg.get("simplify", True)),
        "imgsz": int(export_cfg.get("imgsz") or cfg["imgsz"]),
    }
    if export_cfg.get("device") is not None:
        export_args["device"] = export_cfg["device"]

    print(f"\n  [Edge AI] Exporting {weights} → format='{export_args['format']}' ...")
    model = YOLO(str(weights))
    exported_path = model.export(**export_args)
    print(f"  ✓ Exported model: {exported_path}")
    return exported_path
