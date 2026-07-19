"""
Evaluate detector đã train bằng model.val() của Ultralytics.

⚠ LƯU Ý val ≡ test: với VOC.yaml built-in của Ultralytics, split 'val' và
'test' cùng trỏ tới VOC2007 test (4952 ảnh) → chọn --split val hay test đều
đánh giá trên cùng một tập ảnh. Xem README_DETECTION.md trước khi báo cáo
số liệu để tránh hiểu nhầm val là tập độc lập.
"""
from pathlib import Path

from detection.export_metrics import export_run_to_excel
from detection.metrics_utils import print_detection_metrics


def build_val_args(cfg):
    """Map config YAML → kwargs của model.val()."""
    val_args = {
        "data": cfg["data"],
        "imgsz": int(cfg["imgsz"]),
        "batch": int(cfg["batch"]),
        "workers": int(cfg["workers"]),
    }
    if cfg.get("device") is not None:
        val_args["device"] = cfg["device"]

    eval_cfg = cfg.get("eval") or {}
    # split=None → dùng đúng split mặc định của data.yaml khi val ('val')
    if eval_cfg.get("split"):
        val_args["split"] = eval_cfg["split"]
    if eval_cfg.get("conf") is not None:
        val_args["conf"] = float(eval_cfg["conf"])
    if eval_cfg.get("iou") is not None:
        val_args["iou"] = float(eval_cfg["iou"])
    return val_args


def evaluate_detector(cfg, weights, export_excel=True, run_dir=None):
    """
    Chạy model.val() với weights đã train, in metrics (mAP@0.5, mAP@0.5:0.95,
    Precision, Recall, per-class AP, fitness) và (optional) export Excel.

    Args:
        cfg: dict config detection
        weights: path tới weights đã train (best.pt); bắt buộc — model
                 pretrained COCO thô sẽ lệch số class với VOC (80 vs 20)
        export_excel: xuất luôn file Excel 2 sheet sau khi eval
        run_dir: thư mục run train (chứa results.csv) để ghép sheet PerEpoch;
                 None → tự suy ra từ vị trí weights (<run_dir>/weights/best.pt)

    Returns:
        DetMetrics của Ultralytics.
    """
    from ultralytics import YOLO

    if not weights:
        raise ValueError(
            "Cần weights đã train để eval: truyền --weights path/to/best.pt "
            "(hoặc set `model:` trong config trỏ tới weights đã train trên VOC)."
        )

    eval_split = (cfg.get("eval") or {}).get("split")
    print("\n" + "=" * 70)
    print(" STRATEGY 2 - OBJECT DETECTION EVALUATION (Ultralytics model.val)")
    print("=" * 70)
    print(f"  Weights: {weights}")
    print(f"  Data   : {cfg['data']}")
    print(f"  Split  : {eval_split or 'mặc định theo data.yaml (VOC: val ≡ test — VOC2007)'}")

    model = YOLO(str(weights))
    metrics = model.val(**build_val_args(cfg))

    header = f"EVALUATION RESULTS (split={eval_split or 'default'})"
    print_detection_metrics(metrics, header=header)

    if run_dir is None:
        # Chuẩn Ultralytics: <run_dir>/weights/best.pt → run_dir chứa results.csv
        weights_path = Path(weights)
        if weights_path.parent.name == "weights":
            run_dir = weights_path.parent.parent

    if export_excel:
        # Không có run_dir train → xuất vào thư mục val của lượt eval này
        target_dir = run_dir if run_dir is not None else getattr(metrics, "save_dir", ".")
        export_run_to_excel(target_dir, metrics=metrics, cfg=cfg)

    return metrics
