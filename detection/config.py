"""
Config loader cho pipeline Object Detection (Strategy 2 mở rộng).

Nhánh classification (Strategy2_TinyImageNet) dùng class `Config` trong
config.py gốc. Pipeline detection chuyển sang YAML (configs/detection_voc.yaml)
vì các train-arg của Ultralytics (epochs, imgsz, batch, ...) map thẳng vào
dict kwargs của model.train(), và cho phép đổi model/dataset chỉ bằng cách
sửa 1 dòng YAML mà không đụng vào code. Pattern CLI-override giữ nguyên tinh
thần hàm apply_cli_overrides của main.py.
"""
import copy
import os

import yaml

# Giá trị mặc định — chỉ dùng khi key vắng mặt trong file YAML.
# LƯU Ý: "model" cố tình để None (không chốt cứng version YOLO nào làm mặc
# định); người dùng PHẢI set trong YAML hoặc qua --model.
DEFAULT_CONFIG = {
    "model": None,
    "data": "VOC.yaml",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "device": None,
    "workers": 8,
    "seed": 1,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "patience": 100,
    "pretrained": True,
    "cache": False,
    "resume": False,
    "extra_train_args": {},
    "project": os.path.join("results", "detection"),
    "name": None,
    "exist_ok": False,
    "eval": {"split": None, "conf": None, "iou": None},
    "excel": {"output": None},
    "export": {
        "enabled": False,
        "format": "onnx",
        "half": False,
        "imgsz": None,
        "dynamic": False,
        "simplify": True,
        "device": None,
    },
}

VALID_EVAL_SPLITS = (None, "val", "test", "train")


def _deep_update(base, override):
    """Merge đệ quy: dict con trong YAML chỉ ghi đè key có mặt, giữ default còn lại."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path):
    """Đọc YAML config và merge lên DEFAULT_CONFIG (giá trị trong YAML ưu tiên)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        if not isinstance(user_cfg, dict):
            raise ValueError(f"Config file must be a YAML mapping: {config_path}")
        _deep_update(cfg, user_cfg)
    return cfg


def apply_cli_overrides(cfg, args):
    """CLI override lên config — cùng pattern với apply_cli_overrides của main.py."""
    overrides = [
        ("model", getattr(args, "model", None)),
        ("data", getattr(args, "data", None)),
        ("epochs", getattr(args, "epochs", None)),
        ("imgsz", getattr(args, "imgsz", None)),
        ("batch", getattr(args, "batch", None)),
        ("device", getattr(args, "device", None)),
        ("workers", getattr(args, "workers", None)),
        ("seed", getattr(args, "seed", None)),
        ("optimizer", getattr(args, "optimizer", None)),
        ("lr0", getattr(args, "lr0", None)),
        ("lrf", getattr(args, "lrf", None)),
        ("patience", getattr(args, "patience", None)),
        ("project", getattr(args, "project", None)),
        ("name", getattr(args, "name", None)),
    ]
    for key, value in overrides:
        if value is not None:
            cfg[key] = value

    if getattr(args, "split", None) is not None:
        cfg["eval"]["split"] = args.split
    if getattr(args, "excel_output", None) is not None:
        cfg["excel"]["output"] = args.excel_output
    if getattr(args, "export_after_train", False):
        cfg["export"]["enabled"] = True
    return cfg


def validate_config(cfg, require_model=True):
    """Validate config — tinh thần giống Config.validate_config() của nhánh gốc."""
    if require_model and not cfg.get("model"):
        raise ValueError(
            "Chưa set model. Đặt `model:` trong configs/detection_voc.yaml "
            "(ví dụ: yolov8n.pt, yolo11n.pt, yolov5nu.pt, hoặc .pt/.yaml custom) "
            "hoặc truyền --model khi chạy CLI."
        )

    if not cfg.get("data"):
        raise ValueError("`data` must not be empty (VOC.yaml hoặc path tới data.yaml custom)")

    if int(cfg["epochs"]) <= 0:
        raise ValueError("`epochs` must be positive")

    if int(cfg["imgsz"]) <= 0:
        raise ValueError("`imgsz` must be positive")

    # batch = -1 hợp lệ (auto-batch của Ultralytics); chỉ cấm 0
    if int(cfg["batch"]) == 0:
        raise ValueError("`batch` must be positive, or -1 for Ultralytics auto-batch")

    if int(cfg["workers"]) < 0:
        raise ValueError("`workers` must be non-negative")

    split = (cfg.get("eval") or {}).get("split")
    if split not in VALID_EVAL_SPLITS:
        raise ValueError(f"eval.split must be one of {VALID_EVAL_SPLITS}, got: {split!r}")

    export_cfg = cfg.get("export") or {}
    if export_cfg.get("enabled") and not export_cfg.get("format"):
        raise ValueError("export.enabled=true requires export.format (e.g. onnx, engine)")

    print("[OK] Detection config validated successfully")
    print(f"  Model : {cfg.get('model') or '(chưa set — bắt buộc cho train)'}")
    print(f"  Data  : {cfg['data']}")
    print(f"  Epochs: {cfg['epochs']} | imgsz: {cfg['imgsz']} | batch: {cfg['batch']}")
    print(f"  Eval split: {split or 'mặc định theo data.yaml (VOC: val ≡ test — VOC2007)'}")
    return cfg
