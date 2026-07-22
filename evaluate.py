"""
Evaluation cho Strategy 2 - Instance Segmentation (Ultralytics YOLO + Carparts).

- Metrics đọc TRỰC TIẾP từ SegmentationMetrics của Ultralytics (results.mask.*:
  map50, map, mp, mr, per-class AP, fitness) — không tự tính lại mask AP.
- Strategy 1: best.pt (checkpoint fitness cao nhất trên val', Ultralytics tự chọn).
- Strategy 2: average weights của Top-K checkpoint tốt nhất trên val'
  (giống average_weights của nhánh classification, áp dụng cho ckpt YOLO-seg).
- Export Excel 2 sheet: "Summary" (run info + overall + per-class theo strategy)
  và "PerEpoch" (parse results.csv do Ultralytics tự sinh trong run dir).
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from config import Config

# File ranking do TopKCheckpointManager (train.py) ghi trong <run_dir>/weights/
RANKING_FILE = "strategy2_checkpoints.json"

INDEPENDENT_VAL_NOTE = (
    "Validation set (val') ĐỘC LẬP được tách từ train theo VAL_RATIO — chỉ dùng để "
    "chọn best.pt / rank Top-K checkpoint. Test set (hold-out), giữ nguyên "
    "làm hold-out, chỉ dùng cho báo cáo cuối."
)
VAL_TEST_WARNING = (
    "VAL_RATIO=0: dùng nguyên data.yaml gốc. Nếu dataset không có val độc lập "
    "(val ≡ test), checkpoint được chọn trên chính tập test (nguy cơ leakage)."
)


# ==================== Metrics extraction (từ SegmentationMetrics / DetMetrics) ====================

def extract_overall_metrics(metrics):
    """
    Trả về dict metrics overall từ SegmentationMetrics hoặc DetMetrics.
    - Segmentation: extract từ metrics.mask.*
    - Detection: extract từ metrics.box.*
    """
    # Try segmentation metrics first
    if hasattr(metrics, 'mask') and metrics.mask is not None:
        mask = metrics.mask
        overall = {
            "Precision": float(mask.mp),
            "Recall": float(mask.mr),
            "mAP@0.5": float(mask.map50),
            "mAP@0.75": float(mask.map75),
            "mAP@0.5:0.95": float(mask.map),
        }
    # Fallback to detection metrics (box)
    elif hasattr(metrics, 'box') and metrics.box is not None:
        box = metrics.box
        overall = {
            "Precision": float(box.mp),
            "Recall": float(box.mr),
            "mAP@0.5": float(box.map50),
            "mAP@0.75": float(box.map75),
            "mAP@0.5:0.95": float(box.map),
        }
    else:
        overall = {"Precision": 0, "Recall": 0, "mAP@0.5": 0, "mAP@0.75": 0, "mAP@0.5:0.95": 0}
    
    # fitness = 0.1*mAP50 + 0.9*mAP50-95 (định nghĩa của Ultralytics)
    fitness = getattr(metrics, "fitness", None)
    if fitness is not None:
        overall["Fitness"] = float(fitness)
    return overall


def extract_per_class_metrics(metrics):
    """
    List dict per-class (P, R, AP@0.5, AP@0.5:0.95) từ SegmentationMetrics hoặc DetMetrics.
    - Segmentation: extract từ metrics.mask
    - Detection: extract từ metrics.box
    """
    # Try segmentation metrics first
    if hasattr(metrics, 'mask') and metrics.mask is not None:
        mask = metrics.mask
        names = getattr(metrics, "names", {}) or {}
        rows = []
        for i, class_idx in enumerate(getattr(mask, "ap_class_index", [])):
            class_idx = int(class_idx)
            p, r, ap50, ap = mask.class_result(i)
            rows.append({
                "Class ID": class_idx,
                "Class": str(names.get(class_idx, class_idx)),
                "Precision": float(p),
                "Recall": float(r),
                "AP@0.5": float(ap50),
                "AP@0.5:0.95": float(ap),
            })
        return rows
    # Fallback to detection metrics (box)
    elif hasattr(metrics, 'box') and metrics.box is not None:
        box = metrics.box
        names = getattr(metrics, "names", {}) or {}
        rows = []
        for i, class_idx in enumerate(getattr(box, "ap_class_index", [])):
            class_idx = int(class_idx)
            p, r, ap50, ap = box.class_result(i)
            rows.append({
                "Class ID": class_idx,
                "Class": str(names.get(class_idx, class_idx)),
                "Precision": float(p),
                "Recall": float(r),
                "AP@0.5": float(ap50),
                "AP@0.5:0.95": float(ap),
            })
        return rows
    else:
        return []


def print_detection_metrics(metrics, header="SEGMENTATION EVALUATION RESULTS"):
    """In metrics detection ra console (format banner giống repo gốc)."""
    overall = extract_overall_metrics(metrics)
    per_class = extract_per_class_metrics(metrics)

    print("\n" + "=" * 70)
    print(f" {header}")
    print("=" * 70)
    for key, value in overall.items():
        print(f"  {key:<14}: {value:.4f}")

    if per_class:
        print("-" * 70)
        print(f"  {'Class':<18} {'Precision':>10} {'Recall':>10} {'AP@0.5':>10} {'AP@0.5:0.95':>12}")
        for row in per_class:
            print(
                f"  {row['Class']:<18} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
                f"{row['AP@0.5']:>10.4f} {row['AP@0.5:0.95']:>12.4f}"
            )
    print("=" * 70)
    return overall, per_class


# ==================== model.val() wrapper ====================

def build_val_args(data, split=None):
    """Map Config → kwargs của model.val()."""
    val_args = {
        "data": str(data),
        "imgsz": int(Config.IMGSZ),
        "workers": int(Config.WORKERS),
    }
    # auto-batch (-1) chỉ dành cho train → khi val dùng mặc định nếu BATCH=-1
    if int(Config.BATCH) > 0:
        val_args["batch"] = int(Config.BATCH)
    if Config.DEVICE is not None:
        val_args["device"] = Config.DEVICE
    if split:  # None = dùng split mặc định của data.yaml ('val')
        val_args["split"] = split
    if Config.CONF is not None:
        val_args["conf"] = float(Config.CONF)
    if Config.IOU is not None:
        val_args["iou"] = float(Config.IOU)
    return val_args


def evaluate_weights(weights, data, split=None, header=None):
    """Chạy model.val() với weights đã train, in metrics, trả về DetMetrics."""
    from ultralytics import YOLO

    if not weights:
        raise ValueError(
            "Cần weights đã train để eval: truyền --weights path/to/best.pt "
            "(hoặc set Config.MODEL trỏ tới weights đã train trên VOC)."
        )

    model = YOLO(str(weights))
    metrics = model.val(**build_val_args(data, split))
    print_detection_metrics(
        metrics, header=header or f"EVALUATION (split={split or 'default'}) — {Path(str(weights)).name}"
    )
    return metrics


# ==================== Strategy 2: ranking + weight averaging ====================

def rank_checkpoints(run_dir):
    """
    Rank các checkpoint epoch còn trên disk theo val_loss trên val' (tăng dần).
    Checkpoint có val_loss nhỏ nhất = tốt nhất, giống early stopping của Ultralytics.

    Nguồn chính: strategy2_checkpoints.json (TopKCheckpointManager ghi lúc train).
    Fallback: đọc val/loss từ results.csv (khi run dir được copy từ máy khác mà thiếu file json).

    Returns:
        list[(Path, val_loss, epoch)] sorted theo val_loss tăng dần (nhỏ nhất đứng đầu).
    """
    weights_dir = Path(run_dir) / "weights"
    ranking_path = weights_dir / RANKING_FILE
    records = []

    if ranking_path.exists():
        data = json.loads(ranking_path.read_text())
        for fname, info in data.items():
            path = weights_dir / fname
            if path.exists():
                records.append((path, float(info["val_loss"]), int(info["epoch"])))
    else:
        try:
            df = read_results_csv(run_dir)
        except FileNotFoundError:
            return []
        val_loss_by_epoch = {}
        # Ultralytics lưu val loss dưới cột "val/loss" hoặc tương tự
        val_loss_col = None
        for col in df.columns:
            if "val" in col.lower() and "loss" in col.lower():
                val_loss_col = col
                break
        
        if val_loss_col:
            for _, row in df.iterrows():
                try:
                    val_loss_by_epoch[int(row["epoch"])] = float(row[val_loss_col])
                except (ValueError, KeyError):
                    pass
        
        for path in weights_dir.glob("epoch*.pt"):
            digits = re.sub(r"\D", "", path.stem)
            if not digits:
                continue
            file_epoch = int(digits)
            # Tên file epoch{N}.pt đánh số 0-based, cột epoch results.csv 1-based
            val_loss = val_loss_by_epoch.get(file_epoch + 1, val_loss_by_epoch.get(file_epoch))
            if val_loss is not None:
                records.append((path, float(val_loss), file_epoch))

    # Sort theo val_loss TĂNG DẦN (nhỏ nhất = tốt nhất)
    records.sort(key=lambda r: (r[1], -r[2]))
    return records


def average_checkpoints(ckpt_paths, output_path):
    """
    Average weights của nhiều checkpoint YOLO — tương đương `average_weights`
    của nhánh Strategy2_TinyImageNet, áp dụng cho ckpt Ultralytics.

    Cùng nguyên tắc với classification:
    - Average TẤT CẢ learnable parameter (float): weights, biases, γ/β của BN.
    - KHÔNG average BN running statistics (`running_mean`, `running_var`,
      `num_batches_tracked`) — đây là population stats, không phải learned;
      average chúng làm BN lệch phân phối → giữ nguyên từ checkpoint ĐẦU (đã
      được sort là ckpt có val_loss nhỏ nhất, do đó tốt nhất).
    - Sau khi average, BN stats KHÔNG khớp với weights mới → phải chạy
      update_bn_stats() để re-estimate trên train (xem hàm bên dưới).

    Ckpt input dùng EMA weights (phần Ultralytics thực sự deploy). Ckpt output
    chỉ chứa model (bỏ optimizer) nên nhẹ, load lại bằng YOLO(path) như ckpt
    thường.
    """
    import torch

    ckpts = [torch.load(str(p), map_location="cpu", weights_only=False) for p in ckpt_paths]
    # YOLO(path) load (ckpt['ema'] or ckpt['model']) → average đúng phần EMA
    modules = [(ck.get("ema") or ck["model"]).float() for ck in ckpts]
    state_dicts = [m.state_dict() for m in modules]

    # Skip BN running stats — giống keys_to_keep của Strategy2_TinyImageNet
    def is_bn_stat(key):
        return any(marker in key for marker in ("running_mean", "running_var", "num_batches_tracked"))

    # Short-circuit K=1: khớp hành vi của average_weights bên TinyImageNet
    if len(ckpt_paths) == 1:
        avg_state = {k: v.clone() for k, v in state_dicts[0].items()}
    else:
        avg_state = {}
        for key, ref_tensor in state_dicts[0].items():
            if is_bn_stat(key) or not ref_tensor.dtype.is_floating_point:
                # Giữ nguyên từ checkpoint đầu (đã sort theo fitness giảm dần)
                avg_state[key] = ref_tensor.clone()
            else:
                avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)

    merged_module = modules[0]
    merged_module.load_state_dict(avg_state)

    torch.save(
        {
            "model": merged_module.half(),
            "ema": None,
            "optimizer": None,
            "epoch": -1,
            "train_args": ckpts[0].get("train_args", {}),
            "date": datetime.now().isoformat(timespec="seconds"),
        },
        str(output_path),
    )
    return Path(output_path)


def update_bn_stats(weights_path, data_yaml, num_batches=None, device=None):
    """
    Re-estimate BN running statistics của model đã average — tương đương
    update_bn() của nhánh Strategy2_TinyImageNet, thích ứng cho YOLO.

    Vì sao: sau average_checkpoints() BN running_mean/running_var được giữ từ
    ckpt tốt nhất (không average) nhưng weights của các layer trước BN đã đổi
    → phân phối activation lệch với running stats cũ. Chạy forward pass trên
    train (BN ở mode 'train', momentum=None → cumulative moving average) để
    tính lại running stats khớp weights mới. Ghi đè lại `weights_path`.

    Args:
        weights_path: file .pt đã average (do `average_checkpoints` sinh ra).
        data_yaml: path data yaml (holdout hoặc gốc) — dùng SPLIT TRAIN để
                   ước lượng BN, không đụng val'/test.
        num_batches: số batch forward (mặc định Config.BN_UPDATE_BATCHES).
        device: None = auto GPU nếu có.

    Returns:
        weights_path (đã ghi đè với BN stats mới).
    """
    import torch
    from torch.nn.modules.batchnorm import _BatchNorm
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    weights_path = Path(weights_path)
    num_batches = int(num_batches or Config.BN_UPDATE_BATCHES)

    yolo = YOLO(str(weights_path))
    model = yolo.model
    bn_modules = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    if not bn_modules:
        print("      (Không tìm thấy BN layer nào, bỏ qua update_bn)")
        return weights_path

    device_obj = torch.device(
        device
        or (f"cuda:{Config.DEVICE}" if isinstance(Config.DEVICE, str) and Config.DEVICE.isdigit()
            else "cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.float().to(device_obj)
    stride = int(max(model.stride)) if hasattr(model, "stride") else 32

    # Build train dataloader kiểu Ultralytics (đúng augmentation train)
    data = check_det_dataset(str(data_yaml))
    batch = int(Config.BATCH) if int(Config.BATCH) > 0 else 16
    cfg = get_cfg(overrides={"imgsz": int(Config.IMGSZ), "task": "detect"})
    dataset = build_yolo_dataset(
        cfg, data["train"], batch, data, mode="train", rect=False, stride=stride
    )
    loader = build_dataloader(
        dataset, batch, workers=int(Config.WORKERS), shuffle=True, rank=-1
    )

    # Reset BN running stats + đổi momentum=None (cumulative moving average)
    saved_momentum = {}
    for m in bn_modules:
        saved_momentum[m] = m.momentum
        m.reset_running_stats()
        m.momentum = None

    # Toàn model ở eval() để tắt dropout/random aug ở path forward, riêng BN
    # bật train() để tích lũy running stats — pattern giống update_bn của
    # nhánh classification.
    was_training = model.training
    model.eval()
    for m in bn_modules:
        m.train()

    # Autocast khớp precision train (repo gốc dùng autocast_context BF16 khi
    # USE_AMP=True + CUDA). BN chỉ tích lũy mean/var nên không nhạy cảm với
    # precision, nhưng match train precision cho nhất quán và nhanh hơn trên
    # GPU. Không ảnh hưởng correctness khi Config.AMP=False.
    if getattr(Config, "AMP", True) and device_obj.type == "cuda":
        autocast_ctx = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    else:
        autocast_ctx = torch.autocast(device_type=device_obj.type, enabled=False)

    print(f"      Updating BN stats bằng {num_batches} batch train (device={device_obj})...")
    with torch.no_grad(), autocast_ctx:
        for i, batch_data in enumerate(loader):
            if i >= num_batches:
                break
            img = batch_data["img"].to(device_obj, non_blocking=True).float() / 255.0
            _ = model(img)

    for m, mom in saved_momentum.items():
        m.momentum = mom
    model.train(was_training)

    # Ghi đè weights_path với BN stats mới (giữ nguyên cấu trúc ckpt)
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    ckpt["model"] = model.half().cpu()
    torch.save(ckpt, str(weights_path))
    print(f"      ✓ BN stats updated: {weights_path.name}")
    return weights_path


def run_strategy_evaluation(run_dir, data=None, split=None, seed=None):
    """
    Đánh giá Strategy 1 (best.pt) và Strategy 2 (Top-K average) trên split
    báo cáo (mặc định Config.EVAL_SPLIT='test' — VOC2007 test), in bảng so
    sánh và export Excel.

    Returns:
        dict {strategy_name: DetMetrics}
    """
    run_dir = Path(run_dir)
    data = data or resolve_data_from_run(run_dir) or Config.DATA
    split = split or Config.EVAL_SPLIT

    print("\n" + "=" * 70)
    print(f" STRATEGY EVALUATION (split={split or 'default'}, data={data})")
    print("=" * 70)

    results = {}

    # ----- Strategy 1: best.pt (fitness cao nhất trên val') -----
    best_weights = run_dir / "weights" / "best.pt"
    if best_weights.exists():
        results["Strategy 1 (best.pt)"] = evaluate_weights(
            best_weights, data, split, header=f"Strategy 1 — best.pt (split={split})"
        )
    else:
        print(f"  ⚠ Không tìm thấy {best_weights} — bỏ qua Strategy 1")

    # ----- Strategy 2: average Top-K checkpoint tốt nhất trên val' -----
    if Config.USE_STRATEGY2:
        ranked = rank_checkpoints(run_dir)
        if not ranked:
            print("  ⚠ Không tìm thấy checkpoint epoch nào để average — bỏ qua Strategy 2")
            print("    (cần train với USE_STRATEGY2=True để lưu Top-K checkpoint)")
        else:
            print(f"\n  Checkpoint khả dụng (rank theo fitness val'): "
                  f"{[(p.name, round(f, 4)) for p, f, _ in ranked]}")
            for k in Config.TOP_K_VALUES:
                k = int(k)
                if k > len(ranked):
                    print(f"  ⚠ Top-{k}: chỉ có {len(ranked)} checkpoint — bỏ qua")
                    continue
                avg_path = run_dir / "weights" / f"strategy2_top{k}_avg.pt"
                average_checkpoints([p for p, _, _ in ranked[:k]], avg_path)
                # CRITICAL: BN running stats bị giữ nguyên khi average → phải
                # re-estimate trên train trước khi val, y hệt update_bn của
                # Strategy2_TinyImageNet.
                if Config.USE_BN_UPDATE:
                    update_bn_stats(avg_path, data)
                results[f"Strategy 2 (Top-{k} avg)"] = evaluate_weights(
                    avg_path, data, split, header=f"Strategy 2 — Top-{k} average (split={split})"
                )

    if not results:
        print("  ✗ Không có strategy nào được đánh giá")
        return results

    # ----- Bảng so sánh (giống format tổng hợp của repo gốc) -----
    print("\n" + "=" * 70)
    print(f" STRATEGY COMPARISON (split={split or 'default'})")
    print("=" * 70)
    for name, metrics in results.items():
        m = extract_overall_metrics(metrics)
        print(
            f"  {name:<26} mAP50: {m['mAP@0.5']:.4f} | mAP50-95: {m['mAP@0.5:0.95']:.4f} | "
            f"P: {m['Precision']:.4f} | R: {m['Recall']:.4f}"
        )
    best_name = max(results, key=lambda n: extract_overall_metrics(results[n])["mAP@0.5:0.95"])
    print(f"\n🏆 Best strategy (mAP@0.5:0.95): {best_name}")

    export_to_excel(run_dir, strategy_results=results, data=data, split=split, seed=seed)
    return results


def resolve_data_from_run(run_dir):
    """Lấy path data yaml từ args.yaml mà Ultralytics lưu trong run dir."""
    args_path = Path(run_dir) / "args.yaml"
    if args_path.exists():
        return (yaml.safe_load(args_path.read_text()) or {}).get("data")
    return None


def infer_run_dir(weights):
    """Suy run dir từ vị trí weights chuẩn Ultralytics: <run_dir>/weights/x.pt."""
    weights_path = Path(str(weights))
    if weights_path.parent.name == "weights":
        return weights_path.parent.parent
    return None


# ==================== Excel export (Summary + PerEpoch) ====================

def read_results_csv(run_dir):
    """Đọc results.csv Ultralytics sinh trong run_dir → DataFrame per-epoch."""
    csv_path = os.path.join(str(run_dir), "results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"results.csv not found in run dir: {run_dir}")
    df = pd.read_csv(csv_path)
    # Bản Ultralytics cũ pad khoảng trắng trong header → chuẩn hóa tên cột
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _data_note(data):
    """Chọn note đúng theo cách chuẩn bị data (holdout hay data.yaml gốc)."""
    if "holdout" in os.path.basename(str(data)):
        return INDEPENDENT_VAL_NOTE
    return VAL_TEST_WARNING


def export_to_excel(run_dir, strategy_results=None, data=None, split=None, output_path=None, seed=None):
    """
    Xuất Excel 2 sheet cho một run detection:

    - "Summary" : run info + bảng overall metrics theo strategy + per-class AP
                  theo strategy (cột Strategy — giống Per-Class sheet repo gốc).
                  Nếu strategy_results=None (export offline) → overall lấy từ
                  row cuối results.csv, không có per-class AP.
    - "PerEpoch": toàn bộ results.csv (box/cls/dfl loss train+val, P, R,
                  mAP50, mAP50-95, lr... — mỗi epoch 1 row).
    """
    run_dir = str(run_dir)
    data = data or Config.DATA

    per_epoch_df = None
    try:
        per_epoch_df = read_results_csv(run_dir)
    except FileNotFoundError:
        print(f"  ⚠ Không tìm thấy results.csv trong {run_dir} — bỏ qua sheet PerEpoch")

    # ----- Bảng overall + per-class theo strategy -----
    overall_rows, per_class_rows = [], []
    if strategy_results:
        for name, metrics in strategy_results.items():
            overall_rows.append({"Strategy": name, **extract_overall_metrics(metrics)})
            for row in extract_per_class_metrics(metrics):
                per_class_rows.append({"Strategy": name, **row})
        summary_source = "model.val() — Ultralytics DetMetrics"
    elif per_epoch_df is not None:
        last = per_epoch_df.iloc[-1]
        column_map = {
            "Precision": "metrics/precision(B)",
            "Recall": "metrics/recall(B)",
            "mAP@0.5": "metrics/mAP50(B)",
            "mAP@0.5:0.95": "metrics/mAP50-95(B)",
        }
        row = {"Strategy": "Last epoch (results.csv)"}
        for metric_name, column in column_map.items():
            if column in per_epoch_df.columns:
                row[metric_name] = float(last[column])
        overall_rows.append(row)
        summary_source = "results.csv (epoch cuối) — chạy `strategies`/`eval` để có per-class AP"
    else:
        raise ValueError("Không có strategy results lẫn results.csv — không thể export Excel")

    # ----- Block Run Info (pattern Parameter/Value như repo gốc) -----
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except ImportError:
        ultralytics_version = "N/A"

    run_info_rows = [
        ("model", Config.MODEL or "N/A"),
        ("data", str(data)),
        ("val_ratio (tách từ train)", Config.VAL_RATIO),
        ("epochs", Config.EPOCHS),
        ("imgsz", Config.IMGSZ),
        ("batch", Config.BATCH),
        ("device", Config.DEVICE if Config.DEVICE is not None else "auto"),
        # Ưu tiên seed thực sự dùng cho run này (tham số seed), fallback Config
        ("seed", seed if seed is not None else Config.RANDOM_SEED),
        ("cls_loss", Config.LOSS_FUNCTION + (
            f" (γ={Config.FOCAL_GAMMA}, α={Config.FOCAL_ALPHA})"
            if Config.LOSS_FUNCTION == "focal" else ""
        )),
        ("eval_split", split or Config.EVAL_SPLIT or "mặc định theo data.yaml"),
        ("strategy2", f"Top-K {Config.TOP_K_VALUES}" if Config.USE_STRATEGY2 else "OFF"),
        ("run_dir", run_dir),
        ("summary_source", summary_source),
        ("export_date", datetime.now().isoformat(timespec="seconds")),
        ("ultralytics_version", ultralytics_version),
        ("NOTE data split", _data_note(data)),
    ]

    if output_path is None:
        if seed is not None:
            default_name = f"detection_results_seed{seed}.xlsx"
        else:
            default_name = "detection_results.xlsx"
        output_path = Config.EXCEL_OUTPUT or os.path.join(run_dir, default_name)
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    info_df = pd.DataFrame(run_info_rows, columns=["Parameter", "Value"])
    overall_df = pd.DataFrame(overall_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1 "Summary": Run Info → Overall per strategy → Per-class per strategy
        info_df.to_excel(writer, sheet_name="Summary", index=False, startrow=0)
        next_row = len(info_df) + 2
        overall_df.to_excel(writer, sheet_name="Summary", index=False, startrow=next_row)
        next_row += len(overall_df) + 2
        if not per_class_df.empty:
            per_class_df.to_excel(writer, sheet_name="Summary", index=False, startrow=next_row)

        # Sheet 2 "PerEpoch"
        if per_epoch_df is not None:
            per_epoch_df.to_excel(writer, sheet_name="PerEpoch", index=False)

    print(f"\n  ✓ Excel exported: {output_path}")
    print(f"    - Sheet 'Summary' : {len(overall_rows)} strategy row(s) + "
          f"{len(per_class_rows)} per-class row(s) + run info")
    if per_epoch_df is not None:
        print(f"    - Sheet 'PerEpoch': {len(per_epoch_df)} epochs (từ results.csv)")
    else:
        print("    - Sheet 'PerEpoch': bỏ qua (không có results.csv)")
    return output_path


# ==================== Multi-seed summary export ====================

def export_multi_seed_summary(all_runs_results, output_dir=None):
    """
    Export file Excel tổng hợp tất cả seed runs vào <output_dir>/multi_seed_summary.xlsx.

    Sheets:
    - "Summary"     : mỗi row = 1 seed × 1 strategy; có cột Seed + Run Folder rõ ràng.
    - "Mean ± Std"  : mean/std của từng metric nhóm theo Strategy qua tất cả seeds.
    - "Per-Class AP": AP từng class, nhóm theo Seed × Run Folder × Strategy.

    Args:
        all_runs_results: list[dict] từ train_detector() — mỗi phần tử gồm
            {seed, run_dir, strategy_results, val_metrics, ...}.
        output_dir: thư mục chứa file tổng hợp (mặc định = Config.PROJECT).

    Returns:
        Path file .xlsx đã ghi, hoặc None nếu không có kết quả.
    """
    output_dir = output_dir or Config.PROJECT
    os.makedirs(str(output_dir), exist_ok=True)
    output_path = os.path.join(str(output_dir), "multi_seed_summary.xlsx")

    summary_rows = []   # 1 row per seed × strategy
    per_class_rows = [] # 1 row per seed × strategy × class

    for run in all_runs_results:
        seed_val = run["seed"]
        run_dir_str = run["run_dir"]
        strategy_results = run.get("strategy_results") or {}
        for strat_name, metrics in strategy_results.items():
            if metrics is None:
                continue
            overall = extract_overall_metrics(metrics)
            summary_rows.append({
                "Seed": seed_val,
                "Run Folder": run_dir_str,
                "Strategy": strat_name,
                **overall,
            })
            for row in extract_per_class_metrics(metrics):
                per_class_rows.append({
                    "Seed": seed_val,
                    "Run Folder": run_dir_str,
                    "Strategy": strat_name,
                    **row,
                })

    if not summary_rows:
        print("  ⚠ Không có kết quả nào để export multi-seed summary")
        return None

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    # Mean ± Std grouped by Strategy (giữ thứ tự xuất hiện lần đầu)
    metric_cols = ["mAP@0.5", "mAP@0.5:0.95", "mAP@0.75", "Precision", "Recall"]
    if "Fitness" in summary_df.columns:
        metric_cols.append("Fitness")
    metric_cols = [c for c in metric_cols if c in summary_df.columns]

    # Giữ thứ tự strategy theo thứ tự xuất hiện trong summary_df
    seen_strats = []
    for s in summary_df["Strategy"]:
        if s not in seen_strats:
            seen_strats.append(s)

    mean_std_rows = []
    for strat_name in seen_strats:
        grp = summary_df[summary_df["Strategy"] == strat_name]
        row = {"Strategy": strat_name, "N seeds": len(grp)}
        for col in metric_cols:
            row[f"{col} mean"] = round(float(grp[col].mean()), 6)
            row[f"{col} std"]  = round(float(grp[col].std()),  6)
        mean_std_rows.append(row)
    mean_std_df = pd.DataFrame(mean_std_rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer,   sheet_name="Summary",      index=False)
        mean_std_df.to_excel(writer,  sheet_name="Mean ± Std",   index=False)
        if not per_class_df.empty:
            per_class_df.to_excel(writer, sheet_name="Per-Class AP", index=False)

    seeds_done = [r["seed"] for r in all_runs_results]
    print(f"\n  ✓ Multi-seed summary exported: {output_path}")
    print(f"    Seeds           : {seeds_done}")
    print(f"    Sheet 'Summary'     : {len(summary_rows)} rows ({len(seen_strats)} strategy × {len(seeds_done)} seeds)")
    print(f"    Sheet 'Mean ± Std'  : {len(mean_std_rows)} strategy rows")
    if not per_class_df.empty:
        print(f"    Sheet 'Per-Class AP': {len(per_class_rows)} rows")
    return output_path
