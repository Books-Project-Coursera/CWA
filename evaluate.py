"""
Evaluation cho Strategy 2 - Object Detection (Ultralytics YOLO + Pascal VOC).

- Metrics đọc TRỰC TIẾP từ DetMetrics của Ultralytics (results.box.*:
  map50, map, mp, mr, per-class AP, fitness) — không tự tính lại mAP.
- Strategy 1: best.pt (checkpoint fitness cao nhất trên val', Ultralytics tự chọn).
- Strategy 2: average weights của Top-K checkpoint tốt nhất trên val'
  (giống average_weights của nhánh classification, áp dụng cho ckpt YOLO).
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
    "chọn best.pt / rank Top-K checkpoint. Test = VOC2007 test (4952 ảnh), giữ nguyên "
    "làm hold-out, chỉ dùng cho báo cáo cuối."
)
VAL_TEST_WARNING = (
    "VAL_RATIO=0: dùng nguyên data.yaml gốc. Với VOC.yaml của Ultralytics, split 'val' "
    "và 'test' TRÙNG NHAU (đều là VOC2007 test) — không có val độc lập, checkpoint được "
    "chọn trên chính tập test (nguy cơ leakage khi đọc số liệu)."
)


# ==================== Metrics extraction (từ DetMetrics) ====================

def extract_overall_metrics(metrics):
    """Trả về dict metrics overall từ DetMetrics (metrics.box.*)."""
    box = metrics.box
    overall = {
        "Precision": float(box.mp),
        "Recall": float(box.mr),
        "mAP@0.5": float(box.map50),
        "mAP@0.75": float(box.map75),
        "mAP@0.5:0.95": float(box.map),
    }
    # fitness = 0.1*mAP50 + 0.9*mAP50-95 (định nghĩa của Ultralytics)
    fitness = getattr(metrics, "fitness", None)
    if fitness is not None:
        overall["Fitness"] = float(fitness)
    return overall


def extract_per_class_metrics(metrics):
    """
    List dict per-class (P, R, AP@0.5, AP@0.5:0.95) từ DetMetrics.
    box.ap_class_index = các class-id thực sự xuất hiện trong tập eval;
    box.class_result(i) trả (p, r, ap50, ap) cho phần tử thứ i.
    """
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


def print_detection_metrics(metrics, header="DETECTION EVALUATION RESULTS"):
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
    Rank các checkpoint epoch còn trên disk theo fitness trên val' (giảm dần).

    Nguồn chính: strategy2_checkpoints.json (TopKCheckpointManager ghi lúc train).
    Fallback: tự tính fitness = 0.1*mAP50 + 0.9*mAP50-95 từ results.csv
    (khi run dir được copy từ máy khác mà thiếu file json).

    Returns:
        list[(Path, fitness, epoch)] sorted theo fitness giảm dần.
    """
    weights_dir = Path(run_dir) / "weights"
    ranking_path = weights_dir / RANKING_FILE
    records = []

    if ranking_path.exists():
        data = json.loads(ranking_path.read_text())
        for fname, info in data.items():
            path = weights_dir / fname
            if path.exists():
                records.append((path, float(info["fitness"]), int(info["epoch"])))
    else:
        try:
            df = read_results_csv(run_dir)
        except FileNotFoundError:
            return []
        fitness_by_epoch = {}
        if "metrics/mAP50(B)" in df.columns and "metrics/mAP50-95(B)" in df.columns:
            for _, row in df.iterrows():
                fitness_by_epoch[int(row["epoch"])] = (
                    0.1 * float(row["metrics/mAP50(B)"]) + 0.9 * float(row["metrics/mAP50-95(B)"])
                )
        for path in weights_dir.glob("epoch*.pt"):
            digits = re.sub(r"\D", "", path.stem)
            if not digits:
                continue
            file_epoch = int(digits)
            # Tên file epoch{N}.pt đánh số 0-based, cột epoch results.csv 1-based
            fitness = fitness_by_epoch.get(file_epoch + 1, fitness_by_epoch.get(file_epoch))
            if fitness is not None:
                records.append((path, float(fitness), file_epoch))

    records.sort(key=lambda r: (-r[1], -r[2]))
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
      được sort là ckpt có fitness cao nhất).
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

    print(f"      Updating BN stats bằng {num_batches} batch train (device={device_obj})...")
    with torch.no_grad():
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


def run_strategy_evaluation(run_dir, data=None, split=None):
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

    export_to_excel(run_dir, strategy_results=results, data=data, split=split)
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


def export_to_excel(run_dir, strategy_results=None, data=None, split=None, output_path=None):
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
        ("seed", Config.RANDOM_SEED),
        ("eval_split", split or Config.EVAL_SPLIT or "mặc định theo data.yaml"),
        ("strategy2", f"Top-K {Config.TOP_K_VALUES}" if Config.USE_STRATEGY2 else "OFF"),
        ("run_dir", run_dir),
        ("summary_source", summary_source),
        ("export_date", datetime.now().isoformat(timespec="seconds")),
        ("ultralytics_version", ultralytics_version),
        ("NOTE data split", _data_note(data)),
    ]

    if output_path is None:
        output_path = Config.EXCEL_OUTPUT or os.path.join(run_dir, "detection_results.xlsx")
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
