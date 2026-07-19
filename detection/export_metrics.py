"""
Export metrics detection ra Excel — 1 file .xlsx, 2 sheet:

- Sheet "Summary" : metrics cuối cùng (overall + AP per class) + thông tin run
                    (model, imgsz, epochs, batch, dataset, ngày chạy, ...).
- Sheet "PerEpoch": metrics theo từng epoch để theo dõi training curve —
                    parse từ file results.csv mà Ultralytics TỰ SINH trong
                    thư mục run sau khi train (không tự tính lại gì cả).

Tái sử dụng pattern pd.ExcelWriter + engine='openpyxl' như save_model_results /
export_run_config của nhánh Strategy2_TinyImageNet.
"""
import os
from datetime import datetime

import pandas as pd

from detection.metrics_utils import extract_overall_metrics, extract_per_class_metrics

# Ghi thẳng vào Excel để người đọc số liệu không hiểu nhầm val là tập độc lập.
VAL_TEST_NOTE = (
    "Ultralytics VOC.yaml: split 'val' và 'test' TRÙNG NHAU (đều là VOC2007 test, "
    "4952 ảnh). Metrics 'val' KHÔNG đo trên validation set độc lập với test."
)


def read_results_csv(run_dir):
    """Đọc results.csv Ultralytics sinh ra trong run_dir → DataFrame per-epoch."""
    csv_path = os.path.join(str(run_dir), "results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"results.csv not found in run dir: {run_dir}")
    df = pd.read_csv(csv_path)
    # Bản Ultralytics cũ pad khoảng trắng trong header → chuẩn hóa tên cột
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _summary_from_results_csv(per_epoch_df):
    """
    Fallback khi không có DetMetrics (export offline từ run dir):
    lấy metrics overall từ ROW CUỐI của results.csv. Không có per-class AP —
    muốn có per-class thì chạy `eval` (model.val()) với weights đã train.
    """
    last = per_epoch_df.iloc[-1]
    column_map = {
        "Precision": "metrics/precision(B)",
        "Recall": "metrics/recall(B)",
        "mAP@0.5": "metrics/mAP50(B)",
        "mAP@0.5:0.95": "metrics/mAP50-95(B)",
    }
    overall = {}
    for metric_name, column in column_map.items():
        if column in per_epoch_df.columns:
            overall[metric_name] = float(last[column])
    return overall


def export_run_to_excel(run_dir, metrics=None, cfg=None, output_path=None):
    """
    Xuất Excel 2 sheet (Summary + PerEpoch) cho một run detection.

    Args:
        run_dir: thư mục run của Ultralytics (chứa results.csv, weights/, ...)
        metrics: DetMetrics từ model.val()/model.train(); nếu None → Summary
                 fallback từ row cuối results.csv (không có per-class AP)
        cfg: dict config của run (để ghi block Run Info)
        output_path: đường dẫn .xlsx; None → cfg['excel']['output'] hoặc
                     <run_dir>/detection_results.xlsx

    Returns:
        Đường dẫn file Excel đã ghi.
    """
    run_dir = str(run_dir)
    cfg = cfg or {}

    per_epoch_df = None
    try:
        per_epoch_df = read_results_csv(run_dir)
    except FileNotFoundError:
        print(f"  ⚠ Không tìm thấy results.csv trong {run_dir} — bỏ qua sheet PerEpoch")

    if metrics is not None:
        overall = extract_overall_metrics(metrics)
        per_class = extract_per_class_metrics(metrics)
        summary_source = "model.val() — Ultralytics DetMetrics"
    elif per_epoch_df is not None:
        overall = _summary_from_results_csv(per_epoch_df)
        per_class = []
        summary_source = (
            "results.csv (epoch cuối) — chạy `eval` với weights để có per-class AP"
        )
    else:
        raise ValueError(
            "Không có DetMetrics lẫn results.csv — không có gì để export Excel"
        )

    # ----- Block Run Info (pattern Parameter/Value như export_run_config) -----
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except ImportError:
        ultralytics_version = "N/A"

    eval_split = (cfg.get("eval") or {}).get("split")
    run_info_rows = [
        ("model", cfg.get("model", "N/A")),
        ("data", cfg.get("data", "N/A")),
        ("epochs", cfg.get("epochs", "N/A")),
        ("imgsz", cfg.get("imgsz", "N/A")),
        ("batch", cfg.get("batch", "N/A")),
        ("device", cfg.get("device") if cfg.get("device") is not None else "auto"),
        ("seed", cfg.get("seed", "N/A")),
        ("eval_split", eval_split or "mặc định theo data.yaml ('val')"),
        ("run_dir", run_dir),
        ("summary_source", summary_source),
        ("export_date", datetime.now().isoformat(timespec="seconds")),
        ("ultralytics_version", ultralytics_version),
        ("NOTE val/test", VAL_TEST_NOTE),
    ]

    if output_path is None:
        output_path = (cfg.get("excel") or {}).get("output") or os.path.join(
            run_dir, "detection_results.xlsx"
        )
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    info_df = pd.DataFrame(run_info_rows, columns=["Parameter", "Value"])
    overall_df = pd.DataFrame(list(overall.items()), columns=["Metric", "Value"])
    per_class_df = pd.DataFrame(per_class)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1 "Summary": 3 block xếp dọc — Run Info → Overall → Per-class AP
        info_df.to_excel(writer, sheet_name="Summary", index=False, startrow=0)
        next_row = len(info_df) + 2
        overall_df.to_excel(writer, sheet_name="Summary", index=False, startrow=next_row)
        next_row += len(overall_df) + 2
        if not per_class_df.empty:
            per_class_df.to_excel(
                writer, sheet_name="Summary", index=False, startrow=next_row
            )

        # Sheet 2 "PerEpoch": toàn bộ cột của results.csv (epoch, train/val
        # box_loss, cls_loss, dfl_loss, precision, recall, mAP50, mAP50-95, lr...)
        if per_epoch_df is not None:
            per_epoch_df.to_excel(writer, sheet_name="PerEpoch", index=False)

    print(f"  ✓ Excel exported: {output_path}")
    print(f"    - Sheet 'Summary' : overall + {len(per_class_df)} per-class AP + run info")
    if per_epoch_df is not None:
        print(f"    - Sheet 'PerEpoch': {len(per_epoch_df)} epochs (từ results.csv)")
    else:
        print(f"    - Sheet 'PerEpoch': bỏ qua (không có results.csv)")
    return output_path
