"""
Trích xuất & in metrics detection từ object DetMetrics mà Ultralytics trả về
sau model.val() / model.train().

KHÔNG tự tính lại mAP — mọi số liệu đọc trực tiếp từ results.box.* của
Ultralytics (map50, map, mp, mr, per-class AP), đúng yêu cầu dùng API có sẵn.
"""


def extract_overall_metrics(metrics):
    """Trả về dict metrics overall từ DetMetrics (metrics.box.*)."""
    box = metrics.box
    overall = {
        "Precision": float(box.mp),          # mean precision (mọi class)
        "Recall": float(box.mr),             # mean recall (mọi class)
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
    Trả về list dict per-class (P, R, AP@0.5, AP@0.5:0.95) từ DetMetrics.

    box.ap_class_index là danh sách class-id THỰC SỰ xuất hiện trong tập eval;
    box.class_result(i) trả (p, r, ap50, ap) cho phần tử thứ i của danh sách đó.
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
    """In metrics detection ra console (format banner giống nhánh classification)."""
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
