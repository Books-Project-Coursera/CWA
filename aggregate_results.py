"""
aggregate_results.py

Collect every results/<model>/seed_<seed>/metrics.json and produce exactly 2 report files
under results/:

  1. cifar100_full_results.xlsx  — full per-run detail (every metric in metrics.json)
       - Sheet "Tong_hop" : one row per (model, seed) — ALL models x ALL seeds
       - Sheet "seed_<S>" : one row per model, for that seed only (one sheet per seed)
       6 sheets total (1 + 5 seeds), same column schema throughout.

  2. cifar100_report_format.xlsx (+ .csv) — paper-ready table, EXACTLY matching
     strategy2.pdf Tables 4-6: rows = Baseline (k=1)/k=2..5, cols = 6 models,
     cell = "mean ± std" test accuracy (%) over the 5 seeds, with (+Δ) vs baseline.
     No other columns — ready to paste into the report as-is.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from cifar100_configs import MODEL_ORDER, PRETTY_NAME, SEEDS, TOP_K_VALUES

HERE = os.path.dirname(os.path.abspath(__file__))
STRATEGIES = ["Baseline (k=1)"] + [f"k={k}" for k in TOP_K_VALUES]
ACC_METRIC = "Accuracy (%)"


def load_records(results_dir):
    """One dict per completed (model, seed) run, with every metric for every k."""
    records = []
    found = []
    for mj in glob.glob(os.path.join(results_dir, "*", "seed_*", "metrics.json")):
        with open(mj) as f:
            d = json.load(f)
        if d.get("dry_run"):
            continue
        model, seed = d["model"], d["seed"]
        found.append((model, seed))
        res = d["results"]
        base = res.get("Baseline (k=1)", {})
        row = {
            "Model": PRETTY_NAME.get(model, model),
            "Seed": seed,
            "Epochs Trained": d.get("epochs_trained"),
            "Train Seconds": d.get("train_seconds"),
            "Best Epoch (k=1)": base.get("best_epoch"),
        }
        for strat in STRATEGIES:
            r = res.get(strat, {})
            label = "Baseline" if strat == "Baseline (k=1)" else strat
            row[f"{label} Accuracy (%)"] = r.get("Accuracy (%)")
            if strat != "Baseline (k=1)":
                base_acc, this_acc = base.get("Accuracy (%)"), r.get("Accuracy (%)")
                row[f"{label} Δ Accuracy"] = (
                    round(this_acc - base_acc, 2) if base_acc is not None and this_acc is not None else None
                )
            row[f"{label} Precision (%)"] = r.get("Precision (%)")
            row[f"{label} Recall (%)"] = r.get("Recall (%)")
            row[f"{label} F1 (%)"] = r.get("F1-Score (%)")
            row[f"{label} AUC (%)"] = r.get("AUC (%)")
            row[f"{label} Test Loss"] = r.get("Test Loss")
            row[f"{label} Checkpoint Epochs"] = ",".join(str(e) for e in r.get("checkpoint_epochs", []))
        row["Best k"] = d.get("best_k")
        row["Best-k Accuracy (%)"] = d.get("best_k_accuracy")
        row["_model_key"] = model
        records.append(row)
    return records, sorted(set(found))


def to_df(records):
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["_model_order"] = df["_model_key"].apply(lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999)
    df["_seed_order"] = df["Seed"].apply(lambda s: SEEDS.index(s) if s in SEEDS else 999)
    df = df.sort_values(["_model_order", "_seed_order"]).drop(columns=["_model_key", "_model_order", "_seed_order"])
    return df.reset_index(drop=True)


def build_acc_lookup(results_dir):
    """acc[model][strategy][seed] = accuracy — input for the paper-format table."""
    acc = {m: {s: {} for s in STRATEGIES} for m in MODEL_ORDER}
    for mj in glob.glob(os.path.join(results_dir, "*", "seed_*", "metrics.json")):
        with open(mj) as f:
            d = json.load(f)
        if d.get("dry_run"):
            continue
        model, seed = d["model"], d["seed"]
        if model not in acc:
            continue
        for strat in STRATEGIES:
            if strat in d["results"]:
                acc[model][strat][seed] = d["results"][strat][ACC_METRIC]
    return acc


def mean_std_table(acc):
    """Rows = strategies, cols = pretty model names, cell = 'mean ± std (+Δ)' — Accuracy only."""
    rows = []
    for strat in STRATEGIES:
        row = {"Method": strat}
        for model in MODEL_ORDER:
            vals = [acc[model][strat][s] for s in SEEDS if s in acc[model][strat]]
            base = [acc[model]["Baseline (k=1)"][s] for s in SEEDS if s in acc[model]["Baseline (k=1)"]]
            if not vals:
                row[PRETTY_NAME[model]] = ""
                continue
            mean, std = np.mean(vals), np.std(vals)
            cell = f"{mean:.2f} ± {std:.2f}"
            if strat != "Baseline (k=1)" and base:
                cell += f" ({mean - np.mean(base):+.2f})"
            row[PRETTY_NAME[model]] = cell
        rows.append(row)
    return pd.DataFrame(rows, columns=["Method"] + [PRETTY_NAME[m] for m in MODEL_ORDER])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    records, found = load_records(args.results_dir)
    if not found:
        print("No completed (non-dry-run) metrics.json found yet.")
        return

    # ---------- File 1: full per-run detail ----------
    full_xlsx = os.path.join(args.results_dir, "cifar100_full_results.xlsx")
    with pd.ExcelWriter(full_xlsx, engine="openpyxl") as writer:
        to_df(records).to_excel(writer, sheet_name="Tong_hop", index=False)
        for seed in SEEDS:
            seed_records = [r for r in records if r["Seed"] == seed]
            if seed_records:
                to_df(seed_records).to_excel(writer, sheet_name=f"seed_{seed}", index=False)

    # ---------- File 2: paper-ready report format (Accuracy mean±std only) ----------
    acc = build_acc_lookup(args.results_dir)
    summary = mean_std_table(acc)
    report_xlsx = os.path.join(args.results_dir, "cifar100_report_format.xlsx")
    summary.to_excel(report_xlsx, sheet_name="Mean±Std", index=False)
    summary.to_csv(os.path.join(args.results_dir, "cifar100_report_format.csv"), index=False)

    print("\n================ CIFAR-100 Strategy 2 — report format (Accuracy %, mean ± std) ================")
    print(summary.to_string(index=False))

    print(f"\nRuns found: {len(found)} / {len(MODEL_ORDER) * len(SEEDS)} (models x seeds). Missing:")
    missing = [(m, s) for m in MODEL_ORDER for s in SEEDS if (m, s) not in found]
    print("  " + (", ".join(f"{m}/seed{s}" for m, s in missing) if missing else "none — complete!"))
    print(f"\n✓ Wrote {full_xlsx}")
    print(f"✓ Wrote {report_xlsx}")
    print(f"✓ Wrote {os.path.join(args.results_dir, 'cifar100_report_format.csv')}")


if __name__ == "__main__":
    main()
