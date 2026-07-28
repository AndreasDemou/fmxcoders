"""Fetch training/eval metrics from local files (no wandb API).

Reads:
  - eval_results/<run_name>/sparse_probing/*.json      (SAEBench sparse_probing)
  - eval_results/<run_name>/recon_metrics.json         (standalone crosscoder recon)
  - wandb/local/<run_name>/metrics.jsonl               (training-time metrics; last step)

and prints:
  - ASCII bar chart of per-dataset Top-k F1 (%)
  - Per-layer reconstruction metrics table (MSE / explained variance / cossim)
  - A LaTeX table row matching fetch_run_metrics_table1.py

Usage:
    python fetch_local_run_metrics.py gpt2_m0
    python fetch_local_run_metrics.py gpt2_m0 --k 1
    python fetch_local_run_metrics.py gpt2_m0 --base_dir /path/to/PolySAE_dev
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Map SAEBench dataset_name (with "_results" suffix stripped) to the short
# labels used in the original fetch_run_metrics_table1.py script.
DATASET_LABEL_MAP: dict[str, str] = {
    "Helsinki-NLP/europarl": "europarl",
    "canrager/amazon_reviews_mcauley_1and5_sentiment": "amazon_1and5_sentiment",
    "codeparrot/github-code": "github_code",
    "fancyzhx/ag_news": "ag_news",
    "canrager/amazon_reviews_mcauley_1and5": "amazon_1and5",
    "LabHC/bias_in_bios_class_set1": "bios_class_set1",
    "LabHC/bias_in_bios_class_set2": "bios_class_set2",
    "LabHC/bias_in_bios_class_set3": "bios_class_set3",
    "LabHC/bias_in_bios": "bias_in_bios",
}

DATASETS = [
    "europarl",
    "amazon_1and5_sentiment",
    "github_code",
    "ag_news",
    "amazon_1and5",
]
BIOS_CLASS_SETS = ["bios_class_set1", "bios_class_set2", "bios_class_set3"]


def print_ascii_bar(label: str, value: float, max_val: float = 100, bar_len: int = 40) -> None:
    val_clamped = max(0, min(max_val, value))
    filled = int(round(val_clamped / max_val * bar_len))
    bar = "#" * filled + "." * (bar_len - filled)
    print(f"{label:28} | {bar} {value:5.1f}%")


def load_sparse_probing(run_dir: Path) -> dict[str, Any] | None:
    pattern = str(run_dir / "sparse_probing" / "*.json")
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None
    if len(hits) > 1:
        print(f"[warn] multiple sparse_probing JSONs found under {pattern}, using {hits[0]}",
              file=sys.stderr)
    with open(hits[0]) as f:
        return json.load(f)


def load_recon(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "recon_metrics.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def load_training_last_step(wandb_local_dir: Path) -> dict[str, Any] | None:
    path = wandb_local_dir / "metrics.jsonl"
    if not path.is_file():
        return None
    last: dict[str, Any] | None = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last = json.loads(line)
    return last


def _per_dataset_metrics(sp: dict[str, Any], k: int) -> dict[str, dict[str, float | None]]:
    """Map short-label → {'f1': ..., 'wass': ...} from eval_result_details."""
    out: dict[str, dict[str, float | None]] = {}
    details = sp.get("eval_result_details", [])
    f1_key = f"sae_top_{k}_test_f1"
    wass_key = f"sae_top_{k}_wasserstein"
    for entry in details:
        name = entry.get("dataset_name", "")
        if name.endswith("_results"):
            name = name[: -len("_results")]
        label = DATASET_LABEL_MAP.get(name)
        if label is None:
            continue
        out[label] = {
            "f1": entry.get(f1_key),
            "wass": entry.get(wass_key),
        }
    return out


def _fmt(pair: dict[str, float | None] | None) -> str:
    if pair is None:
        return "- / -"
    f1, w = pair.get("f1"), pair.get("wass")
    if f1 is None or w is None:
        return "- / -"
    return f"{f1 * 100:.1f} / {w * 1000:.1f}"


def report(run_name: str, base_dir: Path, k: int) -> None:
    run_dir = base_dir / "eval_results" / run_name
    wandb_dir = base_dir / "wandb" / "local" / run_name

    print(f"% Run: {run_name}")
    print(f"% eval_results: {run_dir}")
    print(f"% wandb/local:  {wandb_dir}")

    sp = load_sparse_probing(run_dir)
    recon = load_recon(run_dir)
    train_last = load_training_last_step(wandb_dir)

    if sp is None and recon is None and train_last is None:
        print(f"No results found for run '{run_name}'.", file=sys.stderr)
        sys.exit(1)

    # ---- Reconstruction metrics ----
    mse: float | None = None
    ev: float | None = None
    ev_legacy: float | None = None
    cossim: float | None = None
    l0: float | None = None
    if recon is not None:
        agg = recon.get("aggregate", {})
        mse = agg.get("mse")
        ev = agg.get("explained_variance")
        ev_legacy = agg.get("explained_variance_legacy")
        cossim = agg.get("cossim")
        l0 = agg.get("l0")
    elif train_last is not None:
        # Fall back to last-step training metrics
        data = train_last.get("data", {})
        mse = data.get("losses/mse_loss")
        ev = data.get("metrics/explained_variance")
        ev_legacy = data.get("metrics/explained_variance_legacy")
        l0 = data.get("metrics/l0")

    # ---- Sparse probing metrics ----
    per_ds: dict[str, dict[str, float | None]] = {}
    if sp is not None:
        per_ds = _per_dataset_metrics(sp, k)

    # Average the three bios class sets; fall back to primary if present
    bios_f1s = [per_ds[b]["f1"] for b in BIOS_CLASS_SETS
                if b in per_ds and per_ds[b]["f1"] is not None]
    bios_ws = [per_ds[b]["wass"] for b in BIOS_CLASS_SETS
               if b in per_ds and per_ds[b]["wass"] is not None]
    if bios_f1s:
        bios_f1 = float(np.mean(bios_f1s))
        bios_w = float(np.mean(bios_ws)) if bios_ws else None
    elif "bias_in_bios" in per_ds:
        bios_f1 = per_ds["bias_in_bios"]["f1"]
        bios_w = per_ds["bias_in_bios"]["wass"]
    else:
        bios_f1 = bios_w = None

    # F1 percentages for ASCII plot and mean/std
    all_f1s: list[float] = []
    all_ws: list[float] = []   # parallel list, scaled by 1000 (matching per-dataset display)
    bar_entries: list[tuple[str, float]] = []
    for ds in DATASETS:
        if ds in per_ds and per_ds[ds]["f1"] is not None:
            v = per_ds[ds]["f1"] * 100
            all_f1s.append(v)
            bar_entries.append((ds, v))
            if per_ds[ds]["wass"] is not None:
                all_ws.append(per_ds[ds]["wass"] * 1000)
    if bios_f1 is not None:
        v = bios_f1 * 100
        all_f1s.append(v)
        bar_entries.append(("bias_in_bios (avg)", v))
        if bios_w is not None:
            all_ws.append(bios_w * 1000)

    mean_f1 = float(np.mean(all_f1s)) if all_f1s else 0.0
    std_f1 = float(np.std(all_f1s)) if all_f1s else 0.0
    mean_w = float(np.mean(all_ws)) if all_ws else None
    std_w = float(np.std(all_ws)) if all_ws else None

    # ---- Output ----
    if bar_entries:
        print("\n" + "=" * 80)
        print(f"  ASCII PLOT — sparse_probing Top-{k} F1 (%)")
        print("=" * 80)
        for label, v in bar_entries:
            print_ascii_bar(label, v)
        print_ascii_bar("OVERALL MEAN", mean_f1)
    else:
        print("\n[note] no sparse_probing results loaded — skipping F1 plot")

    if recon is not None and recon.get("per_layer"):
        print("\n" + "=" * 80)
        print("  RECONSTRUCTION — per layer (from recon_metrics.json)")
        print("=" * 80)
        print(f"{'hook':30} {'mse':>10} {'cossim':>8} {'EV':>8} {'EV_leg':>8} {'l2_ratio':>10}")
        for row in recon["per_layer"]:
            print(f"{row['hook_name']:30} "
                  f"{row['mse']:10.4f} "
                  f"{row['cossim']:8.4f} "
                  f"{row['explained_variance']:8.4f} "
                  f"{row['explained_variance_legacy']:8.4f} "
                  f"{row['l2_ratio']:10.4f}")

    print("\n" + "=" * 80)
    print("  LATEX TABLE ROW")
    print("=" * 80)
    header = "MSE & Mean F1 & Mean W & Europarl & Bias in Bios & Amazon Sent & Github & AG News & Amazon 1and5 \\\\"
    mse_str = f"{mse:.2f}" if mse is not None else "-"
    f1_mean_std = f"{mean_f1:.1f} \\pm {std_f1:.1f}" if all_f1s else "-"
    w_mean_std = f"{mean_w:.1f} \\pm {std_w:.1f}" if mean_w is not None else "-"
    bios_str = (
        f"{bios_f1 * 100:.1f} / {bios_w * 1000:.1f}"
        if bios_f1 is not None and bios_w is not None else "- / -"
    )
    print(header)
    print(
        f"{mse_str} & {f1_mean_std} & {w_mean_std} & "
        f"{_fmt(per_ds.get('europarl'))} & "
        f"{bios_str} & "
        f"{_fmt(per_ds.get('amazon_1and5_sentiment'))} & "
        f"{_fmt(per_ds.get('github_code'))} & "
        f"{_fmt(per_ds.get('ag_news'))} & "
        f"{_fmt(per_ds.get('amazon_1and5'))} \\\\"
    )

    if recon is not None:
        print("\n[recon aggregate]")
        for k_, v_ in recon.get("aggregate", {}).items():
            if isinstance(v_, (int, float)):
                print(f"  {k_:32} {v_:.6g}")

    if train_last is not None:
        step = train_last.get("step")
        print(f"\n[training last step: {step}]")
        for k_, v_ in (train_last.get("data") or {}).items():
            if isinstance(v_, (int, float)):
                print(f"  {k_:40} {v_:.6g}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch & format local run metrics (no wandb API)."
    )
    p.add_argument("run_name", help="Run name — matches eval_results/<run_name>/ and wandb/local/<run_name>/")
    p.add_argument("--base_dir", default=".", help="Project root (defaults to CWD).")
    p.add_argument("--k", type=int, default=1, help="Top-k for sparse_probing F1 (default 1).")
    args = p.parse_args()

    base = Path(args.base_dir).resolve()
    if not (base / "eval_results").is_dir() and not (base / "wandb" / "local").is_dir():
        print(f"Neither {base}/eval_results nor {base}/wandb/local exists. "
              "Pass --base_dir to point at PolySAE_dev root.", file=sys.stderr)
        sys.exit(1)
    report(args.run_name, base, args.k)


if __name__ == "__main__":
    main()
