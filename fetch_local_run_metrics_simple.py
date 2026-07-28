"""Print one LaTeX table row of metrics for a run, nothing else.

Reads the same files as `fetch_local_run_metrics.py`:
  - eval_results/<run_name>/sparse_probing/*.json   (Top-k F1 per dataset)
  - eval_results/<run_name>/recon_metrics.json      (MSE / EV / cossim aggregates)
  - wandb/local/<run_name>/metrics.jsonl            (fallback for MSE/EV at last step)

Differences vs the full script:
  - Prints only the LaTeX header + row.
  - Mean F1 column is just the mean (no `\\pm std`).
  - Two new columns between MSE and Mean F1: `EV` (explained_variance) and `CosSim`.

Usage:
    python fetch_local_run_metrics_simple.py gpt2_m0
    python fetch_local_run_metrics_simple.py gpt2_m0 --k 1
    python fetch_local_run_metrics_simple.py gpt2_m0 --base_dir /path/to/PolySAE_dev
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

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

DATASETS = ["europarl", "amazon_1and5_sentiment", "github_code", "ag_news", "amazon_1and5"]
BIOS_CLASS_SETS = ["bios_class_set1", "bios_class_set2", "bios_class_set3"]


def _load_sparse_probing(run_dir: Path) -> dict[str, Any] | None:
    hits = sorted(glob.glob(str(run_dir / "sparse_probing" / "*.json")))
    if not hits:
        return None
    with open(hits[0]) as f:
        return json.load(f)


def _load_recon(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "recon_metrics.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _load_training_last_step(wandb_local_dir: Path) -> dict[str, Any] | None:
    path = wandb_local_dir / "metrics.jsonl"
    if not path.is_file():
        return None
    last: dict[str, Any] | None = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def _per_dataset_metrics(sp: dict[str, Any], k: int) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    f1_key = f"sae_top_{k}_test_f1"
    wass_key = f"sae_top_{k}_wasserstein"
    for entry in sp.get("eval_result_details", []):
        name = entry.get("dataset_name", "")
        if name.endswith("_results"):
            name = name[: -len("_results")]
        label = DATASET_LABEL_MAP.get(name)
        if label is None:
            continue
        out[label] = {"f1": entry.get(f1_key), "wass": entry.get(wass_key)}
    return out


def _fmt_pair(pair: dict[str, float | None] | None) -> str:
    if pair is None:
        return "- / -"
    f1, w = pair.get("f1"), pair.get("wass")
    if f1 is None or w is None:
        return "- / -"
    return f"{f1 * 100:.1f} / {w * 1000:.1f}"


def _fmt(v: float | None, places: int = 4) -> str:
    return f"{v:.{places}f}" if v is not None else "-"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_name")
    p.add_argument("--base_dir", default=".", help="Project root (defaults to CWD).")
    p.add_argument("--k", type=int, default=1, help="Top-k for sparse_probing F1.")
    args = p.parse_args()

    base = Path(args.base_dir).resolve()
    run_dir = base / "eval_results" / args.run_name
    wandb_dir = base / "wandb" / "local" / args.run_name

    sp = _load_sparse_probing(run_dir)
    recon = _load_recon(run_dir)
    train_last = _load_training_last_step(wandb_dir)
    if sp is None and recon is None and train_last is None:
        print(f"No results found for run '{args.run_name}'.", file=sys.stderr)
        sys.exit(1)

    # MSE / EV / CosSim — prefer recon_metrics.json, fall back to last training step
    mse = ev = cossim = None
    if recon is not None:
        agg = recon.get("aggregate", {})
        mse = agg.get("mse")
        ev = agg.get("explained_variance")
        cossim = agg.get("cossim")
    if (mse is None or ev is None) and train_last is not None:
        data = train_last.get("data", {}) or {}
        mse = mse if mse is not None else data.get("losses/mse_loss")
        ev = ev if ev is not None else data.get("metrics/explained_variance")

    # Per-dataset F1 / Wasserstein
    per_ds = _per_dataset_metrics(sp, args.k) if sp is not None else {}

    # Average the three bios class sets, fall back to the primary dataset
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

    # Mean F1 / Mean W across reported datasets (no std)
    all_f1s: list[float] = []
    all_ws: list[float] = []
    for ds in DATASETS:
        if ds not in per_ds:
            continue
        f1 = per_ds[ds].get("f1")
        w = per_ds[ds].get("wass")
        if f1 is not None:
            all_f1s.append(f1 * 100)
        if w is not None:
            all_ws.append(w * 1000)
    if bios_f1 is not None:
        all_f1s.append(bios_f1 * 100)
    if bios_w is not None:
        all_ws.append(bios_w * 1000)
    mean_f1 = float(np.mean(all_f1s)) if all_f1s else None
    mean_w = float(np.mean(all_ws)) if all_ws else None

    bios_str = (f"{bios_f1 * 100:.1f} / {bios_w * 1000:.1f}"
                if bios_f1 is not None and bios_w is not None else "- / -")

    header = "MSE & EV & CosSim & Mean F1 & Mean W & Europarl & Bias in Bios & Amazon Sent & Github & AG News & Amazon 1and5 \\\\"
    row = (
        f"{_fmt(mse, 4)} & "
        f"{_fmt(ev, 4)} & "
        f"{_fmt(cossim, 4)} & "
        f"{(f'{mean_f1:.1f}' if mean_f1 is not None else '-')} & "
        f"{(f'{mean_w:.1f}' if mean_w is not None else '-')} & "
        f"{_fmt_pair(per_ds.get('europarl'))} & "
        f"{bios_str} & "
        f"{_fmt_pair(per_ds.get('amazon_1and5_sentiment'))} & "
        f"{_fmt_pair(per_ds.get('github_code'))} & "
        f"{_fmt_pair(per_ds.get('ag_news'))} & "
        f"{_fmt_pair(per_ds.get('amazon_1and5'))} \\\\"
    )
    print(header)
    print(row)


if __name__ == "__main__":
    main()
