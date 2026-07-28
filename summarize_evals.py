"""Summarize RAVEL + absorption metrics across all models under eval_results/.

Walks `eval_results/<model_name>/{absorption,ravel}/*.json`, pulls the headline
metrics from each, and prints one row per model. Models missing one of the two
evals are still reported — the missing columns show as '-'.

Metric direction (so you don't have to look it up):
  absorption (lower is better):
    abs_frac        = mean_absorption_fraction_score
    full_abs        = mean_full_absorption_score
    n_split         = mean_num_split_features
  ravel (higher is better):
    disentangle     = disentanglement_score
    cause           = cause_score
    isolation       = isolation_score

Usage:
    python summarize_evals.py                          # aligned table
    python summarize_evals.py --csv > metrics.csv      # machine-readable
    python summarize_evals.py --root /custom/eval_dir
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ABSORPTION_KEYS = [
    ("mean_absorption_fraction_score", "abs_frac"),
    ("mean_full_absorption_score",     "full_abs"),
    ("mean_num_split_features",        "n_split"),
]

RAVEL_KEYS = [
    ("disentanglement_score", "disentangle"),
    ("cause_score",           "cause"),
    ("isolation_score",       "isolation"),
]


def find_metrics_file(eval_dir: Path) -> Path | None:
    """Return the *_eval_results.json (or any .json) under eval_dir, or None."""
    if not eval_dir.is_dir():
        return None
    for pattern in ("*_eval_results.json", "*.json"):
        hits = sorted(eval_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _nanmean(values):
    valid = [v for v in values if v is not None and not _is_nan(v)]
    return (sum(valid) / len(valid)) if valid else None


def _ravel_recompute_from_unstructured(data: dict) -> dict:
    """Compute disentangle / cause / isolation by NaN-aware averaging across
    every per-attribute score in `eval_result_unstructured`. Used when the
    headline metric in the JSON is NaN (older SAEBench's sum/len propagates
    NaN from attributes with insufficient prompts)."""
    unstructured = data.get("eval_result_unstructured") or {}
    cause, iso, dis = [], [], []
    for class_results in unstructured.values():
        if not isinstance(class_results, dict):
            continue
        for attr_results in class_results.values():
            if not isinstance(attr_results, dict):
                continue
            cause.append(attr_results.get("cause_score"))
            iso.append(attr_results.get("isolation_score"))
            dis.append(attr_results.get("disentangle_score"))
    return {
        "disentangle": _nanmean(dis),
        "cause": _nanmean(cause),
        "isolation": _nanmean(iso),
    }


def read_metrics(path: Path, sub_key: str, keys) -> dict:
    """Pull `eval_result_metrics.<sub_key>.<full>` for each (full, short) pair.

    For RAVEL, if the headline metric is NaN we fall back to recomputing it
    NaN-aware from `eval_result_unstructured` (which retains the per-attribute
    scores) so the user gets a usable number out of older runs.
    """
    with path.open() as f:
        data = json.load(f)
    metrics = data.get("eval_result_metrics", {}).get(sub_key, {}) or {}
    out = {short: metrics.get(full) for full, short in keys}

    if sub_key == "ravel" and any(_is_nan(out.get(s)) for _, s in keys):
        recomputed = _ravel_recompute_from_unstructured(data)
        for _, short in keys:
            if _is_nan(out.get(short)) and recomputed.get(short) is not None:
                out[short] = recomputed[short]

    return out


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=Path(__file__).resolve().parent / "eval_results",
                        type=Path, help="Folder containing per-model subdirectories.")
    parser.add_argument("--csv", action="store_true",
                        help="Print CSV to stdout instead of an aligned table.")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"No such directory: {root}")

    rows: list[dict] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        row: dict = {"model": model_dir.name}

        absorption_file = find_metrics_file(model_dir / "absorption")
        if absorption_file is not None:
            row.update(read_metrics(absorption_file, "mean", ABSORPTION_KEYS))

        ravel_file = find_metrics_file(model_dir / "ravel")
        if ravel_file is not None:
            row.update(read_metrics(ravel_file, "ravel", RAVEL_KEYS))

        # only emit a row if we have *any* metric, to skip empty model dirs
        if any(row.get(short) is not None
               for _, short in ABSORPTION_KEYS + RAVEL_KEYS):
            rows.append(row)

    columns = ["model"] + [s for _, s in ABSORPTION_KEYS] + [s for _, s in RAVEL_KEYS]

    if not rows:
        print(f"No metrics found under {root}")
        return

    if args.csv:
        w = csv.DictWriter(sys.stdout, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({c: fmt(row.get(c)) for c in columns})
        return

    widths = {c: max(len(c), max(len(fmt(r.get(c))) for r in rows)) for c in columns}
    sep = "  "
    header = sep.join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(sep.join(fmt(row.get(c)).ljust(widths[c]) for c in columns))

    print()
    print(f"  {len(rows)} model(s)  |  absorption: lower is better  |  ravel: higher is better")


if __name__ == "__main__":
    main()
