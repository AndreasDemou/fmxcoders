"""Per-latent unique-token diversity from a top-activating-tokens JSON.

Input JSON shape:
    {
        "<latent_id>": [
            {"activation": float, "sentence_id": int, "position": int,
             "token": str, "sentence": str}, ...
        ], ...
    }

For each latent, count how many *distinct* tokens it activates on across the
provided examples. Two normalizations are supported:

  --norm raw      keep the token string exactly as given.
  --norm strip    strip leading/trailing whitespace (default).
  --norm lower    case-fold and strip whitespace.

Reports a summary distribution and (optionally) writes one CSV row per latent
with columns: latent_id, n_examples, n_unique_tokens, top_token, top_share.

NOTE: the unique count is upper-bounded by the number of examples per latent
(the top_k used when generating the input file). A latent that genuinely
activates on a single token will always return count = 1; a latent firing on
diverse contexts will saturate near top_k. Use this as a *relative*
monosemanticity proxy, not an absolute polysemy measure.

Usage:
    python count_latent_tokens.py token_features_top.json
    python count_latent_tokens.py token_features_top.json --norm lower --csv counts.csv
    python count_latent_tokens.py token_features_top.json --top-k-show 20
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\n".join(lines) + "\n")


def _default_coherence_csv(input_path: Path) -> Path:
    """Default coherence path under the assumed layout
       <repo>/eval_results/<run>/feature_analysis/<file>.json  ->
       <repo>/coherence_results/<run>/mask_zero/per_latent_metrics.csv
    """
    run_name = input_path.parent.parent.name
    repo_root = input_path.parent.parent.parent.parent
    return repo_root / "coherence_results" / run_name / "mask_zero" / "per_latent_metrics.csv"


def _load_coherence(path: Path) -> dict[str, float]:
    """Load {latent_id (str): coherence_score (float)} from a 2-column CSV.

    Tolerates a header row by sniffing whether the first row's second cell
    parses as a float.
    """
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    with path.open() as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            lid, score = row[0].strip(), row[1].strip()
            try:
                v = float(score)
            except ValueError:
                if i == 0:
                    continue   # header; skip
                continue
            out[lid] = v
    return out


def normalize(token: str, mode: str) -> str:
    if mode == "raw":
        return token
    if mode == "strip":
        return token.replace("Ġ", " ").replace("▁", " ").strip()
    if mode == "lower":
        return token.replace("Ġ", " ").replace("▁", " ").strip().lower()
    raise ValueError(f"unknown norm mode: {mode}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="Path to the top-activating-tokens JSON.")
    p.add_argument("--norm", choices=["raw", "strip", "lower"], default="strip",
                   help="Token normalization before counting (default: strip).")
    p.add_argument("--csv", type=Path, default=None,
                   help="CSV output path. Defaults to <input_dir>/<input_stem>_token_counts.csv.")
    p.add_argument("--report", type=Path, default=None,
                   help="Text-report output path. Defaults to <input_dir>/<input_stem>_token_counts.txt.")
    p.add_argument("--top-k-show", type=int, default=10,
                   help="Show the K most diverse and K least diverse latents in the report.")
    p.add_argument("--top-active", type=int, default=1000,
                   help="Restrict analysis to the first N latents in the input file "
                        "(assumed pre-sorted by activity). Pass 0 to use all latents "
                        "(default: 1000).")
    p.add_argument("--coherence-csv", type=Path, default=None,
                   help="Path to the per-latent mask-coherence CSV (latent_id, coherence). "
                        "If omitted, derived as "
                        "<repo>/coherence_results/<run_name>/mask_zero/per_latent_metrics.csv "
                        "where <run_name> is the second-level parent of the input file.")
    p.add_argument("--scatter", type=Path, default=None,
                   help="Scatter-plot output PNG (a .pdf with the same stem is also written). "
                        "Defaults to <input_dir>/<input_stem>_unique_vs_coherence.png.")
    args = p.parse_args()

    args.input = args.input.resolve()
    if args.csv is None:
        args.csv = args.input.parent / f"{args.input.stem}_token_counts.csv"
    if args.report is None:
        args.report = args.input.parent / f"{args.input.stem}_token_counts.txt"
    if args.coherence_csv is None:
        args.coherence_csv = _default_coherence_csv(args.input)
    if args.scatter is None:
        args.scatter = args.input.parent / f"{args.input.stem}_unique_vs_coherence.png"
    coherence = _load_coherence(args.coherence_csv)

    # Tee stdout writes into a buffer so we can save them as a report file.
    report_lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        report_lines.append(s)

    with args.input.open() as f:
        data = json.load(f)

    # Take the first N latents in the file (the file is assumed pre-sorted
    # by activity).
    n_total = len(data)
    items = list(data.items())
    if args.top_active and args.top_active > 0:
        items = items[: args.top_active]

    rows = []   # (latent_id, n_examples, n_unique, top_token, top_share, mean_act, coh)
    for latent_id, examples in items:
        if not isinstance(examples, list) or not examples:
            continue
        norm_tokens = [normalize(ex.get("token", ""), args.norm) for ex in examples]
        acts = [ex.get("activation") for ex in examples
                if isinstance(ex.get("activation"), (int, float))]
        mean_act = float(np.mean(acts)) if acts else float("nan")
        c = Counter(norm_tokens)
        n_examples = len(norm_tokens)
        n_unique = len(c)
        top_token, top_count = c.most_common(1)[0]
        coh = coherence.get(str(latent_id), float("nan"))
        rows.append((latent_id, n_examples, n_unique, top_token,
                     top_count / n_examples, mean_act, coh))

    if not rows:
        out("No latents with examples found in the input file.")
        _write_report(args.report, report_lines)
        return

    if args.top_active and args.top_active > 0 and n_total > len(rows):
        out(f"Restricted analysis to the first {len(rows)} latents in the file "
            f"(out of {n_total} total; assumed pre-sorted by activity).")
    else:
        out(f"Analyzing all {len(rows)} latents.")

    n_uniques = np.array([r[2] for r in rows])
    top_shares = np.array([r[4] for r in rows])
    n_examples_arr = np.array([r[1] for r in rows])

    out(f"Loaded {len(rows)} latents from {args.input}")
    out(f"Normalization: {args.norm}")
    out(f"Examples per latent: min={n_examples_arr.min()}  "
        f"median={int(np.median(n_examples_arr))}  max={n_examples_arr.max()}")
    out()

    out("=" * 64)
    out("  Unique-token count per latent")
    out("=" * 64)
    out(f"  mean       {n_uniques.mean():.2f}")
    out(f"  median     {int(np.median(n_uniques))}")
    out(f"  std        {n_uniques.std():.2f}")
    out(f"  min        {n_uniques.min()}")
    out(f"  max        {n_uniques.max()}")
    out(f"  p10/p25/p75/p90  "
        f"{int(np.percentile(n_uniques, 10))} / "
        f"{int(np.percentile(n_uniques, 25))} / "
        f"{int(np.percentile(n_uniques, 75))} / "
        f"{int(np.percentile(n_uniques, 90))}")
    out()

    out("  Top-token share (how concentrated each latent is on its modal token)")
    out(f"  mean       {top_shares.mean():.3f}")
    out(f"  median     {np.median(top_shares):.3f}")
    out(f"  fraction with share >= 0.9   {(top_shares >= 0.9).mean():.3f}")
    out(f"  fraction with share >= 0.5   {(top_shares >= 0.5).mean():.3f}")
    out()

    coh_arr = np.array([r[6] for r in rows], dtype=float)
    coh_valid = coh_arr[~np.isnan(coh_arr)]
    out("  Mask-coherence score")
    if coh_valid.size == 0:
        out(f"  (no coherence values found at {args.coherence_csv})")
    else:
        out(f"  source     {args.coherence_csv}")
        out(f"  matched    {coh_valid.size} / {len(rows)} latents")
        out(f"  mean       {coh_valid.mean():.4f}")
        out(f"  median     {np.median(coh_valid):.4f}")
        out(f"  std        {coh_valid.std():.4f}")
        out(f"  min/max    {coh_valid.min():.4f} / {coh_valid.max():.4f}")
        # Pearson correlation between unique-token count and coherence
        if coh_valid.size >= 2:
            mask = ~np.isnan(coh_arr)
            if mask.sum() >= 2 and np.std(n_uniques[mask]) > 0 and np.std(coh_arr[mask]) > 0:
                r_pearson = float(np.corrcoef(n_uniques[mask], coh_arr[mask])[0, 1])
                out(f"  Pearson r(n_unique, coherence) = {r_pearson:+.3f}")
    out()

    # Histogram of unique counts (text bar chart)
    out("  Histogram of unique-token counts:")
    max_count = int(n_uniques.max())
    bin_edges = np.linspace(1, max_count + 1, min(20, max_count) + 1)
    hist, edges = np.histogram(n_uniques, bins=bin_edges)
    bar_max = max(hist.max(), 1)
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(40 * h / bar_max)
        out(f"    [{int(lo):>4d}, {int(hi):>4d})  {bar:<40s}  {h}")
    out()

    # K most diverse / least diverse latents
    K = args.top_k_show
    rows_sorted = sorted(rows, key=lambda r: r[2])
    coh_str = lambda v: "      -" if (isinstance(v, float) and np.isnan(v)) else f"{v:>7.3f}"
    out(f"  {K} most monosemantic (lowest unique-token count):")
    out(f"    {'latent':>8s}  {'n_ex':>5s}  {'unique':>6s}  {'share':>6s}  {'coh':>7s}  top_token")
    for r in rows_sorted[:K]:
        out(f"    {r[0]:>8s}  {r[1]:>5d}  {r[2]:>6d}  {r[4]:>6.3f}  {coh_str(r[6])}  {r[3]!r}")
    out()
    out(f"  {K} most polysemantic (highest unique-token count):")
    out(f"    {'latent':>8s}  {'n_ex':>5s}  {'unique':>6s}  {'share':>6s}  {'coh':>7s}  top_token")
    for r in rows_sorted[-K:][::-1]:
        out(f"    {r[0]:>8s}  {r[1]:>5d}  {r[2]:>6d}  {r[4]:>6.3f}  {coh_str(r[6])}  {r[3]!r}")
    out()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["latent_id", "n_examples", "n_unique_tokens",
                    "top_token", "top_share", "mean_activation", "mask_coherence"])
        for r in rows:
            w.writerow(r)
    out(f"Wrote {len(rows)} rows to {args.csv}")

    # Scatter: unique-token count vs mask-coherence score.
    if coh_valid.size > 0:
        mask = ~np.isnan(coh_arr)
        x = n_uniques[mask].astype(float)
        y = coh_arr[mask]
        # Small horizontal jitter so integer x-values don't stack into one column
        rng = np.random.default_rng(0)
        x_jit = x + rng.uniform(-0.15, 0.15, size=x.shape)

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.scatter(x_jit, y, s=14, alpha=0.45, edgecolor="none", color="#4c72b0")
        ax.set_xlabel("unique-token count (per latent, normalization=" + args.norm + ")")
        ax.set_ylabel("mask-coherence score")
        ax.set_title(f"unique-token count vs mask-coherence  ({mask.sum()} latents)")

        if mask.sum() >= 2 and np.std(x) > 0 and np.std(y) > 0:
            r = float(np.corrcoef(x, y)[0, 1])
            ax.text(0.98, 0.98, f"Pearson r = {r:+.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="black", alpha=0.85))

        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        scatter_png = args.scatter
        scatter_pdf = scatter_png.with_suffix(".pdf")
        plt.savefig(scatter_png, dpi=180)
        plt.savefig(scatter_pdf)
        plt.close(fig)
        out(f"Wrote scatter to {scatter_png}")
        out(f"Wrote scatter to {scatter_pdf}")
    else:
        out("Skipping scatter plot (no coherence values matched).")

    _write_report(args.report, report_lines)
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
