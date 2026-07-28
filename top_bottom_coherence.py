"""Print the 20 highest- and 20 lowest-coherence latents from the masked TR run,
along with the tokens each one fires on (and their activations) from the
matching feature_analysis pivot file."""

import csv
import json
from collections import Counter
from pathlib import Path

CSV_PATH = Path("coherence_results/gemma2-2b_m0_05_r32-12-545/mask_zero/per_latent_metrics.csv")
NORM_CSV_PATH = Path("coherence_results/gemma2-2b_m0_05_r32-12-545/norm_decoder/per_latent_metrics.csv")
PIVOT_PATH = Path("eval_results/gemma2-2b_m0_05_r32-12-545/feature_analysis/token_features_pivot.json")
N = 20
MIN_EXAMPLES = 10   # keep latents with strictly MORE than this many examples


def load_coherence(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
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
                    continue   # header
                continue
            rows.append((lid, v))
    return rows


SENTENCE_TRUNCATE = 200   # characters; longer sentences get a trailing "..." in stdout only


def _trunc(s: str, n: int = SENTENCE_TRUNCATE) -> str:
    """One-line truncation for printing."""
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= n else s[:n].rstrip() + " ..."


def latent_block(lid: str, score: float, norm_score: float | None,
                 examples: list | None) -> dict:
    """Return a structured dict for one latent."""
    block: dict = {
        "latent_id": lid,
        "coherence": float(score),
        "coherence_x8": float(score) * 8.0,
        "norm_coherence_x8": (float(norm_score) * 8.0 if norm_score is not None else None),
        "tokens": [],
    }
    if not examples:
        return block
    # Aggregate by token: keep the highest-activation example's sentence,
    # plus a count of how many of the latent's top-K examples used that token.
    by_token: dict[str, dict] = {}
    for ex in examples:
        tok = ex.get("token", "")
        act = float(ex.get("activation", 0.0))
        cur = by_token.get(tok)
        if cur is None or act > cur["max_activation"]:
            by_token[tok] = {
                "token": tok,
                "max_activation": act,
                "count": (cur["count"] + 1) if cur else 1,
                "sentence": ex.get("sentence", ""),
            }
        else:
            cur["count"] += 1
    items = sorted(by_token.values(), key=lambda d: d["max_activation"], reverse=True)
    block["tokens"] = items
    return block


def print_latent_block(block: dict) -> None:
    """Render a latent block to stdout."""
    nc = block.get("norm_coherence")
    nc_str = f"{nc:.4f}" if isinstance(nc, (int, float)) else "-"
    print(f"  Latent {block['latent_id']}  coherence = {block['coherence']:.4f}  "
          f"(×8 = {block['coherence_x8']:.4f})  "
          f"norm_coherence = {nc_str}")
    if not block["tokens"]:
        print("    (no examples found in the pivot file)")
        return
    for t in block["tokens"]:
        print(f"    {t['token']!r}  (max_act={t['max_activation']:.4f}, x{t['count']})")
        print(f"        sentence: {_trunc(t['sentence'])}")


def main() -> None:
    rows = load_coherence(CSV_PATH)
    if not rows:
        print(f"No coherence rows found in {CSV_PATH}")
        return
    print(f"Loaded {len(rows)} latents from {CSV_PATH}")

    # Norm coherence: optional secondary signal; not used for ranking/filtering.
    if NORM_CSV_PATH.is_file():
        norm_rows = load_coherence(NORM_CSV_PATH)
        norm_map: dict[str, float] = {lid: v for lid, v in norm_rows}
        print(f"Loaded {len(norm_map)} norm-coherence rows from {NORM_CSV_PATH}")
    else:
        norm_map = {}
        print(f"WARNING: norm-coherence file not found at {NORM_CSV_PATH}")

    # Pivot file: {latent_id (str): [example, ...]}
    if not PIVOT_PATH.is_file():
        print(f"ERROR: pivot file not found at {PIVOT_PATH}")
        return
    with PIVOT_PATH.open() as f:
        pivot = json.load(f)
    print(f"Loaded {len(pivot)} latents from {PIVOT_PATH}")

    # Restrict to latents that have more than MIN_EXAMPLES examples in the pivot.
    pivot_ids = set(pivot.keys())
    filtered = [
        (lid, score) for lid, score in rows
        if str(lid) in pivot_ids
        and isinstance(pivot.get(str(lid)), list)
        and len(pivot[str(lid)]) > MIN_EXAMPLES
    ]
    n_dropped = len(rows) - len(filtered)
    print(f"Restricting to {len(filtered)} latents with > {MIN_EXAMPLES} examples "
          f"(dropped {n_dropped}).")
    print()

    if not filtered:
        print("No latents remain after filtering.")
        return

    filtered.sort(key=lambda r: r[1])
    top = filtered[-N:][::-1]
    bottom = filtered[:N]

    top_blocks = [
        latent_block(lid, score, norm_map.get(str(lid)), pivot.get(str(lid)))
        for lid, score in top
    ]
    bottom_blocks = [
        latent_block(lid, score, norm_map.get(str(lid)), pivot.get(str(lid)))
        for lid, score in bottom
    ]

    print(f"=== Top {N} most coherent (with > {MIN_EXAMPLES} examples) ===")
    for b in top_blocks:
        print_latent_block(b)
        print()

    print(f"=== Bottom {N} least coherent (with > {MIN_EXAMPLES} examples) ===")
    for b in bottom_blocks:
        print_latent_block(b)
        print()

    # Save structured output as JSON next to this script.
    run_name = CSV_PATH.parent.parent.name
    out_path = Path(__file__).resolve().parent / f"top_bottom_coherence_{run_name}.json"
    out_obj = {
        "csv_path": str(CSV_PATH),
        "norm_csv_path": str(NORM_CSV_PATH),
        "pivot_path": str(PIVOT_PATH),
        "run_name": run_name,
        "n_total_coherence_rows": len(rows),
        "n_norm_coherence_rows": len(norm_map),
        "n_with_more_than_min_examples": len(filtered),
        "min_examples": MIN_EXAMPLES,
        "n_top_bottom": N,
        "top_most_coherent": top_blocks,
        "bottom_least_coherent": bottom_blocks,
    }
    with out_path.open("w") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"Wrote structured output to {out_path}")


if __name__ == "__main__":
    main()
