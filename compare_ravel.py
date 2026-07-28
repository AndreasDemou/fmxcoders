"""Run SAEBench RAVEL eval on a crosscoder via single-layer view.

The crosscoder (factorized or not) is wrapped via `CrosscoderMixin.as_single_layer_view`
into a 2-D SAE matching SAEBench's expected interface, then handed to
`sae_bench.evals.ravel.main.run_eval`. Both factored and unfactored crosscoders
go through the same wrapper, so results are directly comparable.

RAVEL only supports a small set of models (see `LLM_NAME_MAP` in
SAEBench/sae_bench/evals/ravel/main.py) — currently `gemma-2-2b` and
`pythia-160m-deduped`. The `--model-name` value passed here must match a key
of that map.

Usage:
    python compare_ravel.py \
        --checkpoint /path/to/inference_sae_dir \
        --layer-idx 4 \
        --model-name pythia-160m-deduped \
        --out-dir eval_results/crosscoder_ravel \
        [--llm-dtype bfloat16] \
        [--llm-batch-size 10] \
        [--device cuda]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from sae_lens.saes.crosscoder_sae import CrosscoderMixin

from _crosscoder_eval_utils import load_crosscoder_inference


def _stage_ravel_prompts_offline(artifact_dir: Path) -> None:
    """Pre-stage `adamkarvonen/ravel_prompts` JSONs from local HF cache.

    SAEBench's RAVEL eval (instance.py:234) calls `snapshot_download` if any
    expected JSON is missing from `<artifact_dir>/base/`. With HF_HUB_OFFLINE=1
    that download fails even when the dataset is cached, because the revision
    lookup hits the Hub. By symlinking the cached JSONs into base_dir up front
    we satisfy the file-existence check and skip the download entirely.
    """
    base_dir = artifact_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    if any(base_dir.glob("ravel_*.json")):
        return  # already populated

    cache_root = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not cache_root:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE as _hub_cache
            cache_root = _hub_cache
        except Exception:
            return  # nothing we can do; let SAEBench try and error informatively

    repo_dir = Path(cache_root) / "datasets--adamkarvonen--ravel_prompts"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return

    snapshot_dirs = sorted(
        (p for p in snapshots.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not snapshot_dirs:
        return
    snapshot = snapshot_dirs[0]

    json_files = list(snapshot.glob("*.json"))
    if not json_files:
        return

    for src in json_files:
        dst = base_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src.resolve())
    print(f"[compare_ravel] staged {len(json_files)} RAVEL prompt files from "
          f"{snapshot} → {base_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to the inference-SAE directory (contains sae_weights.safetensors and cfg.json)")
    parser.add_argument("--layer-idx", required=True, type=int,
                        help="Crosscoder layer index to slice (0..n_layers-1)")
    parser.add_argument("--model-name", default=None,
                        help="Short model name from RAVEL's LLM_NAME_MAP (e.g. 'pythia-160m-deduped', "
                             "'gemma-2-2b'). If omitted, auto-derived from sae.cfg.metadata.model_name "
                             "via reverse lookup in LLM_NAME_MAP.")
    parser.add_argument("--out-dir", default=None, type=Path,
                        help="Where SAEBench writes its JSON results. If omitted, defaults to "
                             "<repo>/eval_results/<run_name>/ravel/ where <run_name> is the "
                             "parent dir of --checkpoint. Created if missing.")
    parser.add_argument("--llm-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--llm-batch-size", default=10, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--name", default=None,
                        help="Name to embed in result filenames (default: <checkpoint>_layer<i>)")
    args = parser.parse_args()

    sae = load_crosscoder_inference(args.checkpoint, device=args.device, dtype=args.llm_dtype)
    if not isinstance(sae, CrosscoderMixin):
        raise ValueError(
            f"Loaded SAE is not a crosscoder ({type(sae).__name__}). "
            "This script is for crosscoders only."
        )

    view = sae.as_single_layer_view(args.layer_idx).to(args.device)

    # Include the run-name (parent dir of the checkpoint) so distinct
    # checkpoints whose step folders share a name (e.g. "final_500002816")
    # don't collide on SAEBench's artifact filenames.
    run_name = args.checkpoint.resolve().parent.name
    name = args.name or f"{run_name}_layer{args.layer_idx}"

    import sae_bench.evals.ravel.main as saebench_ravel
    if args.model_name is not None:
        short_name = args.model_name
    else:
        # Reverse-lookup: SAE metadata stores the full HF id (e.g.
        # "EleutherAI/pythia-160m-deduped"); RAVEL's config wants the short
        # key (e.g. "pythia-160m-deduped"). Accept either form.
        sae_model_name = view.cfg.metadata.model_name
        inverse_map = {full: short for short, full in saebench_ravel.LLM_NAME_MAP.items()}
        if sae_model_name in saebench_ravel.LLM_NAME_MAP:
            short_name = sae_model_name
        elif sae_model_name in inverse_map:
            short_name = inverse_map[sae_model_name]
        else:
            raise ValueError(
                f"Cannot auto-derive RAVEL model from sae.cfg.metadata.model_name="
                f"{sae_model_name!r}. RAVEL only supports {sorted(saebench_ravel.LLM_NAME_MAP)}; "
                "pass --model-name explicitly with one of those keys."
            )

    if short_name not in saebench_ravel.LLM_NAME_MAP:
        raise ValueError(
            f"model_name {short_name!r} is not in RAVEL's LLM_NAME_MAP "
            f"({sorted(saebench_ravel.LLM_NAME_MAP)})."
        )

    if args.out_dir is None:
        repo_root = Path(__file__).resolve().parent
        run_name = args.checkpoint.resolve().parent.name
        out_dir = repo_root / "eval_results" / run_name / "ravel"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    config = saebench_ravel.RAVELEvalConfig(
        model_name=short_name,
        llm_dtype=args.llm_dtype,
        llm_batch_size=args.llm_batch_size,
    )

    # Pre-stage RAVEL prompts from HF cache so SAEBench's snapshot_download
    # is skipped under HF_HUB_OFFLINE=1.
    _stage_ravel_prompts_offline(Path(config.artifact_dir))

    saebench_ravel.run_eval(
        config=config,
        selected_saes=[(name, view)],
        device=args.device,
        output_path=str(out_dir),
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
