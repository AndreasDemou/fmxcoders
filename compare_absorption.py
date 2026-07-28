"""Run SAEBench feature-absorption eval on a crosscoder via single-layer view.

The crosscoder (factorized or not) is wrapped via `CrosscoderMixin.as_single_layer_view`
into a 2-D SAE matching SAEBench's expected interface, then handed to
`sae_bench.evals.absorption.main.run_eval`. Both factored and unfactored crosscoders
go through the same wrapper, so results are directly comparable.

Usage:
    python compare_absorption.py \
        --checkpoint /path/to/inference_sae_dir \
        --layer-idx 4 \
        --out-dir eval_results/crosscoder_absorption \
        [--model-name pythia-410m-deduped] \
        [--llm-dtype bfloat16] \
        [--llm-batch-size 10] \
        [--device cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sae_lens.saes.crosscoder_sae import CrosscoderMixin

from _crosscoder_eval_utils import load_crosscoder_inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to the inference-SAE directory (contains sae_weights.safetensors and cfg.json)")
    parser.add_argument("--layer-idx", required=True, type=int,
                        help="Crosscoder layer index to slice (0..n_layers-1)")
    parser.add_argument("--out-dir", default=None, type=Path,
                        help="Where SAEBench writes its JSON results. If omitted, defaults to "
                             "<repo>/eval_results/<run_name>/absorption/ where <run_name> is the "
                             "parent dir of --checkpoint. Created if missing.")
    parser.add_argument("--model-name", default=None,
                        help="HF model name (e.g. EleutherAI/pythia-410m-deduped). "
                             "If omitted, read from sae.cfg.metadata.model_name.")
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

    model_name = args.model_name or view.cfg.metadata.model_name
    if not model_name:
        raise ValueError(
            "Could not determine model_name from SAE metadata; pass --model-name."
        )
    view.cfg.metadata.model_name = model_name

    # Include the run-name (parent dir of the checkpoint) so distinct
    # checkpoints whose step folders share a name (e.g. "final_500002816")
    # don't collide on SAEBench's artifact filenames.
    run_name = args.checkpoint.resolve().parent.name
    name = args.name or f"{run_name}_layer{args.layer_idx}"

    if args.out_dir is None:
        repo_root = Path(__file__).resolve().parent
        run_name = args.checkpoint.resolve().parent.name
        out_dir = repo_root / "eval_results" / run_name / "absorption"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    import sae_bench.evals.absorption.main as saebench_absorption
    config = saebench_absorption.AbsorptionEvalConfig(
        model_name=model_name,
        llm_dtype=args.llm_dtype,
        llm_batch_size=args.llm_batch_size,
    )
    saebench_absorption.run_eval(
        config=config,
        selected_saes=[(name, view)],
        device=args.device,
        output_path=str(out_dir),
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
