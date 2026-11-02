#!/usr/bin/env python3
"""
train_icml_sae.py

ICML-aligned SAE training script (Neuronpedia-oriented layer choices) using SAELens.

Updates in this version
- Make 16k the default width for Neuronpedia-style runs where appropriate.
- Allow GPT-2 to run at 16k (common Neuronpedia size).
- Make Gemma defaults 16k (keep 65k/131k as scaling options).
- Keep Pythia-160M default at 16k (SAEBench-compatible sweep: 4k/16k/65k).
- Keep Llama default at 32k (consistent with common large-model SAE suites), but you can add 16k if you want.

Key features
- Presets: model + Neuronpedia-style layer + dataset + d_in
- --width selects d_sae (4k/16k/32k/65k/128k)
- --architecture selects sparsity mechanism:
    standard (L1), topk, batchtopk, jumprelu, matryoshka
- Uses SAELens defaults for all runner hyperparams unless overridden via CLI
- Best-effort post-run sparsity report (avg L0, dead feature fraction) on a small sample
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from sae_lens import (
    LanguageModelSAERunnerConfig,
    LoggingConfig,
    LanguageModelSAETrainingRunner,
    # SAE configs (v6+)
    StandardTrainingSAEConfig,
    TopKTrainingSAEConfig,
    BatchTopKTrainingSAEConfig,
    JumpReLUTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAEConfig,
    # Poly variants
    PolyTopKTrainingSAEConfig,
    PolyBatchTopKTrainingSAEConfig,
    PolyJumpReLUTrainingSAEConfig,
    PolyMatryoshkaBatchTopKTrainingSAEConfig,
)


# ----------------------------
# Width parsing
# ----------------------------

WIDTH_TO_DSAE = {
    "4k": 4_096,
    "16k": 16_384,
    "32k": 32_768,
    "65k": 65_536,
    # many works say "128k"; typical exact count is 2^17 = 131072
    "128k": 131_072,
    "131k": 131_072,
}


def norm_width(w: str) -> str:
    w = w.strip().lower()
    if w not in WIDTH_TO_DSAE:
        raise ValueError(f"Unknown --width={w}. Choose from {sorted(WIDTH_TO_DSAE)}")
    return w


# ----------------------------
# Presets (Neuronpedia-oriented layers)
# ----------------------------

@dataclass(frozen=True)
class Preset:
    model_name: str
    hook_name: str
    dataset_path: str
    d_in: int
    allowed_widths: List[str]
    default_width: str


PRESETS: Dict[str, Preset] = {
    # GPT-2 Small: Neuronpedia-oriented choice -> layer 8 resid_post
    # Use Parquet OWT variant to avoid deprecated dataset scripts.
    "gpt2_l8": Preset(
        model_name="gpt2",
        hook_name="blocks.8.hook_resid_post",
        dataset_path="dylanebert/openwebtext",  # parquet-friendly
        d_in=768,
        allowed_widths=["16k", "32k", "65k"],   # UPDATED: allow 16k
        default_width="16k",                   # UPDATED: default 16k
    ),

    # Pythia-70M: mid & late (Neuronpedia-style; 16k default)
    "pythia70m_l3": Preset(
        model_name="EleutherAI/pythia-70m-deduped",
        hook_name="blocks.3.hook_resid_post",
        dataset_path="monology/pile-uncopyrighted",
        d_in=512,
        allowed_widths=["16k", "32k", "65k"],
        default_width="16k",                   # UPDATED
    ),
    "pythia70m_l5": Preset(
        model_name="EleutherAI/pythia-70m-deduped",
        hook_name="blocks.5.hook_resid_post",
        dataset_path="monology/pile-uncopyrighted",
        d_in=512,
        allowed_widths=["16k", "32k", "65k"],
        default_width="16k",                   # UPDATED
    ),

    # Pythia-160M: canonical benchmark layer; SAEBench-style widths
    "pythia160m_l8": Preset(
        model_name="EleutherAI/pythia-160m-deduped",
        hook_name="blocks.8.hook_resid_post",
        dataset_path="monology/pile-uncopyrighted",
        d_in=768,
        allowed_widths=["4k", "16k", "65k"],
        default_width="16k",                   # UPDATED: main paper at 16k; sweep as needed
    ),

    # Gemma-2-2B: mid & late; default 16k for main paper
    "gemma2_2b_l12": Preset(
        model_name="google/gemma-2-2b",
        hook_name="blocks.12.hook_resid_post",
        dataset_path="dylanebert/openwebtext",  # parquet-friendly
        d_in=2304,
        allowed_widths=["16k", "65k", "131k"],
        default_width="16k",                   # UPDATED
    ),
    "gemma2_2b_l19": Preset(
        model_name="google/gemma-2-2b",
        hook_name="blocks.19.hook_resid_post",
        dataset_path="dylanebert/openwebtext",  # parquet-friendly
        d_in=2304,
        allowed_widths=["16k", "65k", "131k"],
        default_width="16k",                   # UPDATED
    ),

    # Llama-3.1-8B: keep 32k default (common in large-model SAE suites)
    "llama31_8b_l25": Preset(
        model_name="meta-llama/Llama-3.1-8B",
        hook_name="blocks.25.hook_resid_post",
        dataset_path="cerebras/SlimPajama-627B",
        d_in=4096,
        allowed_widths=["32k", "128k"],
        default_width="32k",
    ),
}


# ----------------------------
# Matryoshka defaults
# ----------------------------

def default_matryoshka_widths(d_sae: int) -> List[int]:
    if d_sae <= 4096:
        return [d_sae]
    if d_sae <= 16384:
        return [512, 2048, d_sae]
    if d_sae <= 32768:
        return [512, 2048, 8192, d_sae]
    if d_sae <= 65536:
        return [1024, 4096, 16384, 32768, d_sae]
    return [2048, 8192, 32768, 65536, d_sae]


def parse_int_list(csv: Optional[str]) -> Optional[List[int]]:
    if csv is None:
        return None
    items = [x.strip() for x in csv.split(",") if x.strip()]
    if not items:
        return None
    return [int(x) for x in items]


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=sorted(PRESETS.keys()))
    p.add_argument("--width", default=None, help=f"One of {sorted(WIDTH_TO_DSAE.keys())}")

    p.add_argument(
        "--architecture",
        required=True,
        choices=["standard", "topk", "batchtopk", "jumprelu", "matryoshka"],
        help="Sparsity mechanism / SAE architecture.",
    )

    # Sparsity knobs
    p.add_argument("--use_saelens_defaults", action="store_true", help="Do not set sparsity knobs; rely on SAELens defaults.")
    p.add_argument("--l1_coefficient", type=float, default=None, help="For standard (L1) SAEs.")
    p.add_argument("--l0_coefficient", type=float, default=None, help="For JumpReLU SAEs.")
    p.add_argument("--k", type=int, default=None, help="For TopK/BatchTopK/MatryoshkaBatchTopK SAEs.")
    p.add_argument("--matryoshka_widths", type=str, default=None, help='CSV list ending in d_sae, e.g. "512,2048,8192,32768".')

    # PolySAE: polynomial decoder modifier (can be combined with any architecture)
    p.add_argument("--poly", action="store_true", help="Enable polynomial decoder (PolySAE)")
    p.add_argument("--poly_rank", type=int, default=None, help="Rank for PolySAE CP tensors (default: d_in)")
    p.add_argument("--lambda2_init", type=float, default=-0.5, help="Initial λ₂ for PolySAE")
    p.add_argument("--lambda3_init", type=float, default=0.5, help="Initial λ₃ for PolySAE")

    # Optional runner overrides (otherwise SAELens defaults)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train_batch_size_tokens", type=int, default=None)
    p.add_argument("--context_size", type=int, default=None)
    p.add_argument("--training_tokens", type=int, default=None)
    p.add_argument("--n_batches_in_buffer", type=int, default=None)

    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--wandb_project", default="icml_sae")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--n_checkpoints", type=int, default=None)

    # Sparsity reporting
    p.add_argument("--report_sparsity", action="store_true", help="After training, estimate avg L0 and dead features on a small sample.")
    p.add_argument("--sparsity_samples", type=int, default=64, help="Number of sequences to sample for sparsity estimate.")
    p.add_argument("--sparsity_context", type=int, default=None, help="Override context length for sparsity estimate (defaults to cfg.context_size).")

    return p.parse_args()


# ----------------------------
# SAE config factory
# ----------------------------

def build_sae_cfg(
    arch: str,
    d_in: int,
    d_sae: int,
    *,
    use_defaults: bool,
    l1_coefficient: Optional[float],
    l0_coefficient: Optional[float],
    k: Optional[int],
    matryoshka_widths_csv: Optional[str],
    # Poly decoder modifier
    use_poly: bool = False,
    poly_rank: Optional[int] = None,
    lambda2_init: float = -0.5,
    lambda3_init: float = 0.5,
):
    """
    Construct the correct SAE config object. If --use_saelens_defaults is set,
    we omit sparsity knobs (k / l1 / l0) and rely on SAELens defaults.
    
    If use_poly=True, returns the Poly* variant that adds polynomial decoder
    while keeping the same activation function and loss behavior.
    """
    if arch == "standard":
        if use_poly:
            raise SystemExit("--poly is not yet supported with standard architecture (only topk, batchtopk, jumprelu, matryoshka)")
        if use_defaults and l1_coefficient is None:
            return StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae)
        if l1_coefficient is None:
            raise SystemExit("--l1_coefficient required for standard unless --use_saelens_defaults works on your SAELens version.")
        return StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae, l1_coefficient=l1_coefficient)

    if arch == "topk":
        if use_poly:
            if use_defaults and k is None:
                return PolyTopKTrainingSAEConfig(
                    d_in=d_in, d_sae=d_sae,
                    poly_rank=poly_rank,
                    lambda2_init=lambda2_init,
                    lambda3_init=lambda3_init,
                )
            if k is None:
                raise SystemExit("--k required for topk unless --use_saelens_defaults")
            return PolyTopKTrainingSAEConfig(
                d_in=d_in, d_sae=d_sae, k=k,
                poly_rank=poly_rank,
                lambda2_init=lambda2_init,
                lambda3_init=lambda3_init,
            )
        # Non-poly TopK
        if use_defaults and k is None:
            return TopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae)
        if k is None:
            raise SystemExit("--k required for topk unless --use_saelens_defaults")
        return TopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, k=k)

    if arch == "batchtopk":
        if use_poly:
            if use_defaults and k is None:
                return PolyBatchTopKTrainingSAEConfig(
                    d_in=d_in, d_sae=d_sae,
                    poly_rank=poly_rank,
                    lambda2_init=lambda2_init,
                    lambda3_init=lambda3_init,
                )
            if k is None:
                raise SystemExit("--k required for batchtopk unless --use_saelens_defaults")
            return PolyBatchTopKTrainingSAEConfig(
                d_in=d_in, d_sae=d_sae, k=k,
                poly_rank=poly_rank,
                lambda2_init=lambda2_init,
                lambda3_init=lambda3_init,
            )
        # Non-poly BatchTopK
        if use_defaults and k is None:
            return BatchTopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae)
        if k is None:
            raise SystemExit("--k required for batchtopk unless --use_saelens_defaults")
        return BatchTopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, k=k)

    if arch == "jumprelu":
        if use_poly:
            if use_defaults and l0_coefficient is None:
                return PolyJumpReLUTrainingSAEConfig(
                    d_in=d_in, d_sae=d_sae,
                    poly_rank=poly_rank,
                    lambda2_init=lambda2_init,
                    lambda3_init=lambda3_init,
                )
            if l0_coefficient is None:
                raise SystemExit("--l0_coefficient required for jumprelu unless --use_saelens_defaults")
            return PolyJumpReLUTrainingSAEConfig(
                d_in=d_in, d_sae=d_sae,
                l0_coefficient=l0_coefficient,
                poly_rank=poly_rank,
                lambda2_init=lambda2_init,
                lambda3_init=lambda3_init,
            )
        # Non-poly JumpReLU
        if use_defaults and l0_coefficient is None:
            return JumpReLUTrainingSAEConfig(d_in=d_in, d_sae=d_sae)
        if l0_coefficient is None:
            raise SystemExit("--l0_coefficient required for jumprelu unless --use_saelens_defaults")
        return JumpReLUTrainingSAEConfig(d_in=d_in, d_sae=d_sae, l0_coefficient=l0_coefficient)

    if arch == "matryoshka":
        widths = parse_int_list(matryoshka_widths_csv)
        if widths is None:
            widths = default_matryoshka_widths(d_sae)

        if widths[-1] != d_sae:
            raise SystemExit(f"--matryoshka_widths must end with d_sae={d_sae}, got {widths[-1]}")
        if any(w <= 0 for w in widths):
            raise SystemExit(f"All matryoshka widths must be positive, got {widths}")
        if any(widths[i] >= widths[i + 1] for i in range(len(widths) - 1)):
            raise SystemExit(f"Matryoshka widths must be strictly increasing, got {widths}")

        if use_poly:
            if use_defaults and k is None:
                return PolyMatryoshkaBatchTopKTrainingSAEConfig(
                    d_in=d_in, d_sae=d_sae, matryoshka_widths=widths,
                    poly_rank=poly_rank,
                    lambda2_init=lambda2_init,
                    lambda3_init=lambda3_init,
                )
            if k is None:
                raise SystemExit("--k required for matryoshka unless --use_saelens_defaults")
            return PolyMatryoshkaBatchTopKTrainingSAEConfig(
                d_in=d_in, d_sae=d_sae, k=k, matryoshka_widths=widths,
                poly_rank=poly_rank,
                lambda2_init=lambda2_init,
                lambda3_init=lambda3_init,
            )
        # Non-poly Matryoshka
        if use_defaults and k is None:
            return MatryoshkaBatchTopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, matryoshka_widths=widths)
        if k is None:
            raise SystemExit("--k required for matryoshka unless --use_saelens_defaults")
        return MatryoshkaBatchTopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, k=k, matryoshka_widths=widths)

    raise SystemExit(f"Unknown architecture: {arch}")


# ----------------------------
# Best-effort sparsity reporting
# ----------------------------

def estimate_sparsity_stats(
    model_name: str,
    hook_name: str,
    dataset_path: str,
    sae,
    *,
    n_samples: int,
    context_size: int,
    device: str,
) -> Dict[str, float]:
    """
    Best-effort estimation of:
    - avg L0 (active features per token)
    - dead feature fraction
    """
    try:
        from datasets import load_dataset
        from transformer_lens import HookedTransformer
    except Exception as e:
        raise RuntimeError("Missing dependency for sparsity report. Install: pip install datasets transformer-lens") from e

    model = HookedTransformer.from_pretrained(model_name, device=device)

    ds = load_dataset(dataset_path, split="train", streaming=True)

    # Find a text-like field
    text_key = None
    for cand in ["text", "content", "document", "raw", "prompt"]:
        try:
            ex = next(iter(ds.take(1)))
            if cand in ex:
                text_key = cand
                break
        except Exception:
            break
    if text_key is None:
        ex = next(iter(ds.take(1)))
        for k, v in ex.items():
            if isinstance(v, str):
                text_key = k
                break
    if text_key is None:
        raise RuntimeError(f"Could not find a text field in dataset {dataset_path} example keys={list(ex.keys())}")

    prompts: List[str] = []
    for ex in ds:
        if isinstance(ex.get(text_key, None), str) and ex[text_key].strip():
            prompts.append(ex[text_key])
        if len(prompts) >= n_samples:
            break
    if not prompts:
        raise RuntimeError(f"No prompts found in dataset {dataset_path} using field '{text_key}'")

    activations: List[torch.Tensor] = []

    def hook_fn(act, hook):
        activations.append(act.detach())

    toks = model.to_tokens(prompts, truncate=True)
    if toks.shape[1] > context_size:
        toks = toks[:, :context_size]
    batch_size = 8

    all_l0: Optional[torch.Tensor] = None
    all_feat_sum: Optional[torch.Tensor] = None

    for i in range(0, toks.shape[0], batch_size):
        batch = toks[i : i + batch_size]
        activations.clear()
        with torch.no_grad():
            model.run_with_hooks(batch, fwd_hooks=[(hook_name, hook_fn)])
        if not activations:
            raise RuntimeError(f"Failed to capture activations at hook '{hook_name}'.")
        acts = activations[0]  # [b, s, d_in]

        if hasattr(sae, "encode"):
            feats = sae.encode(acts)
        else:
            out = sae(acts)
            if isinstance(out, tuple) and len(out) >= 2:
                feats = out[1]
            else:
                feats = out

        feats = feats.detach()
        l0 = (feats != 0).float().sum(dim=-1).reshape(-1)

        if all_l0 is None:
            all_l0 = l0
            all_feat_sum = feats.abs().sum(dim=(0, 1))
        else:
            all_l0 = torch.cat([all_l0, l0], dim=0)
            all_feat_sum = all_feat_sum + feats.abs().sum(dim=(0, 1))

    assert all_l0 is not None and all_feat_sum is not None
    avg_l0 = all_l0.mean().item()
    dead_frac = (all_feat_sum == 0).float().mean().item()
    return {"avg_l0": float(avg_l0), "dead_feature_frac": float(dead_frac)}


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = parse_args()
    preset = PRESETS[args.exp]

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    width = norm_width(args.width) if args.width else norm_width(preset.default_width)
    if width not in [norm_width(w) for w in preset.allowed_widths]:
        raise SystemExit(f"--width {width} not allowed for {args.exp}. Allowed: {preset.allowed_widths}")
    d_sae = WIDTH_TO_DSAE[width]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sae_cfg = build_sae_cfg(
        args.architecture,
        preset.d_in,
        d_sae,
        use_defaults=args.use_saelens_defaults,
        l1_coefficient=args.l1_coefficient,
        l0_coefficient=args.l0_coefficient,
        k=args.k,
        matryoshka_widths_csv=args.matryoshka_widths,
        use_poly=args.poly,
        poly_rank=args.poly_rank,
        lambda2_init=args.lambda2_init,
        lambda3_init=args.lambda3_init,
    )

    logger_cfg = LoggingConfig(
        log_to_wandb=(not args.no_wandb),
        wandb_project=args.wandb_project,
    )

    # Base runner config: SAELens defaults for everything not set here
    cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=preset.model_name,
        hook_name=preset.hook_name,
        dataset_path=preset.dataset_path,
        logger=logger_cfg,
        device=device,
        seed=args.seed,
        dataset_trust_remote_code=True,
        n_batches_in_buffer=128,
    )

    # Optional overrides
    if args.lr is not None:
        cfg.lr = args.lr
    if args.train_batch_size_tokens is not None:
        cfg.train_batch_size_tokens = args.train_batch_size_tokens
    if args.context_size is not None:
        cfg.context_size = args.context_size
    if args.training_tokens is not None:
        cfg.training_tokens = args.training_tokens
    if args.n_batches_in_buffer is not None:
        cfg.n_batches_in_buffer = args.n_batches_in_buffer
    if args.dtype is not None:
        cfg.dtype = args.dtype
    if args.n_checkpoints is not None:
        cfg.n_checkpoints = args.n_checkpoints

    # Checkpoint path with poly suffix if enabled
    if args.checkpoint_path is not None:
        cfg.checkpoint_path = args.checkpoint_path
    else:
        poly_suffix = "_poly" if args.poly else ""
        cfg.checkpoint_path = f"checkpoints_{args.exp}_{args.architecture}{poly_suffix}_{width}"

    print("=== SAELens run ===")
    print(f"exp:                {args.exp}")
    arch_str = args.architecture + (" + poly" if args.poly else "")
    print(f"architecture:       {arch_str}")
    print(f"model:              {preset.model_name}")
    print(f"hook:               {preset.hook_name}")
    print(f"dataset:            {preset.dataset_path}")
    print(f"d_in / d_sae:       {preset.d_in} / {d_sae} (width={width})")
    print(f"use_saelens_defaults:{args.use_saelens_defaults}")
    if not args.use_saelens_defaults:
        if args.architecture == "standard":
            print(f"l1_coefficient:     {args.l1_coefficient}")
        if args.architecture in {"topk", "batchtopk", "matryoshka"}:
            print(f"k:                  {args.k}")
        if args.architecture == "jumprelu":
            print(f"l0_coefficient:     {args.l0_coefficient}")
    if args.poly:
        print(f"poly_rank:          {args.poly_rank or preset.d_in} (default: d_in)")
        print(f"lambda2_init:       {args.lambda2_init}")
        print(f"lambda3_init:       {args.lambda3_init}")
    print(f"device:             {device}")
    print(f"checkpoint_path:    {cfg.checkpoint_path}")
    print("===================")

    runner = LanguageModelSAETrainingRunner(cfg)
    runner.run()

    if args.report_sparsity:
        sae_obj = getattr(runner, "sae", None)
        if sae_obj is None:
            sae_obj = getattr(getattr(runner, "trainer", None), "sae", None)
        if sae_obj is None:
            print("[warn] Could not find trained SAE object on runner; skipping sparsity estimate.")
            return

        ctx = args.sparsity_context if args.sparsity_context is not None else getattr(cfg, "context_size", 512)
        try:
            stats = estimate_sparsity_stats(
                model_name=preset.model_name,
                hook_name=preset.hook_name,
                dataset_path=preset.dataset_path,
                sae=sae_obj,
                n_samples=args.sparsity_samples,
                context_size=ctx,
                device=device,
            )
            print("=== Sparsity estimate (best-effort) ===")
            print(f"avg_L0_per_token:      {stats['avg_l0']:.2f}")
            print(f"dead_feature_fraction: {stats['dead_feature_frac']:.4f}")
            print("======================================")
        except Exception as e:
            print(f"[warn] Sparsity estimate failed: {e}")

    # Force exit to avoid Python finalization crash (threading/GIL issue with scipy/sklearn)
    os._exit(0)


if __name__ == "__main__":
    main()
