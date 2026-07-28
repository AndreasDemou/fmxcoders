"""Standalone reconstruction-quality eval for crosscoder SAEs.

Aligned with SAEBench Core (sparsity+variance branch) for apples-to-apples
comparison with single-layer SAE numbers. Matches:
  - token provisioning via `ActivationsStore.from_sae` + `shuffle_input_dataset(seed=42)`
  - special-token masking (PAD / EOS / BOS)
  - per-batch accumulation then mean-over-batches (same as
    `get_sparsity_and_variance_metrics` in sae_bench/evals/core/main.py)
  - SAEBench's `explained_variance` formula  ( total_var = E[||x||^2] - (E[x^2])^2 )
    and `explained_variance_legacy`           ( per-token ratio of residual / centred )

The difference from SAEBench Core is scope, not methodology: this script does
only A+B+C metrics (no KL / CE — those need multi-layer model replay) and
reports per-layer plus layer-averaged scalars, matching the crosscoder branch
added to `SAEBench/sae_bench/evals/core/main.py`.

Usage:
    python eval_crosscoder_reconstruction.py \\
        --checkpoint_path checkpoints/xyz/final_XXXXX \\
        --dataset Skylion007/openwebtext \\
        --n_batches 2000 --batch_size_prompts 16 --context_size 128
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import einops
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens.saes.sae import SAE, TrainingSAE
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name


def _load_sae_or_training_sae(path: str, device: str) -> SAE:
    """Crosscoder architectures like `cross_batchtopk` are only registered in
    SAE_TRAINING_CLASS_REGISTRY (see sae_lens/__init__.py), so a plain
    SAE.load_from_disk raises KeyError. Fall back to TrainingSAE's loader
    which hits the training registry."""
    try:
        return SAE.load_from_disk(path, device=device)
    except KeyError:
        return TrainingSAE.load_from_disk(path, device=device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True,
                   help="Path to the saved SAE directory (contains cfg.json + state_dict).")
    p.add_argument("--dataset", default=None,
                   help="HF dataset id. Defaults to the one recorded in the SAE metadata, or Skylion007/openwebtext.")
    p.add_argument("--model_name", default=None,
                   help="Override model id. Defaults to metadata.model_name.")
    p.add_argument("--n_batches", type=int, default=2000,
                   help="Batches to evaluate (SAEBench default for sparsity+variance is 2000).")
    p.add_argument("--batch_size_prompts", type=int, default=16,
                   help="Prompts per batch (SAEBench default is 16).")
    p.add_argument("--context_size", type=int, default=128,
                   help="Tokens per prompt (SAEBench default is 128).")
    p.add_argument("--shuffle_seed", type=int, default=42,
                   help="Seed passed to ActivationsStore.shuffle_input_dataset (SAEBench uses 42).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="Write metrics JSON to this path (stdout only otherwise).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading crosscoder from {args.checkpoint_path}", flush=True)
    sae = _load_sae_or_training_sae(args.checkpoint_path, str(device))
    sae.eval()

    meta = sae.cfg.metadata
    hook_names = getattr(meta, "hook_names", None) or getattr(sae.cfg, "hook_names", None)
    if not hook_names or not isinstance(hook_names, (list, tuple)) or len(hook_names) < 2:
        print(
            f"ERROR: this checkpoint is not a crosscoder (hook_names={hook_names!r}). "
            "Use SAEBench's Core eval for single-layer SAEs.",
            file=sys.stderr,
        )
        return 2
    hook_names = list(hook_names)
    n_layers = len(hook_names)

    model_name = args.model_name or meta.model_name
    dataset = args.dataset or getattr(meta, "dataset_path", None) or "Skylion007/openwebtext"

    print(f"model             : {model_name}")
    print(f"hook_names        : {hook_names}")
    print(f"dataset           : {dataset}")
    print(f"context_size      : {args.context_size}   batch_size_prompts: {args.batch_size_prompts}   n_batches: {args.n_batches}")

    print("Loading model...", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(
        model_name=model_name, device=str(device)
    )
    model.eval()

    # Use ActivationsStore for token provisioning so tokenization, packing and
    # shuffling match SAEBench exactly. from_sae treats the SAE as single-hook
    # but we only call get_batch_tokens, which is hook-agnostic.
    print("Creating ActivationsStore for token provisioning...", flush=True)
    activation_store = ActivationsStore.from_sae(
        model,  # type: ignore
        sae,
        context_size=args.context_size,
        dataset=dataset,
        dataset_trust_remote_code=True,
    )
    activation_store.shuffle_input_dataset(seed=args.shuffle_seed)

    # SAEBench ignore_tokens set
    tokenizer = model.tokenizer
    ignore_tokens: set[int | None] = {
        tokenizer.pad_token_id,  # type: ignore
        tokenizer.eos_token_id,  # type: ignore
        tokenizer.bos_token_id,  # type: ignore
    }
    ignore_tokens_list = [t for t in ignore_tokens if t is not None]
    print(f"ignore_tokens     : {ignore_tokens_list}")

    stop_layers = [extract_stop_at_layer_from_tlens_hook_name(h) for h in hook_names]
    valid_stops = [l for l in stop_layers if l is not None]
    stop_at_layer = max(valid_stops) + 1 if valid_stops else None

    # Per-batch accumulators (matching SAEBench's layout)
    per_layer: list[dict[str, list[torch.Tensor]]] = [
        {k: [] for k in (
            "l2_norm_in", "l2_norm_out", "l2_ratio", "rrb",
            "mse", "cossim", "ev_legacy",
            "mean_sq", "mean_per_dim", "mean_resid_sq",
        )}
        for _ in range(n_layers)
    ]
    l0_vals: list[torch.Tensor] = []
    l1_vals: list[torch.Tensor] = []
    # Per-feature firing counts (across unmasked eval tokens), for dead/below-threshold stats
    d_sae = int(sae.cfg.d_sae)
    feature_fire_count = torch.zeros(d_sae, dtype=torch.long, device=str(device))
    total_unmasked_tokens = 0

    print("Collecting activations and evaluating reconstruction...", flush=True)
    with torch.no_grad():
        for _ in tqdm(range(args.n_batches), desc="batches"):
            batch_tokens = activation_store.get_batch_tokens(args.batch_size_prompts)

            if ignore_tokens_list:
                mask = torch.logical_not(
                    torch.any(
                        torch.stack(
                            [batch_tokens == t for t in ignore_tokens_list], dim=0
                        ),
                        dim=0,
                    )
                )
            else:
                mask = torch.ones_like(batch_tokens, dtype=torch.bool)
            flat_mask = mask.flatten()

            _, cache = model.run_with_cache(
                batch_tokens,
                prepend_bos=False,
                names_filter=hook_names,
                stop_at_layer=stop_at_layer,
            )
            original_act = torch.stack([cache[h] for h in hook_names], dim=2)  # [B, T, L, D]
            del cache

            sae_feat = sae.encode(original_act.to(sae.dtype))         # [B, T, d_sae]
            sae_out = sae.decode(sae_feat).to(original_act.device)    # [B, T, L, D]

            # Sparsity on the shared latent
            feats_flat = einops.rearrange(sae_feat, "b ctx d -> (b ctx) d")[flat_mask]
            l0_vals.append((feats_flat != 0).float().sum(dim=-1))
            l1_vals.append(feats_flat.sum(dim=-1))
            feature_fire_count += (feats_flat != 0).sum(dim=0).long()
            total_unmasked_tokens += int(feats_flat.shape[0])

            for li in range(n_layers):
                m = per_layer[li]
                x = einops.rearrange(original_act[:, :, li, :], "b ctx d -> (b ctx) d")[flat_mask]
                xh = einops.rearrange(sae_out[:, :, li, :], "b ctx d -> (b ctx) d")[flat_mask]

                # L2 / shrinkage
                l2_in = torch.norm(x, dim=-1)
                l2_out = torch.norm(xh, dim=-1)
                l2_div = l2_in.clone()
                l2_div[l2_div.abs() < 1e-4] = 1
                rrb = (
                    torch.norm(xh, dim=-1).pow(2).mean()
                    / (x * xh).sum(dim=-1).mean()
                ).unsqueeze(0)
                m["l2_norm_in"].append(l2_in)
                m["l2_norm_out"].append(l2_out)
                m["l2_ratio"].append(l2_out / l2_div)
                m["rrb"].append(rrb)

                # variance / reconstruction (SAEBench formulas)
                resid_sq = (x - xh).pow(2).sum(dim=-1)
                mse = resid_sq / flat_mask.sum()
                batched_var = (x - x.mean(dim=0)).pow(2).sum(dim=-1)
                ev_legacy = 1 - resid_sq / batched_var.clamp(min=1e-12)
                x_n = x / torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
                xh_n = xh / torch.norm(xh, dim=-1, keepdim=True).clamp(min=1e-8)
                m["mse"].append(mse)
                m["cossim"].append((x_n * xh_n).sum(dim=-1))
                m["ev_legacy"].append(ev_legacy)
                m["mean_sq"].append(x.pow(2).sum(dim=-1).mean(dim=0))
                m["mean_per_dim"].append(x.pow(2).mean(dim=0))
                m["mean_resid_sq"].append(resid_sq.mean(dim=0))

    # Aggregate per-layer scalars exactly like SAEBench does
    per_layer_out: list[dict[str, Any]] = []
    for li in range(n_layers):
        m = per_layer[li]
        out: dict[str, Any] = {"hook_name": hook_names[li]}
        out["l2_norm_in"] = torch.cat(m["l2_norm_in"]).mean().item()
        out["l2_norm_out"] = torch.cat(m["l2_norm_out"]).mean().item()
        out["l2_ratio"] = torch.cat(m["l2_ratio"]).mean().item()
        out["relative_reconstruction_bias"] = torch.cat(m["rrb"]).mean().item()
        out["mse"] = torch.cat(m["mse"]).mean().item()
        out["cossim"] = torch.cat(m["cossim"]).mean().item()
        out["explained_variance_legacy"] = torch.cat(m["ev_legacy"]).mean().item()
        # SAEBench's explained_variance: total_var = E[||x||^2] - (E[x^2])^2
        mss = torch.stack(m["mean_sq"]).mean()
        mapd = torch.cat(m["mean_per_dim"]).mean()
        total_var = mss - mapd ** 2
        resid_var = torch.stack(m["mean_resid_sq"]).mean()
        out["explained_variance"] = (1 - resid_var / total_var).item()
        per_layer_out.append(out)

    # Layer-averaged scalars (canonical, directly comparable to single-layer numbers)
    agg: dict[str, float] = {}
    scalar_keys = [k for k in per_layer_out[0] if k != "hook_name"]
    for k in scalar_keys:
        agg[k] = float(sum(o[k] for o in per_layer_out) / n_layers)
    agg["l0"] = torch.cat(l0_vals).mean().item()
    agg["l1"] = torch.cat(l1_vals).mean().item()

    # Per-feature firing-rate statistics over unmasked eval tokens.
    # "dead_latents" = features that never fired on any eval token.
    # "below_1e-5" / "below_1e-6" mirror the wandb training-time metric but
    # computed on this eval corpus.
    if total_unmasked_tokens > 0:
        fire_rate = feature_fire_count.float() / float(total_unmasked_tokens)
        agg["dead_latents"] = int((feature_fire_count == 0).sum().item())
        agg["below_1e-5"] = int((fire_rate < 1e-5).sum().item())
        agg["below_1e-6"] = int((fire_rate < 1e-6).sum().item())
        agg["frac_dead"] = agg["dead_latents"] / d_sae
        agg["unmasked_eval_tokens"] = int(total_unmasked_tokens)

    out = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "model_name": model_name,
        "dataset": dataset,
        "hook_names": hook_names,
        "n_tokens_evaluated": args.n_batches * args.batch_size_prompts * args.context_size,
        "ignore_tokens": ignore_tokens_list,
        "shuffle_seed": args.shuffle_seed,
        "d_sae": d_sae,
        "aggregate": agg,
        "per_layer": per_layer_out,
    }

    print("\n=== Aggregate (mean over layers) ===")
    for k, v in agg.items():
        print(f"  {k:32s} {v:.6g}")

    print("\n=== Per-layer ===")
    for m in per_layer_out:
        print(f"  {m['hook_name']}")
        for k, v in m.items():
            if k == "hook_name":
                continue
            print(f"    {k:30s} {v:.6g}")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
