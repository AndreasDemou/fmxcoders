#!/usr/bin/env python
"""Evaluate crosscoder latent mask coherence (PolySAE_dev version).

For each latent j and each layer l:
  1. Run inputs through the crosscoder with all layers present → z_j^full
  2. Run the same inputs with layer l masked         → z_j^mask(l)
  3. Stability_j^l = E_x[|z_j^full - z_j^mask(l)| / (|z_j^full| + epsilon)]
  4. Concentration_j = max_l(Stability_j^l) / mean_l(Stability_j^l)
  5. MaskCoherence_j = 1 / Concentration_j

Coherent latent: similar stability across layers → concentration ≈ 1 → coherence ≈ 1
Incoherent latent: one layer dominates          → concentration >> 1 → coherence << 1

Data-dependent counterpart to `eval_norm_coherence.py`. Runs n_layers+1 forward
passes per batch (the unmasked + one per masked layer), so cost scales linearly
with the number of crosscoder layers.

Usage:
    python eval_mask_coherence.py --checkpoint_path /path/to/final_XXXXX
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens.saes.sae import SAE, TrainingSAE
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate crosscoder mask coherence")
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Path to the saved SAE directory.")
    p.add_argument("--model_name", type=str, default=None,
                   help="Base model id. Defaults to metadata.model_name.")
    p.add_argument("--dataset", type=str, default=None,
                   help="HF dataset id. Defaults to metadata.dataset_path, then Skylion007/openwebtext.")
    p.add_argument("--n_batches", type=int, default=100,
                   help="Batches to evaluate.")
    p.add_argument("--batch_size_prompts", type=int, default=16,
                   help="Prompts per batch.")
    p.add_argument("--context_size", type=int, default=128,
                   help="Tokens per prompt.")
    p.add_argument("--shuffle_seed", type=int, default=42)
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--masking_strategy", type=str, default="zero",
                   choices=["zero", "mean"],
                   help="'zero' overwrites the masked layer with zeros; 'mean' overwrites with the batch-mean across tokens.")
    p.add_argument("--output_dir", type=str, default="./mask_coherence_results",
                   help="Directory to write plots + JSON + CSV + NPY.")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="Crosscoders")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _load_sae_or_training_sae(path: str, device: str) -> SAE:
    """Crosscoder architectures like `cross_batchtopk` are only registered in
    SAE_TRAINING_CLASS_REGISTRY. Fall back to TrainingSAE when SAE lookup fails."""
    try:
        return SAE.load_from_disk(path, device=device)
    except KeyError:
        return TrainingSAE.load_from_disk(path, device=device)


def _layer_indices_from_hook_names(hook_names: list[str]) -> list[int]:
    out: list[int] = []
    for h in hook_names:
        m = re.search(r"blocks\.(\d+)", h)
        out.append(int(m.group(1)) if m else len(out))
    return out


@torch.no_grad()
def _encode_with_optional_mask(
    sae: SAE,
    model: HookedTransformer,
    batch_tokens: torch.Tensor,
    hook_names: list[str],
    stop_at_layer: int | None,
    mask_layer_idx: int | None,
    masking_strategy: str,
) -> torch.Tensor:
    """Run the model, stack activations across hook_names, optionally overwrite
    one layer's slice, and run the crosscoder encoder.

    Returns feature activations of shape [batch, seq, d_sae].
    """
    _, cache = model.run_with_cache(
        batch_tokens,
        names_filter=hook_names,
        stop_at_layer=stop_at_layer,
        prepend_bos=False,
    )
    stacked = torch.stack([cache[h] for h in hook_names], dim=2)  # [B, T, L, D]
    del cache

    if mask_layer_idx is not None:
        if masking_strategy == "zero":
            stacked[:, :, mask_layer_idx, :] = 0.0
        elif masking_strategy == "mean":
            layer_slice = stacked[:, :, mask_layer_idx, :]
            stacked[:, :, mask_layer_idx, :] = layer_slice.mean(dim=(0, 1), keepdim=True)
        else:
            raise ValueError(f"Unknown masking_strategy: {masking_strategy!r}")

    feats = sae.encode(stacked.to(sae.dtype))  # [B, T, d_sae]
    return feats


def compute_stability(
    sae: SAE,
    model: HookedTransformer,
    activation_store: ActivationsStore,
    hook_names: list[str],
    n_batches: int,
    batch_size_prompts: int,
    epsilon: float,
    masking_strategy: str,
) -> dict[str, np.ndarray]:
    n_layers = len(hook_names)
    n_latents = sae.cfg.d_sae

    stop_layers = [extract_stop_at_layer_from_tlens_hook_name(h) for h in hook_names]
    valid_stops = [l for l in stop_layers if l is not None]
    stop_at_layer = max(valid_stops) + 1 if valid_stops else None

    stability_sum = torch.zeros(n_latents, n_layers, dtype=torch.float32)

    print(f"\nComputing stability ({n_batches} batches × {1 + n_layers} forward passes each)...")
    for _ in tqdm(range(n_batches), desc="batches"):
        batch_tokens = activation_store.get_batch_tokens(batch_size_prompts)

        z_full = _encode_with_optional_mask(
            sae, model, batch_tokens, hook_names, stop_at_layer,
            mask_layer_idx=None, masking_strategy=masking_strategy,
        )  # [B, T, d_sae]

        for li in range(n_layers):
            z_masked = _encode_with_optional_mask(
                sae, model, batch_tokens, hook_names, stop_at_layer,
                mask_layer_idx=li, masking_strategy=masking_strategy,
            )
            # |z_full - z_masked| / (|z_full| + epsilon), averaged over batch + seq
            stab = (
                (z_full - z_masked).abs()
                / (z_full.abs() + epsilon)
            ).mean(dim=(0, 1)).float().cpu()  # [d_sae]
            stability_sum[:, li] += stab

    stability = (stability_sum / n_batches).numpy()

    stability_max = stability.max(axis=1)
    stability_mean = stability.mean(axis=1)
    concentration = np.maximum(stability_max / (stability_mean + epsilon), 1.0)
    mask_coherence = 1.0 / concentration

    return {
        "stability": stability,
        "concentration": concentration,
        "mask_coherence": mask_coherence,
        "stability_max": stability_max,
        "stability_mean": stability_mean,
    }


def create_visualizations(
    metrics: dict[str, np.ndarray],
    info: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    os.makedirs(output_dir, exist_ok=True)
    plots: dict[str, Any] = {}

    layer_indices = info["layer_indices"]
    n_latents, n_layers = metrics["stability"].shape

    # 1. Mask coherence distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(metrics["mask_coherence"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(metrics["mask_coherence"].mean(), color="red", linestyle="--",
               label=f"Mean: {metrics['mask_coherence'].mean():.3f}")
    ax.axvline(np.median(metrics["mask_coherence"]), color="orange", linestyle="--",
               label=f"Median: {np.median(metrics['mask_coherence']):.3f}")
    ax.set_xlabel("Mask Coherence")
    ax.set_ylabel("Number of Latents")
    ax.set_title("Distribution of Mask Coherence Scores")
    ax.set_xlim([0, 1])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plots["mask_coherence_distribution"] = fig
    fig.savefig(f"{output_dir}/mask_coherence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Concentration distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(metrics["concentration"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(metrics["concentration"].mean(), color="red", linestyle="--",
               label=f"Mean: {metrics['concentration'].mean():.3f}")
    ax.axvline(np.median(metrics["concentration"]), color="orange", linestyle="--",
               label=f"Median: {np.median(metrics['concentration']):.3f}")
    ax.set_xlabel("Concentration")
    ax.set_ylabel("Number of Latents")
    ax.set_title("Distribution of Concentration Scores\n(Higher = more layer-specific)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plots["concentration_distribution"] = fig
    fig.savefig(f"{output_dir}/concentration_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Heatmaps for top coherent / incoherent latents
    coherence_order = np.argsort(metrics["mask_coherence"])[::-1]
    n_show = min(50, n_latents // 2)
    for name, indices in [
        ("coherent", coherence_order[:n_show]),
        ("incoherent", coherence_order[-n_show:]),
    ]:
        fig, ax = plt.subplots(figsize=(12, 10))
        subset = metrics["stability"][indices, :]
        if sns is not None:
            sns.heatmap(
                subset,
                xticklabels=layer_indices,
                yticklabels=[f"L{i}" for i in indices],
                cmap="YlOrRd",
                cbar_kws={"label": "Stability"},
                ax=ax,
            )
        else:
            im = ax.imshow(subset, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(n_layers))
            ax.set_xticklabels(layer_indices)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([f"L{i}" for i in indices])
            plt.colorbar(im, ax=ax, label="Stability")
        ax.set_xlabel("Masked Layer")
        ax.set_ylabel("Latent")
        ax.set_title(f"Stability Matrix - Top {n_show} {'Coherent' if name == 'coherent' else 'Incoherent'} Latents")
        plt.tight_layout()
        plots[f"stability_heatmap_{name}"] = fig
        fig.savefig(f"{output_dir}/stability_heatmap_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. Concentration vs mean stability scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        metrics["stability_mean"], metrics["concentration"],
        c=metrics["mask_coherence"], cmap="viridis", alpha=0.6, s=10,
    )
    ax.set_xlabel("Mean Stability (averaged over layers)")
    ax.set_ylabel("Concentration (max/mean)")
    ax.set_title("Concentration vs Mean Stability\n(Color = Mask Coherence)")
    ax.set_yscale("log")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Mask Coherence")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plots["concentration_vs_stability"] = fig
    fig.savefig(f"{output_dir}/concentration_vs_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Average stability per layer
    fig, ax = plt.subplots(figsize=(10, 6))
    avg = metrics["stability"].mean(axis=0)
    ax.bar(range(n_layers), avg, tick_label=[f"L{idx}" for idx in layer_indices])
    ax.set_xlabel("Masked Layer")
    ax.set_ylabel("Average Stability (over all latents)")
    ax.set_title("Average Stability per Masked Layer")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plots["avg_stability_per_layer"] = fig
    fig.savefig(f"{output_dir}/avg_stability_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return plots


def save_results(
    metrics: dict[str, np.ndarray],
    info: dict[str, Any],
    output_dir: str,
) -> None:
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/stability.npy", metrics["stability"])
    np.save(f"{output_dir}/concentration.npy", metrics["concentration"])
    np.save(f"{output_dir}/mask_coherence.npy", metrics["mask_coherence"])

    def _stats(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }

    summary = {
        **info,
        "mask_coherence": _stats(metrics["mask_coherence"]),
        "concentration": _stats(metrics["concentration"]),
    }
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    df = pd.DataFrame({
        "latent_idx": np.arange(len(metrics["mask_coherence"])),
        "mask_coherence": metrics["mask_coherence"],
        "concentration": metrics["concentration"],
        "stability_mean": metrics["stability_mean"],
        "stability_max": metrics["stability_max"],
    })
    df.to_csv(f"{output_dir}/per_latent_metrics.csv", index=False)

    print(f"\nResults saved to {output_dir}")
    for fn in ("stability.npy", "concentration.npy", "mask_coherence.npy",
               "summary.json", "per_latent_metrics.csv"):
        print(f"  - {fn}")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"mask_coherence_{Path(args.checkpoint_path).name}",
            config=vars(args),
        )

    print(f"Loading crosscoder from {args.checkpoint_path}", flush=True)
    sae = _load_sae_or_training_sae(args.checkpoint_path, str(device))
    sae.eval()

    meta = sae.cfg.metadata
    hook_names = getattr(meta, "hook_names", None) or getattr(sae.cfg, "hook_names", None)
    if not hook_names or not isinstance(hook_names, (list, tuple)) or len(hook_names) < 2:
        print(
            f"ERROR: not a crosscoder (hook_names={hook_names!r}).",
            file=sys.stderr,
        )
        return 2
    hook_names = list(hook_names)
    layer_indices = _layer_indices_from_hook_names(hook_names)

    model_name = args.model_name or meta.model_name
    dataset = args.dataset or getattr(meta, "dataset_path", None) or "Skylion007/openwebtext"

    print(f"model             : {model_name}")
    print(f"hook_names        : {hook_names}")
    print(f"dataset           : {dataset}")
    print(f"context_size      : {args.context_size}   batch_size_prompts: {args.batch_size_prompts}   n_batches: {args.n_batches}")
    print(f"masking_strategy  : {args.masking_strategy}")

    print("Loading model...", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(
        model_name=model_name, device=str(device)
    )
    model.eval()

    print("Creating ActivationsStore for token provisioning...", flush=True)
    activation_store = ActivationsStore.from_sae(
        model,  # type: ignore
        sae,
        context_size=args.context_size,
        dataset=dataset,
        dataset_trust_remote_code=True,
    )
    activation_store.shuffle_input_dataset(seed=args.shuffle_seed)

    metrics = compute_stability(
        sae=sae,
        model=model,
        activation_store=activation_store,
        hook_names=hook_names,
        n_batches=args.n_batches,
        batch_size_prompts=args.batch_size_prompts,
        epsilon=args.epsilon,
        masking_strategy=args.masking_strategy,
    )

    info = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "model_name": model_name,
        "dataset": dataset,
        "hook_names": hook_names,
        "layer_indices": layer_indices,
        "n_layers": len(hook_names),
        "n_latents": int(sae.cfg.d_sae),
        "masking_strategy": args.masking_strategy,
        "n_tokens_evaluated": args.n_batches * args.batch_size_prompts * args.context_size,
    }

    nc = metrics["mask_coherence"]
    conc = metrics["concentration"]

    print("\n" + "=" * 80)
    print("MASK COHERENCE EVALUATION RESULTS")
    print("=" * 80)
    print("\nMask coherence:")
    print(f"  mean / median: {nc.mean():.4f} / {np.median(nc):.4f}")
    print(f"  std:           {nc.std():.4f}")
    print(f"  min / max:     {nc.min():.4f} / {nc.max():.4f}")
    print(f"  P25 / P75:     {np.percentile(nc, 25):.4f} / {np.percentile(nc, 75):.4f}")

    print("\nConcentration:")
    print(f"  mean / median: {conc.mean():.4f} / {np.median(conc):.4f}")
    print(f"  std:           {conc.std():.4f}")
    print(f"  min / max:     {conc.min():.4f} / {conc.max():.4f}")

    best = int(np.argmax(nc))
    worst = int(np.argmin(nc))
    print(f"\nMost coherent latent:   {best}")
    print(f"  coherence: {nc[best]:.4f}   concentration: {conc[best]:.4f}")
    print(f"  stability per layer: {metrics['stability'][best]}")
    print(f"\nMost incoherent latent: {worst}")
    print(f"  coherence: {nc[worst]:.4f}   concentration: {conc[worst]:.4f}")
    print(f"  stability per layer: {metrics['stability'][worst]}")

    print(f"\nCreating visualizations in {args.output_dir} ...")
    plots = create_visualizations(metrics, info, args.output_dir)
    save_results(metrics, info, args.output_dir)

    if args.use_wandb:
        import wandb
        wandb.log({
            "mask_coherence_mean": float(nc.mean()),
            "mask_coherence_median": float(np.median(nc)),
            "mask_coherence_std": float(nc.std()),
            "concentration_mean": float(conc.mean()),
            "concentration_median": float(np.median(conc)),
            "concentration_std": float(conc.std()),
        })
        for name, fig in plots.items():
            wandb.log({name: wandb.Image(fig)})
        wandb.log({
            "mask_coherence_histogram": wandb.Histogram(nc),
            "concentration_histogram": wandb.Histogram(conc),
        })
        wandb.finish()

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())