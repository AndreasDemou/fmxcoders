#!/usr/bin/env python
"""Evaluate crosscoder latent norm coherence (PolySAE_dev version).

For each latent j in a crosscoder, compute the L2 norm of its weight fiber per
layer, then
    Concentration_j = max_l(norm_j^l) / mean_l(norm_j^l)
    NormCoherence_j = 1 / Concentration_j

Coherent latent:   similar norms across layers → concentration ≈ 1 → coherence ≈ 1
Incoherent latent: one layer dominates         → concentration >> 1 → coherence << 1

Supports all three weight layouts used by `sae_lens/saes/crosscoder_sae.py`:
  - Full 3-D:  `W_enc [n_layers, d_in, d_sae]`, `W_dec [d_sae, n_layers, d_in]`
  - CP:        `A_enc/B_enc/C_enc`, `A_dec/B_dec/C_dec` factor matrices
  - TR:        `G_enc_n/G_enc_d/G_enc_s`, `G_dec_n/G_dec_d/G_dec_s` cores

This is a weight-based metric — no model or data required.

Usage:
    python eval_norm_coherence.py --checkpoint_path /path/to/final_XXXXX
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import einops
import numpy as np
import torch

from sae_lens.saes.sae import SAE, TrainingSAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate crosscoder norm coherence")
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Path to the saved SAE directory (cfg.json + state_dict).")
    p.add_argument("--use_encoder", action="store_true",
                   help="Use encoder weights instead of decoder weights.")
    p.add_argument("--epsilon", type=float, default=1e-8,
                   help="Numerical stability constant when dividing by mean norm.")
    p.add_argument("--output_dir", type=str, default="./norm_coherence_results",
                   help="Directory to write plots + JSON + CSV + NPY.")
    p.add_argument("--use_wandb", action="store_true", help="Log to wandb.")
    p.add_argument("--wandb_project", type=str, default="Crosscoders")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for weight tensors. 'cpu' is enough — no forward pass needed.")
    return p.parse_args()


def _load_sae_or_training_sae(path: str, device: str) -> SAE:
    """Crosscoder architectures like `cross_batchtopk` are only registered in
    SAE_TRAINING_CLASS_REGISTRY, so a plain SAE.load_from_disk raises KeyError.
    Fall back to TrainingSAE's loader which hits the training registry."""
    try:
        return SAE.load_from_disk(path, device=device)
    except KeyError:
        return TrainingSAE.load_from_disk(path, device=device)


def _is_cp(sae: SAE) -> bool:
    cfg = sae.cfg
    return (
        getattr(cfg, "use_cross_fact", False)
        and getattr(cfg, "cross_fact_type", "CP") == "CP"
    )


def _is_tr(sae: SAE) -> bool:
    cfg = sae.cfg
    return (
        getattr(cfg, "use_cross_fact", False)
        and getattr(cfg, "cross_fact_type", "CP") == "TR"
    )


def _layer_indices_from_hook_names(hook_names: list[str]) -> list[int]:
    """Extract integer layer indices from TLens hook names like
    'blocks.12.hook_resid_post'. Falls back to positional indices if parsing fails.
    """
    out: list[int] = []
    for h in hook_names:
        m = re.search(r"blocks\.(\d+)", h)
        out.append(int(m.group(1)) if m else len(out))
    return out


def _extract_weights(sae: SAE, use_encoder: bool) -> tuple[torch.Tensor, str]:
    """Return weight tensor shaped [n_layers, d_sae, d_in] and a label for
    which factorization path produced it.
    """
    if _is_cp(sae):
        if use_encoder:
            # W_enc[l, d, s] = Σ_r A_enc[l,r] · B_enc[d,r] · C_enc[s,r]
            A, B, C = sae.A_enc.detach(), sae.B_enc.detach(), sae.C_enc.detach()
            W = einops.einsum(A, B, C, "l r, d r, s r -> l s d")
        else:
            # W_dec[s, l, d] = Σ_r A_dec[s,r] · B_dec[l,r] · C_dec[d,r]
            A, B, C = sae.A_dec.detach(), sae.B_dec.detach(), sae.C_dec.detach()
            W = einops.einsum(A, B, C, "s r, l r, d r -> l s d")
        return W, "CP"

    if _is_tr(sae):
        if use_encoder:
            # W_enc[l, d, s] = einsum('alb, bdc, csa -> lds', G_enc_n, G_enc_d, G_enc_s)
            G_n, G_d, G_s = (
                sae.G_enc_n.detach(), sae.G_enc_d.detach(), sae.G_enc_s.detach(),
            )
            W_lds = einops.einsum(G_n, G_d, G_s, "a l b, b d c, c s a -> l d s")
            W = einops.rearrange(W_lds, "l d s -> l s d")
        else:
            # W_dec[s, l, d] = einsum('xsy, ylz, zdx -> sld', G_dec_s, G_dec_n, G_dec_d)
            G_s, G_n, G_d = (
                sae.G_dec_s.detach(), sae.G_dec_n.detach(), sae.G_dec_d.detach(),
            )
            W_sld = einops.einsum(G_s, G_n, G_d, "x s y, y l z, z d x -> s l d")
            W = einops.rearrange(W_sld, "s l d -> l s d")
        return W, "TR"

    # Full 3-D weights
    if use_encoder:
        W_enc = sae.W_enc.detach()  # [n_layers, d_in, d_sae]
        W = einops.rearrange(W_enc, "l d s -> l s d")
    else:
        W_dec = sae.W_dec.detach()  # [d_sae, n_layers, d_in]
        W = einops.rearrange(W_dec, "s l d -> l s d")
    return W, "3D"


def compute_norm_metrics(
    sae: SAE,
    epsilon: float = 1e-8,
    use_encoder: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Returns (metrics, info). Metrics has layer_norms [n_latents, n_layers],
    concentration [n_latents], norm_coherence [n_latents], plus max/mean.
    """
    meta = sae.cfg.metadata
    hook_names = getattr(meta, "hook_names", None) or getattr(sae.cfg, "hook_names", None)
    if not hook_names or not isinstance(hook_names, (list, tuple)) or len(hook_names) < 2:
        raise ValueError(
            f"Not a crosscoder (hook_names={hook_names!r}); expected a list of >1 hooks."
        )
    hook_names = list(hook_names)
    layer_indices = _layer_indices_from_hook_names(hook_names)

    W, method = _extract_weights(sae, use_encoder=use_encoder)
    n_layers, n_latents, d_in = W.shape

    # Consistency check against cfg
    assert n_layers == len(hook_names), (
        f"W has {n_layers} layers but hook_names has {len(hook_names)}"
    )
    assert n_latents == sae.cfg.d_sae, (
        f"W has {n_latents} latents but cfg.d_sae={sae.cfg.d_sae}"
    )

    # L2 norm per (layer, latent). W: [L, S, D] -> norms: [L, S] -> transpose: [S, L]
    layer_norms = torch.norm(W.float(), dim=2).T.cpu().numpy()  # [n_latents, n_layers]

    norm_max = layer_norms.max(axis=1)
    norm_mean = layer_norms.mean(axis=1)
    concentration = np.maximum(norm_max / (norm_mean + epsilon), 1.0)
    norm_coherence = 1.0 / concentration

    metrics = {
        "layer_norms": layer_norms,
        "concentration": concentration,
        "norm_coherence": norm_coherence,
        "norm_max": norm_max,
        "norm_mean": norm_mean,
    }
    info = {
        "tensor_method": method,
        "hook_names": hook_names,
        "layer_indices": layer_indices,
        "n_layers": n_layers,
        "n_latents": n_latents,
        "d_model": d_in,
        "weight_type": "encoder" if use_encoder else "decoder",
    }
    return metrics, info


def create_visualizations(
    metrics: dict[str, np.ndarray],
    info: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    """All five plots from the original. Requires matplotlib (+ seaborn for one heatmap)."""
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except ImportError:
        sns = None

    os.makedirs(output_dir, exist_ok=True)
    plots: dict[str, Any] = {}

    weight_label = info["weight_type"].capitalize()
    layer_indices = info["layer_indices"]
    n_latents, n_layers = metrics["layer_norms"].shape

    # 1. Norm coherence distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(metrics["norm_coherence"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(metrics["norm_coherence"].mean(), color="red", linestyle="--",
               label=f"Mean: {metrics['norm_coherence'].mean():.3f}")
    ax.axvline(np.median(metrics["norm_coherence"]), color="orange", linestyle="--",
               label=f"Median: {np.median(metrics['norm_coherence']):.3f}")
    ax.set_xlabel("Norm Coherence")
    ax.set_ylabel("Number of Latents")
    ax.set_title(f"Distribution of Norm Coherence Scores ({weight_label} weight-based)")
    ax.set_xlim([0, 1])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plots["norm_coherence_distribution"] = fig
    fig.savefig(f"{output_dir}/norm_coherence_distribution.png", dpi=150, bbox_inches="tight")
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
    ax.set_title(f"Distribution of Concentration Scores ({weight_label} weight-based)\n(Higher = more layer-specific)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plots["concentration_distribution"] = fig
    fig.savefig(f"{output_dir}/concentration_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Heatmaps of per-layer norms for most coherent / incoherent latents
    coherence_order = np.argsort(metrics["norm_coherence"])[::-1]
    n_show = min(50, n_latents // 2)
    for name, indices in [
        ("coherent", coherence_order[:n_show]),
        ("incoherent", coherence_order[-n_show:]),
    ]:
        fig, ax = plt.subplots(figsize=(12, 10))
        norms_subset = metrics["layer_norms"][indices, :]
        if sns is not None:
            sns.heatmap(
                norms_subset,
                xticklabels=layer_indices,
                yticklabels=[f"L{i}" for i in indices],
                cmap="YlOrRd",
                cbar_kws={"label": f"{weight_label} Norm"},
                ax=ax,
            )
        else:
            im = ax.imshow(norms_subset, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(n_layers))
            ax.set_xticklabels(layer_indices)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([f"L{i}" for i in indices])
            plt.colorbar(im, ax=ax, label=f"{weight_label} Norm")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Latent")
        ax.set_title(f"{weight_label} Norms - Top {n_show} {'Coherent' if name == 'coherent' else 'Incoherent'} Latents")
        plt.tight_layout()
        plots[f"layer_norms_heatmap_{name}"] = fig
        fig.savefig(f"{output_dir}/layer_norms_heatmap_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. Concentration vs mean norm scatter (colored by coherence)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        metrics["norm_mean"], metrics["concentration"],
        c=metrics["norm_coherence"], cmap="viridis", alpha=0.6, s=10,
    )
    ax.set_xlabel(f"Mean {weight_label} Norm (averaged over layers)")
    ax.set_ylabel("Concentration (max/mean)")
    ax.set_title(f"Concentration vs Mean {weight_label} Norm\n(Color = Norm Coherence)")
    ax.set_yscale("log")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Norm Coherence")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plots["concentration_vs_mean_norm"] = fig
    fig.savefig(f"{output_dir}/concentration_vs_mean_norm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Average norm per layer
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_norm_per_layer = metrics["layer_norms"].mean(axis=0)
    ax.bar(range(n_layers), avg_norm_per_layer,
           tick_label=[f"L{idx}" for idx in layer_indices])
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Average {weight_label} Norm (over all latents)")
    ax.set_title(f"Average {weight_label} Norm per Layer")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plots["avg_norm_per_layer"] = fig
    fig.savefig(f"{output_dir}/avg_norm_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return plots


def save_results(
    metrics: dict[str, np.ndarray],
    info: dict[str, Any],
    output_dir: str,
) -> None:
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/layer_norms.npy", metrics["layer_norms"])
    np.save(f"{output_dir}/concentration.npy", metrics["concentration"])
    np.save(f"{output_dir}/norm_coherence.npy", metrics["norm_coherence"])

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
        "norm_coherence": _stats(metrics["norm_coherence"]),
        "concentration": _stats(metrics["concentration"]),
    }
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    df = pd.DataFrame({
        "latent_idx": np.arange(len(metrics["norm_coherence"])),
        "norm_coherence": metrics["norm_coherence"],
        "concentration": metrics["concentration"],
        "norm_mean": metrics["norm_mean"],
        "norm_max": metrics["norm_max"],
    })
    df.to_csv(f"{output_dir}/per_latent_metrics.csv", index=False)

    print(f"\nResults saved to {output_dir}")
    for fn in ("layer_norms.npy", "concentration.npy", "norm_coherence.npy",
               "summary.json", "per_latent_metrics.csv"):
        print(f"  - {fn}")


def main() -> int:
    args = parse_args()
    weight_type = "encoder" if args.use_encoder else "decoder"

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"norm_coherence_{weight_type}_{Path(args.checkpoint_path).name}",
            config=vars(args),
        )

    print(f"Loading crosscoder from {args.checkpoint_path}", flush=True)
    sae = _load_sae_or_training_sae(args.checkpoint_path, args.device)
    sae.eval()

    metrics, info = compute_norm_metrics(
        sae, epsilon=args.epsilon, use_encoder=args.use_encoder,
    )

    print("\n" + "=" * 80)
    print(f"NORM COHERENCE EVALUATION RESULTS ({weight_type.upper()} weight-based)")
    print("=" * 80)
    print(f"Tensor method:  {info['tensor_method']}")
    print(f"Layers:         {info['layer_indices']}")
    print(f"d_sae / d_model: {info['n_latents']} / {info['d_model']}")

    nc = metrics["norm_coherence"]
    print("\nNorm coherence:")
    print(f"  mean / median: {nc.mean():.4f} / {np.median(nc):.4f}")
    print(f"  std:           {nc.std():.4f}")
    print(f"  min / max:     {nc.min():.4f} / {nc.max():.4f}")
    print(f"  P25 / P75:     {np.percentile(nc, 25):.4f} / {np.percentile(nc, 75):.4f}")

    conc = metrics["concentration"]
    print("\nConcentration:")
    print(f"  mean / median: {conc.mean():.4f} / {np.median(conc):.4f}")
    print(f"  std:           {conc.std():.4f}")
    print(f"  min / max:     {conc.min():.4f} / {conc.max():.4f}")

    best = int(np.argmax(nc))
    worst = int(np.argmin(nc))
    print(f"\nMost coherent latent:   {best}")
    print(f"  coherence: {nc[best]:.4f}   concentration: {conc[best]:.4f}")
    print(f"  per-layer norms: {metrics['layer_norms'][best]}")
    print(f"\nMost incoherent latent: {worst}")
    print(f"  coherence: {nc[worst]:.4f}   concentration: {conc[worst]:.4f}")
    print(f"  per-layer norms: {metrics['layer_norms'][worst]}")

    print(f"\nCreating visualizations in {args.output_dir} ...")
    plots = create_visualizations(metrics, info, args.output_dir)
    save_results(metrics, info, args.output_dir)

    if args.use_wandb:
        import wandb
        wandb.log({
            "tensor_method": info["tensor_method"],
            "norm_coherence_mean": float(nc.mean()),
            "norm_coherence_median": float(np.median(nc)),
            "norm_coherence_std": float(nc.std()),
            "concentration_mean": float(conc.mean()),
            "concentration_median": float(np.median(conc)),
            "concentration_std": float(conc.std()),
        })
        for name, fig in plots.items():
            wandb.log({name: wandb.Image(fig)})
        wandb.log({
            "norm_coherence_histogram": wandb.Histogram(nc),
            "concentration_histogram": wandb.Histogram(conc),
        })
        wandb.finish()

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())