"""Feature-steering intervention eval for crosscoder SAEs.

Steering *inserts* a feature, so unlike an ablation it is input-independent: we
add a multiple of the feature's decoder direction to the residual stream over a
generic corpus and measure how the output distribution shifts.

This module also owns the shared helpers (_load_model / _load_sae /
_build_specs / _to_increments / _valid_mask / _spearman) that
gen_feature_steering.py imports.

For feature ``i`` at layer ``L`` the unit write direction ``d_{i,L}`` is
recovered mode-agnostically as ``decode(one-hot_i) - decode(0)`` (correct for
unfactored / CP / TR and rescale/folding). We then add::

    steer_{i,L} = coeff * a_ref_i * d_{i,L}

at ``hook_names[L]`` for every position, where ``a_ref_i`` is the feature's max
observed activation (from top_bottom_coherence_*.json), so each feature is
pushed to a comparable multiple of its own natural "on" level.

Direction semantics (--delta_mode):
  The crosscoder reads a CUMULATIVE residual stream (hook_resid_post), so the
  raw d_{i,L} is a snapshot of the feature's accumulated write, largely shared
  across layers for persistent features.
  * increment (default): steer with d_hat_0 = d_0 and d_hat_L = d_L - d_{L-1}.
    --per_layer then injects only the MARGINAL write appearing at layer L
    (block 9's increment = the whole contribution inherited at the span start),
    and --all_layers inserts the feature exactly once (increments telescope).
    Cross-layer token consistency becomes "do this feature's writes at
    different layers promote the same tokens?".
  * cumulative: original behavior. --per_layer inserts the full accumulated
    snapshot at each layer (cross-layer Jaccard is then dominated by the shared
    persistent component); --all_layers over-injects it ~n_layers times.
  Increment injections can be near-zero at layers where the feature is only
  carried; layers with KL below --jaccard_min_frac of the feature's peak-KL
  layer are excluded from the Jaccard so noise token-sets don't dilute it.

Layer targeting: default single layer, --all_layers (inject everywhere at once),
or --per_layer (inject at each layer separately -> one result per layer, a
per-feature "where does this feature act?" profile).

Metrics per (feature, layer, coeff):
  * kl_mean   : mean KL(clean || steered) over corpus positions (effect size)
  * top_tokens: tokens whose mean probability rises most under steering, each
                annotated as "<token> (<mean prob increase>)" (reported for every coeff)
Plus a per-feature cross-layer top-token Jaccard consistency (at the strongest coeff).

Usage:
    python eval_feature_steering.py \\
        --checkpoint_path checkpoints/xyz/final_XXXXX \\
        --contexts_from top_bottom_coherence_xyz.json \\
        --per_layer --coeffs 4 \\
        --output_dir steering_results/xyz
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from _crosscoder_eval_utils import load_crosscoder_inference
from sae_lens.saes.sae import SAE, TrainingSAE
from sae_lens.training.activations_store import ActivationsStore

try:
    from huggingface_hub.errors import OfflineModeIsEnabled
except Exception:  # pragma: no cover - older hub versions
    OfflineModeIsEnabled = OSError  # type: ignore[assignment,misc]


# --- shared loading / layer-targeting helpers (also imported by gen_feature_steering.py) ---


def _load_model(model_name: str, device: str) -> HookedTransformer:
    """Load the base model via TransformerLens.

    Works around a transformers regression where building a tokenizer from a
    *repo id* triggers a network call (``_patch_mistral_regex``/``is_base_mistral``)
    that fails under ``HF_HUB_OFFLINE=1``. On that failure we resolve the local
    HF cache snapshot (a local *path* skips the network check) and hand the
    pre-loaded tokenizer to TransformerLens, which then skips its own lookup.
    """
    try:
        return HookedTransformer.from_pretrained_no_processing(
            model_name=model_name, device=device
        )
    except OfflineModeIsEnabled:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        local_dir = snapshot_download(model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(local_dir, add_bos_token=True)
        print(f"  (offline) loaded tokenizer from local cache: {local_dir}", flush=True)
        return HookedTransformer.from_pretrained_no_processing(
            model_name=model_name, device=device, tokenizer=tokenizer
        )


def _load_sae(path: str, device: str, dtype: str) -> SAE:
    """Load a folded inference crosscoder. Falls back to the SAE/TrainingSAE
    loaders if the dedicated helper cannot handle the checkpoint."""
    try:
        return load_crosscoder_inference(path, device=device, dtype=dtype)
    except Exception:  # noqa: BLE001 - last-resort fallback, mirrors other eval scripts
        try:
            return SAE.load_from_disk(path, device=device)
        except KeyError:
            return TrainingSAE.load_from_disk(path, device=device)


def _to_increments(x: torch.Tensor, layer_dim: int) -> torch.Tensor:
    """Cumulative per-layer writes -> marginal per-layer increments.

    The crosscoder reads a CUMULATIVE residual stream (hook_resid_post), so a
    feature's decoded write at layer L is its accumulated write up to L, not
    what that block alone adds. Differencing along the layer axis recovers the
    marginal part: ``out_0 = x_0``, ``out_L = x_L - x_{L-1}``. Note ``out_0``
    is everything the feature has written by the START of the crosscoder span,
    not the first block's own contribution.

    The increments telescope, so injecting all of them at once accumulates to
    exactly ``x_{L_last}`` -- the feature inserted ONCE across the span, and at
    each intermediate layer the running sum matches that layer's snapshot.
    """
    prev = torch.zeros_like(x.narrow(layer_dim, 0, 1))
    return torch.diff(x, dim=layer_dim, prepend=prev)


def _build_specs(args, hook_names, layer_idx, n_layers):
    """Return list of (label, layer_indices, sort_pos) describing where to inject."""
    if args.per_layer:
        return [(hook_names[l], [l], l) for l in range(n_layers)]
    if args.all_layers:
        return [("all_layers", list(range(n_layers)), -1)]
    return [(hook_names[layer_idx], [layer_idx], layer_idx)]


def _valid_mask(batch_tokens: torch.Tensor, ignore_list: list[int]) -> torch.Tensor:
    mask = torch.ones_like(batch_tokens, dtype=torch.bool)
    for t in ignore_list:
        mask &= batch_tokens != t
    return mask


def _spearman(x, y):
    """Spearman rank correlation (no scipy dependency); None if too few points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return None
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


# --- steering ---


def _make_steer_hook(vec: torch.Tensor):
    """vec: [d_model] constant added at every position of this layer's residual."""

    def hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        return value + vec.to(value.dtype)

    return hook


def _feature_directions(sae, features: list[int], device) -> torch.Tensor:
    """Per-unit decoder write direction d_{i,L} for each selected feature.

    Returns [n_feats, n_layers, d_in] via decode(one-hot) - decode(0), so it is
    correct for factored decoders and rescale/folding.
    """
    d_sae = sae.cfg.d_sae
    eye = torch.zeros(len(features), d_sae, device=device, dtype=sae.dtype)
    for j, i in enumerate(features):
        eye[j, i] = 1.0
    zero = torch.zeros(1, d_sae, device=device, dtype=sae.dtype)
    with torch.no_grad():
        base = sae.decode(zero)          # [1, L, D]  (= b_dec)
        dirs = sae.decode(eye) - base    # [n_feats, L, D]
    return dirs


def _read_features(path: str, groups: list[str], n_features) -> list[dict]:
    """Load (group, latent_id, coherence, a_ref) records from the coherence JSON."""
    obj = json.loads(Path(path).read_text())
    recs = []
    for grp in groups:
        feats = obj.get(grp)
        if feats is None:
            print(f"WARNING: group {grp!r} not in {path}", file=sys.stderr)
            continue
        if n_features is not None:
            feats = feats[:n_features]
        for r in feats:
            toks = r.get("tokens", [])
            a_ref = max((t.get("max_activation", 0.0) for t in toks), default=0.0)
            recs.append({
                "group": grp,
                "latent_id": int(r["latent_id"]),
                "coherence": r.get("coherence"),
                "a_ref": float(a_ref),
            })
    return recs


def _jaccard(sets: list[set]) -> float:
    """Mean pairwise Jaccard overlap across a list of sets."""
    if len(sets) < 2:
        return float("nan")
    vals = []
    for a, b in combinations(sets, 2):
        u = len(a | b)
        vals.append(len(a & b) / u if u else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--contexts_from", required=True,
                   help="top_bottom_coherence_*.json: source of features + each feature's a_ref (max act).")
    p.add_argument("--groups", default="top_most_coherent,bottom_least_coherent")
    p.add_argument("--n_features", type=int, default=20)
    p.add_argument("--coeffs", default="4",
                   help="Comma-separated steering multipliers of a_ref, e.g. '2,4,8'. "
                        "Top tokens are reported per coeff (CSV); the Jaccard uses the largest.")
    # layer targeting
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--model_layer", type=int, default=None)
    p.add_argument("--all_layers", action="store_true",
                   help="Inject at every spanned layer simultaneously.")
    p.add_argument("--per_layer", action="store_true",
                   help="Inject at each layer separately (one result per layer).")
    # corpus / model
    p.add_argument("--dataset", default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--n_batches", type=int, default=10)
    p.add_argument("--batch_size_prompts", type=int, default=8)
    p.add_argument("--context_size", type=int, default=64)
    p.add_argument("--shuffle_seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=15, help="How many promoted tokens to report.")
    p.add_argument("--delta_mode", choices=["increment", "cumulative"], default="increment",
                   help="'increment' (default) steers with marginal per-layer writes "
                        "d_L - d_{L-1}; 'cumulative' steers with raw accumulated snapshots "
                        "(original behavior; over-injects ~n_layers times in --all_layers mode).")
    p.add_argument("--jaccard_min_frac", type=float, default=0.1,
                   help="[per_layer] Layers whose KL at the max coeff is below this fraction of "
                        "the feature's peak-KL layer are excluded from the cross-layer Jaccard.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--output_dir", default="./steering_results")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.all_layers and args.per_layer:
        print("ERROR: choose either --all_layers or --per_layer, not both.", file=sys.stderr)
        return 2
    device = torch.device(args.device)
    coeffs = [float(c) for c in args.coeffs.split(",") if c.strip() != ""]
    max_coeff = max(coeffs)

    print(f"Loading crosscoder from {args.checkpoint_path}", flush=True)
    sae = _load_sae(args.checkpoint_path, str(device), args.dtype)
    sae.eval()
    meta = sae.cfg.metadata
    hook_names = getattr(meta, "hook_names", None) or getattr(sae.cfg, "hook_names", None)
    if not hook_names or len(hook_names) < 2:
        print(f"ERROR: not a crosscoder (hook_names={hook_names!r}).", file=sys.stderr)
        return 2
    hook_names = list(hook_names)
    n_layers = len(hook_names)

    if getattr(sae.cfg, "normalize_activations", "none") == "constant_norm_rescale":
        print("ERROR: normalize_activations='constant_norm_rescale' is not supported.", file=sys.stderr)
        return 2

    layer_idx = args.layer
    if args.model_layer is not None:
        matches = [i for i, h in enumerate(hook_names) if f"blocks.{args.model_layer}." in h]
        if not matches:
            print(f"ERROR: model layer {args.model_layer} not in {hook_names}.", file=sys.stderr)
            return 2
        layer_idx = matches[0]
    specs = _build_specs(args, hook_names, layer_idx, n_layers)

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    records = _read_features(args.contexts_from, groups, args.n_features)
    records = [r for r in records if r["a_ref"] > 0 and 0 <= r["latent_id"] < sae.cfg.d_sae]
    features = [r["latent_id"] for r in records]
    if not features:
        print("ERROR: no usable features.", file=sys.stderr)
        return 2
    a_ref = torch.tensor([r["a_ref"] for r in records], device=device, dtype=torch.float32)

    model_name = args.model_name or meta.model_name
    dataset = args.dataset or getattr(meta, "dataset_path", None) or "Skylion007/openwebtext"
    print(f"model             : {model_name}")
    print(f"hook_names        : {hook_names}")
    print(f"delta_mode        : {args.delta_mode}")
    print(f"injection spec(s) : {[lbl for (lbl, _ly, _p) in specs]}")
    print(f"coeffs (x a_ref)  : {coeffs}   (top tokens @ {max_coeff})")
    print(f"n_features        : {len(features)}  dataset: {dataset}")

    print("Loading model...", flush=True)
    model = _load_model(model_name, str(device))
    model.eval()
    tok = model.tokenizer
    ignore_list = [t for t in {tok.pad_token_id, tok.eos_token_id, tok.bos_token_id} if t is not None]
    vocab = model.cfg.d_vocab

    # Per-unit decoder directions for all selected features.
    dirs = _feature_directions(sae, features, device)  # [F, L, D]
    if args.delta_mode == "increment":
        dirs = _to_increments(dirs, layer_dim=1)
    dir_norms = dirs.float().norm(dim=-1)              # [F, L] injected-vector norms (per unit act)

    # Fixed corpus batches (same text for every feature/layer/coeff).
    store = ActivationsStore.from_sae(
        model, sae, context_size=args.context_size, dataset=dataset,  # type: ignore
        dataset_trust_remote_code=True,
    )
    store.shuffle_input_dataset(seed=args.shuffle_seed)
    batches = [store.get_batch_tokens(args.batch_size_prompts).to(device) for _ in range(args.n_batches)]

    F, S, C = len(features), len(specs), len(coeffs)
    kl_sum = np.zeros((F, S, C))
    pos_total = 0
    prob_delta = torch.zeros(F, S, C, vocab, device=device)  # per-coeff mean prob shift

    with torch.no_grad():
        for tokens in tqdm(batches):
            mask = _valid_mask(tokens, ignore_list)
            clean_logp = torch.log_softmax(model(tokens).float(), dim=-1)
            clean_p = clean_logp.exp()
            pos_total += int(mask.sum().item())

            for j in range(F):
                for s, (_label, layers, _pos) in enumerate(specs):
                    for ci, c in enumerate(coeffs):
                        vecs = [(hook_names[l], _make_steer_hook(c * a_ref[j] * dirs[j, l]))
                                for l in layers]
                        steered_logp = torch.log_softmax(
                            model.run_with_hooks(tokens, prepend_bos=False, fwd_hooks=vecs).float(),
                            dim=-1,
                        )
                        kl = (clean_p * (clean_logp - steered_logp)).sum(dim=-1)[mask]
                        kl_sum[j, s, ci] += float(kl.sum().item())
                        prob_delta[j, s, ci] += (steered_logp.exp() - clean_p)[mask].sum(dim=0)
                        del steered_logp

    # ---- aggregate ----
    kl_mean = kl_sum / max(pos_total, 1)
    mean_delta = (prob_delta / max(pos_total, 1)).cpu()

    def _top_tokens(j, s, ci):
        vals, idx = torch.topk(mean_delta[j, s, ci], args.top_k)
        toks = [tok.decode([int(t)]).replace("\n", "\\n") for t in idx]
        # "<token> (<mean prob increase>)" so each promoted token carries its effect size.
        labeled = " | ".join(f"{t} ({v:.4f})" for t, v in zip(toks, vals.tolist()))
        return labeled, [int(t) for t in idx]

    ci_max = coeffs.index(max_coeff)
    rows = []
    top_sets: dict[tuple[int, int], set] = {}
    for j, rec in enumerate(records):
        for s, (label, layers, pos) in enumerate(specs):
            top_sets[(j, s)] = set(_top_tokens(j, s, ci_max)[1])  # Jaccard uses strongest coeff
            for ci, c in enumerate(coeffs):
                rows.append({
                    "group": rec["group"],
                    "latent_id": rec["latent_id"],
                    "coherence": rec["coherence"],
                    "a_ref": rec["a_ref"],
                    "layer": label,
                    "layer_pos": pos,
                    "coeff": c,
                    "dir_norm": float(dir_norms[j, layers].sum().item()),
                    "kl_mean": kl_mean[j, s, ci],
                    "top_tokens": _top_tokens(j, s, ci)[0],
                })
    df = pd.DataFrame(rows).sort_values(["group", "latent_id", "layer_pos", "coeff"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_feature_steering.csv", index=False)

    # per-feature summary: mean KL (at max coeff) over layers + cross-layer token consistency.
    # The Jaccard only compares layers whose injection has a real effect (KL within
    # jaccard_min_frac of the feature's peak layer); with increment directions, carried-only
    # layers have near-zero injections whose "top tokens" are noise.
    feat_rows = []
    for j, rec in enumerate(records):
        kl_over_layers = kl_mean[j, :, ci_max]
        if args.per_layer:
            peak = float(kl_over_layers.max())
            keep = ([s for s in range(S) if kl_over_layers[s] >= args.jaccard_min_frac * peak]
                    if peak > 0 else [])
            jac = _jaccard([top_sets[(j, s)] for s in keep])
        else:
            keep = []
            jac = float("nan")
        feat_rows.append({
            "group": rec["group"],
            "latent_id": rec["latent_id"],
            "coherence": rec["coherence"],
            "a_ref": rec["a_ref"],
            "kl_mean_over_layers": float(kl_over_layers.mean()),
            "kl_peak_layer": specs[int(kl_over_layers.argmax())][0],
            "crosslayer_token_jaccard": jac,
            "n_jaccard_layers": len(keep),
        })
    feat_summary = pd.DataFrame(feat_rows)
    feat_summary.to_csv(out_dir / "per_feature_summary.csv", index=False)

    def _stats(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return {} if x.size == 0 else {
            "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
        }

    summary: dict[str, Any] = {
        "checkpoint_path": args.checkpoint_path,
        "model_name": model_name,
        "hook_names": hook_names,
        "injection": [lbl for (lbl, _ly, _p) in specs],
        "delta_mode": args.delta_mode,
        "jaccard_min_frac": args.jaccard_min_frac,
        "per_layer": bool(args.per_layer),
        "coeffs": coeffs,
        "n_features": len(features),
        "metric": "KL(clean || steered); top tokens by mean prob increase",
        "kl_by_group": {grp: _stats(g["kl_mean_over_layers"].to_numpy())
                        for grp, g in feat_summary.groupby("group")},
    }
    if args.per_layer:
        summary["token_jaccard_by_group"] = {
            grp: _stats(g["crosslayer_token_jaccard"].to_numpy())
            for grp, g in feat_summary.groupby("group")
        }
        summary["spearman_coherence_vs_token_jaccard"] = _spearman(
            feat_summary["coherence"].to_numpy(), feat_summary["crosslayer_token_jaccard"].to_numpy()
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 80)
    print("STEERING RESULTS")
    print("=" * 80)
    print(feat_summary.sort_values(["group", "kl_mean_over_layers"], ascending=[True, False]).to_string(index=False))
    print("\nKL(clean||steered) by group:")
    for grp, s in summary["kl_by_group"].items():
        print(f"  {grp:24s} median={s.get('median', float('nan')):.5f}  mean={s.get('mean', float('nan')):.5f}")
    if args.per_layer:
        print("\nCross-layer top-token Jaccard by group (higher = steers more consistently):")
        for grp, s in summary["token_jaccard_by_group"].items():
            print(f"  {grp:24s} median={s.get('median', float('nan')):.3f}")
        sp = summary.get("spearman_coherence_vs_token_jaccard")
        if sp is not None:
            print(f"Spearman(coherence, token_jaccard) = {sp:+.3f}")
    print(f"\nWrote {out_dir/'per_feature_steering.csv'}, {out_dir/'per_feature_summary.csv'}, "
          f"{out_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
