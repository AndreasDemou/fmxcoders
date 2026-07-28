"""Steered text-continuation generator for hand-authored probe prompts.

Human-readable counterpart to eval_feature_steering.py. Where that script measures
a feature's steering effect *distributionally* over a training corpus (KL and
promoted-token probability shifts), this one takes the hand-authored prompts in
``steering_prompts.json`` and, for each prompt, generates the ACTUAL continuation
with and without the feature's steering vector -- so the effect can be read as text.

Each prompt in the file is authored to probe a specific parent latent, so each
prompt is steered with that latent's decoder direction. The steering vector is the
same one eval_feature_steering.py injects::

    steer_{i,L} = coeff * a_ref_i * d_{i,L}

where ``d_{i,L}`` is the per-unit decoder write direction (decode(one-hot)-decode(0),
mode-agnostic) and ``a_ref_i`` is the feature's max observed activation (from the
top_bottom_coherence_*.json). ``--delta_mode increment`` (default) converts the
per-layer snapshots into marginal increments so that ``--all_layers`` inserts the
feature EXACTLY ONCE across the crosscoder span (increments telescope). This is the
faithful "turn the feature on" injection and is the recommended invocation.

Generation uses TransformerLens ``model.generate`` wrapped in the ``model.hooks(...)``
context manager, which keeps the additive steer active across every autoregressive
step (contrast run_with_hooks, a single forward pass, used here only for the
target-token boost readout).

``--steer_positions`` controls WHERE the steer is applied, which encodes the question:
  * all   (default) : every position, every step -- ActAdd-style sustained control; must
                      use a low coeff or the residual saturates and the text loops.
  * first           : only the first generated token, then free continuation -- the
                      collapse-proof "does the feature predict concept X next?" readout
                      (the generative twin of target_token_boost).
  * new             : the first + every generated token, prompt interior left clean.
'first'/'new' require the KV cache (they tell prompt pass from gen step by slice length).

Per prompt we record:
  * baseline_output : greedy continuation with no steering
  * steered_outputs : greedy continuation per coeff
  * target_token_boost : for each of the latent's target_tokens, the increase in
                         next-token probability at the first generated position
                         (steered - clean), tying the text back to the eval.

Usage:
    python gen_feature_steering.py \\
        --checkpoint_path checkpoints/xyz/final_XXXXX \\
        --contexts_from top_bottom_coherence_xyz.json \\
        --prompts_from steering_prompts.json \\
        --all_layers --delta_mode increment --coeffs 4,6,8 \\
        --output_dir steering_results/xyz/gen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import transformer_lens
from tqdm import tqdm

from eval_feature_steering import (
    _build_specs,
    _feature_directions,
    _load_model,
    _load_sae,
    _read_features,
    _to_increments,
)


def _make_pos_steer_hook(vec: torch.Tensor, mode: str):
    """Additive steering hook with position selection, for hooked generation.

    mode:
      'all'   : add vec at every position (standard ActAdd; steers the prompt AND every
                generated position -- strongest, saturates/loops if the coeff is high).
      'new'   : add vec only at the LAST position of each forward slice -> steers the first
                generated token (the last prompt position) and every subsequently generated
                token, leaving the prompt interior unsteered.
      'first' : add vec at the last position ONLY on the prompt pass -> steers just the
                first generated token; no steering afterwards (collapse-proof readout).
    'new'/'first' identify the prompt pass by seq>1 and generation steps by seq==1, which
    holds under KV-cache generation (enforced in main). 'first' also leaves a small lingering
    effect: the steered last-prompt-position enters the KV cache that later tokens attend to.
    """

    def hook(value: torch.Tensor, hook) -> torch.Tensor:  # value: [batch, seq, d_model]
        v = vec.to(value.dtype)
        if mode == "all":
            return value + v
        if mode == "first" and value.shape[1] == 1:
            return value  # a generation step -> 'first' does not steer here
        out = value.clone()
        out[:, -1, :] = out[:, -1, :] + v  # last position of this slice
        return out

    return hook


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # sources
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--contexts_from", required=True,
                   help="top_bottom_coherence_*.json: source of each latent's a_ref (max activation).")
    p.add_argument("--prompts_from", default="steering_prompts.json",
                   help="JSON with a 'latents' list; each latent has prompts probing it.")
    p.add_argument("--a_ref_group", default="top_most_coherent",
                   help="Coherence-JSON group to build the {latent_id: a_ref} lookup from.")
    p.add_argument("--a_ref_default", type=float, default=None,
                   help="Fallback a_ref for a prompt latent absent from --a_ref_group "
                        "(default: warn and skip that latent).")
    # steering / layer targeting (same flags as eval_feature_steering.py so _build_specs works)
    p.add_argument("--coeffs", default="0.5,1,2",
                   help="Comma-separated steering multipliers of a_ref. NOTE these are FRACTIONS "
                        "of the feature's peak activation: generation over-steers far below the "
                        "distributional eval's 4-8 (the steer is added at every position and "
                        "compounds across autoregressive steps). Sweet spot usually ~0.5-1.")
    p.add_argument("--delta_mode", choices=["increment", "cumulative"], default="increment",
                   help="'increment' (default) telescopes over --all_layers to inject the feature "
                        "exactly once; 'cumulative' injects the raw accumulated snapshot.")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--model_layer", type=int, default=None)
    p.add_argument("--all_layers", action="store_true",
                   help="Inject at every spanned layer simultaneously (recommended with increment).")
    p.add_argument("--per_layer", action="store_true",
                   help="Inject at each layer separately (one steered output per layer).")
    # generation
    p.add_argument("--steer_positions", choices=["all", "first", "new"], default="all",
                   help="Where to apply the steer: 'all' every position (ActAdd control; "
                        "needs low coeff or it loops); 'first' only the first generated token "
                        "(collapse-proof readout, the generative twin of target_token_boost); "
                        "'new' the first + every generated token, prompt interior left clean. "
                        "'first'/'new' require the KV cache.")
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--do_sample", action="store_true",
                   help="Sample instead of greedy (needs --seed for reproducibility).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--freq_penalty", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_stop_at_eos", action="store_true")
    p.add_argument("--no_kv_cache", action="store_true",
                   help="Disable the generation KV cache (parity smoke check).")
    p.add_argument("--store_full_text", action="store_true",
                   help="Also store prompt+continuation, not just the new continuation.")
    # subsetting (smoke tests)
    p.add_argument("--latents", default=None,
                   help="Comma-separated latent ids to restrict to, e.g. '854,9050'.")
    p.add_argument("--max_prompts_per_latent", type=int, default=None)
    # runtime
    p.add_argument("--model_name", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--output_dir", default="./steering_results")
    return p.parse_args()


def _tl_version() -> str:
    """transformer_lens version, robust across installs (some lack __version__)."""
    v = getattr(transformer_lens, "__version__", None)
    if v:
        return str(v)
    try:
        from importlib.metadata import version
        return version("transformer_lens")
    except Exception:
        return "unknown"


def _first_token_id(model, s: str):
    """First token id of a target-token string (BOS not prepended). None if empty."""
    ids = model.to_tokens(s, prepend_bos=False)
    if ids.numel() == 0:
        return None, None
    tid = int(ids[0, 0])
    return tid, ids.shape[1]  # id, n_tokens (>1 means the string is multi-token)


def main() -> int:
    args = parse_args()
    if args.all_layers and args.per_layer:
        print("ERROR: choose either --all_layers or --per_layer, not both.", file=sys.stderr)
        return 2
    if args.steer_positions != "all" and args.no_kv_cache:
        print("ERROR: --steer_positions first/new require the KV cache (drop --no_kv_cache); "
              "they identify generation steps by slice length, which needs caching.", file=sys.stderr)
        return 2
    device = torch.device(args.device)
    coeffs = [float(c) for c in args.coeffs.split(",") if c.strip() != ""]
    if not coeffs:
        print("ERROR: no coeffs given.", file=sys.stderr)
        return 2
    if args.do_sample:
        torch.manual_seed(args.seed)

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
    single_spec = len(specs) == 1

    # a_ref lookup {latent_id:int -> a_ref:float} from the coherence JSON.
    recs = _read_features(args.contexts_from, [args.a_ref_group], None)
    a_ref_lut = {r["latent_id"]: r["a_ref"] for r in recs}

    # Prompts.
    prompt_obj = json.loads(Path(args.prompts_from).read_text())
    latents = prompt_obj["latents"]
    if args.latents is not None:
        keep = {int(x) for x in args.latents.split(",") if x.strip() != ""}
        latents = [l for l in latents if int(l["latent_id"]) in keep]

    # Resolve which latents are usable (valid id, a_ref available) preserving order.
    d_sae = sae.cfg.d_sae
    usable, a_ref_of = [], {}
    for l in latents:
        lid = int(l["latent_id"])
        if not (0 <= lid < d_sae):
            print(f"WARNING: latent {lid} out of range [0,{d_sae}); skipping.", file=sys.stderr)
            continue
        if lid in a_ref_lut:
            a_ref_of[lid] = a_ref_lut[lid]
        elif args.a_ref_default is not None:
            a_ref_of[lid] = args.a_ref_default
            print(f"WARNING: latent {lid} not in group {args.a_ref_group!r}; "
                  f"using --a_ref_default={args.a_ref_default}.", file=sys.stderr)
        else:
            print(f"WARNING: latent {lid} not in group {args.a_ref_group!r} and no "
                  f"--a_ref_default; skipping.", file=sys.stderr)
            continue
        usable.append(l)
    if not usable:
        print("ERROR: no usable latents.", file=sys.stderr)
        return 2

    features_order = list(dict.fromkeys(int(l["latent_id"]) for l in usable))
    row_of = {lid: j for j, lid in enumerate(features_order)}

    model_name = args.model_name or meta.model_name
    tl_version = _tl_version()
    print(f"model               : {model_name}")
    print(f"transformer_lens    : {tl_version}")
    print(f"hook_names          : {hook_names}")
    print(f"delta_mode          : {args.delta_mode}")
    print(f"injection spec(s)   : {[lbl for (lbl, _ly, _p) in specs]}")
    print(f"steer_positions     : {args.steer_positions}")
    print(f"coeffs (x a_ref)    : {coeffs}")
    print(f"latents / prompts   : {len(usable)} latents, "
          f"{sum(len(l['prompts']) for l in usable)} prompts")

    print("Loading model...", flush=True)
    model = _load_model(model_name, str(device))
    model.eval()
    tok = model.tokenizer

    # Per-unit decoder directions for all selected features.
    dirs = _feature_directions(sae, features_order, device)  # [F, L, D]
    if args.delta_mode == "increment":
        dirs = _to_increments(dirs, layer_dim=1)

    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        freq_penalty=args.freq_penalty,
        stop_at_eos=not args.no_stop_at_eos,
        use_past_kv_cache=not args.no_kv_cache,
        prepend_bos=False,          # input_ids already carry BOS from to_tokens
        return_type="tokens",
        verbose=False,
    )

    def _generate(input_ids, fwd_hooks=None):
        if args.do_sample:
            torch.manual_seed(args.seed)  # same RNG draws for baseline and each steered run
        if fwd_hooks is None:
            return model.generate(input_ids, **gen_kwargs)
        with model.hooks(fwd_hooks=fwd_hooks):
            return model.generate(input_ids, **gen_kwargs)

    out_latents: dict[str, Any] = {}
    txt_lines: list[str] = []

    with torch.no_grad():
        for latent in tqdm(usable, desc="latents"):
            lid = int(latent["latent_id"])
            j = row_of[lid]
            a = a_ref_of[lid]
            target_tokens = latent.get("target_tokens", [])
            # Resolve target-token ids once per latent.
            tgt = {}
            for tstr in target_tokens:
                tid, ntok = _first_token_id(model, tstr)
                if tid is None:
                    continue
                if ntok and ntok > 1:
                    print(f"NOTE: target {tstr!r} (latent {lid}) is multi-token; "
                          f"measuring its first token {tok.decode([tid])!r}.", file=sys.stderr)
                tgt[tstr] = tid

            prompts_out: dict[str, Any] = {}
            prompts = latent["prompts"]
            if args.max_prompts_per_latent is not None:
                prompts = prompts[: args.max_prompts_per_latent]

            for pobj in prompts:
                text = pobj["prompt"]
                input_ids = model.to_tokens(text).to(device)  # [1, T], BOS prepended
                T = input_ids.shape[1]

                # Baseline generation + clean next-token distribution (once per prompt).
                base_ids = _generate(input_ids)
                base_cont = tok.decode(base_ids[0, T:], skip_special_tokens=True)
                clean_p = torch.softmax(model(input_ids)[0, -1].float(), dim=-1)

                steered_outputs: dict[str, Any] = {}
                steered_full: dict[str, Any] = {}
                boost: dict[str, Any] = {}
                for label, layers, _pos in specs:
                    for c in coeffs:
                        vecs = [(hook_names[l], _make_pos_steer_hook(c * a * dirs[j, l], args.steer_positions))
                                for l in layers]
                        s_ids = _generate(input_ids, fwd_hooks=vecs)
                        s_cont = tok.decode(s_ids[0, T:], skip_special_tokens=True)
                        # Target-token boost at the first generated position (single fwd pass).
                        steered_p = torch.softmax(
                            model.run_with_hooks(input_ids, prepend_bos=False, fwd_hooks=vecs)[0, -1].float(),
                            dim=-1,
                        )
                        tb = {}
                        for tstr, tid in tgt.items():
                            cp, spp = float(clean_p[tid]), float(steered_p[tid])
                            tb[tstr] = {"clean": cp, "steered": spp, "boost": spp - cp,
                                        "measured_token": tok.decode([tid])}
                        ck = f"{c:g}"
                        if single_spec:
                            steered_outputs[ck] = s_cont
                            if args.store_full_text:
                                steered_full[ck] = tok.decode(s_ids[0], skip_special_tokens=True)
                            boost[ck] = tb
                        else:
                            steered_outputs.setdefault(label, {})[ck] = s_cont
                            if args.store_full_text:
                                steered_full.setdefault(label, {})[ck] = tok.decode(s_ids[0], skip_special_tokens=True)
                            boost.setdefault(label, {})[ck] = tb

                rec = {
                    "prompt": text,
                    "baseline_expectation": pobj.get("baseline_expectation"),
                    "steered_expectation": pobj.get("steered_expectation"),
                    "baseline_output": base_cont,
                    "steered_outputs": steered_outputs,
                    "target_token_boost": boost,
                }
                if args.store_full_text:
                    rec["baseline_full"] = tok.decode(base_ids[0], skip_special_tokens=True)
                    rec["steered_full"] = steered_full
                prompts_out[pobj["id"]] = rec

                # human-readable log
                txt_lines.append(f"[{pobj['id']}] latent {lid} ({latent.get('label','')})")
                txt_lines.append(f"  prompt   : {text!r}")
                txt_lines.append(f"  targets  : {' '.join(target_tokens)}")
                txt_lines.append(f"  baseline : {base_cont!r}")
                for label, layers, _pos in specs:
                    for c in coeffs:
                        ck = f"{c:g}"
                        s = steered_outputs[ck] if single_spec else steered_outputs[label][ck]
                        tag = f"c={ck}" if single_spec else f"{label} c={ck}"
                        txt_lines.append(f"  steer {tag:>14}: {s!r}")
                txt_lines.append("")

            out_latents[str(lid)] = {
                "label": latent.get("label"),
                "description": latent.get("description"),
                "target_tokens": target_tokens,
                "a_ref": a,
                "prompts": prompts_out,
            }

    result = {
        "metadata": {
            "checkpoint_path": args.checkpoint_path,
            "model_name": model_name,
            "transformer_lens_version": tl_version,
            "hook_names": hook_names,
            "delta_mode": args.delta_mode,
            "injection": [lbl for (lbl, _ly, _p) in specs],
            "steer_positions": args.steer_positions,
            "coeffs": coeffs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "freq_penalty": args.freq_penalty,
            "seed": args.seed,
            "stop_at_eos": not args.no_stop_at_eos,
            "use_past_kv_cache": not args.no_kv_cache,
            "prompts_from": args.prompts_from,
            "contexts_from": args.contexts_from,
            "a_ref_group": args.a_ref_group,
        },
        "latents": out_latents,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "steered_continuations.json").write_text(json.dumps(result, indent=2))
    (out_dir / "continuations.txt").write_text("\n".join(txt_lines))
    print(f"\nWrote {out_dir/'steered_continuations.json'} and {out_dir/'continuations.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
