#!/usr/bin/env python3
"""Extract Top-K SAE/crosscoder features per token.

Works for any HF model supported by TransformerLens, and for any SAE
architecture in this repo:
  - single-layer SAEs   (standard, topk, jumprelu, batchtopk, matryoshka, ...)
  - crosscoders         (cross_topk, cross_jumprelu, cross_batchtopk, ...)
The crosscoder path stacks per-layer activations from all hooks in
`sae.cfg.metadata.hook_names` before calling `sae.encode`; single-layer SAEs
fall through the same code with a 1-element hook list.

Output JSON structure:
    [
        {"sentence": str,
         "tokens": [{"token": str,
                     "position": int,
                     "top_feature_ids":  [int, ...],
                     "top_feature_acts": [float, ...]}, ...]},
        ...
    ]

Usage:
    python extract_token_features.py \\
        --checkpoint /path/to/sae_dir \\
        --n_texts 5000 --top_k 16 \\
        --output token_features.json \\
        [--dataset HFOrg/dataset]   # falls back to sae.cfg.metadata.dataset_path
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens.offline_hf import offline_aware_load_dataset
from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name

from _crosscoder_eval_utils import load_crosscoder_inference


# ---------------------------------------------------------------------------
# Helpers (inlined replacements for the missing polysae_utils module)
# ---------------------------------------------------------------------------


def clean_token(s: str) -> str:
    """Display-clean a tokenizer string for both BPE and SentencePiece."""
    return s.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")


def get_hook_names(sae) -> list[str]:
    """Return the hook names the SAE consumes.

    Length 1 → single-layer SAE; length > 1 → crosscoder.
    """
    hook_names = getattr(sae.cfg.metadata, "hook_names", None)
    if hook_names:
        return list(hook_names)
    hook_name = getattr(sae.cfg.metadata, "hook_name", None)
    if not hook_name:
        raise ValueError(
            "SAE config has neither metadata.hook_names nor metadata.hook_name; "
            "cannot determine which model activation(s) to capture."
        )
    return [hook_name]


def load_models(checkpoint: str | Path, device: str, dtype: str = "float32"):
    """Load the SAE (any architecture, including training-only crosscoder
    variants which are converted to inference via tempdir) and the matching
    HookedTransformer LLM."""
    sae = load_crosscoder_inference(checkpoint, device=device, dtype=dtype)
    sae.eval()

    model_name = getattr(sae.cfg.metadata, "model_name", None)
    if not model_name:
        raise ValueError("SAE config has no metadata.model_name; pass --model_name or fix the checkpoint.")

    model = HookedTransformer.from_pretrained_no_processing(
        model_name=model_name, device=device
    )
    model.eval()
    return model, sae


def stream_texts(dataset_name: str, n_texts: int) -> list[str]:
    """Stream a HF text dataset and collect up to n_texts non-empty entries.

    Uses `offline_aware_load_dataset` which falls back to reading parquet shards
    directly from the local HF cache when `HF_HUB_OFFLINE=1` (no Hub call).
    """
    ds = offline_aware_load_dataset(dataset_name, split="train", streaming=True)
    texts: list[str] = []
    for example in ds:
        if "text" in example:
            t = example["text"]
        elif "content" in example:
            t = example["content"]
        else:
            t = next((v for v in example.values() if isinstance(v, str)), None)
        if t and t.strip():
            texts.append(t)
            if len(texts) >= n_texts:
                break
    return texts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the SAE checkpoint directory (cfg.json + sae_weights.safetensors).")
    parser.add_argument("--n_texts", type=int, default=5000)
    parser.add_argument("--top_k", type=int, default=16,
                        help="Number of top activated features to record per token.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of texts per forward pass.")
    parser.add_argument("--context_size", type=int, default=128,
                        help="Max token length per text.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name. Falls back to sae.cfg.metadata.dataset_path if unset.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path. If omitted, defaults to "
                             "<repo>/eval_results/<run_name>/feature_analysis/token_features.json "
                             "where <run_name> is the parent dir of --checkpoint.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="dtype for the SAE weights at load time.")
    parser.add_argument("--skip_bos", action="store_true", default=True,
                        help="Skip token at position 0 (BOS).")
    parser.add_argument("--min_activation", type=float, default=0.0,
                        help="Drop top-K features whose activation is below this. 0 keeps all top-K.")
    parser.add_argument("--test", action="store_true",
                        help="Process only 20 texts.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_sentence_features(
    model,
    sae,
    texts,
    top_k: int,
    batch_size: int,
    context_size: int,
    device: str,
    skip_bos: bool,
    min_activation: float,
):
    """Yield one result dict per sentence."""
    hook_names = get_hook_names(sae)
    is_multi_hook = len(hook_names) > 1
    stop_layers = [extract_stop_at_layer_from_tlens_hook_name(h) for h in hook_names]
    stop_at_layer = max(l for l in stop_layers if l is not None)

    batch_texts: list[str] = []

    def _process_batch(batch_texts):
        batch_tokens = []
        for text in batch_texts:
            tokens = model.to_tokens(text, truncate=True)
            if tokens.shape[1] > context_size:
                tokens = tokens[:, :context_size]
            if tokens.shape[1] >= 2:
                batch_tokens.append(tokens[0])

        if not batch_tokens:
            return []

        max_len = max(t.shape[0] for t in batch_tokens)
        padded = torch.zeros(len(batch_tokens), max_len, dtype=torch.long, device=device)
        lengths: list[int] = []
        for i, t in enumerate(batch_tokens):
            padded[i, : len(t)] = t.to(device)
            lengths.append(len(t))

        with torch.no_grad():
            _, cache = model.run_with_cache(
                padded,
                names_filter=hook_names,
                stop_at_layer=stop_at_layer,
            )
            if is_multi_hook:
                # [B, seq, n_layers, d_model] — required by the crosscoder encode
                x = torch.stack([cache[h] for h in hook_names], dim=2)
            else:
                x = cache[hook_names[0]]   # [B, seq, d_model]
            z = sae.encode(x)              # [B, seq, d_sae] in both cases
            del cache, x
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        results = []
        for b, length in enumerate(lengths):
            str_toks = model.to_str_tokens(padded[b, :length])
            sentence = (
                "".join(str_toks).replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ").strip()
            )

            start = 1 if skip_bos else 0
            token_records = []
            for pos in range(start, length):
                feat_acts = z[b, pos]   # [d_sae]
                k_actual = min(top_k, feat_acts.shape[0])
                top_vals, top_ids = torch.topk(feat_acts, k=k_actual)

                if min_activation > 0.0:
                    keep = top_vals >= min_activation
                    top_vals = top_vals[keep]
                    top_ids = top_ids[keep]

                token_records.append({
                    "token": clean_token(str_toks[pos]),
                    "position": int(pos),
                    "top_feature_ids":  top_ids.cpu().tolist(),
                    "top_feature_acts": [round(v, 5) for v in top_vals.cpu().tolist()],
                })

            results.append({"sentence": sentence, "tokens": token_records})
        return results

    for text in texts:
        batch_texts.append(text)
        if len(batch_texts) == batch_size:
            yield from _process_batch(batch_texts)
            batch_texts = []
    if batch_texts:
        yield from _process_batch(batch_texts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.test:
        args.n_texts = 20
        print("TEST MODE — processing 20 texts only.")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU.")
        device = "cpu"
    print(f"Device: {device}")
    print(f"Loading SAE + LLM from: {args.checkpoint}")

    model, sae = load_models(args.checkpoint, device=device, dtype=args.dtype)

    dataset_name = args.dataset or getattr(sae.cfg.metadata, "dataset_path", None)
    if not dataset_name:
        raise ValueError(
            "No dataset specified. Pass --dataset or ensure cfg.metadata.dataset_path is set."
        )
    print(f"Dataset: {dataset_name}")

    hook_names = get_hook_names(sae)
    sae_kind = "crosscoder" if len(hook_names) > 1 else "single-layer SAE"
    print(f"SAE kind: {sae_kind} ({len(hook_names)} hook(s))")

    texts = stream_texts(dataset_name, args.n_texts)
    print(f"Loaded {len(texts)} texts.")
    print(f"\nExtracting top-{args.top_k} features per token (batch_size={args.batch_size}) …\n")

    all_results = []
    sentence_gen = extract_sentence_features(
        model=model,
        sae=sae,
        texts=texts,
        top_k=args.top_k,
        batch_size=args.batch_size,
        context_size=args.context_size,
        device=device,
        skip_bos=args.skip_bos,
        min_activation=args.min_activation,
    )

    pbar = tqdm(total=args.n_texts, desc="Sentences")
    for sent_result in sentence_gen:
        all_results.append(sent_result)
        pbar.update(1)
        if len(all_results) >= args.n_texts:
            break
    pbar.close()

    if args.output is None:
        repo_root = Path(__file__).resolve().parent
        run_name = Path(args.checkpoint).resolve().parent.name
        output_path = repo_root / "eval_results" / run_name / "feature_analysis" / "token_features.json"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving {len(all_results)} sentences → {output_path}")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    total_tokens = sum(len(s["tokens"]) for s in all_results)
    print(f"\n{'=' * 60}")
    print(f"  Saved: {output_path}")
    print(f"  Sentences: {len(all_results)}")
    print(f"  Total tokens recorded: {total_tokens}")
    print(f"  Top-K features per token: {args.top_k}")
    print(f"{'=' * 60}")

    if all_results:
        ex = all_results[0]
        print(f"\nExample — first sentence:\n  \"{ex['sentence'][:80]}\"")
        if ex["tokens"]:
            t = ex["tokens"][0]
            print(
                f"  Token '{t['token']}' (pos {t['position']}): "
                f"top features = {t['top_feature_ids'][:5]} …"
            )


if __name__ == "__main__":
    main()
