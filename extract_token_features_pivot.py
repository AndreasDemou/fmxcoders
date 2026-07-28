#!/usr/bin/env python3
"""
Pivot SAE/crosscoder feature activations from (sentence, token) -> top features
into feature -> top-K (sentence, token) examples, suitable for visualization.

Input format (JSON array, or one JSON object per line for .jsonl):
    [
      {
        "sentence": "The quick brown fox ...",
        "tokens": [
          {
            "token": "quick",
            "position": 2,
            "top_feature_ids":  [142, 7801, 33, 512, ...],
            "top_feature_acts": [3.21, 2.87, 1.44, ...]
          },
          ...
        ]
      },
      ...
    ]

Output format:
    {
      "142": [
        {"activation": 9.81, "sentence_id": 17, "position": 4,
         "token": "rapid", "sentence": "..."},
        ...                            # up to top_k examples, sorted desc
      ],
      ...
    }

Usage:
    python pivot_features.py activations.json
    python pivot_features.py activations.json -k 50
"""

import argparse
import heapq
import json
from collections import defaultdict
from pathlib import Path


def load_records(path: Path):
    """Yield records from a .json (array) or .jsonl file."""
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            # Standard JSON array. Loaded fully into memory.
            for record in json.load(f):
                yield record
        else:
            # JSONL: one JSON object per line.
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def pivot(records, top_k: int):
    """
    Build feature_id -> min-heap of size top_k.
    Each entry: (activation, sentence_id, position, token, sentence).
    Min-heap means the weakest example is at the root, so we can cheaply
    decide whether a new example should bump it.
    """
    heaps = defaultdict(list)
    n_sentences = n_tokens = n_activations = 0

    for sent_id, record in enumerate(records):
        sentence = record.get("sentence", "")
        n_sentences += 1
        for tok in record.get("tokens", []):
            n_tokens += 1
            position = tok["position"]
            token_str = tok["token"]
            ids = tok["top_feature_ids"]
            acts = tok["top_feature_acts"]
            for fid, act in zip(ids, acts):
                n_activations += 1
                act = float(act)
                example = (act, sent_id, position, token_str, sentence)
                h = heaps[fid]
                if len(h) < top_k:
                    heapq.heappush(h, example)
                elif act > h[0][0]:
                    heapq.heapreplace(h, example)

    stats = {
        "n_sentences": n_sentences,
        "n_tokens": n_tokens,
        "n_activations": n_activations,
        "n_features": len(heaps),
    }
    return heaps, stats


def heaps_to_dict(heaps):
    """Convert heaps to a JSON-serializable dict, sorted desc by activation."""
    out = {}
    for fid, h in heaps.items():
        sorted_examples = sorted(h, key=lambda x: -x[0])
        out[str(fid)] = [
            {
                "activation": act,
                "sentence_id": sent_id,
                "position": position,
                "token": token_str,
                "sentence": sentence,
            }
            for act, sent_id, position, token_str, sentence in sorted_examples
        ]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Pivot per-token feature activations into per-feature top examples.",
    )
    parser.add_argument("input", type=Path, help="Path to .json or .jsonl input file")
    parser.add_argument(
        "-k", "--top-k", type=int, default=20,
        help="Examples to keep per feature (default: 20)",
    )
    args = parser.parse_args()

    output = args.input.with_name(f"{args.input.stem}_pivot.json")

    print(f"Reading {args.input} ...")
    records = list(load_records(args.input))
    print(f"  loaded {len(records):,} records")

    print(f"Pivoting (top-{args.top_k} per feature) ...")
    heaps, stats = pivot(records, args.top_k)
    print(f"  sentences:          {stats['n_sentences']:,}")
    print(f"  tokens:             {stats['n_tokens']:,}")
    print(f"  activation entries: {stats['n_activations']:,}")
    print(f"  unique features:    {stats['n_features']:,}")

    print(f"Writing {output} ...")
    out = heaps_to_dict(heaps)
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == "__main__":
    main()
