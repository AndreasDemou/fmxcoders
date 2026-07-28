"""Shared loading helper for compare_absorption.py / compare_ravel.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sae_lens.registry import SAE_CLASS_REGISTRY
from sae_lens.saes.sae import SAE, TrainingSAE


def load_crosscoder_inference(
    checkpoint_path: Path | str,
    device: str = "cpu",
    dtype: str = "float32",
) -> SAE:
    """Load a saved crosscoder as an inference SAE.

    Handles two cases:
      - Inference checkpoint (architecture in `SAE_CLASS_REGISTRY`, e.g.
        `cross_jumprelu`, `cross_topk`): loaded directly.
      - Training checkpoint (architecture in `SAE_TRAINING_CLASS_REGISTRY`,
        e.g. `cross_batchtopk`, `cross_matryoshka`, training `cross_topk`):
        loaded via `TrainingSAE`, exported to a tempdir as its inference
        counterpart (which applies fold + topk_threshold→threshold), then
        reloaded as inference. This routes the conversion through the
        canonical `save_inference_model` path.
    """
    checkpoint_path = Path(checkpoint_path)
    cfg_path = checkpoint_path / "cfg.json"
    with cfg_path.open() as f:
        arch = json.load(f)["architecture"]

    if arch in SAE_CLASS_REGISTRY:
        return SAE.load_from_pretrained(checkpoint_path, device=device, dtype=dtype)

    train_sae = TrainingSAE.load_from_pretrained(
        checkpoint_path, device=device, dtype=dtype
    )
    with tempfile.TemporaryDirectory(prefix="crosscoder_inference_") as tmp:
        train_sae.save_inference_model(tmp)
        return SAE.load_from_pretrained(tmp, device=device, dtype=dtype)
