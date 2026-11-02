#!/usr/bin/env python3
"""
train_and_saebench.py

End-to-end script:
1) Train an SAE or crosscoder with SAELens (this repo's fork).
2) Run selected SAEBench evaluations (no AutoInterp):
   - L0 / Loss Recovered
   - Sparse Probing
   - Targeted Probe Perturbation (TPP)
   - RAVEL
   - Spurious Correlation Removal (SCR)
   - Feature Absorption

Assumed layout:
- You are inside the SAELens repo.
- You cloned SAEBench into a subfolder (e.g., ./SAEBench) OR installed it editable.
  Recommended: `pip install -e SAEBench -c SAEBench/constraints.txt` :contentReference[oaicite:1]{index=1}

Usage example (training only):
  python train_and_saebench.py --exp pythia160m_l8 --architecture topk --k 64

Training + SAEBench, explicit SAE selection (recommended):
  python train_and_saebench.py \
    --exp pythia160m_l8 --architecture topk --k 64 \
    --run_saebench \
    --saebench_model_name pythia-160m-deduped \
    --sae_regex_pattern "my_sae_run_name" \
    --sae_block_pattern "blocks.8.hook_resid_post__trainer_0"

If you omit --sae_regex_pattern / --sae_block_pattern, we attempt inference from checkpoint_path and hook_name.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import json
import time
from glob import glob

import wandb
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Work around a transformers 4.57 bug: _patch_mistral_regex probes the HF Hub
# (model_info) for any tokenizer with vocab > 100k referenced by repo-id — including
# Gemma-2 (256k vocab) — which raises OfflineModeIsEnabled under HF_HUB_OFFLINE.
# Gemma isn't Mistral, so skipping the patch is a no-op for every model used here.
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
PreTrainedTokenizerBase._patch_mistral_regex = classmethod(
    lambda cls, tokenizer, *args, **kwargs: tokenizer
)


# ----------------------------
# Local wandb-log mirroring
# ----------------------------

_LOCAL_LOG_DIR: Optional[Path] = None


def _json_sanitize(v: Any) -> Any:
    """Best-effort JSON-safe conversion (scalars, small lists/dicts kept; large/opaque objects stringified)."""
    if isinstance(v, (type(None), bool, int, float, str)):
        return v
    if hasattr(v, "item"):  # 0-d tensor / numpy scalar
        try:
            return v.item()
        except Exception:
            pass
    if hasattr(v, "tolist"):
        try:
            lst = v.tolist()
            if isinstance(lst, list) and len(lst) > 1000:
                return f"<{type(v).__name__} shape={tuple(getattr(v, 'shape', ()))}>"
            return lst
        except Exception:
            pass
    if isinstance(v, dict):
        return {str(k): _json_sanitize(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        if len(v) > 1000:
            return f"<list len={len(v)}>"
        return [_json_sanitize(x) for x in v]
    return f"<{type(v).__name__}>"


def setup_local_logging(
    run_name: str,
    config: Optional[dict] = None,
    wandb_enabled: bool = True,
) -> Path:
    """Mirror every wandb.log call to wandb/local/<run_name>/metrics.jsonl.

    Works regardless of whether wandb is active. The patched wandb.log
    always writes locally, and forwards to wandb only if a run is active.

    When wandb_enabled=False, wandb.init is patched to force mode="disabled"
    so callers can keep log_to_wandb=True (and the gated wandb.log calls in
    the training loop still fire to our local mirror).
    """
    global _LOCAL_LOG_DIR
    _LOCAL_LOG_DIR = Path("wandb/local") / run_name
    _LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

    if config is not None:
        with open(_LOCAL_LOG_DIR / "config.json", "w") as f:
            json.dump(_json_sanitize(config), f, indent=2, default=str)

    # Define the patched wandb.log. Key points:
    #   - Local write is wrapped in try/except so training never dies from a
    #     local-logging bug.
    #   - Forwarding uses wandb.run.log directly. (We can NOT forward to a
    #     captured wandb.log reference: before wandb.init() is called,
    #     wandb.log is a pre-init "stub" that raises on use; calling it
    #     post-init would break training. wandb.run.log is the real, Run-
    #     bound method created during init.)
    _patch_fired = {"done": False}

    def patched_log(data, step=None, commit=None, sync=None):
        if _LOCAL_LOG_DIR is not None and isinstance(data, dict):
            try:
                record = {
                    "timestamp": time.time(),
                    "step": step,
                    "data": _json_sanitize(data),
                }
                with open(_LOCAL_LOG_DIR / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
                if not _patch_fired["done"]:
                    print(f"[local logging] patched wandb.log fired (→ {_LOCAL_LOG_DIR / 'metrics.jsonl'})")
                    _patch_fired["done"] = True
            except Exception as e:  # noqa: BLE001
                if not _patch_fired["done"]:
                    print(f"[local logging] WARN: local write failed: {e}")
                    _patch_fired["done"] = True
        if wandb.run is not None:
            # Use wandb.run.log (the real method), NOT the captured pre-init stub.
            # Run.log accepts step and commit but not sync, so only forward those.
            return wandb.run.log(data, step=step, commit=commit)

    wandb.log = patched_log  # type: ignore[assignment]
    wandb._polysae_patched_log = patched_log  # type: ignore[attr-defined]

    # Always patch wandb.init — for two reasons:
    #   (a) if wandb_enabled=False, force mode="disabled" so the runner's
    #       wandb.init() call succeeds without syncing to the cloud;
    #   (b) wandb.init() internally rebinds wandb.log (to a Run-bound version),
    #       overwriting our patch. We re-apply our patch after init returns.
    if not hasattr(wandb, "_polysae_original_init"):
        wandb._polysae_original_init = wandb.init  # type: ignore[attr-defined]

    def patched_init(*args, **kwargs):
        if not wandb_enabled:
            kwargs["mode"] = "disabled"
        result = wandb._polysae_original_init(*args, **kwargs)  # type: ignore[attr-defined]
        # Re-apply the wandb.log patch after init
        wandb.log = wandb._polysae_patched_log  # type: ignore[assignment, attr-defined]
        return result

    wandb.init = patched_init  # type: ignore[assignment]

    print(f"[local logging] Writing to {_LOCAL_LOG_DIR} (wandb_enabled={wandb_enabled})")
    return _LOCAL_LOG_DIR

import torch

from sae_lens import (
    LanguageModelSAERunnerConfig,
    LoggingConfig,
    LanguageModelSAETrainingRunner,
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
    # Crosscoder variants
    CrossTopKTrainingSAEConfig,
    CrossBatchTopKTrainingSAEConfig,
    CrossMatryoshkaBatchTopKTrainingSAEConfig,
    SAE,
    TrainingSAE,
)

# Make the SAEBench fork bundled with this repo importable even if it has not been
# pip-installed. It goes first on sys.path so it takes precedence over an upstream
# sae_bench, which lacks crosscoder support. --saebench_repo_dir is read here, ahead
# of the main parser, because SAEBench is imported at module load time.
def _resolve_saebench_dir() -> Optional[Path]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--saebench_repo_dir", default="SAEBench")
    raw = Path(pre.parse_known_args()[0].saebench_repo_dir).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, Path(__file__).resolve().parent / raw]
    for c in candidates:
        if (c / "sae_bench").is_dir():
            return c.resolve()
    return None


_SAEBENCH_DIR = _resolve_saebench_dir()
if _SAEBENCH_DIR is not None and str(_SAEBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_SAEBENCH_DIR))

# SAEBench imports for direct API usage
try:
    import sae_bench.evals.core.main as saebench_core
    import sae_bench.evals.scr_and_tpp.main as saebench_scr_tpp
    import sae_bench.evals.sparse_probing.main as saebench_sparse_probing
    import sae_bench.evals.ravel.main as saebench_ravel
    import sae_bench.evals.absorption.main as saebench_absorption
    import sae_bench.sae_bench_utils.general_utils as saebench_utils
    SAEBENCH_AVAILABLE = True
except ImportError:
    SAEBENCH_AVAILABLE = False

# ----------------------------
# Width parsing
# ----------------------------

WIDTH_TO_DSAE = {
    "2k": 2_048,
    "4k": 4_096,
    "8k": 8_192,
    "12k": 12_288,
    "16k": 16_384,
    "32k": 32_768,
    "65k": 65_536,
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
    "gpt2_l8": Preset(
        model_name="gpt2",
        hook_name="blocks.8.hook_resid_post",
        dataset_path="dylanebert/openwebtext",
        d_in=768,
        allowed_widths=["2k", "4k", "8k", "12k", "16k", "32k", "65k"],
        default_width="16k",
    ),
    "pythia70m_l3": Preset(
        model_name="EleutherAI/pythia-70m-deduped",
        hook_name="blocks.3.hook_resid_post",
        dataset_path="monology/pile-uncopyrighted",
        d_in=512,
        allowed_widths=["16k", "32k", "65k"],
        default_width="16k",
    ),
    "pythia70m_l5": Preset(
        model_name="EleutherAI/pythia-70m-deduped",
        hook_name="blocks.5.hook_resid_post",
        dataset_path="monology/pile-uncopyrighted",
        d_in=512,
        allowed_widths=["16k", "32k", "65k"],
        default_width="16k",
    ),
    "pythia160m_l8": Preset(
        model_name="EleutherAI/pythia-160m-deduped",
        hook_name="blocks.8.hook_resid_post",
        dataset_path="monology/pile-uncopyrighted",
        d_in=768,
        allowed_widths=["4k", "16k", "65k"],
        default_width="16k",
    ),
    "gemma2_2b_l12": Preset(
        model_name="google/gemma-2-2b",
        hook_name="blocks.12.hook_resid_post",
        dataset_path="dylanebert/openwebtext",
        d_in=2304,
        allowed_widths=["16k", "65k", "131k"],
        default_width="16k",
    ),
    "gemma2_2b_l19": Preset(
        model_name="google/gemma-2-2b",
        hook_name="blocks.19.hook_resid_post",
        dataset_path="dylanebert/openwebtext",
        d_in=2304,
        allowed_widths=["16k", "65k", "131k"],
        default_width="16k",
    ),
    "gemma2_27b_l22": Preset(
        model_name="google/gemma-2-27b",
        hook_name="blocks.22.hook_resid_post",   
        dataset_path="dylanebert/openwebtext",
        d_in=4608,
        allowed_widths=["16k", "65k", "131k"],
        default_width="16k",
    ),
    "llama31_8b_l25": Preset(
        model_name="meta-llama/Llama-3.1-8B",
        hook_name="blocks.25.hook_resid_post",
        dataset_path="cerebras/SlimPajama-627B",
        d_in=4096,
        allowed_widths=["32k", "128k"],
        default_width="32k",
    ),
    "pythia410m_l15": Preset(
        model_name="EleutherAI/pythia-410m-deduped",
        hook_name="blocks.15.hook_resid_post",
        #hook_name="blocks.15.hook_attn_out",
        dataset_path="monology/pile-uncopyrighted",
        d_in=1024,
        allowed_widths=["8k","16k", "32k", "65k"],
        default_width="16k",
    ),
    "pythia1_4b_l12": Preset(
        model_name="EleutherAI/pythia-1.4b-deduped",
        hook_name="blocks.12.hook_resid_post",
        dataset_path="monology/pile-uncopyrighted",
        d_in=2048,
        allowed_widths=["16k", "32k", "65k"],
        default_width="16k",
    ),
    "llama32_3b_l12": Preset(
        model_name="meta-llama/Llama-3.2-3B",
        hook_name="blocks.12.hook_resid_post",
        #dataset_path="cerebras/SlimPajama-627B",
        dataset_path="monology/pile-uncopyrighted",
        d_in=3072,
        allowed_widths=["16k", "32k", "65k"],
        default_width="16k",
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


def parse_poly_ranks(csv: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """Parse poly_ranks from comma-separated string 'R1,R2,R3'."""
    if csv is None:
        return None
    items = [x.strip() for x in csv.split(",") if x.strip()]
    if len(items) != 3:
        raise SystemExit(f"--poly_ranks must have exactly 3 values (R1,R2,R3), got {len(items)}: {csv}")
    r1, r2, r3 = [int(x) for x in items]
    # Allow any rank combination (removed R1 >= R2 >= R3 constraint)
    return (r1, r2, r3)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Training args (same as your script)
    p.add_argument("--exp", required=True, choices=sorted(PRESETS.keys()))
    p.add_argument("--width", default=None, help=f"One of {sorted(WIDTH_TO_DSAE.keys())}")

    p.add_argument(
        "--architecture",
        required=True,
        choices=["standard", "topk", "batchtopk", "jumprelu", "matryoshka"],
        help="Sparsity mechanism / SAE architecture.",
    )

    p.add_argument("--use_saelens_defaults", action="store_true")
    p.add_argument("--l1_coefficient", type=float, default=None)
    p.add_argument("--l0_coefficient", type=float, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--matryoshka_widths", type=str, default=None)

    # PolySAE: polynomial decoder modifier (can be combined with any architecture)
    p.add_argument("--poly", action="store_true", help="Enable polynomial decoder (PolySAE)")
    p.add_argument("--poly_rank", type=int, default=None, help="Single rank for all PolySAE terms (default: d_in)")
    p.add_argument("--poly_ranks", type=str, default=None, help="Comma-separated ranks R1,R2,R3 for nested truncation (R1>=R2>=R3). Overrides --poly_rank.")
    p.add_argument("--poly_order", type=int, default=3, choices=[1, 2, 3], help="Polynomial decoder order: 1=linear, 2=linear+quadratic, 3=linear+quadratic+cubic (default: 3)")
    p.add_argument("--shared_u", action="store_true", help="Use single shared U matrix for all polynomial orders (requires --poly)")
    p.add_argument("--lambda2_init", type=float, default=-0.5, help="Initial λ₂ for PolySAE")
    p.add_argument("--lambda3_init", type=float, default=0.5, help="Initial λ₃ for PolySAE")

    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train_batch_size_tokens", type=int, default=None)
    p.add_argument("--store_batch_size_prompts", type=int, default=None,
    help="Number of prompts per LLM forward pass when gathering activations (model.run_with_cache batch). Main GPU-memory lever for large models; lower this to fix forward-pass OOM. sae_lens default is 32.")
    p.add_argument("--context_size", type=int, default=None)
    p.add_argument("--training_tokens", type=int, default=None)
    p.add_argument("--n_batches_in_buffer", type=int, default=32)
    p.add_argument("--act_store_device", type=str, default=None, help="Device for the activation mixing buffer. Use 'cpu' to keep the buffer in host RAM and avoid GPU OOM on large crosscoders (trades VRAM for ~5-15%% PCIe overhead). Defaults to the same device as the model.")
    p.add_argument("--eval_batch_size_prompts", type=int, default=None, help="Batch size (prompts) for evaluation. Useful to set lower than training batch size to avoid OOM.")
    p.add_argument("--eval_ce_loss", action="store_true", help="Enable Cross-Entropy loss evaluation during training (expensive, can cause OOM).")
    p.add_argument("--eval_kl", action="store_true", help="Enable KL divergence evaluation during training (expensive, can cause OOM).")

    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    p.add_argument("--model_dtype", default=None, choices=["float32", "float16", "bfloat16"],
    help="dtype to LOAD THE LLM WEIGHTS in. Distinct from --dtype (SAE/activation dtype). Use bfloat16 for Gemma-2.")
    p.add_argument("--n_devices", type=int, default=None,
    help="Shard the LLM across this many GPUs (TransformerLens naive layer/pipeline parallelism). Use to fit a model too large for one GPU. Requires that many visible GPUs.")

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--wandb_project", default="crosscoders_NeurIPS")
    p.add_argument("--wandb_run_name", default=None, help="Custom WandB run name (also used for checkpoint_path if not specified)")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--sae_name", default=None, help="Custom SAE name for absorption eval outputs (overrides checkpoint folder name)")
    p.add_argument("--n_checkpoints", type=int, default=None)
    p.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume training from an existing checkpoint. Requires --wandb_run_name to locate the checkpoint directory.",
    )

    p.add_argument("--report_sparsity", action="store_true")
    p.add_argument("--sparsity_samples", type=int, default=64)
    p.add_argument("--sparsity_context", type=int, default=None)
    
    # Rescaling option (affects TopK, BatchTopK, Matryoshka)
    p.add_argument(
        "--no_rescale_by_decoder_norm",
        action="store_true",
        help="Disable rescale_acts_by_decoder_norm (default is True for TopK variants).",
    )

    # SAEBench args
    p.add_argument("--run_saebench", action="store_true", help="Run SAEBench after training.")
    p.add_argument("--eval_only", action="store_true", help="Skip training; load SAE from --checkpoint_path and run SAEBench evals only. Requires --checkpoint_path and --run_saebench.")
    p.add_argument(
        "--saebench_repo_dir",
        default="SAEBench",
        help="Path to the SAEBench repo to import sae_bench from (default: the fork bundled in ./SAEBench).",
    )
    p.add_argument(
        "--saebench_model_name",
        default=None,
        help=(
            "Model name to pass to SAEBench (often differs from HF id). "
            "Example in SAEBench README: pythia-70m-deduped. If omitted, we try to infer."
        ),
    )
    p.add_argument(
        "--sae_regex_pattern",
        default=None,
        help="SAEBench SAE selection: regex for the SAE group name.",
    )
    p.add_argument(
        "--sae_block_pattern",
        default=None,
        help="SAEBench SAE selection: block pattern, e.g. blocks.8.hook_resid_post__trainer_0",
    )
    p.add_argument(
        "--saebench_python",
        default=sys.executable,
        help="Python executable to use for SAEBench subprocess calls.",
    )
    p.add_argument(
        "--skip_eval",
        action="append",
        default=[],
        help="Optional: repeatable. One of {l0_loss, sparse_probing, tpp, ravel, scr, absorption}.",
    )
    # Memory management for SAEBench (needed for larger models like Pythia 1.4B)
    p.add_argument(
        "--saebench_sae_batch_size",
        type=int,
        default=None,
        help="SAE batch size for SAEBench evals. Default is 125, auto-reduced for large models.",
    )
    p.add_argument(
        "--saebench_llm_batch_size",
        type=int,
        default=None,
        help="LLM batch size for SAEBench evals. Default is 16, auto-reduced for large models.",
    )
    p.add_argument(
        "--saebench_lower_vram",
        action="store_true",
        help="Enable lower VRAM usage for SAEBench (moves model to CPU between batches). Auto-enabled for large models.",
    )
    p.add_argument("--use_crosscoder", action="store_true")
    p.add_argument("--cross_layers", type=str, default=None, help="Comma-separated layer identifiers l1,l2,l3,... for crosscoder training")
    p.add_argument("--layer_masking", type=float, default=0.0, help="Per-token probability of zeroing out each layer's input during crosscoder training. Target reconstruction is still the unmasked activations.")
    p.add_argument("--use_cross_fact", action="store_true", help="Enable tensor factorization of crosscoder W_enc and W_dec.")
    p.add_argument("--cross_fact_type", type=str, default="CP", choices=["CP", "TR"], help="Factorization type: CP (single rank R) or TR (three ranks r1,r2,r3).")
    p.add_argument("--cross_fact_ranks", type=str, default="32", help="Comma-separated rank(s). CP needs 1 value (R). TR needs 3 values (r1,r2,r3) bordering n_layers, d_in, d_sae respectively.")


    return p.parse_args()


# ----------------------------
# SAE config factory (same as your script)
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
    poly_ranks: Optional[Tuple[int, int, int]] = None,  # Overrides poly_rank when set
    poly_order: int = 3,
    shared_u: bool = False,  # Use single shared U matrix for all polynomial orders
    lambda2_init: float = -0.5,
    lambda3_init: float = 0.5,
    # Rescaling option for TopK variants
    rescale_acts_by_decoder_norm: bool = True,
    # Crosscoder configs
    use_crosscoder: bool = False,
    cross_layers: Optional[List[int]] = None,
    layer_masking: float = 0.0,
    use_cross_fact: bool = False,
    cross_fact_type: str = "CP",
    cross_fact_ranks: Tuple[int, ...] = (32,),
):
    """
    Build SAE config for the given architecture.
    
    If use_poly=True, returns the Poly* variant that adds polynomial decoder
    while keeping the same activation function and loss behavior.
    """
    if arch == "standard":
        if use_poly:
            raise SystemExit("--poly is not yet supported with standard architecture (only topk, batchtopk, jumprelu, matryoshka)")
        if use_defaults and l1_coefficient is None:
            return StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae)
        if l1_coefficient is None:
            raise SystemExit("--l1_coefficient required for standard unless --use_saelens_defaults")
        return StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae, l1_coefficient=l1_coefficient)

    if arch == "topk":
        if use_poly:
            if use_defaults and k is None:
                return PolyTopKTrainingSAEConfig(
                    d_in=d_in, d_sae=d_sae,
                    poly_ranks=poly_ranks,
                    poly_order=poly_order,
                    shared_u=shared_u,
                    lambda2_init=lambda2_init,
                    lambda3_init=lambda3_init,
                    rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
                )
            if k is None:
                raise SystemExit("--k required for topk unless --use_saelens_defaults")
            return PolyTopKTrainingSAEConfig(
                d_in=d_in, d_sae=d_sae, k=k,
                poly_ranks=poly_ranks,
                poly_order=poly_order,
                shared_u=shared_u,
                lambda2_init=lambda2_init,
                lambda3_init=lambda3_init,
                rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
            )
        # Crosscoder TopK
        if use_crosscoder:
            if cross_layers is None:
                raise SystemExit("--cross_layers required when using --use_crosscoder")
            n_layers = len(cross_layers)
            if use_defaults and k is None:
                return CrossTopKTrainingSAEConfig(
                    d_in=d_in, d_sae=d_sae, n_layers=n_layers,
                    layer_masking=layer_masking,
                    use_cross_fact=use_cross_fact,
                    cross_fact_type=cross_fact_type,
                    cross_fact_ranks=cross_fact_ranks,
                    rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
                )
            if k is None:
                raise SystemExit("--k required for topk unless --use_saelens_defaults")
            return CrossTopKTrainingSAEConfig(
                d_in=d_in, d_sae=d_sae, k=k, n_layers=n_layers,
                layer_masking=layer_masking,
                use_cross_fact=use_cross_fact,
                cross_fact_type=cross_fact_type,
                cross_fact_ranks=cross_fact_ranks,
                rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
            )
        # Non-poly TopK
        if use_defaults and k is None:
            return TopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm)
        if k is None:
            raise SystemExit("--k required for topk unless --use_saelens_defaults")
        return TopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, k=k, rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm)

    if arch == "batchtopk":
        if use_poly:
            if use_defaults and k is None:
                return PolyBatchTopKTrainingSAEConfig(
                    d_in=d_in, d_sae=d_sae,
                    poly_ranks=poly_ranks,
                    poly_order=poly_order,
                    shared_u=shared_u,
                    lambda2_init=lambda2_init,
                    lambda3_init=lambda3_init,
                    rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
                )
            if k is None:
                raise SystemExit("--k required for batchtopk unless --use_saelens_defaults")
            return PolyBatchTopKTrainingSAEConfig(
                d_in=d_in, d_sae=d_sae, k=k,
                poly_ranks=poly_ranks,
                poly_order=poly_order,
                shared_u=shared_u,
                lambda2_init=lambda2_init,
                lambda3_init=lambda3_init,
                rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
            )
        # Crosscoder BatchTopK
        if use_crosscoder:
            if cross_layers is None:
                raise SystemExit("--cross_layers required when using --use_crosscoder")
            n_layers = len(cross_layers)
            if use_defaults and k is None:
                return CrossBatchTopKTrainingSAEConfig(
                    d_in=d_in, d_sae=d_sae, n_layers=n_layers,
                    layer_masking=layer_masking,
                    use_cross_fact=use_cross_fact,
                    cross_fact_type=cross_fact_type,
                    cross_fact_ranks=cross_fact_ranks,
                    rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
                )
            if k is None:
                raise SystemExit("--k required for batchtopk unless --use_saelens_defaults")
            return CrossBatchTopKTrainingSAEConfig(
                d_in=d_in, d_sae=d_sae, k=k, n_layers=n_layers,
                layer_masking=layer_masking,
                use_cross_fact=use_cross_fact,
                cross_fact_type=cross_fact_type,
                cross_fact_ranks=cross_fact_ranks,
                rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
            )
        # Non-poly BatchTopK
        if use_defaults and k is None:
            return BatchTopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm)
        if k is None:
            raise SystemExit("--k required for batchtopk unless --use_saelens_defaults")
        return BatchTopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, k=k, rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm)

    if arch == "jumprelu":
        if use_poly:
            if use_defaults and l0_coefficient is None:
                return PolyJumpReLUTrainingSAEConfig(
                    d_in=d_in, d_sae=d_sae,
                    poly_ranks=poly_ranks,
                    poly_order=poly_order,
                    shared_u=shared_u,
                    lambda2_init=lambda2_init,
                    lambda3_init=lambda3_init,
                )
            if l0_coefficient is None:
                raise SystemExit("--l0_coefficient required for jumprelu unless --use_saelens_defaults")
            return PolyJumpReLUTrainingSAEConfig(
                d_in=d_in, d_sae=d_sae,
                l0_coefficient=l0_coefficient,
                poly_ranks=poly_ranks,
                poly_order=poly_order,
                shared_u=shared_u,
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
                    poly_ranks=poly_ranks,
                    poly_order=poly_order,
                    shared_u=shared_u,
                    lambda2_init=lambda2_init,
                    lambda3_init=lambda3_init,
                    rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
                )
            if k is None:
                raise SystemExit("--k required for matryoshka unless --use_saelens_defaults")
            return PolyMatryoshkaBatchTopKTrainingSAEConfig(
                d_in=d_in, d_sae=d_sae, k=k, matryoshka_widths=widths,
                poly_ranks=poly_ranks,
                poly_order=poly_order,
                shared_u=shared_u,
                lambda2_init=lambda2_init,
                lambda3_init=lambda3_init,
                rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
            )
        # Crosscoder Matryoshka
        if use_crosscoder:
            if cross_layers is None:
                raise SystemExit("--cross_layers required when using --use_crosscoder")
            n_layers = len(cross_layers)
            if use_defaults and k is None:
                return CrossMatryoshkaBatchTopKTrainingSAEConfig(
                    d_in=d_in, d_sae=d_sae, matryoshka_widths=widths, n_layers=n_layers,
                    layer_masking=layer_masking,
                    use_cross_fact=use_cross_fact,
                    cross_fact_type=cross_fact_type,
                    cross_fact_ranks=cross_fact_ranks,
                    rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
                )
            if k is None:
                raise SystemExit("--k required for matryoshka unless --use_saelens_defaults")
            return CrossMatryoshkaBatchTopKTrainingSAEConfig(
                d_in=d_in, d_sae=d_sae, k=k, matryoshka_widths=widths, n_layers=n_layers,
                layer_masking=layer_masking,
                use_cross_fact=use_cross_fact,
                cross_fact_type=cross_fact_type,
                cross_fact_ranks=cross_fact_ranks,
                rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
            )
        # Non-poly Matryoshka
        if use_defaults and k is None:
            return MatryoshkaBatchTopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, matryoshka_widths=widths, rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm)
        if k is None:
            raise SystemExit("--k required for matryoshka unless --use_saelens_defaults")
        return MatryoshkaBatchTopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, k=k, matryoshka_widths=widths, rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm)

    raise SystemExit(f"Unknown architecture: {arch}")



# ----------------------------
# Parameter Counting Helper
# ----------------------------

def count_decoder_params(sae) -> int:
    """
    Count the number of trainable parameters in the SAE decoder.
    Matches formula used in ICML paper tables.
    """
    total_params = 0
    
    # 1. Decoder Bias (present in all)
    if hasattr(sae, "b_dec"):
         total_params += sae.b_dec.numel()
         
    # 2. Check for PolySAE
    # PolySAE uses PolyDecoderMixin which has specific attributes
    is_poly = hasattr(sae, "C1") # Good indicator
    
    if is_poly:
        # Check Shared U vs Separate U
        if hasattr(sae, "U_shared"):
             # Shared U mode
             total_params += sae.U_shared.numel()
        else:
             # Separate U mode
             if hasattr(sae, "U1"): total_params += sae.U1.numel()
             if hasattr(sae, "U2") and isinstance(sae.U2, torch.nn.Parameter): total_params += sae.U2.numel()
             if hasattr(sae, "U3") and isinstance(sae.U3, torch.nn.Parameter): total_params += sae.U3.numel()
             
        # C matrices
        if hasattr(sae, "C1"): total_params += sae.C1.numel()
        if hasattr(sae, "C2") and isinstance(sae.C2, torch.nn.Parameter): total_params += sae.C2.numel()
        if hasattr(sae, "C3") and isinstance(sae.C3, torch.nn.Parameter): total_params += sae.C3.numel()
        
        # Lambdas
        if hasattr(sae, "lambda2") and isinstance(sae.lambda2, torch.nn.Parameter): total_params += sae.lambda2.numel()
        if hasattr(sae, "lambda3") and isinstance(sae.lambda3, torch.nn.Parameter): total_params += sae.lambda3.numel()
        
    else:
        # Standard SAE: W_dec
        if hasattr(sae, "W_dec"):
            total_params += sae.W_dec.numel()
            
    return total_params

# ----------------------------
# Best-effort sparsity reporting (copied from your script)
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
    try:
        from datasets import IterableDataset
        from transformer_lens import HookedTransformer

        from sae_lens.offline_hf import offline_aware_load_dataset
    except Exception as e:
        raise RuntimeError("Missing dependency for sparsity report. Install: pip install datasets transformer-lens") from e
    if "gemma-2-27b" in model_name.lower():
        # 27B won't fit in fp32 (~108 GB); load weights in bf16.
        # bfloat16 not float16 — Gemma-2's logit soft-capping overflows fp16.
        model = HookedTransformer.from_pretrained(model_name, device=device, dtype="bfloat16")
    else:
        model = HookedTransformer.from_pretrained(model_name, device=device)
    ds = offline_aware_load_dataset(dataset_path, split="train", streaming=True)
    if isinstance(ds, IterableDataset):
        ds = ds.shuffle(seed=42, buffer_size=10_000)
    else:
        ds = ds.shuffle(seed=42)

    text_key = None
    ex = next(iter(ds.take(1)))
    for cand in ["text", "content", "document", "raw", "prompt"]:
        if cand in ex:
            text_key = cand
            break
    if text_key is None:
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
# SAEBench integration helpers
# ----------------------------

def infer_saebench_model_name(hf_model_name: str) -> str:
    """
    SAEBench sometimes wants a short model id (e.g. 'pythia-70m-deduped')
    even if training used 'EleutherAI/pythia-70m-deduped'. Best-effort.
    """
    # Common patterns
    if "pythia-" in hf_model_name and "/" in hf_model_name:
        return hf_model_name.split("/", 1)[1]
    if hf_model_name in {"gpt2"}:
        return "gpt2"
    # Gemma often works with HF id, but SAEBench may have a canonical name in configs.
    # Leave as-is.
    return hf_model_name


def try_infer_regex_and_block(checkpoint_path: str, hook_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Infer SAEBench selection args from checkpoint path and hook name.

    The checkpoint path doesn't need to exist - we construct the patterns
    based on the path that will be created.

    Args:
        checkpoint_path: Path to checkpoint directory (may not exist yet)
        hook_name: Hook name from preset (e.g., "blocks.8.hook_resid_post")

    Returns:
        Tuple of (sae_regex_pattern, sae_block_pattern)
    """
    ckpt_dir = Path(checkpoint_path).expanduser().resolve()
    
    # Use the checkpoint directory basename as the regex pattern
    regex = ckpt_dir.name
    
    # Construct block pattern as: <hook_name>__trainer_0
    # SAELens typically uses trainer_0 for single-trainer runs
    block = f"{hook_name}__trainer_0"
    
    return regex, block


def log_saebench_results_to_wandb(output_dir: str, sae_name: str) -> None:
    """
    Parse SAEBench JSON result files and log key metrics to wandb.
    
    For sparse probing, computes:
    - Macro-F1 (mean across classes) per dataset
    - Median Wasserstein (across classes) per dataset with IQR
    - Per-k breakdowns for k in [1, 2, 5, 10, 20, 50, 100]
    
    Args:
        output_dir: Directory containing SAEBench results
        sae_name: Name of the SAE (used for finding result files)
    """
    import numpy as np

    print("\n[wandb] Logging SAEBench results...")
    print(f"[wandb] Looking for results in: {output_dir}")
    print(f"[wandb] Filtering for SAE: {sae_name}")
    
    # Find result files - look for files matching this specific SAE
    # This prevents parallel runs from accidentally logging each other's results
    result_files = glob(f"{output_dir}/**/*_eval_results.json", recursive=True)
    
    # Filter to only include files for this SAE
    filtered_files = [f for f in result_files if sae_name in f]
    
    print(f"[wandb] Found {len(result_files)} total result files, {len(filtered_files)} matching {sae_name}")
    
    if not filtered_files:
        # Fall back to all files if no match (for backwards compatibility)
        print(f"[wandb] Warning: No files matching '{sae_name}', using all found files")
        filtered_files = result_files
    
    if not filtered_files:
        print("[wandb] No result files found to log")
        return
    
    result_files = filtered_files

    
    logged_metrics = {}
    
    def clean_dataset_name(name: str, aggregate_bios: bool = True) -> str:
        """Shorten dataset names for cleaner metric paths.
        
        If aggregate_bios=True, all bias_in_bios class sets are normalized 
        to just 'bios' so they can be averaged together.
        """
        name = name.replace("_results", "")
        if aggregate_bios:
            # Normalize all bios class sets to just "bias_in_bios" for aggregation
            name = name.replace("LabHC/bias_in_bios_class_set1", "bias_in_bios")
            name = name.replace("LabHC/bias_in_bios_class_set2", "bias_in_bios")
            name = name.replace("LabHC/bias_in_bios_class_set3", "bias_in_bios")
        name = name.replace("LabHC/bias_in_bios_", "bios_")
        name = name.replace("canrager/amazon_reviews_mcauley_", "amazon_")
        name = name.replace("codeparrot/github-code", "github_code")
        name = name.replace("fancyzhx/ag_news", "ag_news")
        name = name.replace("Helsinki-NLP/europarl", "europarl")
        return name
    
    def is_bios_class_set(name: str) -> bool:
        """Check if dataset is one of the bias_in_bios class sets."""
        return "bias_in_bios_class_set" in name or name.startswith("bios_class_set")
    
    for result_file in result_files:
        print(f"[wandb] Processing: {result_file}")
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            eval_type = data.get("eval_type_id", "unknown")
            metrics = data.get("eval_result_metrics", {})
            
            # Flatten nested metrics with prefix (existing behavior)
            for category, category_metrics in metrics.items():
                if isinstance(category_metrics, dict):
                    for metric_name, value in category_metrics.items():
                        # Skip infinity and NaN values
                        if isinstance(value, float) and (value != value or abs(value) == float('inf')):
                            continue
                        if value is None:
                            continue
                        key = f"{eval_type}/{category}/{metric_name}"
                        logged_metrics[key] = value
                elif isinstance(category_metrics, (int, float)):
                    # Skip infinity and NaN values
                    if isinstance(category_metrics, float) and (category_metrics != category_metrics or abs(category_metrics) == float('inf')):
                        continue
                    key = f"{eval_type}/{category}"
                    logged_metrics[key] = category_metrics
            
            # Enhanced sparse_probing logging with per-class aggregation
            if eval_type == "sparse_probing":
                # Use a defaultdict to collect values for averaging
                # This is needed because bios_class_set1/2/3 all map to "bios"
                from collections import defaultdict
                collected_metrics: Dict[str, List[float]] = defaultdict(list)
                
                # Log per-dataset mean metrics from eval_result_details (existing)
                details = data.get("eval_result_details", [])
                for dataset_result in details:
                    raw_name = dataset_result.get("dataset_name", "unknown")
                    dataset_name = clean_dataset_name(raw_name)
                    
                    for metric_name, value in dataset_result.items():
                        if metric_name == "dataset_name":
                            continue
                        if value is None:
                            continue
                        if isinstance(value, float) and (value != value or abs(value) == float('inf')):
                            continue
                        key = f"{eval_type}/datasets/{dataset_name}/{metric_name}"
                        collected_metrics[key].append(value)
                
                # NEW: Extract per-class data from eval_result_unstructured
                # and compute proper aggregations (macro-F1, median Wasserstein)
                unstructured = data.get("eval_result_unstructured", {})
                
                # Track metrics across all datasets for overall summary
                all_dataset_macro_f1: Dict[str, List[float]] = {}
                all_dataset_median_wasserstein: Dict[str, List[float]] = {}
                
                for dataset_key, dataset_metrics in unstructured.items():
                    if not isinstance(dataset_metrics, dict):
                        continue
                    
                    dataset_name = clean_dataset_name(dataset_key)
                    
                    # Process each k value for SAE metrics
                    k_values = [1, 2, 5, 10, 20, 50, 100]
                    
                    for k in k_values:
                        # F1 metrics - compute macro-F1 (mean across classes)
                        f1_key = f"sae_top_{k}_test_f1"
                        if f1_key in dataset_metrics and isinstance(dataset_metrics[f1_key], dict):
                            per_class_f1 = list(dataset_metrics[f1_key].values())
                            per_class_f1 = [v for v in per_class_f1 if isinstance(v, (int, float)) and v == v]
                            
                            if per_class_f1:
                                macro_f1 = float(np.mean(per_class_f1))
                                collected_metrics[f"{eval_type}/datasets/{dataset_name}/k_{k}/macro_f1"].append(macro_f1)
                                
                                # Track for overall summary
                                if f"sae_top_{k}_macro_f1" not in all_dataset_macro_f1:
                                    all_dataset_macro_f1[f"sae_top_{k}_macro_f1"] = []
                                all_dataset_macro_f1[f"sae_top_{k}_macro_f1"].append(macro_f1)
                        
                        # Wasserstein metrics - compute median across classes
                        wasserstein_key = f"sae_top_{k}_wasserstein"
                        if wasserstein_key in dataset_metrics and isinstance(dataset_metrics[wasserstein_key], dict):
                            per_class_wasserstein = list(dataset_metrics[wasserstein_key].values())
                            per_class_wasserstein = [v for v in per_class_wasserstein if isinstance(v, (int, float)) and v == v]
                            
                            if per_class_wasserstein:
                                median_wasserstein = float(np.median(per_class_wasserstein))
                                q75, q25 = np.percentile(per_class_wasserstein, [75, 25])
                                iqr = float(q75 - q25)
                                
                                collected_metrics[f"{eval_type}/datasets/{dataset_name}/k_{k}/median_wasserstein"].append(median_wasserstein)
                                collected_metrics[f"{eval_type}/datasets/{dataset_name}/k_{k}/wasserstein_iqr"].append(iqr)
                                
                                # Track for overall summary
                                if f"sae_top_{k}_median_wasserstein" not in all_dataset_median_wasserstein:
                                    all_dataset_median_wasserstein[f"sae_top_{k}_median_wasserstein"] = []
                                all_dataset_median_wasserstein[f"sae_top_{k}_median_wasserstein"].append(median_wasserstein)
                        
                        # Accuracy metrics - compute mean across classes
                        acc_key = f"sae_top_{k}_test_accuracy"
                        if acc_key in dataset_metrics and isinstance(dataset_metrics[acc_key], dict):
                            per_class_acc = list(dataset_metrics[acc_key].values())
                            per_class_acc = [v for v in per_class_acc if isinstance(v, (int, float)) and v == v]
                            
                            if per_class_acc:
                                mean_acc = float(np.mean(per_class_acc))
                                collected_metrics[f"{eval_type}/datasets/{dataset_name}/k_{k}/mean_accuracy"].append(mean_acc)
                    
                    # Also process full SAE metrics (not top-k)
                    for metric_type, agg_method, metric_suffix in [
                        ("sae_test_f1", "mean", "macro_f1"),
                        ("sae_wasserstein", "median", "median_wasserstein"),
                        ("sae_test_accuracy", "mean", "mean_accuracy"),
                    ]:
                        if metric_type in dataset_metrics and isinstance(dataset_metrics[metric_type], dict):
                            per_class_values = list(dataset_metrics[metric_type].values())
                            per_class_values = [v for v in per_class_values if isinstance(v, (int, float)) and v == v]
                            
                            if per_class_values:
                                if agg_method == "mean":
                                    agg_value = float(np.mean(per_class_values))
                                else:  # median
                                    agg_value = float(np.median(per_class_values))
                                collected_metrics[f"{eval_type}/datasets/{dataset_name}/full_sae/{metric_suffix}"].append(agg_value)
                
                # Average collected metrics (bios will have 3 values from set1/2/3, others will have 1)
                for key, values in collected_metrics.items():
                    if values:
                        logged_metrics[key] = float(np.mean(values))
                
                # Log overall summary metrics (median across datasets)
                for metric_key, values in all_dataset_macro_f1.items():
                    if values:
                        logged_metrics[f"{eval_type}/summary/{metric_key}"] = float(np.mean(values))
                
                for metric_key, values in all_dataset_median_wasserstein.items():
                    if values:
                        logged_metrics[f"{eval_type}/summary/{metric_key}"] = float(np.median(values))
            
            # For TPP and SCR, log per-dataset metrics and compute median/std
            if eval_type in ("tpp", "scr"):
                details = data.get("eval_result_details", [])
                
                # Collect values per metric for median/std computation
                metric_values: Dict[str, List[float]] = {}
                
                for dataset_result in details:
                    # Extract dataset name
                    dataset_name = dataset_result.get("dataset_name", "unknown")
                    dataset_name = dataset_name.replace("_results", "")
                    dataset_name = dataset_name.replace("_tpp", "")
                    dataset_name = dataset_name.replace("_scr", "")
                    # Shorten common prefixes
                    dataset_name = dataset_name.replace("LabHC/bias_in_bios_class_set1_", "bios_")
                    dataset_name = dataset_name.replace("canrager/amazon_reviews_mcauley_1and5_", "amazon_")
                    
                    # Log all metrics for this dataset pair
                    for metric_name, value in dataset_result.items():
                        if metric_name == "dataset_name":
                            continue
                        if value is None:
                            continue
                        if isinstance(value, float) and (value != value or abs(value) == float('inf')):
                            continue
                        
                        # Log per-dataset value
                        key = f"{eval_type}/datasets/{dataset_name}/{metric_name}"
                        logged_metrics[key] = value
                        
                        # Collect for median/std
                        if metric_name not in metric_values:
                            metric_values[metric_name] = []
                        metric_values[metric_name].append(value)
                
                # Compute and log median and std for each metric
                for metric_name, values in metric_values.items():
                    if len(values) > 1:
                        median_val = float(np.median(values))
                        std_val = float(np.std(values))
                        logged_metrics[f"{eval_type}/median/{metric_name}"] = median_val
                        logged_metrics[f"{eval_type}/std/{metric_name}"] = std_val
            
            print(f"  ✓ Parsed {eval_type} results")
            
        except Exception as e:
            print(f"  ⚠ Failed to parse {result_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Log all metrics to wandb
    if logged_metrics:
        print(f"[wandb] Logging {len(logged_metrics)} metrics...")
        # Show some example metrics from different categories
        example_keys = [k for k in logged_metrics.keys() if "macro_f1" in k or "median_wasserstein" in k][:3]
        for key in example_keys:
            print(f"  {key}: {logged_metrics[key]:.4f}")
        other_keys = [k for k in logged_metrics.keys() if k not in example_keys][:2]
        for key in other_keys:
            print(f"  {key}: {logged_metrics[key]}")
        if len(logged_metrics) > 5:
            print(f"  ... and {len(logged_metrics) - 5} more")
        try:
            wandb.log(logged_metrics)
            print(f"[wandb] ✓ Logged {len(logged_metrics)} SAEBench metrics to wandb")
        except wandb.errors.Error:
            print(f"[wandb] No active run — skipping wandb logging ({len(logged_metrics)} metrics extracted)")
    else:
        print("[wandb] No metrics extracted from result files")


def run_saebench_evals_direct(
    sae,  # Accept SAE object directly
    sae_name: str,
    model_name: str,
    device: str,
    skip_evals: List[str],
    output_dir: str = "eval_results",
    dataset: str = "Skylion007/openwebtext",
    sae_batch_size: Optional[int] = None,
    llm_batch_size: Optional[int] = None,
    lower_vram: bool = False,
    absorption_sae_name: Optional[str] = None,
) -> List[str]:
    """
    Run SAEBench evaluations directly using Python API (not CLI).

    This takes a trained SAE object and passes it to SAEBench evaluation functions.

    Returns the names of requested evals that produced no results on disk
    (empty list == everything requested succeeded).

    Args:
        sae: Trained SAE object from runner
        sae_name: Name for the SAE (used in results)
        model_name: Model name for evaluation
        device: Device to run on
        skip_evals: List of eval types to skip
        output_dir: Directory to save eval results
        dataset: Dataset to use for core reconstruction eval (matches training)
        sae_batch_size: SAE encoding batch size (auto-reduced for large models)
        llm_batch_size: LLM forward pass batch size (auto-reduced for large models)
        lower_vram: Move LLM to CPU during SAE encoding (auto-enabled for large models)
    """
    # Auto-detect large models and enable lower_vram mode
    # Models like pythia-1.4b have large LLMs that consume GPU memory during activation collection
    # With lower_vram=True, LLM is moved to CPU during SAE encoding, freeing up memory
    # Batch sizes don't need reduction since SAE output tensors [batch, seq, d_sae] are same size regardless of d_model
    LARGE_MODEL_INDICATORS = ["1.4b", "1b", "2.8b", "2b", "3b", "6.9b", "7b", "12b", "13b", "llama"]
    is_large_model = any(indicator in model_name.lower() for indicator in LARGE_MODEL_INDICATORS)
    
    if is_large_model and not lower_vram:
        lower_vram = True
        print(f"[memory] Auto-enabled lower_vram mode for large model {model_name}")
    
    # Apply defaults
    if sae_batch_size is None:
        sae_batch_size = 125
    if llm_batch_size is None:
        llm_batch_size = 16
    if not SAEBENCH_AVAILABLE:
        print("\n[WARNING] SAEBench not available. Install it with:")
        print("  cd SAEBench && pip install -e .")
        return ["<SAEBench not installed>"]

    print("\n" + "="*70)
    print("Running SAEBench evaluations via direct API...")
    print(f"Using trained SAE: {sae_name}")
    print("="*70)
    
    # Use training SAE directly for evaluation
    # This ensures consistency between training and evaluation behavior
    # for all architectures (TopK, BatchTopK, Matryoshka and their Poly variants)
    training_arch = sae.cfg.architecture() if hasattr(sae.cfg, 'architecture') else 'unknown'
    print(f"[SAE] Using {training_arch} training SAE directly for evaluation")
    
    # Ensure SAE is on the correct device
    sae = sae.to(device)
    
    # Try to fold decoder norms (standard SAEBench approach for pretrained SAEs)
    # This doesn't change the SAE's behavior, just reorganizes the representation
    try:
        sae.fold_W_dec_norm()
        print(f"✓ Decoder norms folded into encoder")
    except (NotImplementedError, AttributeError):
        print(f"⚠ Warning: SAE architecture doesn't support fold_W_dec_norm()")
        print(f"  Some evaluations (e.g., absorption) may fail due to unnormalized decoders")
        print(f"  This is expected for TopK and similar architectures")
    
    selected_saes = [(sae_name, sae)]
    absorption_selected_saes = [(absorption_sae_name or sae_name, sae)]
    print(f"✓ SAE ready for evaluation")
    
    # Determine dtype
    dtype = "float32"  # default
    if hasattr(sae.cfg, "dtype"):
        dtype = sae.cfg.dtype
    
    # Eval configurations
    eval_configs = {
        "l0_loss": {
            "name": "Core (L0/Loss Recovered)",
            "skip_key": "l0_loss",
            "result_subdir": "core",
            "runner": lambda: saebench_core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset=dataset,
                context_size=128,
                output_folder=f"{output_dir}/core",
                verbose=True,
                dtype=dtype,
                device=device,
            ),
        },
        "sparse_probing": {
            "name": "Sparse Probing",
            "skip_key": "sparse_probing",
            "result_subdir": "sparse_probing",
            "runner": lambda: saebench_sparse_probing.run_eval(
                saebench_sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=42,
                    llm_dtype=dtype,
                    llm_batch_size=llm_batch_size,
                    sae_batch_size=sae_batch_size,
                    lower_vram_usage=lower_vram,
                ),
                selected_saes,
                device,
                f"{output_dir}/sparse_probing",
                force_rerun=False,
                clean_up_activations=True,
                save_activations=False,
            ),
        },
        "tpp": {
            "name": "Targeted Probe Perturbation (TPP)",
            "skip_key": "tpp",
            # scr_and_tpp/main.py appends the eval_type to the output path
            "result_subdir": "tpp",
            "runner": lambda: saebench_scr_tpp.run_eval(
                saebench_scr_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=42,
                    perform_scr=False,
                    llm_dtype=dtype,
                    llm_batch_size=llm_batch_size,
                    sae_batch_size=sae_batch_size,
                    lower_vram_usage=lower_vram,  # Move model to CPU when not needed
                ),
                selected_saes,
                device,
                output_dir,
                force_rerun=False,
                clean_up_activations=True,
                save_activations=False,
            ),
        },
        "scr": {
            "name": "Spurious Correlation Removal (SCR)",
            "skip_key": "scr",
            "result_subdir": "scr",
            "runner": lambda: saebench_scr_tpp.run_eval(
                saebench_scr_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=42,
                    perform_scr=True,
                    llm_dtype=dtype,
                    llm_batch_size=llm_batch_size,
                    sae_batch_size=sae_batch_size,
                    lower_vram_usage=lower_vram,  # Move model to CPU when not needed
                ),
                selected_saes,
                device,
                output_dir,
                force_rerun=False,
                clean_up_activations=True,
                save_activations=False,
            ),
        },
        "ravel": {
            "name": "RAVEL",
            "skip_key": "ravel",
            "result_subdir": "ravel",
            "required_models": ["gemma-2-2b", "pythia-160m-deduped"],  # Only these models supported
            "runner": lambda: saebench_ravel.run_eval(
                saebench_ravel.RAVELEvalConfig(
                    # RAVEL expects the short name and maps it to the full HF name internally
                    model_name=next(r for r in ["gemma-2-2b", "pythia-160m-deduped"] if r in model_name),
                    random_seed=42,
                    llm_dtype=dtype,
                    llm_batch_size=llm_batch_size,
                ),
                selected_saes,
                device,
                f"{output_dir}/ravel",
                force_rerun=False,
            ),
        },
        "absorption": {
            "name": "Feature Absorption",
            "skip_key": "absorption",
            "result_subdir": "absorption",
            "runner": lambda: saebench_absorption.run_eval(
                saebench_absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=42,
                    llm_dtype=dtype,
                ),
                absorption_selected_saes,
                device,
                f"{output_dir}/absorption",
                force_rerun=False,
            ),
        },
    }
    
    # Detect crosscoder SAEs and auto-skip incompatible evals
    is_crosscoder = getattr(getattr(sae.cfg, 'metadata', None), 'hook_names', None) is not None
    if not is_crosscoder:
        is_crosscoder = getattr(sae.cfg, 'hook_names', None) is not None
    # Core eval (skip_key "l0_loss") has a crosscoder-aware branch in
    # SAEBench/sae_bench/evals/core/main.py — it skips KL/CE (model-replay)
    # metrics but still reports per-layer and layer-averaged reconstruction,
    # shrinkage and sparsity metrics. No need to blanket-skip it.
    CROSSCODER_INCOMPATIBLE_EVALS = {"tpp", "scr", "absorption", "ravel"}

    # Run evaluations
    skip_set = set(skip_evals)
    # Evals that ran but produced no result file on disk. SAEBench swallows
    # per-SAE exceptions internally (evals/core/main.py wraps each SAE in a
    # broad `except Exception: continue`), so a runner returning normally is
    # NOT evidence that anything was written — the job would otherwise exit 0
    # with an empty results directory. Verify, and fail the job if not.
    failed_evals: list[str] = []
    for eval_key, eval_info in eval_configs.items():
        if eval_info["skip_key"] in skip_set:
            print(f"\n[SKIP] {eval_info['name']}")
            continue

        # Auto-skip evals that are incompatible with crosscoders
        if is_crosscoder and eval_info["skip_key"] in CROSSCODER_INCOMPATIBLE_EVALS:
            print(f"\n[SKIP] {eval_info['name']} - not supported for crosscoder SAEs")
            print(f"       (requires direct W_dec access incompatible with multi-layer/factorized decoders)")
            continue

        # Check model compatibility
        required_models = eval_info.get("required_models")
        if required_models and not any(r in model_name for r in required_models):
            print(f"\n[SKIP] {eval_info['name']} - not supported for model {model_name}")
            print(f"       Supported models: {required_models}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Running: {eval_info['name']}")
        print(f"{'='*70}")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Aggressively free GPU memory before each eval
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            eval_result = eval_info["runner"]()

            # The result file on disk is the authoritative success signal, NOT
            # the runner returning normally. Note that an empty return list is
            # ambiguous on its own: multiple_evals() also returns [] when it
            # skips SAEs whose results already exist (force_rerun=False), so it
            # is only reported as extra context when the file is genuinely absent.
            result_dir = Path(output_dir) / eval_info["result_subdir"]
            result_files = sorted(result_dir.glob("*.json")) if result_dir.is_dir() else []
            if not result_files:
                detail = ""
                if isinstance(eval_result, list) and len(eval_result) == 0:
                    detail = (" and returned no results — SAEBench caught an "
                              "internal exception for every SAE (see stderr traceback)")
                raise RuntimeError(
                    f"eval wrote no result file to {result_dir}{detail}"
                )

            print(f"✓ Completed: {eval_info['name']} "
                  f"({len(result_files)} file(s) in {result_dir})")
        except Exception as e:
            failed_evals.append(eval_info["name"])
            print(f"\n[ERROR] {eval_info['name']} failed:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with other evals

    print("\n" + "="*70)
    if failed_evals:
        print(f"SAEBench evaluations FINISHED WITH {len(failed_evals)} FAILURE(S):")
        for name in failed_evals:
            print(f"  ✗ {name}")
    else:
        print("SAEBench evaluations completed!")
    print(f"Results saved to: {output_dir}")
    print("="*70)

    # Log results to wandb (do this first so partial results are still preserved)
    log_saebench_results_to_wandb(output_dir, sae_name)

    # Returned rather than raised so the caller can still finalize wandb before
    # exiting non-zero.
    return failed_evals


def run_all_requested_saebench_evals(
    args: argparse.Namespace, 
    preset: Preset, 
    cfg_checkpoint_path: str,
    sae,  # Accept the trained SAE object
) -> List[str]:
    """
    Runs the requested SAEBench eval set via direct API calls.

    Returns the names of requested evals that produced no results on disk.
    """
    # Infer model name for SAEBench
    saebench_model_name = infer_saebench_model_name(preset.model_name)
    
    # Determine device (same as training)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get list of evals to skip
    skip_evals = args.skip_eval or []
    
    # Extract SAE name from checkpoint path
    sae_name = Path(cfg_checkpoint_path).name
    
    # Use unique output directory per SAE to prevent parallel run collisions
    # This is critical when running multiple experiments in parallel on SLURM
    output_dir = f"eval_results/{sae_name}"
    
    # Run evaluations directly
    return run_saebench_evals_direct(
        sae=sae,
        sae_name=sae_name,
        model_name=saebench_model_name,
        device=device,
        skip_evals=skip_evals,
        output_dir=output_dir,
        dataset=preset.dataset_path,
        sae_batch_size=args.saebench_sae_batch_size,
        llm_batch_size=args.saebench_llm_batch_size,
        lower_vram=args.saebench_lower_vram,
        absorption_sae_name=args.sae_name,
    )


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = parse_args()
    preset = PRESETS[args.exp]

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # Authenticate with HuggingFace if HF_TOKEN is set (required for gated models like LLaMA)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        print("[auth] Authenticating with HuggingFace via HF_TOKEN...")
        login(token=hf_token)
    elif "llama" in preset.model_name.lower() or "gemma" in preset.model_name.lower():
        print("[warn] HF_TOKEN not set. Gated models (LLaMA, Gemma) require authentication.")
        print("[warn] Set: export HF_TOKEN=your_token_here")

    width = norm_width(args.width) if args.width else norm_width(preset.default_width)
    if width not in [norm_width(w) for w in preset.allowed_widths]:
        raise SystemExit(f"--width {width} not allowed for {args.exp}. Allowed: {preset.allowed_widths}")
    d_sae = WIDTH_TO_DSAE[width]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Build hook name(s) — crosscoder uses multiple layers
    cross_layers = parse_int_list(args.cross_layers)
    if args.use_crosscoder:
        if cross_layers is None:
            raise SystemExit("--use_crosscoder requires --cross_layers (e.g. --cross_layers 3,5,8)")
        hook_base = re.sub(r"blocks\.\d+", "blocks.{}", preset.hook_name)
        hook_names = [hook_base.format(layer) for layer in cross_layers]
    else:
        hook_names = [preset.hook_name]

    # Eval-only mode: skip training, load SAE from checkpoint, run evals, exit.
    if args.eval_only:
        if not args.run_saebench:
            raise SystemExit("--eval_only requires --run_saebench")
        if not args.checkpoint_path:
            raise SystemExit("--eval_only requires --checkpoint_path pointing to the SAE checkpoint")
        ckpt = Path(args.checkpoint_path)
        if not ckpt.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt}")

        # In training mode --checkpoint_path is the *run* directory and SAEs are
        # written to step subdirs (final_<tokens>/, <step>/). load_from_disk
        # needs the directory that actually holds cfg.json, so resolve down to
        # the final (or latest) step when the run directory itself isn't an SAE.
        sae_dir = ckpt
        if not (sae_dir / "cfg.json").exists():
            steps = [d for d in ckpt.iterdir() if d.is_dir() and (d / "cfg.json").exists()]
            if not steps:
                raise SystemExit(
                    f"No SAE checkpoint (cfg.json) found in {ckpt} or its subdirectories"
                )
            finals = [d for d in steps if d.name.startswith("final_")]
            if finals:
                sae_dir = max(finals, key=lambda d: int(d.name.split("_")[-1]))
            else:
                sae_dir = max(steps, key=lambda d: int(d.name) if d.name.isdigit() else -1)
            print(f"[eval_only] Resolved run dir {ckpt.name} -> checkpoint {sae_dir.name}")

        # Name results after the run, not the step dir — otherwise every model
        # would collide in eval_results/final_500002816/.
        run_dir = ckpt
        if sae_dir == ckpt and (ckpt.name.startswith("final_") or ckpt.name.isdigit()):
            run_dir = ckpt.parent

        print(f"[eval_only] Loading training SAE from {sae_dir}")
        sae_obj = TrainingSAE.load_from_disk(sae_dir, device=device)
        print(f"[eval_only] Loaded {type(sae_obj).__name__} (architecture: {sae_obj.cfg.architecture()})")
        print(f"[eval_only] Results will be written to eval_results/{run_dir.name}")
        failed = run_all_requested_saebench_evals(args, preset, str(run_dir), sae_obj)
        if failed:
            print(f"[FATAL] {len(failed)} requested SAEBench eval(s) produced no "
                  f"results: {', '.join(failed)}", file=sys.stderr)
        os._exit(1 if failed else 0)

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
        poly_ranks=parse_poly_ranks(args.poly_ranks),
        poly_order=args.poly_order,
        shared_u=args.shared_u,
        lambda2_init=args.lambda2_init,
        lambda3_init=args.lambda3_init,
        rescale_acts_by_decoder_norm=not args.no_rescale_by_decoder_norm,
        use_crosscoder=args.use_crosscoder,
        cross_layers=parse_int_list(args.cross_layers),
        layer_masking=args.layer_masking,
        use_cross_fact=args.use_cross_fact,
        cross_fact_type=args.cross_fact_type,
        cross_fact_ranks=tuple(parse_int_list(args.cross_fact_ranks) or [32]),
    )

    logger_cfg = LoggingConfig(
        log_to_wandb=(not args.no_wandb),
        wandb_project=args.wandb_project,
        run_name=args.wandb_run_name,
        # Keep wandb run open if we're going to run SAEBench after training
        skip_wandb_finish=args.run_saebench,
    )

    cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=preset.model_name,
        hook_name=preset.hook_name,
        hook_names=hook_names if args.use_crosscoder else None,
        dataset_path=preset.dataset_path,
        logger=logger_cfg,
        device=device,
        seed=args.seed,
        dataset_trust_remote_code=True,
        n_batches_in_buffer=128,
        eval_batch_size_prompts=args.eval_batch_size_prompts,
        eval_compute_ce_loss=args.eval_ce_loss,
        eval_compute_kl=args.eval_kl,
    )

    if args.lr is not None:
        cfg.lr = args.lr
    if args.train_batch_size_tokens is not None:
        cfg.train_batch_size_tokens = args.train_batch_size_tokens
    if args.store_batch_size_prompts is not None:
        cfg.store_batch_size_prompts = args.store_batch_size_prompts
    if args.context_size is not None:
        cfg.context_size = args.context_size
    if args.training_tokens is not None:
        cfg.training_tokens = args.training_tokens
    if args.n_batches_in_buffer is not None:
        cfg.n_batches_in_buffer = args.n_batches_in_buffer
    if args.act_store_device is not None:
        cfg.act_store_device = args.act_store_device
    if args.dtype is not None:
        cfg.dtype = args.dtype
    if args.model_dtype is not None:
        # Preserve sae_lens' default kwargs (center_writing_weights=False) and add the weight dtype.
        cfg.model_from_pretrained_kwargs = {
            **(cfg.model_from_pretrained_kwargs or {}),
            "dtype": args.model_dtype,
        }
    if args.n_devices is not None and args.n_devices > 1:
        # Shard the LLM across n_devices GPUs. load_model forces device="cuda" when
        # n_devices>1, so TransformerLens spreads the blocks over cuda:0..cuda:{n-1}.
        cfg.model_from_pretrained_kwargs = {
            **(cfg.model_from_pretrained_kwargs or {}),
            "n_devices": args.n_devices,
        }


    
    # Set checkpoint defaults - always save periodic checkpoints unless explicitly disabled
    if args.n_checkpoints is not None:
        cfg.n_checkpoints = args.n_checkpoints
    else:
        cfg.n_checkpoints = 5  # Default: 5 periodic checkpoints during training
    cfg.save_final_checkpoint = True  # Always save final checkpoint

    if args.checkpoint_path is not None:
        cfg.checkpoint_path = args.checkpoint_path
    elif args.wandb_run_name is not None:
        # Use wandb_run_name as checkpoint path for consistency
        cfg.checkpoint_path = f"checkpoints/{args.wandb_run_name}"
    else:
        poly_suffix = "_poly" if args.poly else ""
        cross_suffix = f"_cross{'-'.join(str(l) for l in cross_layers)}" if args.use_crosscoder else ""
        fact_suffix = (
            f"_fact{args.cross_fact_type.lower()}{'-'.join(str(r) for r in (parse_int_list(args.cross_fact_ranks) or [32]))}"
            if args.use_crosscoder and args.use_cross_fact else ""
        )
        cfg.checkpoint_path = f"checkpoints_{args.exp}_{args.architecture}{poly_suffix}{cross_suffix}{fact_suffix}_{width}"

    # Handle resume from checkpoint
    if args.resume_from_checkpoint:
        if args.wandb_run_name is None:
            raise SystemExit("--resume_from_checkpoint requires --wandb_run_name to locate the checkpoint directory")
        
        checkpoint_base = Path(cfg.checkpoint_path)
        if not checkpoint_base.exists():
            raise SystemExit(f"Cannot resume: checkpoint directory does not exist: {checkpoint_base}")
        
        # Find the latest checkpoint subdirectory (they are named like "final_123456" or step numbers)
        checkpoint_subdirs = [d for d in checkpoint_base.iterdir() if d.is_dir() and (d / "sae_weights.safetensors").exists()]
        if not checkpoint_subdirs:
            raise SystemExit(f"Cannot resume: no valid checkpoints found in {checkpoint_base}")
        
        # Sort by modification time and get the latest
        latest_checkpoint = max(checkpoint_subdirs, key=lambda d: d.stat().st_mtime)
        print(f"[resume] Found checkpoint: {latest_checkpoint}")
        
        # Extract tokens completed from checkpoint name (e.g., "final_500002816" or "123456")
        checkpoint_name = latest_checkpoint.name
        try:
            if checkpoint_name.startswith("final_"):
                tokens_completed = int(checkpoint_name.split("_")[1])
            else:
                tokens_completed = int(checkpoint_name)
            
            tokens_target = cfg.training_tokens
            tokens_remaining = max(0, tokens_target - tokens_completed)
            
            print(f"[resume] Tokens completed: {tokens_completed:,} / {tokens_target:,}")
            print(f"[resume] Tokens remaining: {tokens_remaining:,}")
            
            if tokens_remaining == 0:
                print(f"[resume] ⚠ Training already complete! Use higher --training_tokens to continue training.")
        except (ValueError, IndexError):
            print(f"[resume] Could not parse step count from checkpoint name: {checkpoint_name}")
        
        # Set resume path in config
        cfg.resume_from_checkpoint = str(latest_checkpoint)
        
        # Configure WandB - the runner will handle wandb.init() after loading model
        logger_cfg = LoggingConfig(
            log_to_wandb=(not args.no_wandb),
            wandb_project=args.wandb_project,
            run_name=args.wandb_run_name,
            skip_wandb_finish=args.run_saebench,
        )
        cfg.logger = logger_cfg
        print(f"[resume] WandB will resume run: {args.wandb_run_name}")

    print("=== SAELens run ===")
    print(f"exp:                 {args.exp}")
    arch_str = args.architecture + (" + poly" if args.poly else "")
    print(f"architecture:        {arch_str}")
    print(f"model:               {preset.model_name}")
    print(f"hook:                {hook_names if args.use_crosscoder else preset.hook_name}")
    print(f"dataset:             {preset.dataset_path}")
    print(f"d_in / d_sae:        {preset.d_in} / {d_sae} (width={width})")
    print(f"use_saelens_defaults:{args.use_saelens_defaults}")
    if not args.use_saelens_defaults:
        if args.architecture == "standard":
            print(f"l1_coefficient:      {args.l1_coefficient}")
        if args.architecture in {"topk", "batchtopk", "matryoshka"}:
            print(f"k:                   {args.k}")
        if args.architecture == "jumprelu":
            print(f"l0_coefficient:      {args.l0_coefficient}")
    if args.poly:
        print(f"poly_rank:           {args.poly_rank or preset.d_in} (default: d_in)")
        print(f"lambda2_init:        {args.lambda2_init}")
        print(f"lambda3_init:        {args.lambda3_init}")
    if args.use_crosscoder:
        print(f"layer_masking:       {args.layer_masking}")
        if args.use_cross_fact:
            print(f"cross_fact:          {args.cross_fact_type} (ranks={args.cross_fact_ranks})")
    print(f"device:              {device}")
    print(f"checkpoint_path:     {cfg.checkpoint_path}")
    if args.resume_from_checkpoint:
        print(f"resume_from:         {cfg.resume_from_checkpoint}")
    print("===================")

    # Set up local mirror of wandb.log calls. Works regardless of --no_wandb.
    run_name_for_local = (
        args.wandb_run_name
        or Path(cfg.checkpoint_path).name
    )
    setup_local_logging(
        run_name_for_local,
        config=cfg.to_dict() if hasattr(cfg, "to_dict") else None,
        wandb_enabled=not args.no_wandb,
    )
    # Force log_to_wandb=True so all gated wandb.log call sites fire; the
    # patched wandb.log mirrors to the local file and only forwards to wandb
    # if a run is actually active.
    cfg.logger.log_to_wandb = True

    runner = LanguageModelSAETrainingRunner(cfg)

    log_to_wandb = getattr(cfg.logger, "log_to_wandb", False)
    
    # Run training
    # If run_saebench is True, skip_wandb_finish is True, so wandb.run stays open
    driver_res = runner.run()

    # Log decoder parameters
    try:
        sae_obj = getattr(runner, "sae", None) or getattr(getattr(runner, "trainer", None), "sae", None)
        if sae_obj:
            n_params = count_decoder_params(sae_obj)
            print(f"Decoder Trainable Parameters: {n_params/1e6:.2f}M")
            wandb.log({"sae/decoder_params": n_params, "sae/decoder_params_M": n_params/1e6}, step=0)
    except Exception as e:
        print(f"[warn] Failed to count/log decoder params: {e}")

    # Optional sparsity report
    if args.report_sparsity:
        sae_obj = getattr(runner, "sae", None)
        if sae_obj is None:
            sae_obj = getattr(getattr(runner, "trainer", None), "sae", None)
        if sae_obj is None:
            print("[warn] Could not find trained SAE object on runner; skipping sparsity estimate.")
        else:
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

    # SAEBench evals
    saebench_failures: List[str] = []
    if args.run_saebench:
        # Get the trained SAE object from the runner
        sae_obj = getattr(runner, "sae", None)
        if sae_obj is None:
            sae_obj = getattr(getattr(runner, "trainer", None), "sae", None)
        
        if sae_obj is None:
            print("[warn] Could not find trained SAE object on runner; skipping SAEBench.")
        else:
            # wandb.run is still active (skip_wandb_finish=True was set)
            if log_to_wandb and wandb.run is not None:
                print(f"[wandb] Using active training run: {wandb.run.id}")
            
            # Free as much training memory as possible before SAEBench
            print("[memory] Cleaning up training memory before SAEBench...")
            del runner
            del driver_res
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[memory] GPU allocated after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            
            # Run SAEBench evaluations
            saebench_failures = run_all_requested_saebench_evals(
                args, preset, cfg.checkpoint_path, sae_obj
            )

            # Now finish wandb after SAEBench completes
            if wandb.run is not None:
                wandb.finish()

    if saebench_failures:
        print(f"[FATAL] {len(saebench_failures)} requested SAEBench eval(s) produced "
              f"no results: {', '.join(saebench_failures)}", file=sys.stderr)

    # Force exit to avoid Python finalization crash (threading/GIL issue with scipy/sklearn)
    os._exit(1 if saebench_failures else 0)


if __name__ == "__main__":
    main()
