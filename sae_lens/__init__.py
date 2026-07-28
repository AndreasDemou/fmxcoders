# ruff: noqa: E402
__version__ = "6.22.3"

import logging

logger = logging.getLogger(__name__)

from sae_lens.saes import (
    SAE,
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
    GatedSAE,
    GatedSAEConfig,
    GatedTrainingSAE,
    GatedTrainingSAEConfig,
    JumpReLUSAE,
    JumpReLUSAEConfig,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
    JumpReLUTranscoder,
    JumpReLUTranscoderConfig,
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
    # Crosscoder variants
    CrossBatchTopKTrainingSAE,
    CrossBatchTopKTrainingSAEConfig,
    CrossJumpReLUSAE,
    CrossJumpReLUSAEConfig,
    CrossTopKSAE,
    CrossTopKSAEConfig,
    CrossTopKTrainingSAE,
    CrossTopKTrainingSAEConfig,
    CrossMatryoshkaBatchTopKTrainingSAE,
    CrossMatryoshkaBatchTopKTrainingSAEConfig,
    # Poly variants
    PolyTopKSAE,
    PolyTopKSAEConfig,
    PolyTopKTrainingSAE,
    PolyTopKTrainingSAEConfig,
    PolyBatchTopKTrainingSAE,
    PolyBatchTopKTrainingSAEConfig,
    PolyJumpReLUSAE,
    PolyJumpReLUSAEConfig,
    PolyJumpReLUTrainingSAE,
    PolyJumpReLUTrainingSAEConfig,
    PolyMatryoshkaBatchTopKTrainingSAE,
    PolyMatryoshkaBatchTopKTrainingSAEConfig,
    # Other
    SAEConfig,
    SkipTranscoder,
    SkipTranscoderConfig,
    StandardSAE,
    StandardSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
    TemporalSAE,
    TemporalSAEConfig,
    TopKSAE,
    TopKSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
    TrainingSAE,
    TrainingSAEConfig,
    Transcoder,
    TranscoderConfig,
)

from .analysis.hooked_sae_transformer import HookedSAETransformer
from .cache_activations_runner import CacheActivationsRunner
from .config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
    LoggingConfig,
    PretokenizeRunnerConfig,
)
from .evals import run_evals
from .llm_sae_training_runner import LanguageModelSAETrainingRunner, SAETrainingRunner
from .loading.pretrained_sae_loaders import (
    PretrainedSaeDiskLoader,
    PretrainedSaeHuggingfaceLoader,
)
from .pretokenize_runner import PretokenizeRunner, pretokenize_runner
from .registry import register_sae_class, register_sae_training_class
from .training.activations_store import ActivationsStore
from .training.upload_saes_to_huggingface import upload_saes_to_huggingface

__all__ = [
    "SAE",
    "SAEConfig",
    "TrainingSAE",
    "TrainingSAEConfig",
    "HookedSAETransformer",
    "ActivationsStore",
    "LanguageModelSAERunnerConfig",
    "LanguageModelSAETrainingRunner",
    "CacheActivationsRunnerConfig",
    "CacheActivationsRunner",
    "PretokenizeRunnerConfig",
    "PretokenizeRunner",
    "pretokenize_runner",
    "run_evals",
    "upload_saes_to_huggingface",
    "PretrainedSaeHuggingfaceLoader",
    "PretrainedSaeDiskLoader",
    "register_sae_class",
    "register_sae_training_class",
    "StandardSAE",
    "StandardSAEConfig",
    "StandardTrainingSAE",
    "StandardTrainingSAEConfig",
    "GatedSAE",
    "GatedSAEConfig",
    "GatedTrainingSAE",
    "GatedTrainingSAEConfig",
    "TopKSAE",
    "TopKSAEConfig",
    "TopKTrainingSAE",
    "TopKTrainingSAEConfig",
    "JumpReLUSAE",
    "JumpReLUSAEConfig",
    "JumpReLUTrainingSAE",
    "JumpReLUTrainingSAEConfig",
    "SAETrainingRunner",
    "LoggingConfig",
    "BatchTopKTrainingSAE",
    "BatchTopKTrainingSAEConfig",
    "Transcoder",
    "TranscoderConfig",
    "SkipTranscoder",
    "SkipTranscoderConfig",
    "JumpReLUTranscoder",
    "JumpReLUTranscoderConfig",
    "MatryoshkaBatchTopKTrainingSAE",
    "MatryoshkaBatchTopKTrainingSAEConfig",
    # Poly variants
    "PolyTopKSAE",
    "PolyTopKSAEConfig",
    "PolyTopKTrainingSAE",
    "PolyTopKTrainingSAEConfig",
    "PolyBatchTopKTrainingSAE",
    "PolyBatchTopKTrainingSAEConfig",
    "PolyJumpReLUSAE",
    "PolyJumpReLUSAEConfig",
    "PolyJumpReLUTrainingSAE",
    "PolyJumpReLUTrainingSAEConfig",
    "PolyMatryoshkaBatchTopKTrainingSAE",
    "PolyMatryoshkaBatchTopKTrainingSAEConfig",
    # Crosscoder variants
    "CrossBatchTopKTrainingSAE",
    "CrossBatchTopKTrainingSAEConfig",
    "CrossJumpReLUSAE",
    "CrossJumpReLUSAEConfig",
    "CrossTopKSAE",
    "CrossTopKSAEConfig",
    "CrossTopKTrainingSAE",
    "CrossTopKTrainingSAEConfig",
    "CrossMatryoshkaBatchTopKTrainingSAE",
    "CrossMatryoshkaBatchTopKTrainingSAEConfig",
    # Other
    "TemporalSAE",
    "TemporalSAEConfig",
]


register_sae_class("standard", StandardSAE, StandardSAEConfig)
register_sae_training_class("standard", StandardTrainingSAE, StandardTrainingSAEConfig)
register_sae_class("gated", GatedSAE, GatedSAEConfig)
register_sae_training_class("gated", GatedTrainingSAE, GatedTrainingSAEConfig)
register_sae_class("topk", TopKSAE, TopKSAEConfig)
register_sae_training_class("topk", TopKTrainingSAE, TopKTrainingSAEConfig)
register_sae_class("jumprelu", JumpReLUSAE, JumpReLUSAEConfig)
register_sae_training_class("jumprelu", JumpReLUTrainingSAE, JumpReLUTrainingSAEConfig)
register_sae_training_class(
    "batchtopk", BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
)
register_sae_training_class(
    "matryoshka_batchtopk",
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)
register_sae_class("transcoder", Transcoder, TranscoderConfig)
register_sae_class("skip_transcoder", SkipTranscoder, SkipTranscoderConfig)
register_sae_class("jumprelu_transcoder", JumpReLUTranscoder, JumpReLUTranscoderConfig)
register_sae_class("temporal", TemporalSAE, TemporalSAEConfig)
# Poly variants
register_sae_class("poly_topk", PolyTopKSAE, PolyTopKSAEConfig)
register_sae_training_class("poly_topk", PolyTopKTrainingSAE, PolyTopKTrainingSAEConfig)
register_sae_training_class("poly_batchtopk", PolyBatchTopKTrainingSAE, PolyBatchTopKTrainingSAEConfig)
register_sae_class("poly_jumprelu", PolyJumpReLUSAE, PolyJumpReLUSAEConfig)
register_sae_training_class("poly_jumprelu", PolyJumpReLUTrainingSAE, PolyJumpReLUTrainingSAEConfig)
register_sae_training_class("poly_matryoshka", PolyMatryoshkaBatchTopKTrainingSAE, PolyMatryoshkaBatchTopKTrainingSAEConfig)
# Crosscoder variants
register_sae_class("cross_jumprelu", CrossJumpReLUSAE, CrossJumpReLUSAEConfig)
register_sae_class("cross_topk", CrossTopKSAE, CrossTopKSAEConfig)
register_sae_training_class("cross_topk", CrossTopKTrainingSAE, CrossTopKTrainingSAEConfig)
register_sae_training_class("cross_batchtopk", CrossBatchTopKTrainingSAE, CrossBatchTopKTrainingSAEConfig)
register_sae_training_class("cross_matryoshka", CrossMatryoshkaBatchTopKTrainingSAE, CrossMatryoshkaBatchTopKTrainingSAEConfig)
