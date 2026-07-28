from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig

EVAL_TYPE_ID_SPARSE_PROBING = "sparse_probing"


@dataclass
class SparseProbingLlmMetrics(BaseMetrics):
    llm_test_accuracy: float = Field(
        title="LLM Test Accuracy",
        description="Linear probe accuracy when training on the full LLM residual stream",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    llm_test_f1: float | None = Field(
        default=None,
        title="LLM Test F1",
        description="Linear probe F1 score when training on the full LLM residual stream",
    )
    llm_wasserstein: float | None = Field(
        default=None,
        title="LLM Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in full LLM residual stream",
    )
    llm_top_1_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 1 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 1 residual stream channel test accuracy",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    llm_top_1_test_f1: float | None = Field(
        default=None,
        title="LLM Top 1 Test F1",
        description="Linear probe F1 score when trained on the LLM top 1 residual stream channel",
    )
    llm_top_1_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 1 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in LLM top 1 residual stream",
    )
    llm_top_2_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 2 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 2 residual stream channels test accuracy",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    llm_top_2_test_f1: float | None = Field(
        default=None,
        title="LLM Top 2 Test F1",
        description="Linear probe F1 score when trained on the LLM top 2 residual stream channels",
    )
    llm_top_2_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 2 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in LLM top 2 residual stream",
    )
    llm_top_5_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 5 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 5 residual stream channels test accuracy",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    llm_top_5_test_f1: float | None = Field(
        default=None,
        title="LLM Top 5 Test F1",
        description="Linear probe F1 score when trained on the LLM top 5 residual stream channels",
    )
    llm_top_5_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 5 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in LLM top 5 residual stream",
    )
    llm_top_10_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 10 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 10 residual stream channels",
    )
    llm_top_10_test_f1: float | None = Field(
        default=None,
        title="LLM Top 10 Test F1",
        description="Linear probe F1 score when trained on the LLM top 10 residual stream channels",
    )
    llm_top_10_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 10 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in LLM top 10 residual stream",
    )
    llm_top_20_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 20 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 20 residual stream channels",
    )
    llm_top_20_test_f1: float | None = Field(
        default=None,
        title="LLM Top 20 Test F1",
        description="Linear probe F1 score when trained on the LLM top 20 residual stream channels",
    )
    llm_top_20_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 20 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in LLM top 20 residual stream",
    )
    llm_top_50_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 50 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 50 residual stream channels",
    )
    llm_top_50_test_f1: float | None = Field(
        default=None,
        title="LLM Top 50 Test F1",
        description="Linear probe F1 score when trained on the LLM top 50 residual stream channels",
    )
    llm_top_50_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 50 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in LLM top 50 residual stream",
    )
    llm_top_100_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 100 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 100 residual stream channels",
    )
    llm_top_100_test_f1: float | None = Field(
        default=None,
        title="LLM Top 100 Test F1",
        description="Linear probe F1 score when trained on the LLM top 100 residual stream channels",
    )
    llm_top_100_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 100 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in LLM top 100 residual stream",
    )


@dataclass
class SparseProbingSaeMetrics(BaseMetrics):
    sae_test_accuracy: float | None = Field(
        default=None,
        title="SAE Test Accuracy",
        description="Linear probe accuracy when trained on all SAE latents",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    sae_test_f1: float | None = Field(
        default=None,
        title="SAE Test F1",
        description="Linear probe F1 score when trained on all SAE latents",
    )
    sae_wasserstein: float | None = Field(
        default=None,
        title="SAE Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in all SAE latents",
    )
    sae_top_1_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 1 Test Accuracy",
        description="Linear probe accuracy when trained on the top 1 SAE latents",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    sae_top_1_test_f1: float | None = Field(
        default=None,
        title="SAE Top 1 Test F1",
        description="Linear probe F1 score when trained on the top 1 SAE latents",
    )
    sae_top_1_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 1 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in top 1 SAE latents",
    )
    sae_top_2_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 2 Test Accuracy",
        description="Linear probe accuracy when trained on the top 2 SAE latents",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    sae_top_2_test_f1: float | None = Field(
        default=None,
        title="SAE Top 2 Test F1",
        description="Linear probe F1 score when trained on the top 2 SAE latents",
    )
    sae_top_2_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 2 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in top 2 SAE latents",
    )
    sae_top_5_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 5 Test Accuracy",
        description="Linear probe accuracy when trained on the top 5 SAE latents",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    sae_top_5_test_f1: float | None = Field(
        default=None,
        title="SAE Top 5 Test F1",
        description="Linear probe F1 score when trained on the top 5 SAE latents",
    )
    sae_top_5_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 5 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in top 5 SAE latents",
    )
    sae_top_10_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 10 Test Accuracy",
        description="Linear probe accuracy when trained on the top 10 SAE latents",
    )
    sae_top_10_test_f1: float | None = Field(
        default=None,
        title="SAE Top 10 Test F1",
        description="Linear probe F1 score when trained on the top 10 SAE latents",
    )
    sae_top_10_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 10 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in top 10 SAE latents",
    )
    sae_top_20_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 20 Test Accuracy",
        description="Linear probe accuracy when trained on the top 20 SAE latents",
    )
    sae_top_20_test_f1: float | None = Field(
        default=None,
        title="SAE Top 20 Test F1",
        description="Linear probe F1 score when trained on the top 20 SAE latents",
    )
    sae_top_20_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 20 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in top 20 SAE latents",
    )
    sae_top_50_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 50 Test Accuracy",
        description="Linear probe accuracy when trained on the top 50 SAE latents",
    )
    sae_top_50_test_f1: float | None = Field(
        default=None,
        title="SAE Top 50 Test F1",
        description="Linear probe F1 score when trained on the top 50 SAE latents",
    )
    sae_top_50_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 50 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in top 50 SAE latents",
    )
    sae_top_100_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 100 Test Accuracy",
        description="Linear probe accuracy when trained on the top 100 SAE latents",
    )
    sae_top_100_test_f1: float | None = Field(
        default=None,
        title="SAE Top 100 Test F1",
        description="Linear probe F1 score when trained on the top 100 SAE latents",
    )
    sae_top_100_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 100 Wasserstein",
        description="Mean Wasserstein distance between positive/negative class activations in top 100 SAE latents",
    )


@dataclass
class SparseProbingMetricCategories(BaseMetricCategories):
    llm: SparseProbingLlmMetrics = Field(
        title="LLM",
        description="LLM metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    sae: SparseProbingSaeMetrics = Field(
        title="SAE",
        description="SAE metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass
class SparseProbingResultDetail(BaseResultDetail):
    dataset_name: str = Field(
        title="Dataset Name",
        description="Dataset name",
    )

    # LLM metrics
    llm_test_accuracy: float = Field(
        title="LLM Test Accuracy",
        description="Linear probe accuracy when trained on all LLM residual stream channels",
    )
    llm_test_f1: float | None = Field(
        default=None,
        title="LLM Test F1",
        description="Linear probe F1 score when trained on all LLM residual stream channels",
    )
    llm_wasserstein: float | None = Field(
        default=None,
        title="LLM Wasserstein",
        description="Mean Wasserstein distance for LLM residual stream",
    )
    llm_top_1_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 1 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 1 residual stream channels",
    )
    llm_top_1_test_f1: float | None = Field(
        default=None,
        title="LLM Top 1 Test F1",
        description="Linear probe F1 score when trained on the LLM top 1 residual stream channels",
    )
    llm_top_1_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 1 Wasserstein",
        description="Mean Wasserstein distance for LLM top 1 residual stream",
    )
    llm_top_2_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 2 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 2 residual stream channels",
    )
    llm_top_2_test_f1: float | None = Field(
        default=None,
        title="LLM Top 2 Test F1",
        description="Linear probe F1 score when trained on the LLM top 2 residual stream channels",
    )
    llm_top_2_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 2 Wasserstein",
        description="Mean Wasserstein distance for LLM top 2 residual stream",
    )
    llm_top_5_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 5 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 5 residual stream channels",
    )
    llm_top_5_test_f1: float | None = Field(
        default=None,
        title="LLM Top 5 Test F1",
        description="Linear probe F1 score when trained on the LLM top 5 residual stream channels",
    )
    llm_top_5_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 5 Wasserstein",
        description="Mean Wasserstein distance for LLM top 5 residual stream",
    )
    llm_top_10_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 10 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 10 residual stream channels",
    )
    llm_top_10_test_f1: float | None = Field(
        default=None,
        title="LLM Top 10 Test F1",
        description="Linear probe F1 score when trained on the LLM top 10 residual stream channels",
    )
    llm_top_10_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 10 Wasserstein",
        description="Mean Wasserstein distance for LLM top 10 residual stream",
    )
    llm_top_20_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 20 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 20 residual stream channels",
    )
    llm_top_20_test_f1: float | None = Field(
        default=None,
        title="LLM Top 20 Test F1",
        description="Linear probe F1 score when trained on the LLM top 20 residual stream channels",
    )
    llm_top_20_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 20 Wasserstein",
        description="Mean Wasserstein distance for LLM top 20 residual stream",
    )
    llm_top_50_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 50 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 50 residual stream channels",
    )
    llm_top_50_test_f1: float | None = Field(
        default=None,
        title="LLM Top 50 Test F1",
        description="Linear probe F1 score when trained on the LLM top 50 residual stream channels",
    )
    llm_top_50_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 50 Wasserstein",
        description="Mean Wasserstein distance for LLM top 50 residual stream",
    )
    llm_top_100_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 100 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 100 residual stream channels",
    )
    llm_top_100_test_f1: float | None = Field(
        default=None,
        title="LLM Top 100 Test F1",
        description="Linear probe F1 score when trained on the LLM top 100 residual stream channels",
    )
    llm_top_100_wasserstein: float | None = Field(
        default=None,
        title="LLM Top 100 Wasserstein",
        description="Mean Wasserstein distance for LLM top 100 residual stream",
    )

    # SAE metrics
    sae_test_accuracy: float | None = Field(
        default=None,
        title="SAE Test Accuracy",
        description="Linear probe accuracy when trained on all SAE latents",
    )
    sae_test_f1: float | None = Field(
        default=None,
        title="SAE Test F1",
        description="Linear probe F1 score when trained on all SAE latents",
    )
    sae_wasserstein: float | None = Field(
        default=None,
        title="SAE Wasserstein",
        description="Mean Wasserstein distance for all SAE latents",
    )
    sae_top_1_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 1 Test Accuracy",
        description="Linear probe accuracy when trained on the top 1 SAE latents",
    )
    sae_top_1_test_f1: float | None = Field(
        default=None,
        title="SAE Top 1 Test F1",
        description="Linear probe F1 score when trained on the top 1 SAE latents",
    )
    sae_top_1_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 1 Wasserstein",
        description="Mean Wasserstein distance for top 1 SAE latents",
    )
    sae_top_2_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 2 Test Accuracy",
        description="Linear probe accuracy when trained on the top 2 SAE latents",
    )
    sae_top_2_test_f1: float | None = Field(
        default=None,
        title="SAE Top 2 Test F1",
        description="Linear probe F1 score when trained on the top 2 SAE latents",
    )
    sae_top_2_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 2 Wasserstein",
        description="Mean Wasserstein distance for top 2 SAE latents",
    )
    sae_top_5_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 5 Test Accuracy",
        description="Linear probe accuracy when trained on the top 5 SAE latents",
    )
    sae_top_5_test_f1: float | None = Field(
        default=None,
        title="SAE Top 5 Test F1",
        description="Linear probe F1 score when trained on the top 5 SAE latents",
    )
    sae_top_5_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 5 Wasserstein",
        description="Mean Wasserstein distance for top 5 SAE latents",
    )
    sae_top_10_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 10 Test Accuracy",
        description="Linear probe accuracy when trained on the top 10 SAE latents",
    )
    sae_top_10_test_f1: float | None = Field(
        default=None,
        title="SAE Top 10 Test F1",
        description="Linear probe F1 score when trained on the top 10 SAE latents",
    )
    sae_top_10_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 10 Wasserstein",
        description="Mean Wasserstein distance for top 10 SAE latents",
    )
    sae_top_20_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 20 Test Accuracy",
        description="Linear probe accuracy when trained on the top 20 SAE latents",
    )
    sae_top_20_test_f1: float | None = Field(
        default=None,
        title="SAE Top 20 Test F1",
        description="Linear probe F1 score when trained on the top 20 SAE latents",
    )
    sae_top_20_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 20 Wasserstein",
        description="Mean Wasserstein distance for top 20 SAE latents",
    )
    sae_top_50_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 50 Test Accuracy",
        description="Linear probe accuracy when trained on the top 50 SAE latents",
    )
    sae_top_50_test_f1: float | None = Field(
        default=None,
        title="SAE Top 50 Test F1",
        description="Linear probe F1 score when trained on the top 50 SAE latents",
    )
    sae_top_50_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 50 Wasserstein",
        description="Mean Wasserstein distance for top 50 SAE latents",
    )
    sae_top_100_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 100 Test Accuracy",
        description="Linear probe accuracy when trained on the top 100 SAE latents",
    )
    sae_top_100_test_f1: float | None = Field(
        default=None,
        title="SAE Top 100 Test F1",
        description="Linear probe F1 score when trained on the top 100 SAE latents",
    )
    sae_top_100_wasserstein: float | None = Field(
        default=None,
        title="SAE Top 100 Wasserstein",
        description="Mean Wasserstein distance for top 100 SAE latents",
    )


@dataclass(config=ConfigDict(title="Sparse Probing"))
class SparseProbingEvalOutput(
    BaseEvalOutput[
        SparseProbingEvalConfig,
        SparseProbingMetricCategories,
        SparseProbingResultDetail,
    ]
):
    # This will end up being the description of the eval in the UI.
    """
    An evaluation using SAEs to probe for supervised concepts in LLMs. We use sparse probing with the top K SAE latents and probe for over 30 different classes across 5 datasets.
    """

    eval_config: SparseProbingEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: SparseProbingMetricCategories
    eval_result_details: list[SparseProbingResultDetail] = Field(
        default_factory=list,
        title="Per-Dataset Sparse Probing Results",
        description="Each object is a stat on the sparse probing results for a dataset.",
    )
    eval_type_id: str = Field(default=EVAL_TYPE_ID_SPARSE_PROBING)
