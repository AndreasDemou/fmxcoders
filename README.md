<div align="center">

# fmxcoders: Factorized Masked Crosscoders for Cross-Layer Feature Discovery

[![arXiv](https://img.shields.io/badge/arXiv-2605.09438-b31b1b.svg)](https://arxiv.org/abs/2605.09438)
[![NeurIPS 2026 Spotlight](https://img.shields.io/badge/NeurIPS%202026-Spotlight-6f42c1.svg)](https://arxiv.org/abs/2605.09438)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Andreas D. Demou, Panagiotis Koromilas, James Oldfield, Yannis Panagakis, Mihalis A. Nicolaou**

🌟 **NeurIPS 2026 Spotlight** 🌟

[📄 Paper](https://arxiv.org/abs/2605.09438) · [Installation](#installation) · [Training](#training-an-fmxcoder) · [Evaluation](#evaluation) · [Citation](#citation)

</div>

---

This repository contains the official code for **fmxcoders**, introduced in
[*fmxcoders: Factorized Masked Crosscoders for Cross-Layer Feature Discovery*](https://arxiv.org/abs/2605.09438),
accepted as a **spotlight at NeurIPS 2026**.

## Overview

Features in pretrained Transformers often span multiple layers. Standard **crosscoders**
(sparse dictionaries trained jointly on activations from several layers) mostly fail to
capture them: their latents tend to be driven by only one or two layers and pick up
surface patterns, such as digit detectors. We trace this to two causes:

1. **Unconstrained cross-layer parameterization.** Every latent has a free, independent
   direction at every layer.
2. **Unregularized cross-layer dependence.** Nothing stops a latent from relying on a
   single layer.

**fmxcoders** address both:

| Component | What it does | Flag |
|---|---|---|
| **Factorized weights** | The 3-D encoder/decoder tensors `W_enc ∈ ℝ^{L×d×s}` and `W_dec ∈ ℝ^{s×L×d}` are replaced by a low-rank **CP** or **Tensor-Ring (TR)** factorization with a basis shared across layers. The full tensors are never materialized in the forward pass. | `--use_cross_fact --cross_fact_type {CP,TR} --cross_fact_ranks ...` |
| **Stochastic layer masking** | During training, each layer's *input* is zeroed independently per token with probability `p`, while the reconstruction target stays unmasked. Latents must then reconstruct a layer from the others. | `--layer_masking p` |

Across GPT-2 Small, Pythia-410M, Pythia-1.4B and Gemma-2-2B, fmxcoders improve sparse-probing
F1, reduce reconstruction error, roughly double functional coherence, and recover
**3–13× more semantically coherent latents** than standard crosscoders (LLM-as-a-judge).
See the [paper](https://arxiv.org/abs/2605.09438) for full results.

### Factorizations

With `L` = number of layers, `d` = `d_in`, `s` = `d_sae`:

- **CP** (`--cross_fact_type CP --cross_fact_ranks R`):
  `W_enc[l,d,s] = Σ_r A[l,r] · B[d,r] · C[s,r]` (one rank `R`).
- **Tensor Ring** (`--cross_fact_type TR --cross_fact_ranks r1,r2,r3`):
  three cores `G_n ∈ ℝ^{r1×L×r2}`, `G_d ∈ ℝ^{r2×d×r3}`, `G_s ∈ ℝ^{r3×s×r1}`;
  the layer mode is bordered by `(r1, r2)`, `d_in` by `(r2, r3)`, and `d_sae` by `(r3, r1)`.

Encoder and decoder are factorized independently. Initialization is variance-matched to
Kaiming-uniform on a per-layer slice. The implementation is in
[`sae_lens/saes/crosscoder_sae.py`](sae_lens/saes/crosscoder_sae.py).

## Repository structure

This codebase is built on [SAELens](https://github.com/decoderesearch/SAELens) (v6.22) for
training and [SAEBench](https://github.com/adamkarvonen/SAEBench) for evaluation. Both are
vendored here with crosscoder support added.

```
.
├── sae_lens/                         # SAELens fork
│   └── saes/crosscoder_sae.py        #   ← crosscoder mixin, CP/TR factorization, layer masking
├── SAEBench/                         # SAEBench fork (core eval has a crosscoder branch)
├── train_and_saebench.py             # main entry point: train (+ optionally evaluate)
├── launch.sh                         # example training launcher
├── validate_crosscoder.py            # end-to-end sanity checks on pythia-70m
│
├── eval_crosscoder_reconstruction.py # per-layer MSE / explained variance / cosine sim
├── eval_norm_coherence.py            # weight-based coherence (no data needed)
├── eval_mask_coherence.py            # data-based coherence via layer ablation
├── extract_token_features.py         # top-k latents per token over a corpus
├── extract_token_features_pivot.py   # pivot to latent → top-activating examples
├── count_latent_tokens.py            # per-latent token-diversity statistics
├── top_bottom_coherence.py           # most/least coherent latents + their contexts
├── compare_absorption.py             # SAEBench feature absorption on a per-layer slice
├── compare_ravel.py                  # SAEBench RAVEL on a per-layer slice
├── steering_prompts.json             # hand-written probe prompts for gen_feature_steering.py
├── eval_feature_steering.py          # distributional steering eval
├── gen_feature_steering.py           # steered text generation
├── summarize_evals.py, fetch_local_run_metrics*.py   # result tables
│
├── run_*.sh                          # SLURM scripts used for the paper's evaluations
├── env.example.sh                    # environment template for run_*.sh (copy to env.sh)
├── legacy/                           # earlier (PolySAE) scripts, kept for reference
├── tests/                            # SAELens tests + crosscoder tests
└── tutorials/                        # upstream SAELens tutorials
```

## Installation

Requires Python ≥ 3.10 and a CUDA GPU for anything beyond toy runs. Tested with
`torch 2.9.1`, `transformers 4.57.3`, `transformer-lens 2.16.1`, and `datasets 3.6.0`.

```bash
git clone https://github.com/AndreasDemou/fmxcoders.git
cd fmxcoders

python -m venv .venv
source .venv/bin/activate

# 1. PyTorch: pick the build that matches your CUDA version (https://pytorch.org)
pip install torch

# 2. This repo's SAELens fork, then this repo's SAEBench fork
pip install -e .
pip install -e ./SAEBench
```

> [!IMPORTANT]
> Use the `sae_lens` and `SAEBench` **bundled in this repo**. The upstream packages from
> PyPI or GitHub do not contain the crosscoder code. `train_and_saebench.py` puts
> `./SAEBench` first on the import path automatically (override with `--saebench_repo_dir`).

To reproduce the exact environment used for the paper, install the pinned versions first,
then the two local packages:

```bash
pip install -r requirements.freeze.txt
pip install -e . --no-deps && pip install -e ./SAEBench --no-deps
```

Gemma-2 and Llama weights are gated on Hugging Face. Run `huggingface-cli login` first.
Log in to W&B (`wandb login`) or pass `--no_wandb`.

### Sanity check

```bash
python validate_crosscoder.py                       # short pythia-70m run, checks shapes, masking, CP/TR, save/load
pytest tests/saes/test_crosscoder_single_layer_view.py
```

## Training an fmxcoder

All training goes through `train_and_saebench.py`. `--exp` selects a preset (model, dataset,
`d_in`, and hook type). `--cross_layers` selects which residual-stream layers the crosscoder
reads and reconstructs.

**fmxcoder (TR-factorized + layer masking)**, Pythia-410M, layers 8–15:

```bash
python train_and_saebench.py \
  --exp pythia410m_l15 --architecture batchtopk \
  --width 16k --k 64 --training_tokens 500000000 --context_size 128 \
  --no_rescale_by_decoder_norm --use_saelens_defaults \
  --use_crosscoder --cross_layers 8,9,10,11,12,13,14,15 \
  --use_cross_fact --cross_fact_type TR --cross_fact_ranks 27,7,302 \
  --layer_masking 0.05 \
  --checkpoint_path checkpoints/pythia410m_m0_05_r27-7-302 \
  --wandb_run_name pythia410m_m0_05_r27-7-302 \
  --seed 42 --run_saebench
```

**Baselines.** Drop flags to get the ablations reported in the paper:

| Model | Flags |
|---|---|
| Standard crosscoder | `--use_crosscoder --cross_layers …` (no factorization, no masking) |
| Masked crosscoder | `… --layer_masking 0.05` |
| Factorized crosscoder | `… --use_cross_fact --cross_fact_type {CP,TR} --cross_fact_ranks …` |
| **fmxcoder** | factorization **and** masking |
| CP fmxcoder (Pythia-410M) | `--cross_fact_type CP --cross_fact_ranks 7707 --layer_masking 0.05` |

Run names in the scripts follow the pattern `<model>_m<mask>_r<ranks>`. For example,
`pythia410m_m0_05_r27-7-302` means masking `p = 0.05` and TR ranks `(27, 7, 302)`, and
`pythia410m_m0` is the unfactorized, unmasked crosscoder.

### Key arguments

| Argument | Description |
|---|---|
| `--exp` | Preset: `gpt2_l8`, `pythia70m_l3`, `pythia70m_l5`, `pythia160m_l8`, `pythia410m_l15`, `pythia1_4b_l12`, `gemma2_2b_l12`, `gemma2_2b_l19`, `gemma2_27b_l22`, `llama31_8b_l25`, `llama32_3b_l12` |
| `--architecture` | Sparsity mechanism: `topk`, `batchtopk`, `jumprelu`, `matryoshka` (crosscoder variants exist for `topk`, `batchtopk`, `matryoshka`) |
| `--width`, `--k` | Dictionary size (`2k` … `131k`) and number of active latents |
| `--use_crosscoder`, `--cross_layers` | Enable the crosscoder and choose layers (comma-separated) |
| `--layer_masking` | Per-token, per-layer input-masking probability (`0` disables) |
| `--use_cross_fact`, `--cross_fact_type`, `--cross_fact_ranks` | Tensor factorization: `CP` takes 1 rank, `TR` takes 3 |
| `--act_store_device cpu` | Keep the activation buffer in host RAM for large crosscoders (avoids GPU OOM) |
| `--resume_from_checkpoint` | Resume from the latest snapshot in `--checkpoint_path` |
| `--run_saebench`, `--eval_only`, `--skip_eval` | Run SAEBench after training, or only evaluate an existing checkpoint |

Checkpoints are written to `<checkpoint_path>/final_<n_tokens>/` (`cfg.json` +
`sae_weights.safetensors`). See `launch.sh` for a resumable launcher (set `CKPT_ROOT` to choose where checkpoints go; default `./checkpoints`).

### Loading a trained fmxcoder

```python
from _crosscoder_eval_utils import load_crosscoder_inference   # run from the repo root

sae = load_crosscoder_inference("checkpoints/pythia410m_m0_05_r27-7-302/final_500002816", device="cuda")
hook_names = sae.cfg.metadata.hook_names        # one hook per crosscoder layer

# x: stacked residual activations [batch, seq, n_layers, d_in]
z = sae.encode(x)                                # [batch, seq, d_sae]
x_hat = sae.decode(z)                            # [batch, seq, n_layers, d_in]

# 2-D single-layer slice, e.g. for single-layer SAE evals
layer_view = sae.as_single_layer_view(layer_idx=3)
```

Training-only architectures such as `cross_batchtopk` are not in the inference registry,
so `SAE.load_from_disk` raises a `KeyError` on them. `load_crosscoder_inference` loads
the checkpoint as a `TrainingSAE`, exports it as its inference counterpart (folding decoder
norms and converting the TopK threshold), and reloads it as an `SAE`.

## Evaluation

All evaluation scripts take a `final_*` checkpoint directory and write to `eval_results/`,
`coherence_results/`, or `steering_results/`. The `run_*.sh` files are the SLURM scripts used
for the paper. To use them, set up your environment once and edit the checkpoint list in each script:

```bash
cp env.example.sh env.sh     # set venv, CKPT_ROOT, HF caches, offline mode (env.sh is git-ignored)
sbatch -A <account> -p <partition> run_coherence_evals.sh   # submit from the repo root (or run with bash)
```

### 1. SAEBench (probing, reconstruction)

```bash
python train_and_saebench.py --exp pythia410m_l15 --architecture batchtopk --width 16k --k 64 \
  --use_crosscoder --cross_layers 8,9,10,11,12,13,14,15 \
  --use_cross_fact --cross_fact_type TR --cross_fact_ranks 27,7,302 \
  --checkpoint_path checkpoints/pythia410m_m0_05_r27-7-302 \
  --eval_only --run_saebench --skip_eval tpp --skip_eval scr --skip_eval ravel --skip_eval absorption
```

SAEBench's core eval has a crosscoder branch that reports per-layer and layer-averaged
metrics. KL and CE-loss metrics are skipped for crosscoders.

**Absorption and RAVEL.** These SAEBench evals expect a single-layer SAE, so crosscoders are
evaluated one layer at a time through `as_single_layer_view`, which treats factorized and
unfactorized crosscoders the same way:

```bash
python compare_absorption.py --checkpoint <ckpt>/final_XXXX --layer-idx 4 --model-name pythia-410m-deduped
python compare_ravel.py      --checkpoint <ckpt>/final_XXXX --layer-idx 4 --model-name gemma-2-2b
```

Results go to `eval_results/<run>/{absorption,ravel}/` and are tabulated by `summarize_evals.py`.
RAVEL only supports the models listed in SAEBench's `LLM_NAME_MAP` (currently `gemma-2-2b` and
`pythia-160m-deduped`).

### 2. Cross-layer reconstruction

```bash
python eval_crosscoder_reconstruction.py --checkpoint_path <ckpt>/final_XXXX \
  --n_batches 2000 --batch_size_prompts 16 --context_size 128 \
  --out eval_results/<run>/recon_metrics.json
```

This follows SAEBench's core methodology and reports per-layer MSE, explained variance, and
cosine similarity. See `run_eval_cross_reconstruction.sh`.

### 3. Cross-layer coherence

For each latent, coherence is `mean_l(·) / max_l(·)` over layers. It is ≈ 1 when a latent
uses all layers evenly and ≪ 1 when a single layer dominates.

```bash
# Weight-based: per-layer norm of each latent's decoder (or encoder) fiber. No data needed.
python eval_norm_coherence.py --checkpoint_path <ckpt> --output_dir coherence_results/<run>/norm_decoder
python eval_norm_coherence.py --checkpoint_path <ckpt> --output_dir coherence_results/<run>/norm_encoder --use_encoder

# Data-based: how much each latent's activation changes when one layer is ablated
python eval_mask_coherence.py --checkpoint_path <ckpt> --output_dir coherence_results/<run>/mask_zero \
  --masking_strategy zero --n_batches 1000 --batch_size_prompts 16 --context_size 128
```

See `run_coherence_evals.sh`.

### 4. Latent interpretability and steering

```bash
# Top-k latents per token, then pivot to latent → top-activating examples
python extract_token_features.py --checkpoint <ckpt> --n_texts 100000
python extract_token_features_pivot.py eval_results/<run>/feature_analysis/token_features.json

# Token diversity per latent (a proxy for surface-pattern / single-token latents)
python count_latent_tokens.py eval_results/<run>/feature_analysis/token_features_pivot.json

# Most / least coherent latents with their contexts → top_bottom_coherence_<run>.json
python top_bottom_coherence.py --run_name <run>

# Steering: distributional effect and steered generations
python eval_feature_steering.py --checkpoint_path <ckpt> --contexts_from top_bottom_coherence_<run>.json \
  --groups top_most_coherent --all_layers --delta_mode increment --coeffs 1,2,4,6,8
python gen_feature_steering.py --checkpoint_path <ckpt> --contexts_from top_bottom_coherence_<run>.json \
  --prompts_from steering_prompts.json --all_layers --delta_mode increment --coeffs 0.5,1,2,4,8
```

`steering_prompts.json` holds hand-written probe prompts for 10 latents of the Gemma-2-2B run
`gemma2-2b_m0_05_r32-12-545`. Latent IDs are specific to that checkpoint, so write your own file
in the same format to steer a different model.

`--all_layers --delta_mode increment` injects a feature exactly once across the crosscoder
span: per-layer increments telescope to the last layer's direction. See the comments in
`run_eval_feature_steering.sh` and `run_gen_feature_steering.sh` for guidance on coefficients.

### Summaries

```bash
python fetch_local_run_metrics.py <run_name>       # probing F1 + per-layer reconstruction, LaTeX row
python summarize_evals.py                          # RAVEL / absorption table across runs
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{demou2026fmxcoders,
  title     = {fmxcoders: Factorized Masked Crosscoders for Cross-Layer Feature Discovery},
  author    = {Demou, Andreas D. and Koromilas, Panagiotis and Oldfield, James and Panagakis, Yannis and Nicolaou, Mihalis A.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Spotlight},
  eprint    = {2605.09438},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2605.09438}
}
```

## Acknowledgements

This code builds on [SAELens](https://github.com/decoderesearch/SAELens) by Joseph Bloom et al.
and [SAEBench](https://github.com/adamkarvonen/SAEBench) by Adam Karvonen et al. Both are
included here in modified form. Please also cite them if you use their components.

## License

This project is released under the [MIT License](LICENSE). The bundled SAELens code keeps its
original MIT copyright notice, and the bundled SAEBench code remains subject to its authors'
terms; see [LICENSE](LICENSE) for details.
