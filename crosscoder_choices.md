# Crosscoder Implementation: Design Choices and Approximations

A complete list of design choices and approximations made during the SAE → crosscoder, layer-masking, CP, and TR factorization implementation. For each: **what it is**, **why we chose it**, and **how it could affect training/eval**.

---

## Crosscoder architecture (non-factorized)

**1. Stacked input shape, not concatenated.**
Input is `[batch, seq, n_layers, d_in]`, weights are 3-D. Each feature has independent encoder/decoder weights per layer.
*Effect:* This is your design choice, not an approximation. But it means feature norms now span all layers — `_crosscoder_decoder_norms` returns `||W_dec[s, :, :]||_F` (Frobenius across both `n_layers` and `d_in`). A "feature" is now defined by its joint multi-layer footprint.

**2. Per-layer Kaiming-uniform init.**
`W_dec` is initialised by calling `kaiming_uniform_` on each `[d_sae, d_in]` slice independently. `W_enc` is initialised as the per-layer transpose of `W_dec`.
*Effect:* Each layer's columns are i.i.d. — there's no correlation between layers at init. Initial decoder Frobenius norms are roughly `sqrt(n_layers) ×` the scalar Kaiming norm, so `rescale_acts_by_decoder_norm` rescales by ~`sqrt(n_layers)` at init. Early training dynamics differ from a single-layer SAE by this factor.

**3. `stop_at_layer = max(layer for hook in hook_names)`.**
ActivationsStore stops the model right after the deepest hooked layer.
*Effect:* If your hooked layers don't include layer 0, the model still computes layers 0…(deepest hook). No wasted work for the hooks you have, but no savings either if you only care about a few middle layers.

**23. MSE is averaged over layers, not summed.**
`training_forward_pass` does `per_item_mse.sum(dim=-1).mean()` — the sum is over `d_in` only, so `.mean()` averages across batch AND `n_layers`. Anthropic's cross-layer crosscoder paper writes the loss as `Σ_l MSE_l`.
*Effect:* Loss optima coincide up to a constant factor `1/L`, but absolute MSE numbers are `1/L` of the Anthropic convention. When comparing to published crosscoder numbers or to a single-layer SAE baseline, the crosscoder's MSE will appear `L×` smaller.

**24. Decoder norm uses Frobenius, not Anthropic's Σ-of-2-norms.**
`_crosscoder_decoder_norms()` returns `(Σ_l Σ_d W_dec[s,l,d]²)^{1/2}`. Anthropic's paper uses `Σ_l ‖W_dec^{l,s}‖_2` (sum of per-layer 2-norms).
*Effect:* For TopK/BatchTopK selection this is a soft reweighting, not a correctness issue. But Anthropic also uses that quantity as an L1 sparsity weight (`λ Σ_i f_i · Σ_l ‖W_dec^{l,i}‖_2`) which pressures features to concentrate on few layers. We use TopK/BatchTopK so there's no L1 and the concentration pressure is absent — fine today, but if you ever switch to an L1-style sparsity you won't get it unless you add it.

---

## Layer masking

**4. Mask applied AFTER `b_dec` subtraction.**
The encoder sees exactly `0` for masked layers (not `−b_dec[l]`).
*Effect:* The "masked" signal is never seen at eval (where `sae_in[l] = x − b_dec`). Encoder must learn to handle a slightly OOD input pattern at training time. The alternative (mask before subtraction) would have masked layers contribute `−b_dec[l] @ W_enc[l]` — also OOD, but in a different way. We discussed this trade-off explicitly.

**5. Per-token Bernoulli mask.**
Each `[batch, seq]` position gets an independent mask of length `n_layers`.
*Effect:* Different tokens in the same sequence may have different masking patterns. Standard dropout-style.

**6. No inverse rescaling on unmasked layers.**
Unlike classical dropout (which rescales by `1/(1-p)` during training so expected magnitude is preserved), we just zero. So at eval, unmasked layer activations are *larger* on average than during training.
*Effect:* Train/eval distribution shift in the encoder's input scale. May bias the SAE toward smaller features at training time.

**7. Target is unmasked.**
The SAE is trained to reconstruct the original (unmasked) activations from a possibly-masked input.
*Effect:* This turns the SAE into a "predict missing layers from the others" model. It's a more demanding task than pure reconstruction and changes the loss landscape. Not the same as classical denoising autoencoder (which would mask both input and target).

**25. BatchTopK→JumpReLU export threshold miscalibrated under masking.**
BatchTopK's `topk_threshold` is an EMA of the min positive activation ([batchtopk_sae.py:116-125](sae_lens/saes/batchtopk_sae.py#L116-L125)), updated during training — where the encoder sees masked inputs. At save time the threshold is copied to the exported JumpReLU unchanged, but inference runs on unmasked inputs (per-layer magnitudes are ~`1/(1−p)` larger than what the EMA saw).
*Effect:* Exported JumpReLU systematically fires too often at inference. Barely noticeable at `p=0.1`, significant at `p=0.5`. Easy fix: run a post-training threshold recalibration on an unmasked batch before exporting.

**26. No guarantee at least one layer is kept.**
Per-token per-layer independent Bernoulli means `P(all layers masked) = p^L`. At `p=0.5, L=2` that's 25% of tokens; at `p=0.3, L=3` it's ~2.7%.
*Effect:* All-masked tokens degenerate into `hidden_pre = b_enc` and only train the mean-reconstruction path (`b_enc`, `b_dec`, `W_dec`). Not broken — just a chunk of each batch "learns the mean" instead of layer-interpolation. Sampling `k ≥ 1` layers to keep would remove it.

**27. Aux loss uses masked `hidden_pre` but full (unmasked) residual.**
`calculate_topk_aux_loss` picks dead-feature activations from the masked `hidden_pre` and asks them to reconstruct `sae_in − sae_out` (full target).
*Effect:* Dead features inherit the denoising objective — trained to reconstruct the full residual from a corrupted pre-activation. Not a bug, but couples auxk behavior to `layer_masking` in a non-obvious way. If auxk convergence looks strange with masking on, this is a suspect.

---

## CP factorization

**8. Variance-matching init: `std = (target_var / R)^(1/6)`.**
Designed so the *materialised* `W` has variance `target_var` per element. The algebra is right: for independent zero-mean factors, covariance across different `r` vanishes, so `Var[W] = R · σ^6` → `σ = (target_var/R)^{1/6}`.
*Effect:* The docstring's "matching Kaiming-uniform" claim is misattributed. `target_var = 6/(d_in+d_sae)` is Xavier-uniform's *squared bound*, not the Kaiming variance `1/(3·d_in)`. The factored init ends up ~2–3× *smaller* than a true Kaiming slice. Training recovers, but the absolute init scale is not what the comment claims.

**9. No orthogonalization between factors.**
CP has scale/permutation ambiguity (you can rescale columns of `A` and inverse-rescale columns of `B`, etc., without changing `W`). Init is i.i.d. Normal.
*Effect:* Optimization can wander in the gauge orbit. Convergence may be slower than with orthogonalized cores, though in practice training usually finds a working configuration.

**10. `decoder_init_norm` silently ignored under CP/TR.**
`TrainingSAE.initialize_weights` normalises the 2-D `W_dec` to `cfg.decoder_init_norm` right before `_init_crosscoder_weights` deletes it and creates factors. A user who sets `decoder_init_norm=0.1` expecting unit-ish decoder rows at init gets the variance-matching factor init instead, with no warning.
*Effect:* Silent footgun when switching between standard and factored runs. The init scale is set by the formula at #8/#12, not by the user's config. (Note: the stale 2-D `W_enc`/`W_dec` ARE deleted at [crosscoder_sae.py:146-154](sae_lens/saes/crosscoder_sae.py#L146-L154) — earlier versions of this doc claimed otherwise.)

---

## TR factorization

**11. Cheap-proxy decoder norm: `||G_dec_s[:, s, :]||_F` instead of true Frobenius norm of `W_dec[s, :, :]`.**
This was your explicit choice (Q3, option C). The true Frobenius norm via Gram matrices is `O(d_sae · R^6)` per step — too expensive. The proxy ignores contributions from `G_dec_n` and `G_dec_d`.
*Effect: BIG one.* `rescale_acts_by_decoder_norm` under TR rescales features by something that's monotonic in only one core's magnitude, not by the true reconstruction magnitude. Could cause:
- Features whose magnitude is dominated by `G_dec_n` or `G_dec_d` to be under/over-rescaled
- `fold_W_dec_norm` produces a model where the cheap proxy is 1, not the true Frobenius norm. Inference checkpoints will have unnormalized true `W_dec` rows.
- If you compare CP vs TR with the same `cross_fact_ranks` and `rescale_acts_by_decoder_norm=True`, the comparison isn't apples-to-apples.

**12. Variance-matching init: `std = (target_var / (r1·r2·r3))^(1/6)`.**
Same approximation as CP, generalised to three ranks.
*Effect:* Same caveat — independence assumption at init.

**13. No gauge fixing on TR cores.**
TR has gauge freedom on each rank index: invertible R×R matrices can be inserted between any two adjacent cores. Init doesn't fix this.
*Effect:* Optimization can drift through the gauge orbit. May affect convergence speed; doesn't affect correctness.

**14. Encoder and decoder cores are NOT shared.**
We have 6 independent core tensors. Same dimensions per axis (your constraint), but separate parameters.
*Effect:* This is a design choice, not an approximation. If you wanted weight tying for additional regularization / param savings, that would be a separate change.

---

## Common to CP and TR

**28. Clamp inconsistency in decoder norms.**
Standard-mode clamp: `1e-8` ([crosscoder_sae.py:436](sae_lens/saes/crosscoder_sae.py#L436)). CP/TR: `1e-16` ([crosscoder_sae.py:424](sae_lens/saes/crosscoder_sae.py#L424), [434](sae_lens/saes/crosscoder_sae.py#L434)).
*Effect:* In bf16 both round to zero (eps ≈ 1e-2), so moot today. In fp32, switching a checkpoint between factored and standard modes can cross the floor in pathological cases. Pick one.

**29. `k_aux = d_in // 2` in TopK aux loss doesn't reflect factored rank.**
`calculate_topk_aux_loss` sets `k_aux = sae_in.shape[-1] // 2 = d_in // 2` — unchanged from single-layer SAE. For CP at `R=32` the decoder's effective rank is bounded by `R`, while `d_in/2` is typically ~384.
*Effect:* Auxk reserves more "dead feature slots" than the factorization can populate with independent directions. Wastes auxk capacity at low rank; may also inflate the aux loss term. No correctness impact.

---

## Eval changes

**15. `get_recons_loss` two-pass approach for crosscoder.**
Pass 1: cache all hooked activations from a clean forward pass; run the crosscoder offline to get reconstructions. Pass 2: re-run with hooks that return the **pre-computed** reconstructions.
*Effect:* Reconstructions are based on the *original* activations, not the cascade-modified ones. In a real "ablation" sense, replacing layer 3's activation should change layer 5's input → layer 5's activation should change → its reconstruction should change too. We ignore this cascade. Standard practice for multi-site patching, but worth knowing.

**16. `get_sparsity_and_variance_metrics` flattens `[n_layers, d_in] → [n_layers*d_in]`.**
For MSE, cosine similarity, L2 norms, etc.
*Effect:* Per-layer reconstruction quality isn't reported separately. A feature that reconstructs layer 3 well but layer 8 poorly would look "average" in these metrics. Comparison with single-layer SAE metrics is not apples-to-apples — the implicit feature dimension is `n_layers × d_in`.

**17. `get_decoder_directions` for crosscoder = `mean over layers, normalized to unit row`.**
For SAEBench compatibility (which expects `[d_sae, d_in]`).
*Effect:* SAEBench analyses (cosine similarity, absorption, meta-structure) operate on the per-feature mean direction, not the full multi-layer decoder. Geometric structure that exists *between* layers is invisible to SAEBench.

---

## Possible bugs / untested paths

**18. Matryoshka crosscoder: inherited `training_forward_pass` computes `inv_W_dec_norm = 1 / self.W_dec.norm(dim=-1)` assuming 2-D `W_dec`.**
For crosscoder (3-D `W_dec`), `norm(dim=-1)` returns shape `[d_sae, n_layers]`, not `[d_sae]`. The downstream `inv_W_dec_norm[:width] * inner_feature_acts` would shape-mismatch and crash.
*Effect:* **Likely bug**. We never trained Matryoshka crosscoder end-to-end (the validation script uses BatchTopK). If you launch a matryoshka crosscoder run, this will probably error in the inner-level decode. Easy fix: override `training_forward_pass` in `CrossMatryoshkaBatchTopKTrainingSAE` to use `_crosscoder_decoder_norms()`.

**19. Matryoshka crosscoder + factorization untested.**
Same caveat as #18; even more so since CP/TR add another layer of indirection.

**20. JumpReLU encoder pre-existing bug fixed.**
Inheriting `JumpReLUSAE.encode` did a 2-D matmul. We added an override. Mention only because if you ever encounter old saved checkpoints saved as `CrossJumpReLU` and load them with a fresh codebase, the loaded SAE will use the new encode (which is correct). No backward-compat issue, just calling it out.

**30. No test coverage for layer masking, CP, or TR.**
`grep tests/` finds no references to `layer_masking`, `p_mask`, `A_enc`, `G_enc_n`, `use_cross_fact`, or `cross_fact_type`.
*Effect:* Silent regressions on these paths are likely. Highest-leverage tests:
- *Factored forward identity*: materialise `W_enc`/`W_dec` from factors and compare against the factored einsum (shapes + values).
- *Fold identity*: reconstruction before and after `fold_W_dec_norm` matches under `rescale_acts_by_decoder_norm=True` (bit-close for CP, ranking-preserving for TR).
- *Mask invariants*: zero at `p=0`, target-is-unmasked, eval-mode bypass, shape broadcasts over `d_in`.

---

## Local logging

**21. Sanitization stringifies tensors with >1000 elements.**
The `_json_sanitize` helper writes large tensors as `<Tensor shape=(...)>` instead of dumping values, to avoid bloating `metrics.jsonl`.
*Effect:* If you ever log a per-feature density vector or a confusion matrix, the local file won't have the actual numbers — only wandb cloud will.

**22. Drops `sync` kwarg when forwarding to `wandb.run.log`.**
`Run.log` doesn't accept `sync`. No caller in the codebase passes it, so this is moot today, but worth noting if you ever import 3rd-party callbacks that do.

---

## Summary: which choices most affect training quality

If you're trying to debug suspicious training behavior, investigate in this order:

1. **TR cheap-proxy norm (#11)** — biggest deviation, most likely to cause unexpected feature scales
2. **BatchTopK→JumpReLU threshold miscalibration under masking (#25)** — easy to hit if you use both; explains "fires too often at eval"
3. **Layer masking train/eval shift (#6)** — could be a culprit if eval looks much worse than training
4. **Init scale for CP/TR (#8, #12)** — if loss takes oddly long to start decreasing; note factored init is ~2–3× smaller than a true Kaiming slice
5. **Matryoshka crosscoder bug (#18)** — *will* error if exercised
6. **`decoder_init_norm` silently ignored under CP/TR (#10)** — only matters if you set it explicitly
7. **All-masked token degeneracy (#26)** — relevant at high `p` or small `L`
8. **Mask-after-`b_dec` semantics (#4)** — only matters if you observe weird `b_dec` behavior
9. **No test coverage for masking / factorization (#30)** — preventive; failures elsewhere are harder to diagnose without these

Everything else is pretty minor.
