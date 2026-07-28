source /nvme/h/ademou/envs/poly_env/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH="/nvme/h/ademou/envs/poly_env:/nvme/h/ademou/data_aiml/ademou/MN5/code/SAEBench"
export HF_HUB_CACHE="/nvme/h/ademou/data_aiml/ademou/MN5/hf_cache/hub"
export HF_DATASETS_CACHE="/nvme/h/ademou/data_aiml/ademou/MN5/hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

CKPT=/nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/gemma2-2b_m0_05_r32-12-545/final_500002816
RUN=gemma2-2b_m0_05_r32-12-545

# MEANING-READOUT SHOWCASE for the high-coherence latents: insert each feature ONCE
# across the whole crosscoder span and read the tokens it promotes, to show that
# steering causally reproduces what each coherent feature fires on.
#
# --all_layers + --delta_mode increment: the per-layer increments d_L - d_{L-1}
# TELESCOPE (d_0 + sum(d_L - d_{L-1}) = d_last), so the feature is injected exactly
# ONCE with its natural accumulated magnitude, and the partial sum at each layer equals
# the feature's natural level there. This is the faithful "turn the feature on" readout.
# (cumulative + --all_layers would OVER-inject the snapshot ~n_layers times -- avoid.)
# One promoted-token list per feature; no cross-layer Jaccard (that's per_layer only).
# coeffs 4,6,8: top tokens are reported at the largest coeff (8) -- strong enough for a
# crisp concept, below the ~10x point where off-distribution junk tokens creep in.
#
# For per-layer LOCALIZATION (which layer a feature is written at) use the other run:
# --per_layer --delta_mode increment (steering_results/${RUN}/per_layer_increment).
python eval_feature_steering.py \
    --checkpoint_path "$CKPT" \
    --contexts_from "top_bottom_coherence_${RUN}.json" \
    --groups top_most_coherent \
    --n_features 20 \
    --all_layers \
    --coeffs 1,2,4,6,8,10,20 \
    --delta_mode increment \
    --n_batches 10 --batch_size_prompts 8 --context_size 128 \
    --top_k 15 \
    --output_dir "steering_results/${RUN}/alllayers_increment_topcoh" \
    2>&1 | tee "steering_results_${RUN}_alllayers_increment_topcoh.log"