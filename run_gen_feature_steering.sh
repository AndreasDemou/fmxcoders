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

# Generate steered TEXT continuations for the hand-authored probe prompts in
# steering_prompts.json. For every prompt we produce the greedy continuation with no
# steering (baseline) and with the prompt's parent latent inserted at coeffs 4/6/8,
# plus the next-token probability boost of that latent's target_tokens.
#
# --all_layers + --delta_mode increment injects the feature EXACTLY ONCE across the
# crosscoder span (increments telescope) -- the faithful "turn the feature on" method.
# NOTE: with increment, forgetting --all_layers injects only block-9's increment.
#
# Coeff guidance (GENERATION): coeff scales a_ref, the feature's PEAK activation (~11-26),
# and the vector is added at every position AND every autoregressive step, so the effect
# COMPOUNDS. Coeffs tuned on the single-forward-pass distributional eval (4-8) massively
# over-steer text -- it collapses into repeating the promoted token ("steps steps steps")
# or an off-distribution junk token. For fluent-but-steered generation use FRACTIONS of
# a_ref: sweep ~0.25-2, sweet spot usually ~0.5-1. Optional anti-repetition lever:
# --freq_penalty 0.5-1.0. Sanity control: steer an UNRELATED latent at the same coeff
# (--latents <other_id>) and confirm the effect is direction-specific.

# ---- smoke test first: 2 easy-to-read latents, 2 prompts each ----
# python gen_feature_steering.py \
#     --checkpoint_path "$CKPT" \
#     --contexts_from "top_bottom_coherence_${RUN}.json" \
#     --prompts_from steering_prompts.json \
#     --all_layers --delta_mode increment --coeffs 0.25,0.5,1,2 \
#     --latents 854,9050 --max_prompts_per_latent 2 --max_new_tokens 30 \
#     --output_dir "steering_results/${RUN}/gen_smoke" \
#     2>&1 | tee "gen_steering_${RUN}_smoke.log"

# --steer_positions controls WHERE the steer lands (which claim you're making):
#   all   (default): every position -> sustained ActAdd control; needs LOW coeff or it loops.
#   first          : only the first generated token -> collapse-proof "does the feature
#                    predict the concept next?" readout; robust to high coeff (no compounding).
#   new            : first + every generated token, prompt read left clean.
# Readout demo (can push coeff higher since there's no compounding loop):
# python gen_feature_steering.py \
#     --checkpoint_path "$CKPT" --contexts_from "top_bottom_coherence_${RUN}.json" \
#     --prompts_from steering_prompts.json \
#     --all_layers --delta_mode increment --steer_positions first --coeffs 1,2,4 \
#     --output_dir "steering_results/${RUN}/gen_first"

# ---- full run (sustained control): all 10 latents x 10 prompts x coeffs 0.25,0.5,1,2 ----
python gen_feature_steering.py \
    --checkpoint_path "$CKPT" \
    --contexts_from "top_bottom_coherence_${RUN}.json" \
    --prompts_from steering_prompts.json \
    --all_layers --delta_mode increment --steer_position all --coeffs 0.5,1,2,4,8 \
    --max_new_tokens 40 \
    --output_dir "steering_results/${RUN}/gen_all" \
    2>&1 | tee "gen_steering_${RUN}.log"

    #--all_layers --delta_mode increment --steer_position first --coeffs 2,4,8,16 \
