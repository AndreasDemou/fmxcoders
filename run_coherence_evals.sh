#!/bin/bash
#SBATCH --job-name=coherence
#SBATCH --account=aiml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --partition=rtx
#SBATCH --time=24:00:00

set -e  # stop on first hard error; remove if you want one-checkpoint failures to not abort the rest

source /nvme/h/ademou/envs/poly_env/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH="/nvme/h/ademou/envs/poly_env:/nvme/h/ademou/data_aiml/ademou/MN5/code/SAEBench"
export HF_HUB_CACHE="/nvme/h/ademou/data_aiml/ademou/MN5/hf_cache/hub"
export HF_DATASETS_CACHE="/nvme/h/ademou/data_aiml/ademou/MN5/hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# ---- edit this list ----
CKPTS=(
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_r27-7-302/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_02_r27-7-302/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_05_r27-7-302/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_1_r27-7-302/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_r19-5-241/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_02_r19-5-241/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_05_r19-5-241/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_1_r19-5-241/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_r13-3-151/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pthia410m_m0_02_r13-3-151/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_05_r13-3-151/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_1_r13-3-151/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_r9-2-107/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_02_r9-2-107/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_05_r9-2-107/final_500002816
    /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia410m_m0_1_r9-2-107/final_500002816
)

OUT_ROOT="coherence_results"
# ------------------------

mkdir -p "$OUT_ROOT"

for ckpt in "${CKPTS[@]}"; do
    name=$(basename $(dirname "$ckpt"))
    out_dir="$OUT_ROOT/$name"
    mkdir -p "$out_dir"

    echo ""
    echo "============================================================"
    echo "  $name"
    echo "  ckpt: $ckpt"
    echo "============================================================"

    # --- Norm coherence: decoder weights ---
    echo ""
    echo "[$name] norm coherence (decoder)"
    python eval_norm_coherence.py \
        --checkpoint_path "$ckpt" \
        --output_dir "$out_dir/norm_decoder" \
        --device cpu \
        2>&1 | tee "$out_dir/norm_decoder.log"

    # --- Norm coherence: encoder weights ---
    echo ""
    echo "[$name] norm coherence (encoder)"
    python eval_norm_coherence.py \
        --checkpoint_path "$ckpt" \
        --output_dir "$out_dir/norm_encoder" \
        --use_encoder \
        --device cpu \
        2>&1 | tee "$out_dir/norm_encoder.log"

    # --- Mask coherence: zero ablation ---
    echo ""
    echo "[$name] mask coherence (zero ablation)"
    python eval_mask_coherence.py \
        --checkpoint_path "$ckpt" \
        --output_dir "$out_dir/mask_zero" \
        --masking_strategy zero \
        --n_batches 1000 --batch_size_prompts 16 --context_size 128 \
        --device cuda \
        2>&1 | tee "$out_dir/mask_zero.log"

    # --- Mask coherence: mean ablation ---
    #echo ""
    #echo "[$name] mask coherence (mean ablation)"
    #python eval_mask_coherence.py \
    #    --checkpoint_path "$ckpt" \
    #    --output_dir "$out_dir/mask_mean" \
    #    --masking_strategy mean \
    #    --n_batches 100 --batch_size_prompts 16 --context_size 128 \
    #    --device cuda \
    #    2>&1 | tee "$out_dir/mask_mean.log"
done

echo ""
echo "============================================================"
echo "All coherence evals complete. Results under $OUT_ROOT/"
echo "============================================================"

