#!/bin/bash
#SBATCH --job-name=coherence
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --time=24:00:00
# Account/partition are cluster-specific: sbatch -A <account> -p <partition> run_coherence_evals.sh

set -e  # stop on first hard error; remove if you want one-checkpoint failures to not abort the rest

# Run from the repository root (under SLURM: submit from the repo root).
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f env.sh ]; then echo "env.sh not found: cp env.example.sh env.sh and edit it" >&2; exit 1; fi
source ./env.sh

# ---- edit this list ----
CKPTS=(
    $CKPT_ROOT/pythia410m_m0/final_500002816
    $CKPT_ROOT/pythia410m_m0_r27-7-302/final_500002816
    $CKPT_ROOT/pythia410m_m0_02_r27-7-302/final_500002816
    $CKPT_ROOT/pythia410m_m0_05_r27-7-302/final_500002816
    $CKPT_ROOT/pythia410m_m0_1_r27-7-302/final_500002816
    $CKPT_ROOT/pythia410m_m0_r19-5-241/final_500002816
    $CKPT_ROOT/pythia410m_m0_02_r19-5-241/final_500002816
    $CKPT_ROOT/pythia410m_m0_05_r19-5-241/final_500002816
    $CKPT_ROOT/pythia410m_m0_1_r19-5-241/final_500002816
    $CKPT_ROOT/pythia410m_m0_r13-3-151/final_500002816
    $CKPT_ROOT/pythia410m_m0_02_r13-3-151/final_500002816
    $CKPT_ROOT/pythia410m_m0_05_r13-3-151/final_500002816
    $CKPT_ROOT/pythia410m_m0_1_r13-3-151/final_500002816
    $CKPT_ROOT/pythia410m_m0_r9-2-107/final_500002816
    $CKPT_ROOT/pythia410m_m0_02_r9-2-107/final_500002816
    $CKPT_ROOT/pythia410m_m0_05_r9-2-107/final_500002816
    $CKPT_ROOT/pythia410m_m0_1_r9-2-107/final_500002816
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

