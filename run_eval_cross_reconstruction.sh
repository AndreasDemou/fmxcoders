#!/bin/bash
#SBATCH --job-name=CrossRecon
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --time=24:00:00
# Account/partition are cluster-specific: sbatch -A <account> -p <partition> run_eval_cross_reconstruction.sh

# Run from the repository root (under SLURM: submit from the repo root).
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f env.sh ]; then echo "env.sh not found: cp env.example.sh env.sh and edit it" >&2; exit 1; fi
source ./env.sh

BASE="$CKPT_ROOT"

RUNS=(
    pythia410m_topk
    pythia410m_Lall
    pythia410m_L12
    pythia410m_L4
    pythia410m_k128
    pythia410m_k32
    pythia410m_d32k
    pythia410m_d8k
    pythia410m_attn
    #pythia410m_seed1
    #pythia410m_seed2
    #pythia410m_seed3
    #pythia410m_seed4
    #gpt2_seed1
    #gpt2_seed2
    #gpt2_seed3
    #gpt2_seed4
    #gemma2-2b_m0_r16153
    #gemma2-2b_m0_05_r16153
    #gemma2-2b_m0_1_r16153
    #gemma2-2b_m0_2_r16153
    #pythia410m_m0_r7707
    #pythia410m_m0_05_r7707
    #pythia410m_m0_1_r7707
    #pythia410m_m0_2_r7707
    #pythia1_4b_m0_r14557
    #pythia1_4b_m0_05_r14557
    #pythia1_4b_m0_1_r14557
    #pythia1_4b_m0_2_r14557
)

for name in "${RUNS[@]}"; do
    ckpt=$(echo "$BASE/$name"/final_*)   # expands to the one final_<step> dir
    if [[ ! -d "$ckpt" ]]; then
        echo "SKIP: $name (no final_* checkpoint)"
        continue
    fi
    python eval_crosscoder_reconstruction.py \
        --checkpoint_path "$ckpt" \
        --n_batches 2000 \
        --batch_size_prompts 16 \
        --context_size 128 \
        --out "eval_results/${name}/recon_metrics.json"
done


#for ckpt in $CKPT_ROOT/pythia410m_m*_r*matched/final_*; do
#    name=$(basename $(dirname "$ckpt"))
#    python eval_crosscoder_reconstruction.py \
#        --checkpoint_path "$ckpt" \
#        --n_batches 2000 \
#        --batch_size_prompts 16 \
#        --context_size 128 \
#        --out "eval_results/${name}/recon_metrics.json"
#done

