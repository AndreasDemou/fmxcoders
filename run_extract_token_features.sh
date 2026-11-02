#!/bin/bash
#SBATCH --job-name=TokenFeatures
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --time=24:00:00
# Account/partition are cluster-specific: sbatch -A <account> -p <partition> run_extract_token_features.sh


# Run from the repository root (under SLURM: submit from the repo root).
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f env.sh ]; then echo "env.sh not found: cp env.example.sh env.sh and edit it" >&2; exit 1; fi
source ./env.sh

#python extract_token_features.py \
#        --checkpoint $CKPT_ROOT/gemma2-2b_m0/final_500002816 \
#        --n_texts 100_000
#
#python extract_token_features.py \
#        --checkpoint $CKPT_ROOT/gemma2-2b_m0_r32-12-545/final_500002816 \
#        --n_texts 100_000
#
#python extract_token_features.py \
#        --checkpoint $CKPT_ROOT/gemma2-2b_m0_05_r32-12-545/final_500002816 \
#        --n_texts 100_000
#
#python extract_token_features.py \
#        --checkpoint $CKPT_ROOT/gpt2_m0/final_300003328 \
#        --n_texts 100_000
#
#python extract_token_features.py \
#        --checkpoint $CKPT_ROOT/gpt2_m0_05_r25-5-244/final_300003328 \
#        --n_texts 100_000
#
python extract_token_features.py \
        --checkpoint $CKPT_ROOT/pythia1_4b_m0/final_500002816 \
        --n_texts 100_000

python extract_token_features.py \
        --checkpoint $CKPT_ROOT/pythia1_4b_m0_05_r31-11-501/final_500002816 \
        --n_texts 100_000
