#!/bin/bash
#SBATCH --job-name=CrossRecon
#SBATCH --account=aiml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --partition=rtx
#SBATCH --time=24:00:00

source /nvme/h/ademou/envs/poly_env/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH="/nvme/h/ademou/envs/poly_env:/nvme/h/ademou/data_aiml/ademou/MN5/code/SAEBench"
export HF_HUB_CACHE="/nvme/h/ademou/data_aiml/ademou/MN5/hf_cache/hub"
export HF_DATASETS_CACHE="/nvme/h/ademou/data_aiml/ademou/MN5/hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

BASE=/nvme/h/ademou/data_aiml/ademou/MN5/checkpoints

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


#for ckpt in /gpfs/scratch/ehpc289/NeurIPS2026/checkpoints/pythia410m_m*_r*matched/final_*; do
#    name=$(basename $(dirname "$ckpt"))
#    python eval_crosscoder_reconstruction.py \
#        --checkpoint_path "$ckpt" \
#        --n_batches 2000 \
#        --batch_size_prompts 16 \
#        --context_size 128 \
#        --out "eval_results/${name}/recon_metrics.json"
#done

