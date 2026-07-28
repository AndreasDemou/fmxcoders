#!/bin/bash
#SBATCH --job-name=TokenFeatures
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
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

#python extract_token_features.py \
#        --checkpoint /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/gemma2-2b_m0/final_500002816 \
#        --n_texts 100_000
#
#python extract_token_features.py \
#        --checkpoint /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/gemma2-2b_m0_r32-12-545/final_500002816 \
#        --n_texts 100_000
#
#python extract_token_features.py \
#        --checkpoint /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/gemma2-2b_m0_05_r32-12-545/final_500002816 \
#        --n_texts 100_000
#
#python extract_token_features.py \
#        --checkpoint /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/gpt2_m0/final_300003328 \
#        --n_texts 100_000
#
#python extract_token_features.py \
#        --checkpoint /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/gpt2_m0_05_r25-5-244/final_300003328 \
#        --n_texts 100_000
#
python extract_token_features.py \
        --checkpoint /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia1_4b_m0/final_500002816 \
        --n_texts 100_000

python extract_token_features.py \
        --checkpoint /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/pythia1_4b_m0_05_r31-11-501/final_500002816 \
        --n_texts 100_000
