#!/bin/bash
#SBATCH --job-name=ravel
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

#python compare_ravel.py \
#        --checkpoint /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/gemma2-2b_m0/final_500002816 \
#	--layer-idx 4
#
#python compare_ravel.py \
#        --checkpoint /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/gemma2-2b_m0_r32-12-545/final_500002816 \
#	--layer-idx 4
#
python compare_ravel.py \
        --checkpoint /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/gemma2-2b_m0_05_r32-12-545/final_500002816 \
	--layer-idx 4

python compare_ravel.py \
        --checkpoint /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/gemma2-2b_m0_r16153/final_500002816 \
	--layer-idx 4

python compare_ravel.py \
        --checkpoint /nvme/h/ademou/data_aiml/ademou/MN5/checkpoints/gemma2-2b_m0_05_r16153/final_500002816 \
	--layer-idx 4
