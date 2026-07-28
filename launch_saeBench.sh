#!/bin/bash
#SBATCH --job-name=pp0
#SBATCH --account=aiml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --partition=gpu
#SBATCH --reservation=simea
#SBATCH --exclusive

export HF_HUB_CACHE="/nvme/h/ademou/data_aiml/ademou/PolySAE_dev/hub"
export HF_DATASETS_CACHE="/nvme/h/ademou/data_aiml/ademou/PolySAE_dev/hub"
#export PYTHONNOUSERSITE=1  # Keep ~/.local out of it
#export HF_HUB_OFFLINE=1
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
#export HF_HUB_DISABLE_TELEMETRY=1
#export HF_HUB_ENABLE_HF_TRANSFER=0
#export WANDB_MODE=offline
#export WANDB_DIR="/gpfs/projects/ehpc289/saeLens_modified/PolySAE_dev/"
#export WANDB_CACHE_DIR="/gpfs/projects/ehpc289/saeLens_modified/PolySAE_dev/"
#export PYTHONUNBUFFERED=1

source load_modules.sh
#export WANDB_API_KEY=69cfc86a959a671df4409912f46418fd7c475bd6
#export WANDB_SERVICE_WAIT=300
wandb login 

# Vanilla
#python train_and_saebench.py --exp gemma2_2b_l12 --architecture matryoshka \
#  --eval_only \
#  --width 16k --k 64 --training_tokens 0 --context_size 128 \
#  --no_rescale_by_decoder_norm --n_batches_in_buffer 32 --train_batch_size_tokens 4096 \
#  --wandb_run_name topk_gemma2 --run_saebench --use_saelens_defaults \
#  --checkpoint_path /nvme/h/ademou/data_aiml/koromilas/repos/PolySAE_dev/checkpoints/matryoshka_gemma2/final_500002816/ \
#  --resume_from_checkpoint \
#  --sae_name vanilla_matryoshka_gemma2 \
#  --skip_eval l0_loss \
#  --skip_eval sparse_probing \
#  --skip_eval tpp \
#  --skip_eval scr \
#  --skip_eval ravel

#mv eval_results/final_500002816/* eval_results/final_500002816_vanilla_matryoshka/.

# PolySAE
python train_and_saebench.py --exp gemma2_2b_l12 --architecture matryoshka \
  --eval_only \
  --width 16k --k 64 --training_tokens 500000000 --context_size 128 \
  --no_rescale_by_decoder_norm --n_batches_in_buffer 32 --train_batch_size_tokens 4096 \
  --wandb_run_name poly_o3_r2304_128_128_topk_gemma2 --run_saebench --use_saelens_defaults \
  --poly --shared_u --poly_order 3 --poly_ranks 2304,128,128 \
  --checkpoint_path /nvme/h/ademou/data_aiml/koromilas/repos/PolySAE_dev/checkpoints/poly_o3_r2304_128_128_matryoshka_gemma2/final_500002816/ \
  --resume_from_checkpoint \
  --sae_name poly_matryoshka_gemma2 \
  --skip_eval l0_loss \
  --skip_eval sparse_probing \
  --skip_eval tpp \
  --skip_eval scr \
  --skip_eval ravel
 
#mv eval_results/final_500002816/* eval_results/final_500002816_poly_matryoshka/.

# Vanilla
python train_and_saebench.py --exp gemma2_2b_l12 --architecture batchtopk \
  --eval_only \
  --width 16k --k 64 --training_tokens 0 --context_size 128 \
  --no_rescale_by_decoder_norm --n_batches_in_buffer 32 --train_batch_size_tokens 4096 \
  --wandb_run_name topk_gemma2 --run_saebench --use_saelens_defaults \
  --checkpoint_path /nvme/h/ademou/data_aiml/koromilas/repos/PolySAE_dev/checkpoints/batchtopk_gemma2/final_500002816/ \
  --resume_from_checkpoint \
  --sae_name vanilla_batchtopk_gemma2 \
  --skip_eval l0_loss \
  --skip_eval sparse_probing \
  --skip_eval tpp \
  --skip_eval scr \
  --skip_eval ravel

#mv eval_results/final_500002816/* eval_results/final_500002816_vanilla_batchtopk/.

# PolySAE
python train_and_saebench.py --exp gemma2_2b_l12 --architecture batchtopk \
  --eval_only \
  --width 16k --k 64 --training_tokens 500000000 --context_size 128 \
  --no_rescale_by_decoder_norm --n_batches_in_buffer 32 --train_batch_size_tokens 4096 \
  --wandb_run_name poly_o3_r2304_128_128_topk_gemma2 --run_saebench --use_saelens_defaults \
  --poly --shared_u --poly_order 3 --poly_ranks 2304,128,128 \
  --checkpoint_path /nvme/h/ademou/data_aiml/koromilas/repos/PolySAE_dev/checkpoints/poly_o3_r2304_128_128_batchtopk_gemma2/final_500002816/ \
  --resume_from_checkpoint \
  --sae_name poly_batchtopk_gemma2 \
  --skip_eval l0_loss \
  --skip_eval sparse_probing \
  --skip_eval tpp \
  --skip_eval scr \
  --skip_eval ravel
 
#mv eval_results/final_500002816/* eval_results/final_500002816_poly_batchtopk/.