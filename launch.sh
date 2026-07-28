#!/bin/bash
#SBATCH --job-name=g27
#SBATCH --account=aiml
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --partition=rtx
#SBATCH --time=24:00:00
###SBATCH --reservation=neurips


echo "--- GPU allocated for $1 ."
echo "Starting at `date`"
echo "Running on hosts: $SLURM_JOB_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Job name: $SLURM_JOB_NAME"
echo "Job id: $SLURM_JOBID"

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
export WANDB_MODE=offline
export WANDB_DIR="/nvme/h/ademou/data_aiml/ademou/MN5"
export WANDB_CACHE_DIR="/nvme/h/ademou/data_aiml/ademou/MN5"
export WANDB_DATA_DIR="/nvme/h/ademou/data_aiml/ademou/MN5"
export PYTHONUNBUFFERED=1
# Reduce CUDA fragmentation (helps reclaim reserved-but-unallocated VRAM on the big crosscoder).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --eval_only: load the trained SAE from --checkpoint_path and run evals only.
EVAL_ONLY_FLAGS="--eval_only \
  --skip_eval tpp --skip_eval scr \
  --skip_eval ravel --skip_eval absorption"
  #--skip_eval l0_loss \
  #--skip_eval probing \

FAILED=""
CKPT_ROOT=/nvme/h/ademou/data_aiml/ademou/MN5/checkpoints

# ---------------------------------------------------------------------------
# resume_flag <checkpoint_dir>
# Generalized auto-resume for TRAINING runs. Prints "--resume_from_checkpoint"
# on STDOUT when the checkpoint dir already holds a snapshot (so a requeued job
# continues from the latest one), and nothing on a fresh run. Progress messages
# go to STDERR so they are NOT captured when the function is called via $(...).
# Usage — same structure for every model (resume needs --wandb_run_name):
#   RUN_NAME=my_run ; CKPT=$CKPT_ROOT/$RUN_NAME
#   python train_and_saebench.py ... \
#     --checkpoint_path $CKPT --wandb_run_name $RUN_NAME $(resume_flag "$CKPT")
# In --eval_only blocks it is a harmless no-op: eval mode exits before the
# resume handler, so the emitted flag is simply ignored.
# ---------------------------------------------------------------------------
resume_flag() {
  local ckpt_dir="$1"
  if compgen -G "$ckpt_dir/*/sae_weights.safetensors" > /dev/null; then
    echo "[resume] Found snapshot(s) in $ckpt_dir — resuming from the latest." >&2
    echo "--resume_from_checkpoint"
  else
    echo "[resume] No snapshot in $ckpt_dir — starting a fresh run." >&2
  fi
}

#RUN_NAME=pythia410m_Lall
#CKPT=$CKPT_ROOT/$RUN_NAME
#python train_and_saebench.py --exp pythia410m_l15 --architecture batchtopk \
#  --width 16k --k 64 --training_tokens 500000000 --context_size 128 \
#  --no_rescale_by_decoder_norm \
#  --checkpoint_path $CKPT \
#  --use_crosscoder --cross_layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
#  --layer_masking 0.05 \
#  --use_cross_fact --cross_fact_type TR --cross_fact_ranks 27,7,302 \
#  --use_saelens_defaults \
#  --seed 42 \
#  --run_saebench \
#  --wandb_run_name $RUN_NAME \
#  $(resume_flag "$CKPT") \
#  $EVAL_ONLY_FLAGS
#[ $? -ne 0 ] && FAILED="$FAILED $RUN_NAME"
#  #--use_cross_fact --cross_fact_type TR --cross_fact_ranks 9,2,107 \
#  #--use_cross_fact --cross_fact_type CP --cross_fact_ranks 7707 \

#RUN_NAME=gpt2_m0_2_r5866
#CKPT=$CKPT_ROOT/$RUN_NAME
#python train_and_saebench.py --exp gpt2_l8 --architecture batchtopk \
#  --width 16k --k 64 --training_tokens 300000000 --context_size 128 \
#  --no_rescale_by_decoder_norm \
#  --checkpoint_path $CKPT \
#  --use_crosscoder --cross_layers 2,3,4,5,6,7,8,9 \
#  --layer_masking 0.2 \
#  --use_cross_fact --cross_fact_type CP --cross_fact_ranks 5866 \
#  --use_saelens_defaults \
#  --wandb_run_name $RUN_NAME \
#  --run_saebench \
#  $(resume_flag "$CKPT") \
#  $EVAL_ONLY_FLAGS
#[ $? -ne 0 ] && FAILED="$FAILED $RUN_NAME"
#  #--checkpoint_path $CKPT_ROOT/gpt2_m0_2_r25-5-244 \
#  #--use_cross_fact --cross_fact_type TR --cross_fact_ranks 25,5,244 \
#

#RUN_NAME=pythia1_4b_seed1
#CKPT=$CKPT_ROOT/$RUN_NAME
#python train_and_saebench.py --exp pythia1_4b_l12 --architecture batchtopk \
#  --width 16k --k 64 --training_tokens 500000000 --context_size 128 \
#  --no_rescale_by_decoder_norm \
#  --checkpoint_path $CKPT \
#  --use_crosscoder --cross_layers 8,9,10,11,12,13,14,15 \
#  --layer_masking 0.05 \
#  --use_cross_fact --cross_fact_type TR --cross_fact_ranks 31,11,501 \
#  --use_saelens_defaults \
#  --wandb_run_name $RUN_NAME \
#  --run_saebench \
#  --seed 1042 \
#  $EVAL_ONLY_FLAGS
#  #$(resume_flag "$CKPT")
#[ $? -ne 0 ] && FAILED="$FAILED $RUN_NAME"
#  #$EVAL_ONLY_FLAGS
#  #--use_cross_fact --cross_fact_type TR --cross_fact_ranks 31,11,501 \
#  #--use_cross_fact --cross_fact_type CP --cross_fact_ranks 14557 \
#
RUN_NAME=gemma2-2b_seed4
CKPT=$CKPT_ROOT/$RUN_NAME
python train_and_saebench.py --exp gemma2_2b_l19 --architecture batchtopk \
  --width 16k --k 64 --training_tokens 500000000 --context_size 128 \
  --no_rescale_by_decoder_norm \
  --checkpoint_path $CKPT \
  --use_crosscoder --cross_layers 9,10,11,12,13,14,15,16 \
  --layer_masking 0.05 \
  --use_cross_fact --cross_fact_type TR --cross_fact_ranks 32,12,545 \
  --use_saelens_defaults \
  --act_store_device cpu \
  --wandb_run_name $RUN_NAME \
  --run_saebench \
  --seed 4042 \
  $(resume_flag "$CKPT") 
[ $? -ne 0 ] && FAILED="$FAILED $RUN_NAME"
#  #--use_cross_fact --cross_fact_type TR --cross_fact_ranks 32,12,545 \
#  #--use_cross_fact --cross_fact_type CP --cross_fact_ranks 16153 \
#  #$EVAL_ONLY_FLAGS

# ============================================================================
# Gemma-2-27b crosscoder TRAINING. Auto-resumes from the latest snapshot on
# requeue via resume_flag() (defined at top). RUN_NAME drives the checkpoint
# dir, the WandB run name, and the resume detection — keep them in sync.
#RUN_NAME=gemma2-27b-cross
#RUN_NAME=gemma2-27b-crossRUN_NAME=gemma2-27b
#CKPT=$CKPT_ROOT/$RUN_NAME
#python train_and_saebench.py --exp gemma2_27b_l22 --architecture batchtopk \
#  --width 16k --k 64 --training_tokens 500000000 --context_size 128 \
#  --n_checkpoints 20 \
#  --no_rescale_by_decoder_norm \
#  --checkpoint_path $CKPT \
#  --model_dtype bfloat16 \
#  --use_crosscoder --cross_layers 20,21,22,23,24,25,26,27 \
#  --layer_masking 0.05 \
#  --use_cross_fact --cross_fact_type TR --cross_fact_ranks 37,19,877 \
#  --run_saebench --use_saelens_defaults \
#  --act_store_device cpu \
#  --store_batch_size_prompts 8 \
#  --eval_batch_size_prompts 2 \
#  --train_batch_size_tokens 512 \
#  --wandb_run_name $RUN_NAME \
#  $(resume_flag "$CKPT")
#[ $? -ne 0 ] && FAILED="$FAILED $RUN_NAME"
#
#  --layer_masking 0.05 \
#  --use_cross_fact --cross_fact_type TR --cross_fact_ranks 37,19,877 \

# Llama was set up with a different dataset 
# LEFT DISABLED: $CKPT_ROOT/llama32-3b_m0_2_r34-15-667 does not exist —
# this SAE was never trained (or its checkpoint was deleted). Nothing to eval.
#RUN_NAME=llama32-3b_m0_2_r34-15-667
#CKPT=$CKPT_ROOT/$RUN_NAME
#python train_and_saebench.py --exp llama32_3b_l12 --architecture batchtopk \
#  --width 16k --k 64 --training_tokens 500000000 --context_size 128 \
#  --no_rescale_by_decoder_norm \
#  --checkpoint_path $CKPT \
#  --use_crosscoder --cross_layers 10,11,12,13,14,15,16,17 \
#  --layer_masking 0.2 \
#  --use_cross_fact --cross_fact_type TR --cross_fact_ranks 34,15,667 \
#  --use_saelens_defaults \
#  --act_store_device cpu \
#  --wandb_run_name $RUN_NAME \
#  $(resume_flag "$CKPT") \
#  $EVAL_ONLY_FLAGS
#[ $? -ne 0 ] && FAILED="$FAILED $RUN_NAME"

if [ -n "$FAILED" ]; then
  echo "=== FAILED RUNS:$FAILED ==="
  RC=1
else
  echo "=== All runs produced results ==="
  RC=0
fi
echo "Program finished with exit code $RC at: `date`"
# Propagate the exit code so SLURM marks the job FAILED when an eval produced
# no results (a bare trailing `echo` would always exit 0).
exit $RC
