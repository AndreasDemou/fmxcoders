# --eval_only: load the trained SAE from --checkpoint_path and run evals only.
EVAL_ONLY_FLAGS="--eval_only \
  --skip_eval tpp --skip_eval scr \
  --skip_eval ravel --skip_eval absorption"
  #--skip_eval l0_loss \
  #--skip_eval probing \

FAILED=""
CKPT_ROOT= {checkpoint root}

resume_flag() {
  local ckpt_dir="$1"
  if compgen -G "$ckpt_dir/*/sae_weights.safetensors" > /dev/null; then
    echo "[resume] Found snapshot(s) in $ckpt_dir — resuming from the latest." >&2
    echo "--resume_from_checkpoint"
  else
    echo "[resume] No snapshot in $ckpt_dir — starting a fresh run." >&2
  fi
}

RUN_NAME=pythia410m
CKPT=$CKPT_ROOT/$RUN_NAME
python train_and_saebench.py --exp pythia410m_l15 --architecture batchtopk \
  --width 16k --k 64 --training_tokens 500000000 --context_size 128 \
  --no_rescale_by_decoder_norm \
  --checkpoint_path $CKPT \
  --use_crosscoder --cross_layers 8,9,10,11,12,13,14,15 \
  --layer_masking 0.05 \
  --use_cross_fact --cross_fact_type TR --cross_fact_ranks 27,7,302 \
  --use_saelens_defaults \
  --seed 42 \
  --run_saebench \
  --wandb_run_name $RUN_NAME \
  $(resume_flag "$CKPT") 
[ $? -ne 0 ] && FAILED="$FAILED $RUN_NAME"
#  $EVAL_ONLY_FLAGS
#  #--use_cross_fact --cross_fact_type CP --cross_fact_ranks 7707 \

if [ -n "$FAILED" ]; then
  echo "=== FAILED RUNS:$FAILED ==="
  RC=1
else
  echo "=== All runs produced results ==="
  RC=0
fi

echo "Program finished with exit code $RC at: `date`"
exit $RC
