# Environment for the run_*.sh scripts.
#   cp env.example.sh env.sh   # then edit env.sh (it is git-ignored)
#
# SLURM account/partition are cluster-specific, so they are not set in the scripts.
# Pass them when submitting:  sbatch -A <account> -p <partition> run_coherence_evals.sh
# or export SBATCH_ACCOUNT / SBATCH_PARTITION in your shell.

# Python virtualenv to activate (leave empty to use the currently active one)
VENV=""
if [ -n "$VENV" ]; then source "$VENV/bin/activate"; fi

# Folder holding trained runs: $CKPT_ROOT/<run_name>/final_<n_tokens>/
export CKPT_ROOT="${CKPT_ROOT:-$PWD/checkpoints}"

# Import sae_bench from the fork bundled in this repo
export PYTHONPATH="$PWD/SAEBench${PYTHONPATH:+:$PYTHONPATH}"

# Hugging Face caches
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"

# Set to 1 on compute nodes without internet access (download models/datasets beforehand)
HF_OFFLINE=0
if [ "$HF_OFFLINE" = 1 ]; then
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
