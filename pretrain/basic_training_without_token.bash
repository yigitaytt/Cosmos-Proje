#!/bin/bash
#SBATCH -J "Turkish-GPT2-NoMods"   # job name
#SBATCH -A idm001                  # account / project name
#SBATCH -p gpu2dq                  # partition/queue name
#SBATCH -n 64                      # number of cores/processors
#SBATCH -N 1                       # number of nodes
#SBATCH --gres=gpu:1               # additional resource (1 GPU required)

# Start Miniconda
source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate turkish-gpt2-training

# ============================================================================
# CONFIGURATION
# ============================================================================

export TRAINING_WORKSPACE="$HOME"
export JOB_ID="${SLURM_JOB_ID}"

# Model and dataset (LOCAL PATHS)
export MODEL_PATH="${TRAINING_WORKSPACE}/turkish-gpt2-medium"
export DATASET_PATH="${TRAINING_WORKSPACE}/bilgem_dataset.jsonl"

# Output configuration
export OUTPUT_DIR="${TRAINING_WORKSPACE}/jobs/${SLURM_JOB_ID}/output"
export CACHE_DIR="${TRAINING_WORKSPACE}/cache"

# Dataset configuration
export MAX_LENGTH="4096"

# Training hyperparameters
export NUM_EPOCHS="3"
export LEARNING_RATE="5e-5"
export WARMUP_STEPS="2000"
export WEIGHT_DECAY="0.01"
export MAX_GRAD_NORM="1.0"

# Batch size and gradient accumulation
export BATCH_SIZE="16"
export GRADIENT_ACCUM="8"
export DATALOADER_NUM_WORKERS="8"

export EVAL_SPLIT_RATIO="0.001"           # 0.001% for evaluation
export EVAL_STEPS="500"                  # Evaluate every 500 steps
export EVAL_STRATEGY="steps"             # or "epoch"
export LOAD_BEST_MODEL="true"            # Save best model based on eval loss
export METRIC_FOR_BEST_MODEL="loss"

# Mixed precision
export USE_BF16="true"

# Logging and checkpointing
export LOGGING_STEPS="10"
export SAVE_STEPS="500"
export SAVE_TOTAL_LIMIT="2"

# Resume training
export RESUME_FROM_CHECKPOINT="auto"

# Random seed
export SEED="42"

# HuggingFace cache
export HF_HOME="${CACHE_DIR}/huggingface_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# ============================================================================
# SETUP
# ============================================================================

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${CACHE_DIR}"
mkdir -p "${HF_HOME}"

# ============================================================================
# VALIDATION
# ============================================================================

echo "=========================================="
echo "Validating paths..."
echo "=========================================="

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset file not found: ${DATASET_PATH}"
    exit 1
fi

echo "✓ Model path validated: ${MODEL_PATH}"
echo "✓ Dataset path validated: ${DATASET_PATH}"

# ============================================================================
# RUN TRAINING
# ============================================================================

echo ""
echo "=========================================="
echo "Turkish GPT-2 Training (No Tokenizer Mods)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=========================================="
echo "Start time: $(date)"
echo "Working directory: ${TRAINING_WORKSPACE}"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Cache: ${CACHE_DIR}"
echo "=========================================="
echo ""
echo "GPU Information:"
nvidia-smi
echo ""
echo "=========================================="

python basic_training_without_token.py

echo ""
echo "=========================================="
echo "Training job completed at $(date)"
echo "=========================================="