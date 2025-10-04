#!/bin/bash
# Multi-GPU training script using Hugging Face TRL
#
# This script automatically uses all available GPUs for distributed training.
# Works with any number of GPUs (1, 2, 4, 8, etc.)

set -x

# Configuration
MODEL_NAME=${1:-"Qwen/Qwen3-30B-A3B-Instruct-2507"}
OUTPUT_DIR=${2:-"./models/prompt_distillation_trl"}
TRAIN_FILE=${3:-"./data/prompt_distillation_lang.jsonl"}

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "============================================"
echo "Multi-GPU Training with TRL"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Train file: $TRAIN_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "============================================"
echo ""

# Check if training file exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Training file not found: $TRAIN_FILE"
    echo "Please run data generation first:"
    echo "  python create_data.py"
    exit 1
fi

# Multi-GPU training using torchrun
# torchrun automatically handles distributed training setup
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    train_sft_trl.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "$TRAIN_FILE" \
    --use_lora \
    --lora_rank 32 \
    --lora_alpha 16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_length 2048 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine_with_min_lr

echo ""
echo "============================================"
echo "✅ Training Complete!"
echo "============================================"
echo "Model saved to: $OUTPUT_DIR"
echo "Number of GPUs used: $NUM_GPUS"
echo ""
echo "Effective batch size: $((4 * 4 * NUM_GPUS))"
echo "  = per_device_batch_size (4)"
echo "  × gradient_accumulation_steps (4)"
echo "  × num_gpus ($NUM_GPUS)"
echo ""
echo "To test the model, run:"
echo "  python evaluate_trl.py --model_path $OUTPUT_DIR"
echo "============================================"

