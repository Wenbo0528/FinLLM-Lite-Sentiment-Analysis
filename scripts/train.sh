#!/bin/bash

# 设置错误处理
set -e

# 定义日志函数
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# 检查Python环境
if ! command -v python &> /dev/null; then
    log "Error: Python is not installed"
    exit 1
fi

# 检查必要的目录和文件
if [ ! -d "FinLLM-Instruction-tuning/train" ]; then
    log "Error: Training directory not found"
    exit 1
fi

# 设置默认参数
MODEL_NAME="default_model"
BATCH_SIZE=8
LEARNING_RATE=2e-5
EPOCHS=3

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            log "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# 创建日志目录
mkdir -p logs

# 开始训练
log "Starting training process..."
log "Model: $MODEL_NAME"
log "Batch size: $BATCH_SIZE"
log "Learning rate: $LEARNING_RATE"
log "Epochs: $EPOCHS"

python FinLLM-Instruction-tuning/train/train.py \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    2>&1 | tee "logs/train_$(date +'%Y%m%d_%H%M%S').log"

if [ $? -eq 0 ]; then
    log "Training completed successfully!"
else
    log "Error: Training failed"
    exit 1
fi 