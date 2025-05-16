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
if [ ! -d "FinLLM-Instruction-tuning/Inference" ]; then
    log "Error: Inference directory not found"
    exit 1
fi

# 设置默认参数
MODEL_PATH="FinLLM-Instruction-tuning/model_lora"
INPUT_FILE="FinLLM-Instruction-tuning/data/test_queries.txt"
OUTPUT_DIR="FinLLM-Instruction-tuning/results"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            log "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# 开始推理
log "Starting inference process..."
log "Model path: $MODEL_PATH"
log "Input file: $INPUT_FILE"
log "Output directory: $OUTPUT_DIR"

python FinLLM-Instruction-tuning/Inference/inference.py \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "logs/inference_$(date +'%Y%m%d_%H%M%S').log"

if [ $? -eq 0 ]; then
    log "Inference completed successfully!"
    log "Results saved to: $OUTPUT_DIR"
else
    log "Error: Inference failed"
    exit 1
fi 