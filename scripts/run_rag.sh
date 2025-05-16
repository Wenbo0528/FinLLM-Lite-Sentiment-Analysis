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
if [ ! -d "FinLLM-RAG/inference" ]; then
    log "Error: RAG inference directory not found"
    exit 1
fi

# 设置默认参数
MODEL_PATH="FinLLM-Instruction-tuning/model_lora"
QUERY_FILE="FinLLM-RAG/data/queries.txt"
OUTPUT_DIR="FinLLM-RAG/results"
TOP_K=3

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --query_file)
            QUERY_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
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

# 开始RAG推理
log "Starting RAG inference..."
log "Model path: $MODEL_PATH"
log "Query file: $QUERY_FILE"
log "Output directory: $OUTPUT_DIR"
log "Top-K: $TOP_K"

python FinLLM-RAG/inference/rag_retrieve_and_infer.py \
    --model_path "$MODEL_PATH" \
    --query_file "$QUERY_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --top_k "$TOP_K" \
    2>&1 | tee "logs/rag_$(date +'%Y%m%d_%H%M%S').log"

if [ $? -eq 0 ]; then
    log "RAG inference completed successfully!"
else
    log "Error: RAG inference failed"
    exit 1
fi 