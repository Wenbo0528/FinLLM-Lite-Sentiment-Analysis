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
    log "Error: Evaluation directory not found"
    exit 1
fi

# 设置默认参数
PREDICTIONS_DIR="FinLLM-RAG/results"
GROUND_TRUTH_FILE="FinLLM-RAG/data/ground_truth.json"
METRICS_FILE="FinLLM-RAG/results/metrics.json"
EVAL_TYPE="all"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --predictions_dir)
            PREDICTIONS_DIR="$2"
            shift 2
            ;;
        --ground_truth_file)
            GROUND_TRUTH_FILE="$2"
            shift 2
            ;;
        --metrics_file)
            METRICS_FILE="$2"
            shift 2
            ;;
        --eval_type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        *)
            log "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p "$(dirname "$METRICS_FILE")"
mkdir -p logs

# 开始评估
log "Starting evaluation process..."
log "Predictions directory: $PREDICTIONS_DIR"
log "Ground truth file: $GROUND_TRUTH_FILE"
log "Metrics file: $METRICS_FILE"
log "Evaluation type: $EVAL_TYPE"

python FinLLM-RAG/inference/evaluate_rag_results.py \
    --predictions_dir "$PREDICTIONS_DIR" \
    --ground_truth_file "$GROUND_TRUTH_FILE" \
    --metrics_file "$METRICS_FILE" \
    --eval_type "$EVAL_TYPE" \
    2>&1 | tee "logs/evaluate_$(date +'%Y%m%d_%H%M%S').log"

if [ $? -eq 0 ]; then
    log "Evaluation completed successfully!"
    log "Results saved to: $METRICS_FILE"
else
    log "Error: Evaluation failed"
    exit 1
fi 