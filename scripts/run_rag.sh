#!/bin/bash

# Set error handling
set -e

# Define logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check Python environment
if ! command -v python &> /dev/null; then
    log "Error: Python is not installed"
    exit 1
fi

# Check required directories and files
if [ ! -d "FinLLM-RAG/inference" ]; then
    log "Error: RAG inference directory not found"
    exit 1
fi

# Set default parameters
MODEL_PATH="FinLLM-Instruction-tuning/model_lora"
QUERY_FILE="FinLLM-Instruction-tuning/data/validation_data.jsonl"
KNOWLEDGE_BASE="FinLLM-RAG/data/phrasebank_75_agree.json"
OUTPUT_DIR="FinLLM-RAG/results"
TOP_K=3

# Parse command line arguments
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
        --knowledge_base)
            KNOWLEDGE_BASE="$2"
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

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Start RAG inference
log "Starting RAG inference..."
log "Model path: $MODEL_PATH"
log "Query file: $QUERY_FILE"
log "Knowledge base: $KNOWLEDGE_BASE"
log "Output directory: $OUTPUT_DIR"
log "Top-K: $TOP_K"

python FinLLM-RAG/inference/rag_retrieve_and_infer.py \
    --model_path "$MODEL_PATH" \
    --query_file "$QUERY_FILE" \
    --knowledge_base "$KNOWLEDGE_BASE" \
    --output_dir "$OUTPUT_DIR" \
    --top_k "$TOP_K" \
    2>&1 | tee "logs/rag_$(date +'%Y%m%d_%H%M%S').log"

if [ $? -eq 0 ]; then
    log "RAG inference completed successfully!"
else
    log "Error: RAG inference failed"
    exit 1
fi 