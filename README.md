# FinLLM-Sentiment-Analysis

A financial text sentiment analysis system based on large language models, combining instruction tuning and Retrieval-Augmented Generation (RAG) technologies to provide more accurate and reliable financial text sentiment analysis capabilities.

## Project Structure

```
FinLLM-Sentiment-Analysis/
├── FinLLM-Instruction-tuning/  # Instruction tuning module
│   ├── data/                  # Training data
│   ├── train/                # Training related code
│   ├── Inference/           # Inference related code
│   └── model_lora/          # Fine-tuned model
├── FinLLM-RAG/              # RAG enhancement module
│   ├── data/               # Knowledge base data
│   ├── inference/         # RAG inference code
│   └── results/          # Output results
└── scripts/              # Common scripts
    ├── train.sh         # Training script
    ├── train.bat       # Windows training script
    ├── inference.sh    # Inference script
    ├── inference.bat   # Windows inference script
    ├── run_rag.sh      # RAG inference script
    └── run_rag.bat     # Windows RAG inference script
```

## Module Description

### 1. FinLLM-Instruction-tuning

A model fine-tuning module based on instruction tuning and LoRA (Low-Rank Adaptation) technology. It enhances the model's understanding of financial text sentiment through carefully designed instruction templates and financial domain data.

Key Features:
- Uses DeepSeek-R1-Distill-Qwen-1.5B as the base model
- Employs LoRA technology for efficient fine-tuning
- Supports 4-bit quantization training
- Provides complete training and inference scripts

For detailed information, please refer to [FinLLM-Instruction-tuning/README.md](FinLLM-Instruction-tuning/README.md)

### 2. FinLLM-RAG

A sentiment analysis enhancement module based on Retrieval-Augmented Generation (RAG) technology. It provides more accurate and reliable financial text sentiment analysis capabilities by incorporating external knowledge bases.

Key Features:
- Supports offline knowledge base (Phrasebank) and online crawling data
- Similarity retrieval based on vector database
- Configurable retrieval parameters
- Comprehensive evaluation metrics

For detailed information, please refer to [FinLLM-RAG/README.md](FinLLM-RAG/README.md)

## Path Configuration

The project supports both Google Colab and local environment paths. You need to configure the paths in the respective Python files before running the scripts:

1. Data Preparation (`FinLLM-Instruction-tuning/data/data_preparation.py`):
```python
# For local environment (uncomment to use):
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TRAIN_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "instruction_formatted_data.jsonl")
# TEST_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "validation_data.jsonl")

# For Google Colab (default):
DRIVE_ROOT = "/content/drive/FinLLM-Sentiment-Analysis/FinLLM-Instruction-tuning"
TRAIN_OUTPUT_PATH = os.path.join(DRIVE_ROOT, "data", "instruction_formatted_data.jsonl")
TEST_OUTPUT_PATH = os.path.join(DRIVE_ROOT, "data", "validation_data.jsonl")
```

2. Training Script (`FinLLM-Instruction-tuning/train/train.py`):
   - Similar path configuration for model checkpoints and outputs
   - Adjust paths based on your environment

3. Inference Script (`FinLLM-Instruction-tuning/Inference/inference.py`):
   - Configure input/output paths for your environment
   - Update model path according to your setup

Note: The scripts in the `scripts/` directory use relative paths, but the Python files they call may need path adjustments based on your environment.

## Quick Start

Important Note: All scripts in this project are for reference only. To use them:
1. First modify the paths in the Python files according to your environment (local or Google Colab)
2. Then adjust the corresponding scripts accordingly

1. Environment Setup:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure paths in Python files
# 1. Open data_preparation.py and adjust paths
# 2. Open train.py and adjust paths
# 3. Open inference.py and adjust paths
```

2. Model Training:
```bash
# Linux/Mac
./scripts/train.sh --model_name "my_model" --batch_size 4 --learning_rate 1e-4 --epochs 2

# Windows
scripts\train.bat --model_name "my_model" --batch_size 4 --learning_rate 1e-4 --epochs 2

# Default parameters if not specified:
# --model_name: default_model
# --batch_size: 8
# --learning_rate: 2e-5
# --epochs: 3
```

3. Model Inference:
```bash
# Using original model inference
# Linux/Mac
./scripts/inference.sh --model_path "FinLLM-Instruction-tuning/model_lora" --input_file "FinLLM-Instruction-tuning/data/validation_data.jsonl"

# Windows
scripts\inference.bat --model_path "FinLLM-Instruction-tuning\model_lora" --input_file "FinLLM-Instruction-tuning\data\validation_data.jsonl"

# Using RAG-enhanced inference
# Linux/Mac
./scripts/run_rag.sh --model_path "FinLLM-Instruction-tuning/model_lora" --query_file "FinLLM-Instruction-tuning/data/validation_data.jsonl" --knowledge_base "FinLLM-RAG/data/phrasebank_75_agree.json"

# Windows
scripts\run_rag.bat --model_path "FinLLM-Instruction-tuning\model_lora" --query_file "FinLLM-Instruction-tuning\data\validation_data.jsonl" --knowledge_base "FinLLM-RAG\data\phrasebank_75_agree.json"
```

Note: The inference process automatically generates evaluation results along with the predictions. No separate evaluation step is required.

Important: The file paths in the examples above are placeholders. You need to replace them with your actual file paths:
- `FinLLM-Instruction-tuning/data/validation_data.jsonl`: Replace with your validation data file path
- `FinLLM-Instruction-tuning/model_lora`: Replace with your trained model path
- `FinLLM-RAG/data/phrasebank_75_agree.json`: Replace with your knowledge base path (for RAG inference)

Note for RAG Knowledge Base:
- By default, the system uses offline Phrasebank data (`phrasebank_75_agree.json`) as the knowledge base
- You can also use online data by setting `--knowledge_base "online"`, which will use `FinLLM-RAG/data/get_rag_context_data.py` to fetch real-time financial data
- Example for online data:
  ```bash
  # Linux/Mac
  ./scripts/run_rag.sh --model_path "FinLLM-Instruction-tuning/model_lora" --query_file "FinLLM-Instruction-tuning/data/validation_data.jsonl" --knowledge_base "online"
  
  # Windows
  scripts\run_rag.bat --model_path "FinLLM-Instruction-tuning\model_lora" --query_file "FinLLM-Instruction-tuning\data\validation_data.jsonl" --knowledge_base "online"
  ```

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Minimum 8GB GPU memory
- 16GB+ system memory

## Important Notes

1. Data Preparation:
   - Ensure correct training data format
   - Choose between offline or online knowledge base as needed

2. Model Training:
   - GPU recommended for training
   - Performance can be optimized by adjusting training parameters

3. Inference Deployment:
   - Supports both CPU and GPU inference
   - Batch size can be adjusted based on requirements

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Issues and Pull Requests are welcome to help improve the project. Before submitting code, please ensure:
1. Code follows the project's coding standards
2. Necessary comments and documentation are added
3. Related test cases are updated