# FinLLM: Financial News Sentiment Analysis System

A comprehensive system for financial news sentiment analysis using Instruction Tuning and RAG (Retrieval-Augmented Generation).

## Project Structure

```
FinLLM/
├── FinLLM-Instruction-tuning/    # Instruction tuning module
│   ├── data/                     # Training data
│   ├── train/                    # Training scripts
│   ├── evaluation/               # Evaluation scripts
│   └── model_lora/              # Fine-tuned model
│
├── FinLLM-RAG/                   # RAG module
│   ├── data_sources/            # News data collection
│   ├── inference/               # RAG inference
│   └── evaluation/              # RAG evaluation
│
├── scripts/                      # Quick start scripts
│   ├── train.bat                # Windows training script
│   ├── train.sh                 # Linux/Mac training script
│   ├── run_rag.bat              # Windows RAG script
│   ├── run_rag.sh               # Linux/Mac RAG script
│   ├── evaluate.bat             # Windows evaluation script
│   └── evaluate.sh              # Linux/Mac evaluation script
│
└── requirements.txt             # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FinLLM-Sentiment-Analysis.git
cd FinLLM-Sentiment-Analysis
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training

1. Windows users:
```bash
scripts\train.bat
```

2. Linux/Mac users:
```bash
chmod +x scripts/*.sh
./scripts/train.sh
```

### Running RAG Inference

1. Windows users:
```bash
scripts\run_rag.bat
```

2. Linux/Mac users:
```bash
./scripts/run_rag.sh
```

### Evaluation

1. Windows users:
```bash
scripts\evaluate.bat
```

2. Linux/Mac users:
```bash
./scripts/evaluate.sh
```

## System Features

1. **Instruction Tuning**
   - Fine-tunes BLOOM model for sentiment analysis
   - Uses LoRA for efficient training
   - Supports custom instruction formats

2. **RAG Enhancement**
   - Real-time financial news retrieval
   - Context-aware sentiment analysis
   - Improved accuracy with relevant context

3. **Evaluation**
   - Comprehensive metrics
   - Comparative analysis
   - Visualization support

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 20GB+ free disk space

## Important Notes

1. **Hardware Requirements**
   - GPU with 8GB+ VRAM recommended
   - Sufficient disk space for model storage

2. **Data Requirements**
   - Internet connection for news retrieval
   - Storage space for news corpus

3. **Usage Tips**
   - Start with small batch sizes
   - Monitor GPU memory usage
   - Regular model evaluation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue in the GitHub repository.
