# FinLLM: Financial News Sentiment Analysis System

FinLLM is a financial news sentiment analysis system based on large language models, consisting of two main modules: instruction fine-tuning and RAG enhancement. The system can accurately analyze the sentiment of financial news to provide reference for investment decisions.

## Project Structure

```
FinLLM/
├── FinLLM-Instruction-tuning/    # Instruction Fine-tuning Module
│   ├── data/                     # Training Data
│   ├── train/                    # Training Scripts
│   ├── evaluation/              # Evaluation Scripts
│   ├── model_lora/              # Fine-tuned Model
│   └── README.md               # Module Documentation
│
├── FinLLM-RAG/                  # RAG Enhancement Module
│   ├── data_sources/           # Data Sources and Crawlers
│   ├── inference/              # Inference Scripts
│   ├── evaluation/            # Evaluation Scripts
│   ├── results/               # Evaluation Results
│   └── README.md             # Module Documentation
│
└── README.md                  # Project Documentation
```

## System Architecture

The system consists of two main modules:

1. **Instruction Fine-tuning Module**:
   - Based on BLOOM-560M model
   - Efficient fine-tuning using LoRA
   - Supports 8-bit quantization training
   - Optimized for financial sentiment analysis tasks

2. **RAG Enhancement Module**:
   - Real-time financial news crawling
   - Intelligent context retrieval
   - Multi-source data fusion
   - Comparative evaluation system

## Requirements

- Python 3.8+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- Stable internet connection

## Quick Start

1. Clone the project:
```bash
git clone https://github.com/yourusername/FinLLM.git
cd FinLLM
```

2. Install dependencies:
```bash
# Install base dependencies
pip install -r requirements.txt

# Install instruction fine-tuning module dependencies
cd FinLLM-Instruction-tuning
pip install -r requirements.txt

# Install RAG module dependencies
cd ../FinLLM-RAG
pip install -r requirements.txt
```

3. Train the model:
```bash
cd FinLLM-Instruction-tuning/train
python train.py
```

4. Run the RAG system:
```bash
cd FinLLM-RAG/data_sources
python get_rag_context_data.py  # Get news data

cd ../inference
python rag_retrieve_and_infer.py  # Run inference
```

5. Evaluate results:
```bash
cd FinLLM-RAG/evaluation
python evaluate_rag_results.py
```

## Key Features

1. **Efficient Training**:
   - LoRA fine-tuning reduces VRAM usage
   - 8-bit quantization support
   - Gradient checkpoint optimization

2. **Real-time Data**:
   - Multi-source news crawling
   - Intelligent anti-crawling
   - Automatic data updates

3. **Intelligent Analysis**:
   - Context-enhanced inference
   - Multi-source data fusion
   - Sentiment distribution analysis

4. **Comprehensive Evaluation**:
   - RAG vs non-RAG comparison
   - Detailed evaluation metrics
   - Visualization analysis

## Performance Metrics

- Model size: 560M parameters
- Training VRAM: ~4GB
- Inference VRAM: ~2GB
- Supports FP16 training and inference

## Usage Examples

1. Basic sentiment analysis:
```python
from FinLLM_Instruction_tuning.inference import analyze_sentiment
result = analyze_sentiment("Tesla stock rose 5% after strong earnings")
```

2. RAG-enhanced analysis:
```python
from FinLLM_RAG.inference import analyze_with_rag
result = analyze_with_rag("Tesla stock rose 5% after strong earnings")
```

## Project Characteristics

1. **Modular Design**:
   - Clear module separation
   - Independent configuration management
   - Easy to extend and maintain

2. **Resource Optimization**:
   - VRAM usage optimization
   - Training speed optimization
   - Inference efficiency optimization

3. **Practical Features**:
   - Complete evaluation system
   - Detailed documentation
   - Rich usage examples

## Important Notes

1. **Hardware Requirements**:
   - GPU VRAM ≥ 8GB
   - CUDA support
   - SSD recommended

2. **Data Requirements**:
   - Regular training data updates
   - Ensure data source accessibility
   - Monitor data quality

3. **Usage Recommendations**:
   - Regular model performance evaluation
   - Timely news data updates
   - Monitor system resource usage

## Future Plans

1. Feature Expansion:
   - Support more financial data sources
   - Add more analysis dimensions
   - Optimize retrieval algorithms

2. Performance Optimization:
   - Reduce resource usage
   - Improve inference speed
   - Optimize training efficiency

3. Deployment Optimization:
   - Provide API interface
   - Support batch processing
   - Add monitoring system

## Contributing

Issues and Pull Requests are welcome. Before submitting code, please ensure:
1. Code follows PEP 8 standards
2. Necessary comments and documentation are added
3. Related test cases are updated
4. All tests pass

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

- Project Maintainer: [Your Name]
- Email: [Your Email]
- GitHub: [Your GitHub]

## Acknowledgments

Thanks to all developers who have contributed to this project.
