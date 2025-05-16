# FinLLM-RAG: Financial News Sentiment Analysis RAG System

This project implements a RAG (Retrieval-Augmented Generation) based financial news sentiment analysis system, providing more accurate sentiment analysis results by combining real-time financial news data with fine-tuned language models.

## Project Structure

```
FinLLM-RAG/
├── data_sources/              # Data Source Directory
│   ├── get_rag_context_data.py    # Web Crawler Script
│   └── rag_context_data.jsonl     # Crawled News Data
├── inference/                 # Inference Module
│   └── rag_retrieve_and_infer.py  # RAG Inference Script
├── evaluation/               # Evaluation Module
│   └── evaluate_rag_results.py    # RAG Results Evaluation Script
├── results/                  # Results Directory
│   ├── evaluation_report.json     # Evaluation Report
│   ├── sentiment_comparison.png   # Sentiment Distribution Comparison Chart
│   └── non_rag_predictions.json  # Non-RAG Prediction Results
└── requirements.txt         # Project Dependencies
```

## Environment Configuration

Install project dependencies:
```bash
pip install -r requirements.txt
```

Main dependencies include:
- transformers
- peft
- torch
- beautifulsoup4
- requests
- fake-useragent
- scikit-learn
- matplotlib
- seaborn

## Data Acquisition

1. Run the web crawler script to get financial news data:
```bash
cd data_sources
python get_rag_context_data.py
```

The crawler script will get data from the following sources:
- Yahoo Finance
- MarketWatch
- Seeking Alpha

Data will be saved in the `data_sources/rag_context_data.jsonl` file.

## Model Inference

1. Ensure all dependencies are installed
2. Ensure LoRA model is trained (located in `FinLLM-Instruction-tuning/model_lora`)
3. Run RAG inference script:
```bash
cd inference
python rag_retrieve_and_infer.py
```

Inference results will be saved in the `inference/rag_inference_results.json` file.

## Results Evaluation

Run evaluation script to analyze RAG and non-RAG results:
```bash
cd evaluation
python evaluate_rag_results.py
```

Evaluation results will be saved in the `results` directory, including:
- `evaluation_report.json`: Detailed evaluation report
- `sentiment_comparison.png`: Sentiment distribution comparison chart between RAG and non-RAG
- `non_rag_predictions.json`: Prediction results without using RAG

## System Features

1. Real-time data acquisition:
   - Automatically crawl latest financial news
   - Support multiple data sources
   - Include anti-crawling measures

2. Intelligent retrieval:
   - Context retrieval based on similarity
   - Support multi-source data fusion
   - Preserve source information

3. Comparison evaluation:
   - RAG vs non-RAG performance comparison
   - Detailed evaluation metrics
   - Visual analysis results

## Notes

1. Hardware requirements:
   - GPU memory: Recommended at least 8GB
   - Support FP16 training and inference

2. Network requirements:
   - Stable network connection required
   - Proxy configuration may be needed

3. Data requirements:
   - Ensure data source is accessible
   - Regularly update news data

## Usage Example

1. Get news data:
```python
from data_sources.get_rag_context_data import scrape_financial_news
news_data = scrape_financial_news()
```

2. Run RAG inference:
```python
from inference.rag_retrieve_and_infer import run_inference
result = run_inference("Tesla stock rose 5% after strong earnings")
```

3. Evaluate results:
```python
from evaluation.evaluate_rag_results import RAGEvaluator
evaluator = RAGEvaluator("inference/rag_inference_results.json")
evaluator.generate_report()
```

## Contribution Guidelines

Welcome to submit issues and Pull Requests to improve the project. Before submitting code, please ensure:
1. Code conforms to PEP 8 standards
2. Add necessary comments and documentation
3. Update related test cases

## License

This project uses the MIT License. See [LICENSE](LICENSE) file for details. 