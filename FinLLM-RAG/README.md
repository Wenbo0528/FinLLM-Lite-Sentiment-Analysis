# FinLLM-RAG

A financial sentiment analysis enhancement module based on Retrieval-Augmented Generation (RAG). This module combines the fine-tuned FinLLM model with external knowledge bases to provide more accurate and reliable financial text sentiment analysis capabilities.

## Project Features

- Combines fine-tuned FinLLM model with RAG technology
- Supports both offline Phrasebank data and online financial news retrieval
- Provides more accurate sentiment analysis results
- Extensible knowledge base support

## Technical Architecture

### Core Components
1. Retrieval System
   - Similarity search based on vector database
   - Supports both offline and online knowledge bases
   - Configurable retrieval parameters

2. Generation System
   - Based on fine-tuned FinLLM model
   - Supports context-enhanced generation
   - Adjustable generation parameters

### Workflow
1. Input Processing: Receive user queries
2. Knowledge Retrieval: Retrieve relevant documents from knowledge base
3. Context Construction: Integrate retrieval results with user queries
4. Enhanced Generation: Generate analysis results using FinLLM model

## Project Structure

```
FinLLM-RAG/
├── data/                    # Data directory
│   ├── Phrasebank_data_preprocess.py  # Data preprocessing script
│   ├── get_rag_context_data.py       # Online knowledge base data acquisition script
│   ├── phrasebank_75_agree.json      # Default offline knowledge base
│   └── api_rag_context_data_sample.jsonl  # API knowledge base data sample
├── inference/              # Inference related code
│   ├── rag_retrieve_and_infer.py   # RAG inference main program
│   └── analyze_rag_context.py  # Context analysis tool
└── results/               # Results output directory
```

## Usage

### Environment Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare knowledge base data:
   The project provides two knowledge base data sources:

   a. Offline Data (Default):
   - Uses preprocessed Phrasebank data (`data/phrasebank_75_agree.json`)
   - No additional configuration needed, ready to use
   - Recommended for quick start and stable performance

   b. Online Data (Optional):
   - Use `get_rag_context_data.py` script to crawl real-time financial news
   - Supports multiple financial news sources
   - Data will be saved in JSONL format
   - Requires internet connection

### RAG Inference

Use the provided scripts for RAG-enhanced inference:

```bash
# Linux/Mac
./scripts/run_rag.sh --model_path "FinLLM-Instruction-tuning/model_lora" --query_file "FinLLM-Instruction-tuning/data/validation_data.jsonl" --knowledge_base "FinLLM-RAG/data/phrasebank_75_agree.json"

# Windows
scripts\run_rag.bat --model_path "FinLLM-Instruction-tuning\model_lora" --query_file "FinLLM-Instruction-tuning\data\validation_data.jsonl" --knowledge_base "FinLLM-RAG\data\phrasebank_75_agree.json"
```

For online data retrieval:
```bash
# Linux/Mac
./scripts/run_rag.sh --model_path "FinLLM-Instruction-tuning/model_lora" --query_file "FinLLM-Instruction-tuning/data/validation_data.jsonl" --knowledge_base "online"

# Windows
scripts\run_rag.bat --model_path "FinLLM-Instruction-tuning\model_lora" --query_file "FinLLM-Instruction-tuning\data\validation_data.jsonl" --knowledge_base "online"
```

### Parameter Description

- `--model_path`: Path to the fine-tuned FinLLM model
- `--query_file`: Path to the validation data file for analysis
- `--knowledge_base`: Path to the knowledge base file or "online" for real-time data
- `--output_dir`: Output results directory (default: results/)
- `--top_k`: Number of documents to retrieve (default: 3)

## Performance Optimization

1. Retrieval Optimization:
   - Efficient vector retrieval algorithms
   - Batch processing support
   - Configurable caching mechanism

2. Generation Optimization:
   - Context length optimization
   - Parallel processing support
   - Memory usage optimization

## Notes

1. Knowledge Base Requirements:
   - Offline mode: Ensure phrasebank_75_agree.json is available
   - Online mode: Requires internet connection and API access
   - Regular knowledge base updates recommended

2. System Requirements:
   - SSD storage recommended for knowledge base
   - Sufficient memory for vector retrieval
   - GPU acceleration recommended

3. Usage Recommendations:
   - Start with offline mode for stable performance
   - Use online mode for real-time financial news analysis
   - Adjust retrieval parameters based on needs

## Evaluation Metrics

1. Retrieval Quality:
   - Retrieval accuracy
   - Retrieval recall
   - Retrieval latency

2. Generation Quality:
   - Sentiment analysis accuracy
   - Generation result relevance
   - Response time

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

Issues and Pull Requests are welcome to help improve the project. Before submitting code, please ensure:
1. Code follows the project's coding standards
2. Necessary comments and documentation are added
3. Related test cases are updated 