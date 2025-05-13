# FinLLM Instruction Fine-tuning Module

This module implements instruction fine-tuning for the FinLLM model, enabling it to perform financial sentiment analysis tasks.

## Project Structure

```
FinLLM-Instruction-tuning/
├── data/                     # Training Data
│   └── instruction_formatted_data.jsonl
├── train/                    # Training Scripts
│   └── train.py
├── evaluation/              # Evaluation Scripts
│   └── evaluate.py
├── model_lora/              # Fine-tuned Model
└── README.md               # Module Documentation
```

## Features

1. **Efficient Training**:
   - LoRA-based fine-tuning
   - 8-bit quantization support
   - Gradient checkpoint optimization
   - Memory-efficient training

2. **Model Architecture**:
   - Base model: BLOOM-560M
   - LoRA configuration optimized for financial tasks
   - FP16 training support

3. **Training Process**:
   - Custom instruction dataset
   - Optimized training parameters
   - Automatic model saving

## Requirements

- Python 3.8+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4+

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare training data:
```bash
# Ensure your data is in JSONL format
# Each line should contain:
# {
#   "instruction": "Your instruction here",
#   "output": "Expected output here"
# }
```

3. Start training:
```bash
cd train
python train.py
```

4. Evaluate the model:
```bash
cd evaluation
python evaluate.py
```

## Training Configuration

The training script uses the following default parameters:
- Learning rate: 2e-4
- Batch size: 2
- Gradient accumulation steps: 4
- Training epochs: 1
- LoRA rank: 16
- LoRA alpha: 32

## Model Output

The fine-tuned model will be saved in the `model_lora` directory with the following structure:
- `adapter_config.json`: LoRA configuration
- `adapter_model.bin`: LoRA weights
- `training_args.bin`: Training arguments

## Evaluation

The evaluation script provides:
- Model performance metrics
- Prediction examples
- Detailed evaluation report

## Notes

1. **Hardware Requirements**:
   - GPU VRAM ≥ 8GB
   - CUDA support
   - SSD recommended

2. **Training Tips**:
   - Monitor GPU memory usage
   - Adjust batch size if needed
   - Use gradient checkpointing for large models

3. **Best Practices**:
   - Regular model evaluation
   - Save checkpoints
   - Monitor training loss

## Future Improvements

1. Model Optimization:
   - Support for larger models
   - Advanced quantization methods
   - Distributed training

2. Training Enhancement:
   - Curriculum learning
   - Advanced data augmentation
   - Multi-task learning

3. Evaluation Metrics:
   - More comprehensive metrics
   - Automated evaluation pipeline
   - Performance visualization

## Contributing

Issues and Pull Requests are welcome. Before submitting code, please ensure:
1. Code follows PEP 8 standards
2. Necessary comments and documentation are added
3. Related test cases are updated
4. All tests pass

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 