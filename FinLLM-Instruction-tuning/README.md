# FinLLM-Instruction-tuning

A financial sentiment analysis instruction tuning project based on the DeepSeek-R1-Distill-Qwen-1.5B model. This project uses QLoRA (4-bit quantization + LoRA) technology for efficient fine-tuning, enabling the model to better understand and analyze sentiment tendencies in financial text.

## Project Features

- Efficient fine-tuning using QLoRA technology, significantly reducing GPU memory requirements
- Instruction-tuning approach to improve model understanding of financial text
- Optimized for financial sentiment analysis tasks
- Supports model quantization for practical deployment
- Provides complete training and inference scripts

## Technical Details

### Model Architecture
- Base Model: DeepSeek-R1-Distill-Qwen-1.5B
- Fine-tuning Method: QLoRA (4-bit quantization + LoRA)
- LoRA Configuration:
  - rank (r) = 4
  - alpha = 8
  - Target Modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  - dropout = 0.05

### Training Configuration
- Training Epochs: 3 (default)
- Batch Size: 8 (default)
- Gradient Accumulation Steps: 8
- Learning Rate: 2e-5 (default)
- Weight Decay: 0.01
- Warmup Steps: 10000
- Save Steps: 1000
- Training with FP16

## Project Structure

```
FinLLM-Instruction-tuning/
├── data/                    # Training data directory
│   ├── data_preparation.py  # Data preparation script
│   ├── instruction_formatted_data.jsonl  # Training data
│   └── validation_data.jsonl  # Validation data
├── train/                   # Training related code
│   └── train.py            # Training script
├── model_lora/             # Saved LoRA model weights
├── results/                # Training results and evaluation metrics
└── Inference/             # Inference related code
    └── inference.py       # Inference script
```

## Usage

### Environment Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure paths in Python files:
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

### Model Training

Use the provided training script to train the model:

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

### Model Inference

After training, use the trained model for inference:

```bash
# Linux/Mac
./scripts/inference.sh --model_path "FinLLM-Instruction-tuning/model_lora" --input_file "FinLLM-Instruction-tuning/data/validation_data.jsonl"

# Windows
scripts\inference.bat --model_path "FinLLM-Instruction-tuning\model_lora" --input_file "FinLLM-Instruction-tuning\data\validation_data.jsonl"
```

Note: The inference process automatically generates evaluation results along with the predictions. No separate evaluation step is required.

## Performance Optimization

1. Memory Optimization:
   - 4-bit quantization to reduce base model memory usage
   - LoRA technology to minimize trainable parameters
   - Gradient accumulation for handling large batch sizes

2. Training Efficiency:
   - FP16 mixed precision training
   - Checkpoint mechanism for resuming training
   - Optimized data loading and processing pipeline

## Important Notes

1. Memory Requirements:
   - Recommended GPU with at least 8GB memory
   - Adjust batch_size and gradient_accumulation_steps to accommodate different memory sizes

2. Data Format:
   - Training data: `instruction_formatted_data.jsonl`
   - Validation data: `validation_data.jsonl`
   - Both files should be in JSONL format

3. Model Saving:
   - Model weights are saved in the model_lora directory
   - Supports training resumption from checkpoints

4. Path Configuration:
   - Modify paths in Python files according to your environment
   - Update script paths if needed
   - Ensure all required directories exist

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Issues and Pull Requests are welcome to help improve the project. Before submitting code, please ensure:
1. Code follows the project's coding standards
2. Necessary comments and documentation are added
3. Related test cases are updated 