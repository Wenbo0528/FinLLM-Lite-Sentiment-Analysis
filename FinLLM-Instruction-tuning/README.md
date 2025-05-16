# FinLLM-Instruction-tuning

A financial sentiment analysis instruction tuning project based on the DeepSeek-R1-Distill-Qwen-1.5B model. This project uses QLoRA (4-bit quantization + LoRA) technology for efficient fine-tuning, enabling the model to better understand and analyze sentiment tendencies in financial text.

## Project Features

- Efficient fine-tuning using QLoRA technology, significantly reducing GPU memory requirements
- Instruction-tuning approach to improve model understanding of financial text
- Optimized for financial sentiment analysis tasks
- Supports model quantization for practical deployment

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
- Training Epochs: 2
- Batch Size: 4 (per device)
- Gradient Accumulation Steps: 8
- Learning Rate: 1e-4
- Weight Decay: 0.01
- Warmup Steps: 10000
- Save Steps: 1000
- Training with FP16

## Project Structure

```
FinLLM-Instruction-tuning/
├── data/                    # Training data directory
│   └── instruction_formatted_data.jsonl  # Instruction-formatted training data
├── train/                   # Training related code
│   └── train.py            # Training script
├── model_lora/             # Saved LoRA model weights
├── results/                # Training results and evaluation metrics
├── Inference/             # Inference related code
│   ├── inference.py       # Inference script for fine-tuned model
│   └── inference_origin.py # Inference script for original model
└── backup_py/             # Backup code
```

## Usage

### Environment Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare training data:
- Place training data in `data/instruction_formatted_data.jsonl`
- Data should be in JSONL format, with each line containing instruction and output fields

### Model Training

Use the provided training script to train the model:

```bash
# Linux/Mac
./scripts/train.sh --model_name "my_model" --batch_size 4 --learning_rate 1e-4 --epochs 2

# Windows
scripts\train.bat --model_name "my_model" --batch_size 4 --learning_rate 1e-4 --epochs 2
```

### Model Inference

After training, use the trained model for inference:

```bash
# Linux/Mac
./scripts/inference.sh --model_path "FinLLM-Instruction-tuning/model_lora" --input_file "data/test_queries.txt"

# Windows
scripts\inference.bat --model_path "FinLLM-Instruction-tuning\model_lora" --input_file "data\test_queries.txt"
```

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
   - Recommended GPU with at least 16GB memory
   - Adjust batch_size and gradient_accumulation_steps to accommodate different memory sizes

2. Data Format:
   - Ensure correct training data format
   - Recommended to preprocess and clean the data

3. Model Saving:
   - Model weights are saved in the model_lora directory
   - Supports training resumption from checkpoints

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Issues and Pull Requests are welcome to help improve the project. 