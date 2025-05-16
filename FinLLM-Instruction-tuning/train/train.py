import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoConfig,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import Dataset
import logging
from transformers.trainer_utils import get_last_checkpoint

# ==== Configuration ====
# Original paths (commented for local environment)
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_MODEL = "deepseek-ai/deepseek-llm-1.3b"
# TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "instruction_formatted_data.jsonl")
# OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_lora")

# Google Drive paths and model configuration
DRIVE_ROOT = "/content/drive/MyDrive/Columbia_Semester3/5293/Final/FinLLM-Sentiment-Analysis"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Updated model
TRAIN_DATA_PATH = os.path.join(DRIVE_ROOT, "FinLLM-Instruction-tuning/data/instruction_formatted_data.jsonl")
OUTPUT_DIR = os.path.join(DRIVE_ROOT, "FinLLM-Instruction-tuning/model_lora")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== Training Configuration ====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_steps=500,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=10000,
    save_steps=1000,
    fp16=True,
    torch_compile=False,
    remove_unused_columns=False,
    save_total_limit=10,
    report_to="none",
    # checkpoint相关
    save_safetensors=True,
    save_only_model=True,
    overwrite_output_dir=False,  # 不覆盖输出目录，这样可以保存checkpoint
    load_best_model_at_end=False,  # 不加载最佳模型
    metric_for_best_model=None,    # 不使用指标选择最佳模型
    greater_is_better=None,        # 不使用指标比较
)

# ==== Model Configuration ====
model_config = AutoConfig.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    use_cache=False,  # Disable KV cache to save GPU memory
)

# ==== QLoRA Configuration (4-bit Quantization + LoRA) ====
# QLoRA = 4-bit quantized base model + LoRA fine-tuning, LoRA layers in float16
lora_config = LoraConfig(
    r=4,  # Smaller rank
    lora_alpha=8,  # Smaller alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # QLoRA key parameters
    use_rslora=False     # Disable RSLora (enable for extreme memory saving)
)

# ==== Quantization Configuration ====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Use NF4 quantization type
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computation
    bnb_4bit_use_double_quant=True,  # Use double quantization
)

# ==== Model Loading ====
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    config=model_config,
    quantization_config=bnb_config,  # QLoRA key: 4-bit quantization of base model
    device_map="auto",  # Automatic device mapping
    trust_remote_code=True,
)

# ==== Tokenizer Configuration ====
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    padding_side="right",  # Right-side padding
    use_fast=False,  # Use slow tokenizer for better compatibility
)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token

# ==== Custom Dataset Class ====
class InstructionDataset:
    def __init__(self, data_path, tokenizer):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer

    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format the instruction and output
        prompt = f"Human: {item['instruction']}\nAssistant: {item['output']}"
        
        # Tokenize the text
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None  # Changed from "pt" to None
        )
        
        # Convert to format expected by trainer
        return {
            "input_ids": torch.tensor(encodings["input_ids"]),
            "attention_mask": torch.tensor(encodings["attention_mask"]),
            "labels": torch.tensor(encodings["input_ids"])  # For causal LM, labels are same as input_ids
        }

# ==== Main Training Function ====
def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model (4-bit quantization)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,  # QLoRA key
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(base_model)

    # QLoRA LoRA configuration
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=False,
    )

    # Get PEFT model (QLoRA)
    model = get_peft_model(model, lora_config)

    # Load and prepare dataset
    dataset = InstructionDataset(TRAIN_DATA_PATH, tokenizer)
    
    # Create training dataset
    train_dataset = Dataset.from_dict({
        "input_ids": [item["input_ids"] for item in dataset],
        "attention_mask": [item["attention_mask"] for item in dataset],
        "labels": [item["labels"] for item in dataset]
    })

    # Initialize trainer with custom data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
    )

    # 检查是否存在检查点
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint is not None:
            logger.info(f"Found checkpoint: {last_checkpoint}")
            # 尝试从检查点恢复训练
            try:
                trainer.train(resume_from_checkpoint=last_checkpoint)
                logger.info("Successfully resumed training from checkpoint")
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
                logger.info("Starting training from scratch")
                trainer.train()
        else:
            logger.info("No checkpoint found, starting training from scratch")
            trainer.train()
    else:
        logger.info("Output directory does not exist, starting training from scratch")
        trainer.train()

    # Save final model
    trainer.save_model()
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    # Add logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    main() 