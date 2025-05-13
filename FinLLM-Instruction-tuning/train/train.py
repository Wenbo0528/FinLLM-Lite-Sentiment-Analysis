import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import Dataset

# ==== Configuration ====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = "bigscience/bloom-560m"
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "instruction_formatted_data.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_lora")

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
        prompt = f"Human: {item['instruction']}\nAssistant: {item['output']}"
        return {"text": prompt}

# ==== Main Training Function ====
def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(base_model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)

    # Load dataset
    dataset = InstructionDataset(TRAIN_DATA_PATH, tokenizer)
    train_dataset = Dataset.from_list([{"text": item["text"]} for item in dataset])

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Start training
    trainer.train()

    # Save model
    trainer.save_model()
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 