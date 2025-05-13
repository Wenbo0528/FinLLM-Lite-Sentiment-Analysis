import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import load_dataset
import sys

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载JSONL数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # 对文本进行编码
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }

def main():
    # 配置参数
    model_name = "bigscience/bloom-560m"  # 使用较小的模型进行测试
    data_path = os.path.join(PROJECT_ROOT, "data", "instruction_formatted_data.jsonl")
    # 修改输出目录为 model_lora
    output_dir = os.path.join(PROJECT_ROOT, "model_lora")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config
    )
    
    # 准备LoRA配置
    lora_config = LoraConfig(
        r=4,  # 降低LoRA秩以节省显存
        lora_alpha=8,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 准备模型进行LoRA训练
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # 准备数据集
    dataset = InstructionDataset(data_path, tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # 修改为1轮训练
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    print(f"✅ 模型已保存到 {output_dir}")

if __name__ == "__main__":
    main() 