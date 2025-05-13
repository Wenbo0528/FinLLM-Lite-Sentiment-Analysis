import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import os
import sys

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class EvaluationDataset(Dataset):
    @staticmethod
    def clean_prediction(pred_text):
        text = pred_text.lower()
        if "positive" in text:
            return "positive"
        elif "negative" in text:
            return "negative"
        elif "neutral" in text:
            return "neutral"
        else:
            return "unknown"
        
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载测试数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # 分离输入和标签
        input_text = text.split("Assistant:")[0].strip()
        label = text.split("Assistant:")[1].strip()
        
        # 对输入文本进行编码
        encodings = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'label': label
        }

def load_model_and_tokenizer(base_model_path, lora_model_path):
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 配置量化参数
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, lora_model_path)
    model.eval()
    
    return model, tokenizer

def generate_predictions(model, tokenizer, dataset, device):
    predictions = []
    true_labels = []
    
    for item in tqdm(dataset, desc="Generating predictions"):
        input_ids = item['input_ids'].unsqueeze(0).to(device)
        attention_mask = item['attention_mask'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 解码预测结果
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_output = pred_text.split("Assistant:")[-1].strip()
        pred_label = EvaluationDataset.clean_prediction(raw_output)


        #pred_label = pred_text.split("Assistant:")[-1].strip()
        
        predictions.append(pred_label)
        true_labels.append(item['label'])
    
    return predictions, true_labels

def calculate_metrics(predictions, true_labels):
    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    
    # 计算精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, 
        predictions, 
        average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    # 配置参数
    base_model_path = "bigscience/bloom-560m"  # 使用较小的模型
    lora_model_path = os.path.join(PROJECT_ROOT, "model_lora")
    test_data_path = os.path.join(PROJECT_ROOT, "data", "instruction_formatted_data.jsonl")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型和tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(base_model_path, lora_model_path)
    
    # 准备测试数据集
    print("Preparing test dataset...")
    test_dataset = EvaluationDataset(test_data_path, tokenizer)
    
    # 生成预测
    print("Generating predictions...")
    predictions, true_labels = generate_predictions(model, tokenizer, test_dataset, device)
    
    # 计算评估指标
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions, true_labels)
    
    # 打印评估结果
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # 创建结果目录
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)


    # 保存评估结果
    results = {
        'metrics': metrics,
        'predictions': predictions,
        'true_labels': true_labels
    }
    
    results_path = os.path.join(results_dir, "evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评估结果已保存到 {results_path}")

if __name__ == "__main__":
    main() 