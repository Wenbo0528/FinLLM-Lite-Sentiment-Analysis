import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

# ==== Configuration ====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = "bigscience/bloom-560m"
LORA_PATH = os.path.join(PROJECT_ROOT, "model_lora")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "instruction_formatted_data.jsonl")
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== Setup Logging ====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# ==== Model Evaluation Class ====
class ModelEvaluator:
    def __init__(self, test_data_path):
        self.test_data_path = test_data_path
        self.test_data = self.load_test_data()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def load_test_data(self):
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def load_model(self):
        logger.info("Loading model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Check if LoRA path exists
        if not os.path.exists(LORA_PATH):
            raise ValueError(f"LoRA model path does not exist: {LORA_PATH}")

        # Load LoRA model
        self.model = PeftModel.from_pretrained(base_model, LORA_PATH)
        self.model.eval()
        logger.info("Model and tokenizer loaded successfully")

    def evaluate(self):
        if not self.model:
            self.load_model()

        logger.info("Starting evaluation...")
        results = []

        for item in self.test_data:
            instruction = item["instruction"]
            expected_output = item["output"]

            # Build prompt
            prompt = f"Human: {instruction}\nAssistant:"
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=100)
            
            # Decode response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()

            # Save result
            results.append({
                "instruction": instruction,
                "expected_output": expected_output,
                "model_output": response
            })

        # Save evaluation results
        output_path = OUTPUT_DIR / "evaluation_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation results saved to {output_path}")

        return results

# ==== Main Program ====
if __name__ == "__main__":
    evaluator = ModelEvaluator(TEST_DATA_PATH)
    evaluator.evaluate() 