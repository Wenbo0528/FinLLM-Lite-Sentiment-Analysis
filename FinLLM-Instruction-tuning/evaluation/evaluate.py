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
from collections import Counter

# ==== Configuration ====
# Original paths (commented for local environment)
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  
# LORA_PATH = os.path.join(PROJECT_ROOT, "model_lora")
# TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "validation_data.jsonl")
# OUTPUT_DIR = Path(__file__).parent / "results"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Google Drive paths (commented out)
DRIVE_ROOT = "/content/drive/MyDrive/Columbia_Semester3/5293/Final/FinLLM-Sentiment-Analysis/FinLLM-Instruction-tuning"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  
LORA_PATH = os.path.join(DRIVE_ROOT, "model_lora")
TEST_DATA_PATH = os.path.join(DRIVE_ROOT, "data", "validation_data.jsonl")
OUTPUT_DIR = Path(__file__).parent / "results"

# ==== Model Configuration ====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Updated to 4-bit quantization
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

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
        
        # Load test data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the instruction for input
        input_text = f"Human: {item['instruction']}\nAssistant:"
        label = item['output']
        
        # Encode input text
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
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights
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
        
        # Decode prediction results
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_output = pred_text.split("Assistant:")[-1].strip()
        pred_label = EvaluationDataset.clean_prediction(raw_output)

        predictions.append(pred_label)
        true_labels.append(item['label'])
    
    return predictions, true_labels

def calculate_metrics(predictions, true_labels):
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    # Calculate precision, recall, and F1 score
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

def print_dataset_stats(dataset, name):
    labels = [item['output'] for item in dataset]
    label_counts = Counter(labels)
    print(f"\n{name} dataset statistics:")
    print(f"Total samples: {len(dataset)}")
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"- {label}: {count} ({count/len(dataset)*100:.1f}%)")

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
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            padding_side="right",
            use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
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

        for item in tqdm(self.test_data, desc="Evaluating"):
            instruction = item["instruction"]
            expected_output = item["output"]

            # Build prompt
            prompt = f"Human: {instruction}\nAssistant:"
            
            # Generate response
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response and clean prediction
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            pred_label = EvaluationDataset.clean_prediction(response)

            # Save result
            results.append({
                "instruction": instruction,
                "expected_output": expected_output,
                "model_output": response,
                "predicted_label": pred_label
            })

        # Save evaluation results
        output_path = OUTPUT_DIR / "evaluation_results.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logger.info(f"Evaluation results saved to {output_path}")

        # Calculate and log metrics
        predictions = [r["predicted_label"] for r in results]
        true_labels = [r["expected_output"] for r in results]
        metrics = calculate_metrics(predictions, true_labels)
        logger.info(f"Evaluation metrics: {metrics}")

        return results, metrics

# ==== Main Program ====
if __name__ == "__main__":
    evaluator = ModelEvaluator(TEST_DATA_PATH)
    evaluator.evaluate() 