import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

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
OUTPUT_DIR = Path(DRIVE_ROOT) / "results"  # Modify to results directory under FinLLM-Instruction-tuning
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

# ==== Model Configuration ====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
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
            'label': label,
            'instruction': item['instruction']  # Add instruction to return value
        }

def load_model(self):
    logger.info("Loading model and tokenizer...")
    
    # Load tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="left",  # Ensure using left padding
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
                max_new_tokens=5,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                num_beams=1,
                early_stopping=False  # Disable early stopping
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
            padding_side="left",  # Ensure using left padding
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
        
        # Increase batch size
        batch_size = 32  # This size should be fine for A100
        dataset = EvaluationDataset(self.test_data_path, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in tqdm(dataloader, desc="Evaluating"):
            # Prepare batch input
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            instructions = batch['instruction']  # Get instructions
            labels = batch['label']  # Get labels
            
            # Generate predictions
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5,  # Further reduce maximum token generation
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,  # Enable KV cache
                    num_beams=1,     # Use greedy search
                    early_stopping=False  # Disable early stopping
                )
            
            # Process each sample's prediction results
            for i, output in enumerate(outputs):
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                prompt = f"Human: {instructions[i]}\nAssistant:"
                response = response.replace(prompt, "").strip()
                pred_label = EvaluationDataset.clean_prediction(response)
                
                results.append({
                    "instruction": instructions[i],
                    "expected_output": labels[i],
                    "model_output": response,
                    "predicted_label": pred_label
                })

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. Save inference results
        inference_path = OUTPUT_DIR / "inference_results.jsonl"
        with open(inference_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Inference results saved to {inference_path}")

        # 2. Calculate and save evaluation metrics
        predictions = [r["predicted_label"] for r in results]
        true_labels = [r["expected_output"] for r in results]
        metrics = calculate_metrics(predictions, true_labels)
        
        # Save evaluation metrics
        metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
        metrics_data = {
            "accuracy": float(metrics['accuracy']),
            "precision": float(metrics['precision']),
            "recall": float(metrics['recall']),
            "f1_score": float(metrics['f1']),
            "prediction_distribution": dict(Counter(predictions))
        }
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=4, ensure_ascii=False)
        print(f"Evaluation metrics saved to {metrics_path}")

        # Print evaluation metrics
        print("\n=== Evaluation Metrics ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Print prediction distribution
        print("\n=== Prediction Distribution ===")
        pred_counts = Counter(predictions)
        total = len(predictions)
        for label, count in pred_counts.items():
            print(f"{label}: {count} ({count/total*100:.1f}%)")

        # 3. Create and save visualizations
        # 3.1 Confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(true_labels, predictions)
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add value labels
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # Set axis labels
        classes = sorted(set(true_labels))
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        confusion_matrix_path = OUTPUT_DIR / "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        plt.close()

        # 3.2 Evaluation metrics bar chart
        plt.figure(figsize=(10, 6))
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metrics_values = [metrics['accuracy'], metrics['precision'], 
                         metrics['recall'], metrics['f1']]
        bars = plt.bar(metrics_names, metrics_values)
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        metrics_plot_path = OUTPUT_DIR / "metrics_bar.png"
        plt.savefig(metrics_plot_path)
        plt.close()

        # 3.3 Prediction distribution pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(pred_counts.values(), labels=pred_counts.keys(), autopct='%1.1f%%')
        plt.title('Prediction Distribution')
        plt.tight_layout()
        distribution_plot_path = OUTPUT_DIR / "prediction_distribution.png"
        plt.savefig(distribution_plot_path)
        plt.close()

        print("\nVisualizations have been saved to:")
        print(f"- {confusion_matrix_path}")
        print(f"- {metrics_plot_path}")
        print(f"- {distribution_plot_path}")

        return results, metrics

# ==== Main Program ====
if __name__ == "__main__":
    evaluator = ModelEvaluator(TEST_DATA_PATH)
    evaluator.evaluate() 