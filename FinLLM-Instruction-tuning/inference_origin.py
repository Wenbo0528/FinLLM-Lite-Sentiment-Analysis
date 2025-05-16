import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# ==== Configuration ====
# Google Drive paths
DRIVE_ROOT = "/content/drive/MyDrive/Columbia_Semester3/5293/Final/FinLLM-Sentiment-Analysis/FinLLM-Instruction-tuning"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  
TEST_DATA_PATH = os.path.join(DRIVE_ROOT, "data", "validation_data.jsonl")
OUTPUT_DIR = Path(DRIVE_ROOT) / "results_origin"  # Changed to results_origin directory
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

class InferenceDataset(Dataset):
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

def calculate_metrics(predictions, true_labels):
    # Calculate overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        true_labels, 
        predictions, 
        average=None
    )
    
    # Calculate macro and micro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, 
        predictions, 
        average='macro'
    )
    
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        true_labels, 
        predictions, 
        average='micro'
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Calculate error analysis metrics
    total_samples = len(true_labels)
    correct_predictions = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    error_rate = 1 - (correct_predictions / total_samples)
    
    # Calculate class-wise error rates
    class_error_rates = {}
    for i, label in enumerate(sorted(set(true_labels))):
        class_samples = sum(1 for t in true_labels if t == label)
        class_errors = sum(1 for p, t in zip(predictions, true_labels) if t == label and p != t)
        class_error_rates[label] = class_errors / class_samples if class_samples > 0 else 0
    
    return {
        'accuracy': accuracy,
        'error_rate': error_rate,
        'per_class_metrics': {
            label: {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(sup),
                'error_rate': float(class_error_rates[label])
            }
            for label, precision, recall, f1, sup in zip(
                sorted(set(true_labels)),
                precision_per_class,
                recall_per_class,
                f1_per_class,
                support
            )
        },
        'macro_metrics': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1': float(macro_f1)
        },
        'micro_metrics': {
            'precision': float(micro_precision),
            'recall': float(micro_recall),
            'f1': float(micro_f1)
        },
        'confusion_matrix': cm.tolist()
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
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        logger.info("Model and tokenizer loaded successfully")

    def plot_detailed_metrics(self, metrics, output_dir):
        # 1. Per-class metrics visualization
        plt.figure(figsize=(15, 8))
        classes = list(metrics['per_class_metrics'].keys())
        x = np.arange(len(classes))
        width = 0.25

        precision = [m['precision'] for m in metrics['per_class_metrics'].values()]
        recall = [m['recall'] for m in metrics['per_class_metrics'].values()]
        f1 = [m['f1'] for m in metrics['per_class_metrics'].values()]

        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1, width, label='F1 Score')

        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Per-class Performance Metrics')
        plt.xticks(x, classes)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_metrics_origin.png')
        plt.close()

        # 2. Macro vs Micro metrics comparison
        plt.figure(figsize=(10, 6))
        metrics_names = ['Precision', 'Recall', 'F1']
        macro_values = [metrics['macro_metrics'][m.lower()] for m in metrics_names]
        micro_values = [metrics['micro_metrics'][m.lower()] for m in metrics_names]

        x = np.arange(len(metrics_names))
        width = 0.35

        plt.bar(x - width/2, macro_values, width, label='Macro Average')
        plt.bar(x + width/2, micro_values, width, label='Micro Average')

        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Macro vs Micro Average Metrics')
        plt.xticks(x, metrics_names)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'macro_micro_comparison_origin.png')
        plt.close()

        # 3. Error rate visualization
        plt.figure(figsize=(12, 6))
        classes = list(metrics['per_class_metrics'].keys())
        error_rates = [m['error_rate'] for m in metrics['per_class_metrics'].values()]
        
        plt.bar(classes, error_rates)
        plt.xlabel('Classes')
        plt.ylabel('Error Rate')
        plt.title('Class-wise Error Rates')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'error_rates_origin.png')
        plt.close()

    def evaluate(self):
        if not self.model:
            self.load_model()

        logger.info("Starting inference...")
        results = []
        
        # Increase batch size
        batch_size = 32  # This size should be fine for A100
        dataset = InferenceDataset(self.test_data_path, self.tokenizer)
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
            
            # Process prediction results for each sample
            for i, output in enumerate(outputs):
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                prompt = f"Human: {instructions[i]}\nAssistant:"
                response = response.replace(prompt, "").strip()
                pred_label = InferenceDataset.clean_prediction(response)
                
                results.append({
                    "instruction": instructions[i],
                    "expected_output": labels[i],
                    "model_output": response,
                    "predicted_label": pred_label
                })

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. Save inference results
        inference_path = OUTPUT_DIR / "inference_results_origin.jsonl"
        with open(inference_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Inference results saved to {inference_path}")

        # 2. Calculate and save evaluation metrics
        predictions = [r["predicted_label"] for r in results]
        true_labels = [r["expected_output"] for r in results]
        metrics = calculate_metrics(predictions, true_labels)
        
        # Save evaluation metrics
        metrics_path = OUTPUT_DIR / "inference_metrics_origin.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"Inference metrics saved to {metrics_path}")

        # Print evaluation metrics
        print("\n=== Inference Metrics ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Error Rate: {metrics['error_rate']:.4f}")
        
        print("\n=== Per-class Metrics ===")
        for label, class_metrics in metrics['per_class_metrics'].items():
            print(f"\n{label}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  F1 Score: {class_metrics['f1']:.4f}")
            print(f"  Error Rate: {class_metrics['error_rate']:.4f}")
            print(f"  Support: {class_metrics['support']}")
        
        print("\n=== Macro Average Metrics ===")
        print(f"Precision: {metrics['macro_metrics']['precision']:.4f}")
        print(f"Recall: {metrics['macro_metrics']['recall']:.4f}")
        print(f"F1 Score: {metrics['macro_metrics']['f1']:.4f}")
        
        print("\n=== Micro Average Metrics ===")
        print(f"Precision: {metrics['micro_metrics']['precision']:.4f}")
        print(f"Recall: {metrics['micro_metrics']['recall']:.4f}")
        print(f"F1 Score: {metrics['micro_metrics']['f1']:.4f}")
        
        # Print prediction distribution
        print("\n=== Prediction Distribution ===")
        pred_counts = Counter(predictions)
        total = len(predictions)
        for label, count in pred_counts.items():
            print(f"{label}: {count} ({count/total*100:.1f}%)")

        # 3. Create and save visualization
        # 3.1 Confusion matrix
        plt.figure(figsize=(10, 8))
        cm = np.array(metrics['confusion_matrix'])
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
        confusion_matrix_path = OUTPUT_DIR / "confusion_matrix_origin.png"
        plt.savefig(confusion_matrix_path)
        plt.close()

        # 3.2 Evaluation metrics bar chart
        plt.figure(figsize=(10, 6))
        metrics_names = ['Accuracy', 'Error Rate']
        metrics_values = [metrics['accuracy'], metrics['error_rate']]
        bars = plt.bar(metrics_names, metrics_values)
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        metrics_plot_path = OUTPUT_DIR / "metrics_bar_origin.png"
        plt.savefig(metrics_plot_path)
        plt.close()

        # 3.3 Prediction distribution pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(pred_counts.values(), labels=pred_counts.keys(), autopct='%1.1f%%')
        plt.title('Prediction Distribution')
        plt.tight_layout()
        distribution_plot_path = OUTPUT_DIR / "prediction_distribution_origin.png"
        plt.savefig(distribution_plot_path)
        plt.close()

        # 3.4 Create detailed evaluation metrics visualization
        self.plot_detailed_metrics(metrics, OUTPUT_DIR)

        print("\nVisualizations have been saved to:")
        print(f"- {confusion_matrix_path}")
        print(f"- {metrics_plot_path}")
        print(f"- {distribution_plot_path}")
        print(f"- {OUTPUT_DIR / 'per_class_metrics_origin.png'}")
        print(f"- {OUTPUT_DIR / 'macro_micro_comparison_origin.png'}")
        print(f"- {OUTPUT_DIR / 'error_rates_origin.png'}")

        return results, metrics

# ==== Main Program ====
if __name__ == "__main__":
    evaluator = ModelEvaluator(TEST_DATA_PATH)
    evaluator.evaluate() 