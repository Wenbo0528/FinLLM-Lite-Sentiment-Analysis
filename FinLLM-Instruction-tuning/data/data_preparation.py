import json
import os
from pathlib import Path
from datasets import load_dataset
import random

# ==== Configuration ====
# Original paths (commented for local environment)
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TRAIN_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "instruction_formatted_data.jsonl")
# TEST_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "validation_data.jsonl")

# Google Drive paths (commented out)
DRIVE_ROOT = "/content/drive/MyDrive/Columbia_Semester3/5293/Final/FinLLM-Sentiment-Analysis/FinLLM-Instruction-tuning"
TRAIN_OUTPUT_PATH = os.path.join(DRIVE_ROOT, "data", "instruction_formatted_data.jsonl")
TEST_OUTPUT_PATH = os.path.join(DRIVE_ROOT, "data", "validation_data.jsonl")

# Model configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Updated model name

# Dataset configuration
DATASETS = [
    {
        "name": "FinGPT/fingpt-sentiment-train",
        "split": "train",
        "input_field": "input",
        "output_field": "output",
        "label_type": "text",  # Original dataset uses text labels
        "train_ratio": 0.8  # 80% for training, 20% for testing
    },
    {
        "name": "zeroshot/twitter-financial-news-sentiment",
        "split": "train",
        "input_field": "text",
        "output_field": "label",
        "label_type": "numeric",  # Twitter dataset uses numeric labels (0: negative, 1: neutral, 2: positive)
        "label_mapping": {
            "0": "negative",
            "1": "neutral",
            "2": "positive"
        },
        "train_ratio": 0.8  # 80% for training, 20% for testing
    }
]

# Create output directories if they don't exist
os.makedirs(os.path.dirname(TRAIN_OUTPUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(TEST_OUTPUT_PATH), exist_ok=True)

# Set random seed for reproducibility
random.seed(42)

# ==== Label Normalization ====
def normalize_to_three_class(label: str) -> str:
    """
    Normalize different label formats to three classes: positive, negative, or neutral.
    Handles both text labels and numeric labels (0,1,2).
    
    Args:
        label: Input label (can be text or numeric)
    
    Returns:
        Normalized label as one of: "positive", "negative", "neutral", or "unknown"
    """
    label = str(label).lower()
    if "positive" in label or label == "2":
        return "positive"
    elif "negative" in label or label == "0":
        return "negative"
    elif "neutral" in label or label == "1":
        return "neutral"
    else:
        return "unknown"

def format_instruction(instruction, input_text, output_text):
    """
    Format instruction data for training.
    Creates a structured prompt with system, user, and assistant messages.
    
    Args:
        instruction: The task instruction
        input_text: The input text to analyze
        output_text: The expected output/sentiment label
    
    Returns:
        Formatted instruction string with chat template
    """
    return f"""<|im_start|>system
You are a helpful AI assistant that analyzes financial sentiment. Your task is to classify the sentiment of financial text into one of the following categories: positive, negative, or neutral. Please provide your analysis in a clear and concise manner.

<|im_start|>user
{instruction}

Input: {input_text}

<|im_start|>assistant
{output_text}<|im_end|>"""

def process_dataset(dataset_config):
    """
    Process a single dataset and split it into train and test portions.
    
    Args:
        dataset_config: Configuration for the dataset to process
        
    Returns:
        Tuple of (train_data, test_data) lists
    """
    train_data = []
    test_data = []
    print(f"Processing dataset: {dataset_config['name']}")
    
    try:
        dataset = load_dataset(dataset_config["name"], split=dataset_config["split"])
        # Convert to list and shuffle
        dataset_list = list(dataset)
        random.shuffle(dataset_list)
        
        # Calculate split point
        split_idx = int(len(dataset_list) * dataset_config["train_ratio"])
        
        # Process each item
        for idx, item in enumerate(dataset_list):
            # Get input and output fields based on dataset configuration
            input_text = item[dataset_config["input_field"]]
            output = item[dataset_config["output_field"]]
            
            # Handle different label types
            if dataset_config["label_type"] == "numeric":
                # Convert numeric label to text using mapping
                output = dataset_config["label_mapping"][str(output)]
            else:
                # Use existing normalization for text labels
                output = normalize_to_three_class(output)
            
            if output == "unknown":
                continue  # Skip unrecognized labels

            # Format instruction
            instruction = f"What is the sentiment of the financial text? Please choose an answer from {{negative/neutral/positive}}: {input_text}".strip()

            formatted_item = {
                "instruction": instruction,
                "output": output
            }
            
            # Add to appropriate list based on split
            if idx < split_idx:
                train_data.append(formatted_item)
            else:
                test_data.append(formatted_item)
            
        print(f"[SUCCESS] Successfully processed {len(dataset)} samples from {dataset_config['name']}")
        print(f"   - Training samples: {len(train_data)}")
        print(f"   - Testing samples: {len(test_data)}")
        
    except Exception as e:
        print(f"[ERROR] Error processing dataset {dataset_config['name']}: {str(e)}")
    
    return train_data, test_data

def prepare_instruction_data():
    """
    Prepare instruction-formatted data for fine-tuning and evaluation.
    Processes all datasets and splits them into train and test portions.
    
    The function:
    1. Processes each dataset and splits it into train/test portions
    2. Combines all training portions into one file
    3. Combines all testing portions into another file
    """
    all_train_data = []
    all_test_data = []
    
    # Process each dataset
    for dataset_config in DATASETS:
        train_data, test_data = process_dataset(dataset_config)
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)
    
    # Save training data
    with open(TRAIN_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in all_train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\n[SUCCESS] Training data saved to: {TRAIN_OUTPUT_PATH}")
    print(f"[INFO] Total training samples: {len(all_train_data)}")

    # Save test data
    with open(TEST_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in all_test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"[SUCCESS] Test data saved to: {TEST_OUTPUT_PATH}")
    print(f"[INFO] Total test samples: {len(all_test_data)}")

# ==== Main Program ====
if __name__ == "__main__":
    Path(TRAIN_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(TEST_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    prepare_instruction_data()
