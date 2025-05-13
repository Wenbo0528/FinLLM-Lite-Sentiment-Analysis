import json
import os
from pathlib import Path

# ==== Configuration ====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw_data.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "instruction_formatted_data.jsonl")

# ==== Data Preparation Function ====
def prepare_instruction_data():
    """
    Prepare instruction-formatted data for fine-tuning.
    Converts raw data into instruction-output pairs.
    """
    # Load raw data
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Format data into instruction-output pairs
    formatted_data = []
    for item in raw_data:
        instruction = f"Determine the sentiment of the financial news as negative, neutral or positive: {item['text']}"
        output = item['sentiment']
        
        formatted_data.append({
            "instruction": instruction,
            "output": output
        })

    # Save formatted data
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Data preparation completed. Output saved to: {OUTPUT_PATH}")
    print(f"📊 Total samples: {len(formatted_data)}")

# ==== Main Program ====
if __name__ == "__main__":
    # Ensure output directory exists
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Run data preparation
    prepare_instruction_data()
