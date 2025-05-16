import json
import os

def convert_to_json():
    # Input and output file paths
    input_file = 'Sentences_AllAgree.txt'
    output_file = 'phrasebank_all_agree.json'
    
    # Get the current directory (data folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create full paths
    input_path = os.path.join(current_dir, input_file)
    output_path = os.path.join(current_dir, output_file)
    
    # Read the input file and process each line
    data = []
    with open(input_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split the line into text and sentiment
                text, sentiment = line.rsplit('@', 1)
                
                # Create a document entry
                doc = {
                    'text': text.strip(),
                    'sentiment': sentiment.strip(),
                    'metadata': {
                        'source': 'Phrasebank',
                        'type': 'financial_news'
                    }
                }
                data.append(doc)
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully converted {len(data)} entries to JSON format")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    convert_to_json()
