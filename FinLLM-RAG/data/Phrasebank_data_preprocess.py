import json
import os
from collections import Counter
import random

def convert_to_json():
    # Input and output file paths
    input_file = 'Sentences_50Agree.txt'
    output_file = 'phrasebank_50_agree.json'
    
    # Get the current directory (data folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create full paths
    input_path = os.path.join(current_dir, input_file)
    output_path = os.path.join(current_dir, output_file)
    
    # Read the input file and process each line
    data = []
    sentiment_data = {'positive': [], 'negative': [], 'neutral': []}
    
    with open(input_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split the line into text and sentiment
                text, sentiment = line.rsplit('@', 1)
                sentiment = sentiment.strip()
                
                # Create a document entry
                doc = {
                    'text': text.strip(),
                    'sentiment': sentiment,
                    'metadata': {
                        'source': 'Phrasebank',
                        'type': 'financial_news'
                    }
                }
                sentiment_data[sentiment].append(doc)
    
    # Get negative count as baseline
    negative_count = len(sentiment_data['negative'])
    print(f"\nOriginal data distribution:")
    for sentiment, docs in sentiment_data.items():
        print(f"{sentiment}: {len(docs)}")
    
    # Randomly sample positive and neutral to match negative count
    balanced_data = sentiment_data['negative'].copy()  # Keep all negative samples
    
    # Sample from positive
    if len(sentiment_data['positive']) > negative_count:
        balanced_data.extend(random.sample(sentiment_data['positive'], negative_count))
    else:
        balanced_data.extend(sentiment_data['positive'])
    
    # Sample from neutral
    if len(sentiment_data['neutral']) > negative_count:
        balanced_data.extend(random.sample(sentiment_data['neutral'], negative_count))
    else:
        balanced_data.extend(sentiment_data['neutral'])
    
    # Randomly shuffle the data
    random.shuffle(balanced_data)
    
    # Count distribution after balancing
    balanced_sentiments = Counter(doc['sentiment'] for doc in balanced_data)
    print(f"\nBalanced data distribution:")
    for sentiment, count in balanced_sentiments.items():
        print(f"{sentiment}: {count}")
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSuccessfully converted and balanced {len(balanced_data)} entries to JSON format")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    convert_to_json()
