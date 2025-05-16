import os

def analyze_sentiment_distribution(file_path):
    # Read txt file
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split text and sentiment label
                text, sentiment = line.rsplit('@', 1)
                sentiment = sentiment.strip()
                sentiment_counts[sentiment] += 1
    
    # Print statistics
    print("\n=== Sentiment Distribution Statistics ===")
    total = sum(sentiment_counts.values())
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total) * 100
        print(f"{sentiment}: {count} ({percentage:.2f}%)")
    print(f"\nTotal samples: {total}")

if __name__ == "__main__":
    # Use correct local path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(os.path.dirname(current_dir), "data", "Sentences_AllAgree.txt")
    analyze_sentiment_distribution(input_file) 