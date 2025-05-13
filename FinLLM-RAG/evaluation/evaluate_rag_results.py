import json
import os
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# ==== Configuration ====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = "bigscience/bloom-560m"
LORA_PATH = os.path.join(PROJECT_ROOT, "..", "FinLLM-Instruction-tuning", "model_lora")
INFERENCE_RESULTS_PATH = os.path.join(PROJECT_ROOT, "inference", "rag_inference_results.json")
OUTPUT_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== Setup Logging ====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==== RAG Evaluator Class ====
class RAGEvaluator:
    def __init__(self, results_path):
        self.results_path = results_path
        self.results = self.load_results()
        self.output_dir = OUTPUT_DIR
        self.non_rag_predictions = self.get_non_rag_predictions()

    def load_results(self):
        with open(self.results_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_non_rag_predictions(self):
        logger.info("Getting non-RAG predictions...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token

        # Use BitsAndBytesConfig for quantization
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        # Check if LoRA path exists
        if not os.path.exists(LORA_PATH):
            raise ValueError(f"LoRA model path does not exist: {LORA_PATH}")
        
        model = PeftModel.from_pretrained(base, LORA_PATH)
        model.eval()

        non_rag_predictions = []
        for item in self.results:
            query = item["query"]
            prompt = f"Human: Determine the sentiment of the financial news as negative, neutral or positive: {query}\nAssistant:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=20)
            
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            prediction = prediction.split("Assistant:")[-1].strip()
            
            non_rag_predictions.append({
                "query": query,
                "prediction": prediction
            })

        # Save non-RAG predictions
        output_path = self.output_dir / "non_rag_predictions.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(non_rag_predictions, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved non-RAG predictions to {output_path}")

        return non_rag_predictions

    def analyze_retrieval_quality(self):
        logger.info("Analyzing retrieval quality...")
        
        # Calculate average similarity score
        scores = [item["retrieval"]["score"] for item in self.results]
        avg_score = sum(scores) / len(scores)
        logger.info(f"Average similarity score: {avg_score:.4f}")

        # Analyze source distribution
        sources = [item["retrieval"]["source"] for item in self.results]
        source_dist = Counter(sources)
        logger.info("\nSource distribution:")
        for source, count in source_dist.items():
            logger.info(f"{source}: {count}")

        # Plot retrieval scores distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(scores, bins=20)
        plt.title("Distribution of Retrieval Similarity Scores")
        plt.xlabel("Similarity Score")
        plt.ylabel("Count")
        plt.savefig(self.output_dir / "retrieval_scores_distribution.png")
        plt.close()

        # Save retrieval quality analysis
        analysis = {
            "average_similarity_score": avg_score,
            "source_distribution": dict(source_dist)
        }
        output_path = self.output_dir / "retrieval_quality.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved retrieval quality analysis to {output_path}")

    def analyze_sentiment_distribution(self):
        logger.info("Analyzing sentiment distribution...")
        
        # Analyze RAG predictions
        rag_sentiments = [item["prediction"].lower() for item in self.results]
        rag_dist = Counter(rag_sentiments)
        
        # Analyze non-RAG predictions
        non_rag_sentiments = [item["prediction"].lower() for item in self.non_rag_predictions]
        non_rag_dist = Counter(non_rag_sentiments)

        # Plot sentiment distribution comparison
        plt.figure(figsize=(12, 6))
        x = np.arange(len(set(rag_sentiments + non_rag_sentiments)))
        width = 0.35

        plt.bar(x - width/2, [rag_dist[s] for s in sorted(set(rag_sentiments + non_rag_sentiments))], 
                width, label='RAG')
        plt.bar(x + width/2, [non_rag_dist[s] for s in sorted(set(rag_sentiments + non_rag_sentiments))], 
                width, label='Non-RAG')

        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title('Sentiment Distribution Comparison')
        plt.xticks(x, sorted(set(rag_sentiments + non_rag_sentiments)))
        plt.legend()
        plt.savefig(self.output_dir / "sentiment_comparison.png")
        plt.close()

        # Save sentiment distributions
        distributions = {
            "rag_sentiment_distribution": dict(rag_dist),
            "non_rag_sentiment_distribution": dict(non_rag_dist)
        }
        output_path = self.output_dir / "sentiment_distribution.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(distributions, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved sentiment distribution analysis to {output_path}")

    def generate_report(self):
        logger.info("Generating evaluation report...")
        
        report = {
            "total_queries": len(self.results),
            "retrieval_quality": {
                "average_similarity_score": sum(item["retrieval"]["score"] for item in self.results) / len(self.results),
                "source_distribution": dict(Counter(item["retrieval"]["source"] for item in self.results))
            },
            "sentiment_distribution": {
                "rag": dict(Counter(item["prediction"].lower() for item in self.results)),
                "non_rag": dict(Counter(item["prediction"].lower() for item in self.non_rag_predictions))
            }
        }

        output_path = self.output_dir / "evaluation_report.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved evaluation report to {output_path}")

# ==== Main Program ====
if __name__ == "__main__":
    evaluator = RAGEvaluator(INFERENCE_RESULTS_PATH)
    evaluator.analyze_retrieval_quality()
    evaluator.analyze_sentiment_distribution()
    evaluator.generate_report() 