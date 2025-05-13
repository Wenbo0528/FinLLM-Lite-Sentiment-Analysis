import json
import os
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGEvaluator:
    def __init__(self, results_path):
        self.results_path = results_path
        self.results = self._load_results()
        # 修改输出目录为 results
        self.output_dir = Path(__file__).parent.parent / "results"
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _load_results(self):
        """加载推理结果"""
        with open(self.results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_non_rag_predictions(self):
        """获取不使用 RAG 的预测结果"""
        # 加载模型
        base_model = "bigscience/bloom-560m"
        lora_path = Path(__file__).parent.parent.parent / "FinLLM-Instruction-tuning" / "results" / "finllm-lora"
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token
        
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        model = PeftModel.from_pretrained(base, lora_path)
        model.eval()
        
        non_rag_results = []
        for item in self.results:
            query = item['query']
            # 构建不带上下文的 prompt
            prompt = f"Human: Determine the sentiment of the financial news as negative, neutral or positive: {query}\nAssistant:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=20)
            
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            prediction = prediction.split("Assistant:")[-1].strip()
            
            non_rag_results.append({
                "query": query,
                "prediction": prediction
            })
        
        # 保存非 RAG 预测结果
        non_rag_path = self.output_dir / "non_rag_predictions.json"
        with open(non_rag_path, 'w', encoding='utf-8') as f:
            json.dump(non_rag_results, f, ensure_ascii=False, indent=2)
        logging.info(f"非 RAG 预测结果已保存到: {non_rag_path}")
        
        return non_rag_results
    
    def analyze_retrieval_quality(self):
        """分析检索质量"""
        scores = [item['retrieval']['score'] for item in self.results]
        sources = [item['retrieval']['source'] for item in self.results]
        
        avg_score = sum(scores) / len(scores)
        logging.info(f"\n📊 检索质量分析:")
        logging.info(f"平均相似度分数: {avg_score:.3f}")
        
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logging.info("\n来源分布:")
        for source, count in source_counts.items():
            logging.info(f"{source}: {count} 篇文章")
        
        # 保存检索质量分析结果
        retrieval_quality = {
            "average_score": avg_score,
            "source_distribution": source_counts
        }
        retrieval_path = self.output_dir / "retrieval_quality.json"
        with open(retrieval_path, 'w', encoding='utf-8') as f:
            json.dump(retrieval_quality, f, ensure_ascii=False, indent=2)
        logging.info(f"检索质量分析结果已保存到: {retrieval_path}")
        
        return scores, sources
    
    def analyze_sentiment_distribution(self, results, label="RAG"):
        """分析情感分布"""
        sentiments = [item['prediction'].lower() for item in results]
        sentiment_counts = {}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        logging.info(f"\n🎯 {label} 情感分布:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(sentiments)) * 100
            logging.info(f"{sentiment}: {count} 条 ({percentage:.1f}%)")
        
        # 保存情感分布分析结果
        sentiment_path = self.output_dir / f"{label.lower()}_sentiment_distribution.json"
        with open(sentiment_path, 'w', encoding='utf-8') as f:
            json.dump(sentiment_counts, f, ensure_ascii=False, indent=2)
        logging.info(f"{label} 情感分布分析结果已保存到: {sentiment_path}")
        
        return sentiment_counts
    
    def plot_sentiment_comparison(self, rag_counts, non_rag_counts):
        """绘制 RAG 和非 RAG 的情感分布对比图"""
        plt.figure(figsize=(12, 6))
        
        # 准备数据
        sentiments = sorted(set(rag_counts.keys()) | set(non_rag_counts.keys()))
        rag_values = [rag_counts.get(s, 0) for s in sentiments]
        non_rag_values = [non_rag_counts.get(s, 0) for s in sentiments]
        
        x = range(len(sentiments))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], rag_values, width, label='With RAG')
        plt.bar([i + width/2 for i in x], non_rag_values, width, label='Without RAG')
        
        plt.xlabel('情感类别')
        plt.ylabel('数量')
        plt.title('RAG vs Non-RAG 情感分布对比')
        plt.xticks(x, sentiments)
        plt.legend()
        
        plt.savefig(self.output_dir / 'sentiment_comparison.png')
        plt.close()
        logging.info(f"情感分布对比图已保存到: {self.output_dir / 'sentiment_comparison.png'}")
    
    def generate_report(self):
        """生成评估报告"""
        # 分析检索质量
        scores, sources = self.analyze_retrieval_quality()
        
        # 获取非 RAG 预测结果
        non_rag_results = self.get_non_rag_predictions()
        
        # 分析情感分布
        rag_sentiment_counts = self.analyze_sentiment_distribution(self.results, "RAG")
        non_rag_sentiment_counts = self.analyze_sentiment_distribution(non_rag_results, "Non-RAG")
        
        # 绘制对比图
        self.plot_sentiment_comparison(rag_sentiment_counts, non_rag_sentiment_counts)
        
        # 保存详细报告
        report = {
            "total_queries": len(self.results),
            "retrieval_quality": {
                "average_score": sum(scores) / len(scores),
                "source_distribution": {source: sources.count(source) for source in set(sources)}
            },
            "sentiment_distribution": {
                "with_rag": rag_sentiment_counts,
                "without_rag": non_rag_sentiment_counts
            }
        }
        
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logging.info(f"\n📝 评估报告已保存到: {report_path}")

def main():
    # 获取推理结果文件路径
    inference_dir = Path(__file__).parent.parent / "inference"
    results_path = inference_dir / "rag_inference_results.json"
    
    if not results_path.exists():
        logging.error(f"找不到推理结果文件: {results_path}")
        return
    
    # 创建评估器并生成报告
    evaluator = RAGEvaluator(results_path)
    evaluator.generate_report()

if __name__ == "__main__":
    main() 