import json
import torch
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==== 配置项 ====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = "bigscience/bloom-560m"
LORA_PATH = os.path.join(PROJECT_ROOT, "..", "FinLLM-Instruction-tuning", "model_lora")
RAG_DATA_PATH = os.path.join(PROJECT_ROOT, "data_sources", "rag_context_data.jsonl")
EVAL_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_inference_results.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== 背景检索函数 ====
def retrieve_context(query, rag_data_path):
    def overlap(query, context):
        q_words = set(query.lower().split())
        c_words = set(context.lower().split())
        return len(q_words & c_words) / (min(len(q_words), len(c_words)) + 1e-6)

    with open(rag_data_path, 'r', encoding='utf-8') as f:
        rag_data = [json.loads(line) for line in f]

    best_context, best_score = "", 0
    best_source = ""
    best_title = ""
    best_url = ""

    for item in rag_data:
        if item["query"] == query:
            context = item["retrieved"]["context"]
            score = overlap(query, context)
            if score > best_score:
                best_score = score
                best_context = context
                best_source = item["retrieved"]["source"]
                best_title = item["retrieved"]["title"]
                best_url = item["retrieved"]["url"]

    return {
        "context": best_context,
        "score": best_score,
        "source": best_source,
        "title": best_title,
        "url": best_url
    }

# ==== Prompt 拼接 ====
def build_prompt(context, query):
    return f"{context}\n\nHuman: Determine the sentiment of the financial news as negative, neutral or positive: {query}\nAssistant:"

# ==== 模型加载 + 推理 ====
def run_inference(prompt):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # 使用 BitsAndBytesConfig 进行量化
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
    
    # 检查 LoRA 路径是否存在
    if not os.path.exists(LORA_PATH):
        raise ValueError(f"LoRA model path does not exist: {LORA_PATH}")
    
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.split("Assistant:")[-1].strip()

# ==== 保存评估结果 ====
def save_eval_results(results, output_path):
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# ==== 主程序 ====
if __name__ == "__main__":
    # 读取所有查询
    with open(RAG_DATA_PATH, 'r', encoding='utf-8') as f:
        queries = [json.loads(line)["query"] for line in f]

    eval_results = []
    
    for query in queries:
        print(f"\n🟡 处理查询：{query}")
        
        # 检索背景信息
        retrieval_result = retrieve_context(query, RAG_DATA_PATH)
        context = retrieval_result["context"]
        
        if not context:
            print(f"❌ 未找到相关背景信息")
            continue
            
        print(f"\n🟢 检索到的背景信息：")
        print(f"来源: {retrieval_result['source']}")
        print(f"标题: {retrieval_result['title']}")
        print(f"URL: {retrieval_result['url']}")
        print(f"相似度分数: {retrieval_result['score']:.2f}")
        print(f"内容: {context[:200]}...")

        # 构建 prompt 并运行推理
        prompt = build_prompt(context, query)
        prediction = run_inference(prompt)
        print(f"\n✅ 模型预测结果：{prediction}")

        # 保存结果
        eval_results.append({
            "query": query,
            "retrieval": retrieval_result,
            "prediction": prediction
        })

    # 保存所有评估结果
    save_eval_results(eval_results, EVAL_RESULTS_PATH)
    print(f"\n📊 已保存 {len(eval_results)} 条评估结果到 {EVAL_RESULTS_PATH}")
