import json
import torch
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ==== Configuration ====
# Original paths (commented for local environment)
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_MODEL = "deepseek-ai/deepseek-llm-1.3b-base"
# LORA_PATH = os.path.join(PROJECT_ROOT, "..", "FinLLM-Instruction-tuning", "model_lora")
# RAG_DATA_PATH = os.path.join(PROJECT_ROOT, "data_sources", "rag_context_data.jsonl")
# EVAL_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_inference_results.json")

# Google Drive paths and model configuration
DRIVE_ROOT = "/content/drive/MyDrive/Columbia_Semester3/5293/Final/FinLLM-Sentiment-Analysis"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LORA_PATH = os.path.join(DRIVE_ROOT, "FinLLM-Instruction-tuning/model_lora")
RAG_DATA_PATH = os.path.join(DRIVE_ROOT, "FinLLM-RAG/data_sources/rag_context_data.jsonl")
EVAL_RESULTS_PATH = os.path.join(DRIVE_ROOT, "FinLLM-RAG/inference/rag_inference_results.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Model Configuration ====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 更新为4-bit量化
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# ==== Context Retrieval Function ====
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

# ==== Prompt Construction ====
def build_prompt(context, query):
    return f"{context}\n\nHuman: Determine the sentiment of the financial news as negative, neutral or positive: {query}\nAssistant:"

# ==== Model Loading and Inference ====
def run_inference(prompt):
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Check if LoRA path exists
    if not os.path.exists(LORA_PATH):
        raise ValueError(f"LoRA model path does not exist: {LORA_PATH}")
    
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.replace(prompt, "").strip()

# ==== Save Evaluation Results ====
def save_eval_results(results, output_path):
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# ==== Main Program ====
if __name__ == "__main__":
    # Read all queries
    with open(RAG_DATA_PATH, 'r', encoding='utf-8') as f:
        queries = [json.loads(line)["query"] for line in f]

    eval_results = []
    
    for query in queries:
        print(f"\n🟡 Processing query: {query}")
        
        # Retrieve context
        retrieval_result = retrieve_context(query, RAG_DATA_PATH)
        context = retrieval_result["context"]
        
        if not context:
            print(f"❌ No relevant context found")
            continue
            
        print(f"\n🟢 Retrieved context:")
        print(f"Source: {retrieval_result['source']}")
        print(f"Title: {retrieval_result['title']}")
        print(f"URL: {retrieval_result['url']}")
        print(f"Similarity score: {retrieval_result['score']:.2f}")
        print(f"Content: {context[:200]}...")

        # Build prompt and run inference
        prompt = build_prompt(context, query)
        prediction = run_inference(prompt)
        print(f"\n✅ Model prediction: {prediction}")

        # Save results
        eval_results.append({
            "query": query,
            "retrieval": retrieval_result,
            "prediction": prediction
        })

    # Save all evaluation results
    save_eval_results(eval_results, EVAL_RESULTS_PATH)
    print(f"\n📊 Saved {len(eval_results)} evaluation results to {EVAL_RESULTS_PATH}")
