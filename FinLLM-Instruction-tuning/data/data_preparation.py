import json
from pathlib import Path

# 手动写几条测试样本（你可以替换成从 CSV/JSON 加载）
examples = [
    {"text": "Tesla stock rose 5% after strong earnings.", "label": "positive"},
    {"text": "Company announces layoffs of 1000 employees.", "label": "negative"},
    {"text": "The market closed with little change today.", "label": "neutral"},
]

# 模板（指令）
instruction = "Determine the sentiment of the financial news as negative, neutral or positive: "

# 格式化为 instruction-tuning 风格
formatted = [
    {
        "text": f"Human: {instruction}{ex['text']}\nAssistant: {ex['label']}"
    }
    for ex in examples
]

# 写入 JSONL 文件
output_path = Path("data/instruction_formatted_data.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for item in formatted:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ 写入 {len(formatted)} 条数据到 {output_path}")
