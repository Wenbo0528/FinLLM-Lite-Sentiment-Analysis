# FinLLM-RAG: 金融新闻情感分析 RAG 系统

这个项目实现了一个基于 RAG (Retrieval-Augmented Generation) 的金融新闻情感分析系统，通过结合实时金融新闻数据和微调的语言模型，提供更准确的情感分析结果。

## 项目结构

```
FinLLM-RAG/
├── data_sources/              # 数据源目录
│   ├── get_rag_context_data.py    # 网页爬虫脚本
│   └── rag_context_data.jsonl     # 爬取的新闻数据
├── inference/                 # 推理模块
│   └── rag_retrieve_and_infer.py  # RAG 推理脚本
├── evaluation/               # 评估模块
│   └── evaluate_rag_results.py    # RAG 结果评估脚本
├── results/                  # 结果目录
│   ├── evaluation_report.json     # 评估报告
│   ├── sentiment_comparison.png   # 情感分布对比图
│   └── non_rag_predictions.json  # 非 RAG 预测结果
└── requirements.txt         # 项目依赖
```

## 环境配置

安装项目依赖：
```bash
pip install -r requirements.txt
```

主要依赖包括：
- transformers
- peft
- torch
- beautifulsoup4
- requests
- fake-useragent
- scikit-learn
- matplotlib
- seaborn

## 数据获取

1. 运行网页爬虫脚本获取金融新闻数据：
```bash
cd data_sources
python get_rag_context_data.py
```

爬虫脚本会从以下来源获取数据：
- Yahoo Finance
- MarketWatch
- Seeking Alpha

数据将保存在 `data_sources/rag_context_data.jsonl` 文件中。

## 模型推理

1. 确保已安装所有依赖
2. 确保 LoRA 模型已训练完成（位于 `FinLLM-Instruction-tuning/model_lora`）
3. 运行 RAG 推理脚本：
```bash
cd inference
python rag_retrieve_and_infer.py
```

推理结果将保存在 `inference/rag_inference_results.json` 文件中。

## 结果评估

运行评估脚本分析 RAG 和非 RAG 的结果：
```bash
cd evaluation
python evaluate_rag_results.py
```

评估结果将保存在 `results` 目录下，包括：
- `evaluation_report.json`: 详细的评估报告
- `sentiment_comparison.png`: RAG 和非 RAG 的情感分布对比图
- `non_rag_predictions.json`: 不使用 RAG 的预测结果

## 系统特点

1. 实时数据获取：
   - 自动爬取最新金融新闻
   - 支持多个数据源
   - 包含反爬虫措施

2. 智能检索：
   - 基于相似度的上下文检索
   - 支持多源数据融合
   - 保留来源信息

3. 对比评估：
   - RAG vs 非 RAG 性能对比
   - 详细的评估指标
   - 可视化分析结果

## 注意事项

1. 硬件要求：
   - GPU 显存：建议至少 8GB
   - 支持 FP16 训练和推理

2. 网络要求：
   - 需要稳定的网络连接
   - 可能需要配置代理

3. 数据要求：
   - 确保数据源可访问
   - 定期更新新闻数据

## 使用示例

1. 获取新闻数据：
```python
from data_sources.get_rag_context_data import scrape_financial_news
news_data = scrape_financial_news()
```

2. 运行 RAG 推理：
```python
from inference.rag_retrieve_and_infer import run_inference
result = run_inference("Tesla stock rose 5% after strong earnings")
```

3. 评估结果：
```python
from evaluation.evaluate_rag_results import RAGEvaluator
evaluator = RAGEvaluator("inference/rag_inference_results.json")
evaluator.generate_report()
```

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目。在提交代码前，请确保：
1. 代码符合 PEP 8 规范
2. 添加必要的注释和文档
3. 更新相关的测试用例

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 