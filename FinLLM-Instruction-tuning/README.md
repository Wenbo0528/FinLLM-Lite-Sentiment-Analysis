# FinLLM Instruction Tuning

这个项目使用LoRA方法对大型语言模型进行指令微调，用于金融情感分析任务。

## 项目结构

```
FinLLM-Instruction-tuning/
├── data/                      # 数据目录
│   └── instruction_formatted_data.jsonl  # 指令格式化的数据
├── evaluation/                # 评估模块
│   └── evaluate.py           # 评估脚本
├── train/                     # 训练模块
│   └── train.py              # 训练脚本
├── results/                   # 结果目录
│   └── finllm-lora/          # 微调后的模型权重
└── requirements.txt          # 项目依赖
```

## 环境配置

安装项目依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

数据格式为JSONL，每条数据包含以下字段：
```json
{
    "text": "Human: Determine the sentiment of the financial news as negative, neutral or positive: Tesla stock rose 5% after strong earnings.\nAssistant: positive"
}
```

## 模型训练

1. 配置训练参数：
   - 基础模型：bigscience/bloom-560m（测试用）
   - LoRA参数：rank=8, alpha=16
   - 训练轮数：3 epochs
   - 学习率：2e-4
   - Batch size：8
   - 梯度累积步数：2

2. 开始训练：
```bash
cd train
python train.py
```

训练完成后，模型权重将保存在 `results/finllm-lora` 目录下。

## 模型评估

1. 进入评估目录：
```bash
cd evaluation
```

2. 运行评估：
```bash
python evaluate.py
```

评估结果将保存在 `results/evaluation_results.json` 文件中，包含以下指标：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数

## 注意事项

1. 硬件要求：
   - GPU显存：建议至少8GB（使用小模型）
   - 支持FP16训练

2. 数据要求：
   - 确保数据格式正确
   - 建议使用平衡的数据集

3. 模型选择：
   - 当前使用BLOOM-560M作为基础模型（用于测试）
   - 正式训练时可更换为更大的模型（如ChatGLM3-6B）

## 常见问题

1. 显存不足：
   - 减小batch size
   - 增加gradient accumulation steps
   - 使用更小的基础模型

2. 训练效果不理想：
   - 检查数据质量
   - 调整LoRA参数
   - 增加训练轮数

## 引用

如果您使用了本项目，请引用：
```
@misc{finllm2024,
    title={FinLLM: Financial Sentiment Analysis with Instruction Tuning},
    author={Your Name},
    year={2024}
}
``` 