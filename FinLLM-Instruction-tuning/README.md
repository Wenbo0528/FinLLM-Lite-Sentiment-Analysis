# FinLLM-Instruction-tuning

基于DeepSeek-R1-Distill-Qwen-1.5B模型的金融情绪分析指令微调项目。本项目使用QLoRA（4-bit量化+LoRA）技术对模型进行高效微调，使其能够更好地理解和分析金融文本中的情绪倾向。

## 项目特点

- 使用QLoRA技术进行高效微调，显著降低显存需求
- 采用Instruction-tuning方法，提高模型对金融文本的理解能力
- 针对金融领域情绪分析任务进行优化
- 支持模型量化部署，便于实际应用

## 技术细节

### 模型架构
- 基础模型：DeepSeek-R1-Distill-Qwen-1.5B
- 微调方法：QLoRA (4-bit量化 + LoRA)
- LoRA配置：
  - rank (r) = 4
  - alpha = 8
  - 目标模块：["q_proj", "k_proj", "v_proj", "o_proj"]
  - dropout = 0.05

### 训练配置
- 训练轮数：2 epochs
- 批次大小：4 (per device)
- 梯度累积步数：8
- 学习率：1e-4
- 权重衰减：0.01
- 预热步数：10000
- 保存步数：1000
- 使用FP16训练

## 项目结构

```
FinLLM-Instruction-tuning/
├── data/                    # 训练数据目录
│   └── instruction_formatted_data.jsonl  # 指令格式的训练数据
├── train/                   # 训练相关代码
│   └── train.py            # 训练脚本
├── model_lora/             # 保存的LoRA模型权重
├── results/                # 训练结果和评估指标
├── Inference/             # 推理相关代码
│   ├── inference.py       # 微调后模型的推理脚本
│   └── inference_origin.py # 原始模型的推理脚本
└── backup_py/             # 备份代码
```

## 使用方法

### 环境配置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 准备训练数据：
- 将训练数据放在 `data/instruction_formatted_data.jsonl` 中
- 数据格式应为JSONL，每行包含instruction和output字段

### 训练模型

使用提供的训练脚本进行模型训练：

```bash
# Linux/Mac
./scripts/train.sh --model_name "my_model" --batch_size 4 --learning_rate 1e-4 --epochs 2

# Windows
scripts\train.bat --model_name "my_model" --batch_size 4 --learning_rate 1e-4 --epochs 2
```

### 模型推理

训练完成后，可以使用训练好的模型进行推理：

```bash
# Linux/Mac
./scripts/inference.sh --model_path "FinLLM-Instruction-tuning/model_lora" --input_file "data/test_queries.txt"

# Windows
scripts\inference.bat --model_path "FinLLM-Instruction-tuning\model_lora" --input_file "data\test_queries.txt"
```

## 性能优化

1. 显存优化：
   - 使用4-bit量化降低基础模型显存占用
   - 采用LoRA技术减少可训练参数
   - 使用梯度累积处理大批量数据

2. 训练效率：
   - 使用FP16混合精度训练
   - 采用checkpoint机制支持断点续训
   - 优化数据加载和处理流程

## 注意事项

1. 显存要求：
   - 建议使用至少16GB显存的GPU
   - 可以通过调整batch_size和gradient_accumulation_steps来适应不同显存大小

2. 数据格式：
   - 确保训练数据格式正确
   - 建议对数据进行预处理和清洗

3. 模型保存：
   - 模型权重保存在model_lora目录
   - 支持断点续训，可以从checkpoint恢复训练

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来帮助改进项目。 