# FinLLM-Sentiment-Analysis

基于大语言模型的金融文本情绪分析系统，结合了指令微调（Instruction-tuning）和检索增强生成（RAG）技术，提供更准确、更可靠的金融文本情绪分析能力。

## 项目结构

```
FinLLM-Sentiment-Analysis/
├── FinLLM-Instruction-tuning/  # 指令微调模块
│   ├── data/                  # 训练数据
│   ├── train/                # 训练相关代码
│   ├── Inference/           # 推理相关代码
│   └── model_lora/          # 微调后的模型
├── FinLLM-RAG/              # RAG增强模块
│   ├── data/               # 知识库数据
│   ├── inference/         # RAG推理代码
│   └── results/          # 结果输出
└── scripts/              # 通用脚本
    ├── train.sh         # 训练脚本
    ├── train.bat       # Windows训练脚本
    ├── inference.sh    # 推理脚本
    ├── inference.bat   # Windows推理脚本
    ├── run_rag.sh      # RAG推理脚本
    ├── run_rag.bat     # Windows RAG推理脚本
    ├── evaluate.sh     # 评估脚本
    └── evaluate.bat    # Windows评估脚本
```

## 模块说明

### 1. FinLLM-Instruction-tuning

基于指令微调（Instruction-tuning）和LoRA（Low-Rank Adaptation）技术的模型微调模块。通过精心设计的指令模板和金融领域数据，提升模型对金融文本情绪的理解能力。

主要特点：
- 使用 DeepSeek-R1-Distill-Qwen-1.5B 作为基础模型
- 采用 LoRA 技术进行高效微调
- 支持 4-bit 量化训练
- 提供完整的训练和推理脚本

详细说明请参考 [FinLLM-Instruction-tuning/README.md](FinLLM-Instruction-tuning/README.md)

### 2. FinLLM-RAG

基于检索增强生成（RAG）技术的情绪分析增强模块。通过结合外部知识库，提供更准确、更可靠的金融文本情绪分析能力。

主要特点：
- 支持离线知识库（Phrasebank）和在线爬取数据
- 基于向量数据库的相似度检索
- 可配置的检索参数
- 完整的评估指标

详细说明请参考 [FinLLM-RAG/README.md](FinLLM-RAG/README.md)

## 快速开始

1. 环境配置：
```bash
# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

2. 模型训练：
```bash
# Linux/Mac
./scripts/train.sh

# Windows
scripts\train.bat
```

3. 模型推理：
```bash
# 使用原始模型推理
# Linux/Mac
./scripts/inference.sh

# Windows
scripts\inference.bat

# 使用RAG增强推理
# Linux/Mac
./scripts/run_rag.sh

# Windows
scripts\run_rag.bat
```

4. 结果评估：
```bash
# Linux/Mac
./scripts/evaluate.sh

# Windows
scripts\evaluate.bat
```

## 系统要求

- Python 3.8+
- CUDA 支持的 GPU（推荐）
- 至少 8GB GPU 显存
- 16GB+ 系统内存

## 注意事项

1. 数据准备：
   - 确保训练数据格式正确
   - 根据需要选择使用离线或在线知识库

2. 模型训练：
   - 建议使用 GPU 进行训练
   - 可以通过修改训练参数优化性能

3. 推理部署：
   - 支持 CPU 和 GPU 推理
   - 可以根据需求调整批处理大小

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进项目。在提交代码前，请确保：
1. 代码符合项目的编码规范
2. 添加必要的注释和文档
3. 更新相关的测试用例
