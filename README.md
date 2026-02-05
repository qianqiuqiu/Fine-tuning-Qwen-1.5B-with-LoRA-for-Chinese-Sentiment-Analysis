# 🎯 Qwen-1.5B 中文情感分析微调项目

基于 LoRA/QLoRA 技术微调 Qwen2.5-1.5B 模型，实现高效的中文情感分析任务。

## 📋 项目简介

本项目是一个完整的端到端中文情感分析解决方案，展示了如何使用参数高效微调（PEFT）技术训练大语言模型。项目特点：

- **高效微调**：使用 LoRA/QLoRA 技术，仅需微调少量参数（约 1% 模型参数）
- **低显存占用**：QLoRA 模式下仅需 4GB 显存即可训练
- **完整工作流**：涵盖数据处理、模型训练、评估、推理全流程
- **详细评估**：提供多维度模型评估工具，包括置信度分析、鲁棒性测试等

### 技术栈

| 组件 | 技术/框架 |
|------|----------|
| 基础模型 | Qwen2.5-1.5B |
| 微调方法 | LoRA / QLoRA (4-bit) |
| 训练框架 | HuggingFace Transformers + Trainer |
| 参数高效微调 | PEFT (Parameter-Efficient Fine-Tuning) |
| 量化库 | BitsAndBytes |
| 数据集 | ChnSentiCorp (中文情感语料) |

### 项目结构

```
Fine-tuning-Qwen-1.5B/
├── configs/                      # 配置模块
│   ├── lora_config.py           # LoRA 超参数配置
│   └── training_config.py       # 训练参数配置
├── data/                         # 数据处理模块
│   ├── data_loader.py           # 数据集加载
│   └── preprocessing.py         # 数据预处理
├── model_evaluation/             # 模型评估模块
│   ├── run_full_eval.py         # 完整评估流程
│   ├── confidence_analysis.py   # 置信度分析
│   ├── robustness_test.py       # 鲁棒性测试
│   ├── benchmark.py             # 基准对比
│   ├── report_generator.py      # 评估报告生成
│   └── outputs/                 # 评估结果输出（详见下方说明）
├── outputs/                      # 训练输出目录
│   └── lora_adapter/            # 微调后的 LoRA 适配器
├── train.py                      # 训练主脚本
├── eval.py                       # 基础评估脚本
├── inference.py                  # 推理脚本
├── requirements.txt              # 项目依赖
└── README.md                     # 项目文档
```

## 🚀 Quick Start

### 环境配置

**1. 创建虚拟环境（推荐）**

```bash
# # Windows
# python -m venv .venv
# .venv\Scripts\activate
建议用conda,一些支持会更好。
# # Linux/Mac
# python -m venv .venv
# source .venv/bin/activate
```

**2. 安装依赖**

```bash
pip install -r requirements.txt
```

**硬件要求**

| 模式 | 最小显存 | 推荐显存 | 训练速度 |
|------|---------|---------|---------|
| LoRA | 8GB | 16GB | 快 |
| QLoRA (4-bit) | 4GB | 8GB | 较慢 |

### 模型训练

**基础训练**

```bash
# 使用默认配置（LoRA，r=8）
python train.py

# 使用 QLoRA 节省显存（推荐显存不足时使用）
python train.py --use_qlora
```

**自定义参数训练**

```bash
python train.py \
    --lora_r 16 \              # LoRA 秩（rank）
    --lora_alpha 32 \          # LoRA alpha 参数
    --num_epochs 5 \           # 训练轮数
    --batch_size 8 \           # 批次大小
    --learning_rate 2e-4       # 学习率
```

**断点续训**

```bash
# 自动检测并从最新 checkpoint 恢复
python train.py --resume_from_checkpoint auto

# 从指定 checkpoint 恢复
python train.py --resume_from_checkpoint ./outputs/checkpoint-500

# 强制从头开始训练
python train.py --resume_from_checkpoint none
```

**训练输出**

训练完成后，模型将保存在 `outputs/lora_adapter/` 目录：
- `adapter_model.safetensors` - LoRA 适配器权重
- `adapter_config.json` - 适配器配置
- `tokenizer.json` 等 - 分词器文件

### 模型推理

**单条文本预测**

```bash
python inference.py --model_path ./outputs/lora_adapter --text "这个产品质量非常好，值得购买！"
```

**交互式预测**

```bash
python inference.py --model_path ./outputs/lora_adapter --interactive
```

在交互模式下，可以持续输入文本进行预测，输入 `quit` 或 `exit` 退出。

**批量预测**

```python
from inference import SentimentPredictor

# 初始化预测器
predictor = SentimentPredictor(
    base_model_name="Qwen/Qwen2.5-1.5B",
    lora_path="./outputs/lora_adapter"
)

# 批量预测
texts = ["产品很好", "质量太差了", "物流速度快"]
results = predictor.predict_batch(texts)

for text, result in zip(texts, results):
    print(f"文本: {text}")
    print(f"预测: {result['label']} (置信度: {result['confidence']:.2%})")
```

## 📊 模型评估结果说明

### 评估输出文件

项目在 `model_evaluation/outputs/` 目录下生成以下评估结果文件：

#### 1. **metrics.json** - 基础性能指标

包含模型在测试集上的核心性能指标：

```json
{
  "accuracy": 0.95,           // 准确率
  "precision": 0.94,          // 精确率（宏平均）
  "recall": 0.95,             // 召回率（宏平均）
  "f1": 0.94,                 // F1 分数（宏平均）
  "auc": 0.98,                // ROC-AUC 分数
  "confusion_matrix": [[...], [...]]  // 混淆矩阵
}
```

**评价指标说明：**
- **Accuracy（准确率）**：正确预测的样本占总样本的比例
- **Precision（精确率）**：预测为正面的样本中真正为正面的比例
- **Recall（召回率）**：所有正面样本中被正确预测的比例
- **F1 Score**：精确率和召回率的调和平均值
- **AUC**：ROC 曲线下面积，衡量分类器性能

#### 2. **confidence_analysis.json** - 置信度分析

分析模型预测的置信度分布和可靠性：

```json
{
  "avg_confidence": 0.92,              // 平均置信度
  "high_confidence_ratio": 0.85,       // 高置信度样本比例（>0.9）
  "low_confidence_samples": [...],     // 低置信度样本列表
  "confidence_distribution": {         // 置信度区间分布
    "0.5-0.6": 50,
    "0.6-0.7": 120,
    "0.9-1.0": 1500
  }
}
```

**用途：**
- 评估模型预测的可靠性
- 识别模型不确定的样本
- 为生产环境设置置信度阈值

#### 3. **robustness_result.json** - 鲁棒性测试结果

测试模型对输入扰动的抵抗能力：

```json
{
  "original_accuracy": 0.95,
  "perturbed_accuracy": 0.89,
  "robustness_score": 0.94,           // 鲁棒性得分
  "perturbation_tests": {
    "synonym_replace": 0.92,          // 同义词替换
    "typo_insertion": 0.88,           // 错别字插入
    "punctuation_change": 0.93        // 标点变化
  }
}
```

**测试类型：**
- 同义词替换：测试语义理解能力
- 错别字干扰：测试对拼写错误的容忍度
- 标点符号变化：测试对格式变化的鲁棒性

#### 4. **baseline_comparison.json** - 基线模型对比

将微调模型与基线模型进行对比：

```json
{
  "fine_tuned_model": {
    "accuracy": 0.95,
    "f1": 0.94
  },
  "baseline_model": {
    "accuracy": 0.75,
    "f1": 0.72
  },
  "improvement": {
    "accuracy": "+20%",
    "f1": "+22%"
  }
}
```

**对比维度：**
- 微调模型 vs 未微调的基础模型
- 各项指标的绝对提升和相对提升

### 运行完整评估

```bash
# 执行所有评估测试并生成报告
python model_evaluation/run_full_eval.py --model_path ./outputs/lora_adapter

# 仅运行特定评估
python model_evaluation/confidence_analysis.py --model_path ./outputs/lora_adapter
python model_evaluation/robustness_test.py --model_path ./outputs/lora_adapter
```

## 📝 参数说明

### LoRA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_r` | 8 | LoRA 秩，控制参数量（越大效果越好但参数越多） |
| `--lora_alpha` | 16 | LoRA 缩放因子（通常设为 r 的 2 倍） |
| `--lora_dropout` | 0.05 | Dropout 比例，防止过拟合 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_epochs` | 3 | 训练轮数 |
| `--batch_size` | 8 | 每批次样本数 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--use_qlora` | False | 是否使用 QLoRA (4-bit 量化) |

## 📚 更多资源

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [Qwen2.5 模型文档](https://github.com/QwenLM/Qwen2.5)
- [HuggingFace PEFT 库](https://github.com/huggingface/peft)

## 📄 License

本项目仅供学习交流使用。

### 推理预测

```bash
# 交互式模式
python inference.py --model_path ./outputs/lora_adapter --interactive

# 单条预测
python inference.py --model_path ./outputs/lora_adapter --text "这家餐厅的菜很好吃！"

# 运行示例
python inference.py --model_path ./outputs/lora_adapter --demo
```

## 📚 核心概念

### LoRA 原理

LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法：

```
原始权重: W (d × k)
LoRA 分解: W + ΔW = W + BA
其中: B (d × r), A (r × k), r << min(d, k)
```

**核心思想**：冻结预训练权重，只训练低秩分解矩阵

```python
# LoRA 配置示例
LoraConfig(
    r=8,                    # 秩（rank）
    lora_alpha=32,          # 缩放因子
    lora_dropout=0.1,       # Dropout
    target_modules=[        # 目标模块
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
)
```

### QLoRA

QLoRA 在 LoRA 基础上增加了 4-bit 量化：

- **NF4 量化**：使用正态分布最优量化
- **双重量化**：进一步压缩量化常数
- **分页优化器**：处理内存峰值

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

### 训练流程

```
┌─────────────┐
│  加载模型   │
└──────┬──────┘
       ▼
┌─────────────┐
│  应用 LoRA  │  ← 冻结原始权重，添加低秩适配器
└──────┬──────┘
       ▼
┌─────────────┐
│  加载数据   │  ← ChnSentiCorp 数据集
└──────┬──────┘
       ▼
┌─────────────┐
│  微调训练   │  ← 只更新 LoRA 参数 (~0.1% 参数)
└──────┬──────┘
       ▼
┌─────────────┐
│  保存适配器 │  ← 只保存 LoRA 权重 (~10MB)
└─────────────┘
```

## 📊 数据集

### ChnSentiCorp

中文酒店评论情感分析数据集：

| 分割 | 样本数 | 正面 | 负面 |
|------|--------|------|------|
| 训练集 | 9,600 | 4,800 | 4,800 |
| 验证集 | 1,200 | 600 | 600 |
| 测试集 | 1,200 | 600 | 600 |

**样本示例**：
```
正面: "酒店位置很好，服务态度也非常好，房间干净整洁。"
负面: "服务态度太差，房间也很脏，再也不会来了。"
```

## ⚙️ 配置说明

### LoRA 参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `r` | 秩（rank） | 4-32 |
| `lora_alpha` | 缩放因子 | r 的 2-4 倍 |
| `lora_dropout` | Dropout | 0.05-0.1 |
| `target_modules` | 目标层 | q,k,v,o_proj |

### 训练参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `learning_rate` | 学习率 | 1e-4 ~ 5e-4 |
| `batch_size` | 批次大小 | 4-16 |
| `num_epochs` | 训练轮数 | 3-5 |
| `warmup_ratio` | 预热比例 | 0.1 |

## 📈 预期效果

使用默认配置训练后，在测试集上的预期效果：

| 指标 | 数值 |
|------|------|
| 准确率 | ~93% |
| F1 分数 | ~93% |
| AUC-ROC | ~97% |

## 🔧 常见问题

### Q1: 显存不足怎么办？

```bash
# 使用 QLoRA
python train.py --use_qlora

# 减小批次大小
python train.py --batch_size 2

# 启用梯度检查点
python train.py --gradient_checkpointing
```

### Q2: 训练中断了怎么办？

训练会自动保存 checkpoint（每100步），可以断点续训：

```bash
# 自动检测并恢复最新的 checkpoint
python train.py --resume_from_checkpoint auto

# 手动指定 checkpoint
python train.py --resume_from_checkpoint ./outputs/checkpoint-500
```

**注意**：checkpoint 保存在 `outputs/checkpoint-*` 目录，最多保留3个最新的。

### Q3: 如何使用自己的数据？

在 `data/data_loader.py` 中修改 `load_local_dataset` 函数，准备 JSON Lines 格式数据：

```json
{"text": "评论文本", "label": 0}
{"text": "评论文本", "label": 1}
```

### Q4: 如何部署模型？

```python
from inference import SentimentPredictor

predictor = SentimentPredictor(
    base_model_name="Qwen/Qwen2.5-1.5B",
    lora_path="./outputs/lora_adapter"
)

result = predictor.predict("这个产品很好用！")
print(result["label"])  # 正面
```

## 📖 学习资源

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [Qwen 模型](https://huggingface.co/Qwen)

## 📄 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**Happy Learning! 🎉**
