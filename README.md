# 🎯 中文情感分析微调项目

使用 LoRA/QLoRA 技术微调 Qwen-1.5B 模型进行中文情感分析任务。

## 📋 项目概述

本项目是一个完整的中文情感分析微调示例，旨在帮助学习：
- 如何使用 LoRA (Low-Rank Adaptation) 高效微调大语言模型
- 如何使用 HuggingFace Trainer 进行模型训练
- 如何使用 PEFT 库进行参数高效微调

## 🏗️ 项目结构

```
project/
├── configs/                    # 配置文件
│   ├── __init__.py
│   ├── lora_config.py         # LoRA 超参数配置
│   └── training_config.py     # 训练参数配置
├── data/                       # 数据处理
│   ├── __init__.py
│   ├── data_loader.py         # 数据加载
│   └── preprocessing.py       # 数据预处理
├── train.py                    # 微调主脚本
├── eval.py                     # 评估脚本
├── inference.py                # 推理示例
├── requirements.txt            # 依赖
└── README.md                   # 项目说明
```

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 基础模型 | Qwen2.5-1.5B |
| 微调技术 | LoRA / QLoRA |
| 训练框架 | HuggingFace Transformers + Trainer |
| 参数高效微调 | PEFT |
| 数据集 | ChnSentiCorp |

## 📦 环境配置

### 1. 创建虚拟环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 硬件要求

| 配置 | LoRA | QLoRA (4-bit) |
|------|------|---------------|
| 最小显存 | ~8GB | ~4GB |
| 推荐显存 | 16GB | 8GB |

## 🚀 快速开始

### 训练模型

```bash
# 使用默认配置训练
python train.py

# 使用 QLoRA（4-bit 量化，节省显存）
python train.py --use_qlora

# 自定义参数
python train.py \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_epochs 5 \
    --batch_size 4 \
    --learning_rate 1e-4

# 断点续训（自动检测最新 checkpoint）
python train.py --resume_from_checkpoint auto

# 从指定 checkpoint 恢复
python train.py --resume_from_checkpoint ./outputs/checkpoint-500

# 强制从头开始训练（不使用 checkpoint）
python train.py --resume_from_checkpoint none
```

### 评估模型

```bash
python eval.py --model_path ./outputs/lora_adapter
```

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
