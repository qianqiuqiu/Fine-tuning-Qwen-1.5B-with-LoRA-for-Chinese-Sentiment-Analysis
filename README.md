# 🎯 Qwen-1.5B 中文情感分析微调项目

> 基于 LoRA/QLoRA 技术微调 Qwen2.5-1.5B 模型，实现高效的中文情感分析任务

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.36+-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 项目亮点

- 🏆 **高性能**：Label Scoring 方案达到 **95.75%** 准确率
- 💡 **双方案对比**：判别式分类头 vs 生成式 Label Scoring 全面对比
- ⚡ **高效训练**：LoRA/QLoRA 技术，仅需微调 ~1% 参数
- 💾 **低显存占用**：QLoRA 模式下仅需 4GB 显存即可训练
- 📊 **完整评估**：置信度分析、鲁棒性测试、多方案基准对比
- 🔧 **生产就绪**：完整的训练、评估、推理流程

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
├── README.md                     # 项目文档
├── pyproject.toml               # 项目配置
├── requirements.txt             # 依赖列表
├── requirements_autodl.txt      # AutoDL 环境依赖
│
├── configs/                     # 配置模块
│   ├── __init__.py
│   ├── lora_config.py          # LoRA 超参数配置
│   └── training_config.py      # 训练参数配置
│
├── data/                        # 数据处理模块
│   ├── __init__.py
│   ├── data_loader.py          # 数据集加载
│   └── preprocessing.py        # 数据预处理
│
├── scripts/                     # 训练和推理脚本
│   ├── train_classifier.py     # 方案一：LoRA + 分类头
│   ├── train_label_scoring.py  # 方案二：LoRA + Label Scoring
│   ├── evaluate.py             # 基础评估脚本
│   └── inference.py            # 推理脚本
│
├── evaluation/                  # 模型评估模块
│   ├── run_full_eval.py        # 完整评估流程
│   ├── confidence_analysis.py  # 置信度分析
│   ├── robustness_test.py      # 鲁棒性测试
│   ├── benchmark.py            # 基准对比
│   ├── zero_shot_baseline.py  # Zero-shot 基线
│   ├── report_generator.py     # 评估报告生成
│   └── outputs/                # 评估结果输出
│       ├── baseline_comparison.json
│       ├── confidence_analysis.json
│       ├── metrics.json
│       └── robustness_result.json
│
└── experiments/                 # 实验结果目录
    ├── classifier_head/        # 方案一训练输出
    │   ├── lora_adapter/      # LoRA 适配器
    │   ├── train_results.json
    │   └── test_results.json
    └── label_scoring/          # 方案二训练输出
        ├── lora_adapter/      # LoRA 适配器
        ├── train_results.json
        └── test_results.json
```

## 🚀 Quick Start

### 两种微调方案对比

本项目实现了两种不同的微调方案，适用于不同的应用场景：

#### **方案一：LoRA + 判别式分类头**（`train_classifier.py`）

使用 `AutoModelForSequenceClassification`，在模型顶部添加线性分类层。

```bash
python scripts/train_classifier.py --lora_r 8 --num_epochs 3
```

**特点：**
- 🎯 使用传统分类头（Linear layer: hidden_dim → 2）
- 📊 输出二维 logits，通过 softmax 得到概率
- ⚡ 训练和推理速度快
- 🎓 适合标准分类任务

**性能指标：**
| 指标 | 数值 |
|---|---|
| 准确率 | 93.8% |
| F1 分数 | 93.7% |
| 精确率 | 97.1% |
| 召回率 | 90.6% |

#### **方案二：LoRA + 生成式 Label Scoring**（`train_label_scoring.py`）

使用 `AutoModelForCausalLM`，通过比较候选标签的生成概率进行分类。

```bash
python scripts/train_label_scoring.py --lora_r 8 --num_epochs 3
```

**特点：**
- 🔮 复用语言模型的 LM Head（无额外分类层）
- 📝 Prompt 模板：`"评论：{text}\n情感倾向："`
- 🎲 比较 "正面" 和 "负面" 的 token log-probability
- 🌟 更自然的语义对齐，适合少样本和跨领域场景

**性能指标：**
| 指标 | 数值 |
|---|---|
| **准确率** | **95.75%** ⭐ |
| **F1 分数** | **95.80%** ⭐ |
| **精确率** | **95.88%** |
| **召回率** | **95.72%** |

#### **Zero-shot 基线**（`zero_shot_baseline.py`）

不进行任何微调，直接使用预训练模型 + prompt 进行预测。

```bash
python evaluation/zero_shot_baseline.py
```

**性能指标：**
| 指标 | 数值 |
|---|---|
| 准确率 | 88.0% |
| F1 分数 | 88.2% |

---

**📊 方案对比总结：**

| 方案 | 准确率 | F1 | 相对提升 | 推荐场景 |
|---|---|---|---|---|
| **Label Scoring** | **95.75%** | **95.80%** | +1.95% | 少样本、跨领域、语义对齐要求高 |
| **Classifier Head** | 93.8% | 93.7% | 基准 | 标准分类任务、追求推理速度 |
| **Zero-shot** | 88.0% | 88.2% | -5.8% | 无标注数据、快速验证 |

> 💡 **选择建议**：Label Scoring 方案在本项目中表现最佳（+1.95%），且无需额外分类层参数，推荐作为默认方案。

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

> **Windows 用户注意**：项目已配置自动安装 `bitsandbytes-windows` 以解决 Windows 兼容性问题。如遇到 bitsandbytes 相关错误，请手动执行：
> ```bash
> pip uninstall bitsandbytes -y
> pip install bitsandbytes-windows
> ```

**硬件要求**

| 模式 | 最小显存 | 推荐显存 | 训练速度 |
|------|---------|---------|---------|
| LoRA | 8GB | 16GB | 快 |
| QLoRA (4-bit) | 4GB | 8GB | 较慢 |

### 模型训练

**基础训练（方案一：分类头）**

```bash
# 使用默认配置（LoRA，r=8）
python scripts/train_classifier.py

# 使用 QLoRA 节省显存（推荐显存不足时使用）
python scripts/train_classifier.py --use_qlora
```

**Label Scoring 训练（方案二：推荐）**

```bash
# 使用默认配置
python scripts/train_label_scoring.py

# 使用 QLoRA
python scripts/train_label_scoring.py --use_qlora
```

**自定义参数训练**

```bash
python scripts/train_classifier.py \
    --lora_r 16 \              # LoRA 秩（rank）
    --lora_alpha 32 \          # LoRA alpha 参数
    --num_epochs 5 \           # 训练轮数
    --batch_size 8 \           # 批次大小
    --learning_rate 2e-4       # 学习率
```

**断点续训**

```bash
# 自动检测并从最新 checkpoint 恢复
python scripts/train_classifier.py --resume_from_checkpoint auto

# 从指定 checkpoint 恢复
python scripts/train_classifier.py --resume_from_checkpoint ./experiments/classifier_head/checkpoint-500

# 强制从头开始训练
python scripts/train_classifier.py --resume_from_checkpoint none
```

**训练输出**

训练完成后，模型将保存在相应的实验目录：
- **方案一**：`experiments/classifier_head/lora_adapter/`
- **方案二**：`experiments/label_scoring/lora_adapter/`

输出文件包括：
- `adapter_model.safetensors` - LoRA 适配器权重
- `adapter_config.json` - 适配器配置
- `tokenizer.json` 等 - 分词器文件
- `label_scoring_meta.json` - Label Scoring 方案的元信息（仅方案二）

### 模型推理

**单条文本预测**

```bash
# 方案一：分类头模型
python scripts/inference.py \
    --model_path ./experiments/classifier_head/lora_adapter \
    --text "这个产品质量非常好，值得购买！"

# 方案二：Label Scoring 模型
python scripts/inference.py \
    --model_path ./experiments/label_scoring/lora_adapter \
    --text "这个产品质量非常好，值得购买！"
```

**交互式预测**

```bash
python scripts/inference.py \
    --model_path ./experiments/classifier_head/lora_adapter \
    --interactive
```

在交互模式下，可以持续输入文本进行预测，输入 `quit` 或 `exit` 退出。

**批量预测**

```python
from scripts.inference import SentimentPredictor

# 初始化预测器
predictor = SentimentPredictor(
    base_model_name="Qwen/Qwen2.5-1.5B",
    lora_path="./experiments/classifier_head/lora_adapter"
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

项目在 `evaluation/outputs/` 目录下生成以下评估结果文件：

#### 1. **baseline_comparison.json** - 多方案对比

完整对比以下三种方案（还有一种Qwen-1.5B 冻结 + LoRA分类头列在evaluation\outputs\baseline_comparison.json里面，不太具有参考价值，只能说明lora训练的hidden state和分类头强绑定）的性能表现：

```json
{
  "qwen_lora_label_scoring": {
    "name": "Qwen-1.5B + LoRA微调 (Label Scoring)",
    "description": "使用 CausalLM + LoRA 微调，生成式分类接口",
    "metrics": {
      "accuracy": 0.9575,    // 准确率 95.75%
      "precision": 0.9588,   // 精确率
      "recall": 0.9572,      // 召回率
      "f1": 0.9580          // F1 分数
    }
  },
  "qwen_lora": {
    "name": "Qwen-1.5B + LoRA微调",
    "description": "LoRA微调的完整模型（基础模型+分类头）",
    "metrics": {
      "accuracy": 0.938,     // 准确率 93.8%
      "f1": 0.937           // F1 分数
    }
  },
  "qwen_zero_shot": {
    "name": "Qwen-1.5B Zero-shot",
    "description": "不微调，使用 prompt 模板直接预测",
    "metrics": {
      "accuracy": 0.88,      // 准确率 88.0%
      "f1": 0.882           // F1 分数
    }
  }
}
```

**对比洞察：**
- ✨ **Label Scoring 方案** 取得最佳性能（95.75%），比传统分类头高 1.95%
- 📈 相比 Zero-shot 基线提升 7.75 个百分点
- 🚀 两种微调方案都显著超越未微调模型


### 以下性能指标均基于方案一：分类头模型，方案二下面测试的评估代码还没来得及改，暂时挖个坑。
#### 2. **metrics.json** - 基础性能指标

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

#### 3. **confidence_analysis.json** - 置信度分析

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

#### 4. **robustness_result.json** - 鲁棒性测试结果

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

### 运行完整评估

```bash
# 执行所有评估测试并生成报告（方案一）
python evaluation/run_full_eval.py \
    --model_path ./experiments/classifier_head/lora_adapter

# 执行所有评估测试并生成报告（方案二）
python evaluation/run_full_eval.py \
    --model_path ./experiments/label_scoring/lora_adapter

# 仅运行特定评估
python evaluation/confidence_analysis.py \
    --model_path ./experiments/classifier_head/lora_adapter
    
python evaluation/robustness_test.py \
    --model_path ./experiments/classifier_head/lora_adapter
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
# 交互式模式（方案一）
python scripts/inference.py \
    --model_path ./experiments/classifier_head/lora_adapter \
    --interactive

# 单条预测（方案二）
python scripts/inference.py \
    --model_path ./experiments/label_scoring/lora_adapter \
    --text "这家餐厅的菜很好吃！"

# 运行示例
python scripts/inference.py \
    --model_path ./experiments/classifier_head/lora_adapter \
    --demo
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

## 📈 实验结果

使用默认配置训练后，在测试集上的实际效果：

### 方案对比

| 方案 | 准确率 | 精确率 | 召回率 | F1 分数 | 备注 |
|------|--------|--------|--------|---------|------|
| **LoRA + Label Scoring** | **95.75%** | **95.88%** | **95.72%** | **95.80%** | 🏆 最佳方案 |
| LoRA + Classifier Head | 93.8% | 97.1% | 90.6% | 93.7% | 标准方案 |
| Zero-shot Baseline | 88.0% | 89.6% | 86.9% | 88.2% | 无微调 |
| Frozen Base + LoRA Head | 51.2% | 51.1% | 99.2% | 67.5% | 仅分类头 |

### 关键发现

1. **Label Scoring 显著优于传统分类头**
   - 准确率提升：95.75% vs 93.8% (+1.95%)
   - 无需额外分类层参数，语义对齐更自然
   
2. **微调效果显著**
   - 相比 Zero-shot 提升 7.75% 准确率
   - F1 分数提升 7.6 个百分点

3. **LoRA 高效性验证**
   - 仅微调 ~1% 参数即可达到优异效果
   - 训练时间：约 56 分钟（3 epochs）

## 🔧 常见问题

### Q1: Windows 系统 bitsandbytes 报错怎么办？

**问题症状**：
```
packaging.version.InvalidVersion: Invalid version: '"r"):read("*a"))()'
```

**解决方案**：
```bash
# 卸载原版 bitsandbytes
pip uninstall bitsandbytes -y

# 安装 Windows 兼容版本
pip install bitsandbytes-windows
```

项目已在 `requirements.txt` 中配置自动检测操作系统并安装对应版本，新环境安装时会自动处理。

### Q2: 它消失了

### Q3: 显存不足怎么办？

```bash
# 使用 QLoRA
python scripts/train_classifier.py --use_qlora

# 减小批次大小
python scripts/train_classifier.py --batch_size 2

# 启用梯度检查点
python scripts/train_classifier.py --gradient_checkpointing
```

### Q4: 训练中断了怎么办？

训练会自动保存 checkpoint（每100步），可以断点续训：

```bash
# 自动检测并恢复最新的 checkpoint
python scripts/train_classifier.py --resume_from_checkpoint auto

# 手动指定 checkpoint
python scripts/train_classifier.py --resume_from_checkpoint \
    ./experiments/classifier_head/checkpoint-500
```

**注意**：checkpoint 保存在 `experiments/*/checkpoint-*` 目录，最多保留3个最新的。

### Q5: 如何使用自己的数据？

在 `data/data_loader.py` 中修改 `load_local_dataset` 函数，准备 JSON Lines 格式数据：

```json
{"text": "评论文本", "label": 0}
{"text": "评论文本", "label": 1}
```

### Q6: 如何部署模型？

```python
from scripts.inference import SentimentPredictor

predictor = SentimentPredictor(
    base_model_name="Qwen/Qwen2.5-1.5B",
    lora_path="./experiments/classifier_head/lora_adapter"
)

result = predictor.predict("这个产品很好用！")
print(result["label"])  # 正面
```

### Q7: 两种方案如何选择？

**推荐使用 Label Scoring 方案（方案二）**，因为：
- ✅ 准确率更高（95.75% vs 93.8%）
- ✅ 无需额外分类层，参数更少
- ✅ 语义对齐更自然，泛化能力更强

**使用 Classifier Head（方案一）的场景：**
- 需要极致的推理速度
- 标准二分类任务，不需要语义对齐
- 与现有系统集成更方便

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
