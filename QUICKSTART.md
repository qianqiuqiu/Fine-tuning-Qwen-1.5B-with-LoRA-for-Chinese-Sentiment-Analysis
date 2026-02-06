# 快速开始指南

## 🚀 5分钟上手

### 1. 环境准备

```bash
# 克隆项目（或已有项目跳过）
cd "Fine-tuning Qwen-1.5B with LoRA for Chinese Sentiment Analysis"

# 安装依赖
pip install -r requirements.txt
```

### 2. 选择训练方案

#### 🥇 推荐：方案二 - Label Scoring（准确率 95.75%）

```bash
python scripts/train_label_scoring.py \
    --lora_r 8 \
    --num_epochs 3 \
    --batch_size 8
```

**为什么选择这个方案？**
- ✅ 最高准确率（95.75% vs 93.8%）
- ✅ 无需额外分类层，参数更少
- ✅ 语义对齐更自然，泛化能力强

#### 方案一 - 经典分类头（准确率 93.8%）

```bash
python scripts/train_classifier.py \
    --lora_r 8 \
    --num_epochs 3 \
    --batch_size 8
```

### 3. 模型推理

```bash
# 单条预测
python scripts/inference.py \
    --model_path ./experiments/label_scoring/lora_adapter \
    --text "这个产品质量非常好，值得购买！"

# 交互式模式（推荐测试）
python scripts/inference.py \
    --model_path ./experiments/label_scoring/lora_adapter \
    --interactive
```

### 4. 模型评估

```bash
# 完整评估（生成详细报告）
python evaluation/run_full_eval.py \
    --model_path ./experiments/label_scoring/lora_adapter

# 查看评估结果
cat evaluation/outputs/baseline_comparison.json
```

---

## 💡 使用技巧

### 显存不足？使用 QLoRA

```bash
# 4GB 显存即可训练
python scripts/train_label_scoring.py --use_qlora
```

### 训练中断？自动恢复

```bash
# 自动从最新 checkpoint 恢复
python scripts/train_label_scoring.py --resume_from_checkpoint auto
```

### 自定义参数

```bash
python scripts/train_label_scoring.py \
    --lora_r 16 \              # 增加 LoRA 秩以提升容量
    --num_epochs 5 \           # 延长训练轮数
    --learning_rate 3e-4       # 调整学习率
```

---

## 📊 预期结果

| 方案 | 准确率 | F1 分数 | 训练时间 |
|------|--------|---------|---------|
| **Label Scoring** | **95.75%** | **95.80%** | ~56 分钟 |
| Classifier Head | 93.8% | 93.7% | ~50 分钟 |
| Zero-shot | 88.0% | 88.2% | - |

---

## 🔍 目录结构一览

```
项目根目录/
├── scripts/              # 👈 所有可执行脚本在这里
│   ├── train_label_scoring.py    # 🥇 推荐训练脚本
│   ├── train_classifier.py       # 传统分类头训练
│   ├── inference.py              # 推理脚本
│   └── evaluate.py               # 评估脚本
│
├── experiments/          # 👈 训练结果保存在这里
│   ├── label_scoring/           # 方案二输出
│   └── classifier_head/         # 方案一输出
│
├── evaluation/           # 👈 评估工具和结果
│   └── outputs/                 # 评估报告
│
└── configs/             # 配置文件
    └── data/            # 数据处理模块
```

---

## ❓ 常见问题速查

### Q: 两种方案有什么区别？

| 维度 | Label Scoring | Classifier Head |
|------|---------------|-----------------|
| 模型类型 | CausalLM | SequenceClassification |
| 输出方式 | 比较标签 logprob | Softmax 分类 |
| 准确率 | 95.75% | 93.8% |
| 推理速度 | 稍慢 | 快 |
| 推荐场景 | 少样本、跨领域 | 标准分类任务 |

### Q: 显存不够怎么办？

```bash
# 方案1：使用 QLoRA（推荐）
python scripts/train_label_scoring.py --use_qlora

# 方案2：减小批次
python scripts/train_label_scoring.py --batch_size 4

# 方案3：启用梯度检查点
python scripts/train_label_scoring.py --gradient_checkpointing
```

### Q: 如何查看训练进度？

训练过程会自动同步到 WandB（需登录）：
```bash
wandb login
python scripts/train_label_scoring.py
```

也可以查看本地日志：
```bash
tail -f experiments/label_scoring/train_results.json
```

---

## 📚 更多信息

- 📖 完整文档：查看 [README.md](README.md)
- 🔄 项目重组说明：查看 [RESTRUCTURE.md](RESTRUCTURE.md)
- 📊 评估报告：查看 `evaluation/outputs/baseline_comparison.json`

---

**祝您使用愉快！如有问题欢迎提 Issue 💬**
