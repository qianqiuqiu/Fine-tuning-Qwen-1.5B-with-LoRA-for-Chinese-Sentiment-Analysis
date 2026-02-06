# 项目重组说明

## 📁 目录结构优化

本项目已完成全面的目录结构重组，使其更加专业、清晰和易于维护。

### ✨ 主要变更

#### 1. **脚本集中管理** (`scripts/` 目录)

**原结构**（分散在根目录）：
```
├── train.py
├── train2.py
├── eval.py
└── inference.py
```

**新结构**（集中在 scripts/）：
```
scripts/
├── train_classifier.py      # 方案一：LoRA + 分类头
├── train_label_scoring.py   # 方案二：LoRA + Label Scoring
├── evaluate.py              # 评估脚本
└── inference.py             # 推理脚本
```

**命名优化**：
- `train.py` → `train_classifier.py`（更明确的方案命名）
- `train2.py` → `train_label_scoring.py`（语义化命名）
- `eval.py` → `evaluate.py`（完整词汇，更专业）

#### 2. **评估模块重命名** (`evaluation/`)

```
model_evaluation/  →  evaluation/
```

- 更简洁的目录名
- 内部包名从 `model_evaluation` 更新为 `evaluation`
- 所有导入路径已同步更新

#### 3. **实验结果整合** (`experiments/`)

**原结构**（两个分离的输出目录）：
```
├── outputs/
└── outputs_label_scoring/
```

**新结构**（集中管理，方案清晰）：
```
experiments/
├── classifier_head/        # 方案一训练输出
│   ├── lora_adapter/      # LoRA 适配器
│   ├── train_results.json
│   └── test_results.json
└── label_scoring/          # 方案二训练输出
    ├── lora_adapter/      # LoRA 适配器
    ├── label_scoring_meta.json
    ├── train_results.json
    └── test_results.json
```

**优势**：
- ✅ 多方案并行管理，互不干扰
- ✅ 目录名语义化，一目了然
- ✅ 便于切换和对比不同实验结果

### 🔄 路径更新映射表

| 旧路径 | 新路径 | 说明 |
|-------|--------|------|
| `train.py` | `scripts/train_classifier.py` | 方案一训练脚本 |
| `train2.py` | `scripts/train_label_scoring.py` | 方案二训练脚本 |
| `eval.py` | `scripts/evaluate.py` | 评估脚本 |
| `inference.py` | `scripts/inference.py` | 推理脚本 |
| `model_evaluation/` | `evaluation/` | 评估模块目录 |
| `outputs/` | `experiments/classifier_head/` | 方案一输出 |
| `outputs_label_scoring/` | `experiments/label_scoring/` | 方案二输出 |

### 🚀 使用方法更新

#### 训练模型

**方案一（分类头）**：
```bash
# 旧命令
python train.py --lora_r 8 --num_epochs 3

# 新命令
python scripts/train_classifier.py --lora_r 8 --num_epochs 3
```

**方案二（Label Scoring，推荐）**：
```bash
# 旧命令
python train2.py --lora_r 8 --num_epochs 3

# 新命令
python scripts/train_label_scoring.py --lora_r 8 --num_epochs 3
```

#### 模型推理

```bash
# 方案一
python scripts/inference.py \
    --model_path ./experiments/classifier_head/lora_adapter \
    --text "这个产品很好用！"

# 方案二
python scripts/inference.py \
    --model_path ./experiments/label_scoring/lora_adapter \
    --text "这个产品很好用！"
```

#### 评估模型

```bash
# 旧命令
python model_evaluation/run_full_eval.py --model_path ./outputs/lora_adapter

# 新命令
python evaluation/run_full_eval.py \
    --model_path ./experiments/classifier_head/lora_adapter
```

### 📊 代码自动更新

以下内容已自动更新，无需手动修改：

✅ **脚本默认路径**
- `train_classifier.py` 默认输出：`./experiments/classifier_head`
- `train_label_scoring.py` 默认输出：`./experiments/label_scoring`
- `inference.py` 默认模型路径：`./experiments/classifier_head/lora_adapter`
- `evaluate.py` 默认模型路径：`./experiments/classifier_head/lora_adapter`

✅ **评估模块导入**
- 所有 `from model_evaluation import ...` 已更新为 `from evaluation import ...`
- 包名 `__package__ = "model_evaluation"` 已更新为 `__package__ = "evaluation"`

✅ **README 文档**
- 项目结构图已更新
- 所有示例命令已更新
- 添加了方案对比结果表格

### 💡 最佳实践

1. **从项目根目录运行所有命令**
   ```bash
   # ✅ 正确
   python scripts/train_classifier.py
   
   # ❌ 避免
   cd scripts && python train_classifier.py
   ```

2. **使用相对路径（从根目录开始）**
   ```bash
   --model_path ./experiments/classifier_head/lora_adapter
   ```

3. **区分不同方案的实验结果**
   - 方案一（分类头）：`experiments/classifier_head/`
   - 方案二（Label Scoring）：`experiments/label_scoring/`

### 🎯 重组目标达成

✅ **模块化清晰**：scripts、configs、data、evaluation、experiments 各司其职

✅ **命名语义化**：所有文件和目录名都能清晰表达其用途

✅ **易于扩展**：新增方案只需在 scripts/ 和 experiments/ 添加对应文件/目录

✅ **专业规范**：符合 Python 项目的最佳实践和行业标准

✅ **向后兼容**：已有的训练结果已迁移，无需重新训练

---

**变更日期**：2026年2月7日  
**影响范围**：所有训练、推理、评估相关脚本  
**兼容性**：完全向后兼容（已有训练结果已自动迁移）
