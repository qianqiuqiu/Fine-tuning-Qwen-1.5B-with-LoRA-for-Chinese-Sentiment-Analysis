"""
微调主脚本（方案二：LoRA + Label Scoring / 生成式分类）
使用 LoRA 微调 Qwen-1.5B 的 CausalLM，通过比较标签 token 的 log-prob 进行分类

与 train.py（方案一：LoRA + classifier head）的核心区别：
┌─────────────────────────────┬─────────────────────────────────────────┐
│ train.py (判别式分类头)       │ train2.py (生成式 Label Scoring)          │
├─────────────────────────────┼─────────────────────────────────────────┤
│ AutoModelForSeqCls           │ AutoModelForCausalLM                    │
│ 额外 Linear(d→2) 分类头      │ 复用 LM Head（词表投影，与词嵌入共享权重）  │
│ 输出: softmax(Wh+b), 2 维    │ 输出: 比较 "正面"/"负面" 的 logprob       │
│ TaskType.SEQ_CLS             │ TaskType.CAUSAL_LM                      │
│ loss = CE(分类 logits, label) │ loss = CE(next-token logits, label_ids) │
└─────────────────────────────┴─────────────────────────────────────────┘

Prompt 模板:
    输入: "评论：{text}\n情感倾向："
    标签: "正面" (label=1) / "负面" (label=0)

训练: 将 prompt + 标签 拼接，仅对标签 token 计算 loss；只更新 LoRA 权重
推理: 在 prompt 末尾比较 "正面" 与 "负面" 的生成 log-prob，取较大者

使用方法:
    python train2.py                          # 使用默认配置
    python train2.py --use_qlora              # 使用 QLoRA（4-bit 量化）
    python train2.py --lora_r 16              # 自定义 LoRA 秩
    python train2.py --num_epochs 5           # 自定义训练轮数
"""

import os

# 修复 torch 导入卡死问题 (Intel MKL 库冲突)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import wandb
import glob
import json
from dataclasses import dataclass

# 设置 HuggingFace 镜像（用于在线下载时加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# 导入项目配置（复用 training_config 中的数据类）
from configs import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    QLoRAConfig,
    get_training_args,
)
from data import load_sentiment_dataset

# 评估指标
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ==================== 标签定义 ====================

# 中文情感标签映射
LABEL_TEXTS = {
    0: "负面",   # negative
    1: "正面",   # positive
}

# Prompt 模板
PROMPT_TEMPLATE = "评论：{text}\n情感倾向："


# ==================== 工具函数 ====================

def get_local_model_path(model_name: str) -> str:
    """
    检测并返回本地模型路径

    Args:
        model_name: HuggingFace 模型名称，如 "Qwen/Qwen2.5-1.5B"

    Returns:
        本地模型路径或原始模型名称
    """
    cache_folder = "models--" + model_name.replace("/", "--")

    if os.path.exists(cache_folder):
        snapshot_pattern = os.path.join(cache_folder, "snapshots", "*")
        snapshots = glob.glob(snapshot_pattern)
        if snapshots:
            model_path = snapshots[0]
            print(f"✅ 检测到本地模型: {model_path}")
            return model_path

    print(f"🌐 本地模型不存在，将从 HuggingFace 下载: {model_name}")
    return model_name


def get_local_dataset_path(dataset_name: str) -> tuple:
    """
    检测并返回本地数据集路径
    """
    cache_folder = "datasets--" + dataset_name.replace("/", "--")

    if os.path.exists(cache_folder):
        snapshot_pattern = os.path.join(cache_folder, "snapshots", "*")
        snapshots = glob.glob(snapshot_pattern)
        if snapshots:
            dataset_path = snapshots[0]
            print(f"✅ 检测到本地数据集: {dataset_path}")
            return True, dataset_path

    print(f"🌐 本地数据集不存在，将从 HuggingFace 下载: {dataset_name}")
    return False, dataset_name


def get_last_checkpoint(output_dir: str) -> Optional[str]:
    """获取最新的 checkpoint 路径"""
    if not os.path.isdir(output_dir):
        return None

    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]

    if not checkpoints:
        return None

    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


# ==================== 数据处理（Label Scoring 专用） ====================

def get_label_token_ids(tokenizer: PreTrainedTokenizer) -> Dict[int, List[int]]:
    """
    获取各标签字符串对应的 token ID 列表

    Args:
        tokenizer: 分词器

    Returns:
        {label_int: [token_id, ...]}  例如 {0: [负, 面], 1: [正, 面]}
    """
    label_token_ids = {}
    for label_int, label_text in LABEL_TEXTS.items():
        # 用 encode 获取纯文本的 token ID（不加特殊 token）
        ids = tokenizer.encode(label_text, add_special_tokens=False)
        label_token_ids[label_int] = ids
        print(f"  标签 {label_int} ('{label_text}') -> token IDs: {ids}  "
              f"(decoded: {tokenizer.decode(ids)})")
    return label_token_ids


def build_label_scoring_examples(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    text_column: str = "text",
    label_column: str = "label",
) -> Dict[str, List]:
    """
    为 Label Scoring 构建训练样本

    每条数据 -> prompt + label_text，拼接为完整序列
    labels 中，prompt 部分置为 -100（不参与 loss），仅标签 token 计算 loss

    Args:
        examples: 原始数据批次
        tokenizer: 分词器
        max_length: 最大序列长度
        text_column: 文本列名
        label_column: 标签列名

    Returns:
        {input_ids, attention_mask, labels}
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for text, label in zip(examples[text_column], examples[label_column]):
        # 1) 构造 prompt 和完整文本
        prompt = PROMPT_TEMPLATE.format(text=text)
        label_text = LABEL_TEXTS[label]
        full_text = prompt + label_text

        # 2) 分别 tokenize prompt 和完整文本
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # 3) 截断（从 prompt 部分截断，保留标签 token）
        label_ids_raw = tokenizer.encode(label_text, add_special_tokens=False)
        label_len = len(label_ids_raw)

        if len(full_ids) > max_length:
            # 保留末尾的标签 token，截断 prompt 部分
            max_prompt_len = max_length - label_len
            prompt_ids = prompt_ids[:max_prompt_len]
            full_ids = prompt_ids + label_ids_raw

        prompt_len = len(full_ids) - label_len  # 重新计算 prompt 长度

        # 4) Padding
        seq_len = len(full_ids)
        pad_len = max_length - seq_len

        input_ids = full_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len

        # 5) 构造 labels：prompt 部分 = -100，标签 token 保留，padding = -100
        #    Causal LM 的 label 是右移的，即 labels[i] 是 input_ids[i+1] 的目标
        #    所以 labels 长度与 input_ids 相同，含义是 position i 的预测目标
        #    HuggingFace CausalLM 内部会处理 shift：
        #      - logits = model(input_ids)   # shape [seq_len, vocab]
        #      - shift_logits = logits[..., :-1, :]
        #      - shift_labels = labels[..., 1:]
        #      - loss = CE(shift_logits, shift_labels)
        #    因此 labels[i] 应该等于 input_ids[i]（模型内部负责 shift）
        labels = [-100] * prompt_len + full_ids[prompt_len:] + [-100] * pad_len

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def create_label_scoring_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    text_column: str = "text",
    label_column: str = "label",
    num_proc: int = 4,
) -> DatasetDict:
    """
    将原始数据集转换为 Label Scoring 格式

    Args:
        dataset: 原始 DatasetDict
        tokenizer: 分词器
        max_length: 最大序列长度
        text_column / label_column: 列名
        num_proc: 并行处理进程数

    Returns:
        处理后的 DatasetDict
    """

    def transform_fn(examples):
        return build_label_scoring_examples(
            examples=examples,
            tokenizer=tokenizer,
            max_length=max_length,
            text_column=text_column,
            label_column=label_column,
        )

    # 获取需要移除的列
    sample_split = list(dataset.keys())[0]
    columns_to_remove = dataset[sample_split].column_names

    tokenized_dataset = dataset.map(
        transform_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=columns_to_remove,
        desc="Building label-scoring dataset",
    )

    return tokenized_dataset


# ==================== 自定义 Trainer ====================

class LabelScoringTrainer(Trainer):
    """
    自定义 Trainer，为 Label Scoring 方案提供：
    1. 标准 Causal LM loss（训练，只对标签 token 计算 loss）
    2. 基于 logprob 的评估（比较候选标签的 log-probability）
    """

    def __init__(self, *args, label_token_ids: Dict[int, List[int]] = None,
                 eval_dataset_raw=None, eval_tokenizer=None,
                 eval_max_length: int = 256, **kwargs):
        """
        Args:
            label_token_ids: {label_int: [token_ids]} 各标签的 token ID
            eval_dataset_raw: 原始（未 tokenize 的）验证/测试集，用于 logprob 评估
            eval_tokenizer: 分词器
            eval_max_length: 评估时的最大长度
        """
        super().__init__(*args, **kwargs)
        self.label_token_ids = label_token_ids or {}
        self.eval_dataset_raw = eval_dataset_raw
        self.eval_tokenizer = eval_tokenizer
        self.eval_max_length = eval_max_length

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        标准 Causal LM loss，labels 中 -100 的位置自动被忽略
        """
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        重写 evaluate：用 label scoring（logprob 比较）做分类评估

        对验证集中的每条样本：
          1. 构建 prompt: "评论：{text}\n情感倾向："
          2. 对于每个候选标签，计算其 token 序列的条件 log-prob
          3. 选择 log-prob 最大的标签作为预测
          4. 计算 accuracy / precision / recall / f1
        """
        model = self.model
        model.eval()
        device = next(model.parameters()).device
        tokenizer = self.eval_tokenizer

        # 确定评估数据集
        raw_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset_raw
        if raw_dataset is None:
            # 回退到父类行为
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        all_preds = []
        all_labels = []

        for sample in raw_dataset:
            text = sample["text"]
            gold_label = sample["label"]

            prompt = PROMPT_TEMPLATE.format(text=text)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

            # 截断 prompt（为标签 token 留空间）
            max_label_len = max(len(ids) for ids in self.label_token_ids.values())
            if len(prompt_ids) > self.eval_max_length - max_label_len:
                prompt_ids = prompt_ids[:self.eval_max_length - max_label_len]

            best_label = -1
            best_logprob = float('-inf')

            for label_int, label_ids in self.label_token_ids.items():
                # 拼接 prompt + label
                full_ids = prompt_ids + label_ids
                input_tensor = torch.tensor([full_ids], device=device)
                attention_mask = torch.ones_like(input_tensor)

                outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
                logits = outputs.logits  # [1, seq_len, vocab_size]

                # 计算标签 token 的 log-prob
                # logits[t] 预测的是 position t+1 的 token
                # 标签 token 在 full_ids 中的位置: prompt_len ~ prompt_len + label_len - 1
                # 对应的 logits: prompt_len - 1 ~ prompt_len + label_len - 2
                prompt_len = len(prompt_ids)
                log_prob = 0.0
                for i, token_id in enumerate(label_ids):
                    logit_pos = prompt_len - 1 + i
                    log_probs = F.log_softmax(logits[0, logit_pos, :], dim=-1)
                    log_prob += log_probs[token_id].item()

                if log_prob > best_logprob:
                    best_logprob = log_prob
                    best_label = label_int

            all_preds.append(best_label)
            all_labels.append(gold_label)

        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )

        # 同时获取 causal LM loss（在 tokenized eval dataset 上）
        loss_metrics = {}
        if self.eval_dataset is not None:
            loss_output = super().evaluate(
                eval_dataset=self.eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            loss_metrics = loss_output

        metrics = {
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_precision": precision,
            f"{metric_key_prefix}_recall": recall,
            f"{metric_key_prefix}_f1": f1,
        }

        # 合并 loss
        if f"{metric_key_prefix}_loss" in loss_metrics:
            metrics[f"{metric_key_prefix}_loss"] = loss_metrics[f"{metric_key_prefix}_loss"]

        # 日志
        self.log(metrics)
        print(f"\n{'='*40} 评估结果 {'='*40}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"{'='*90}\n")

        return metrics


# ==================== 模型 & LoRA 初始化 ====================

def setup_model_and_tokenizer(
    model_config: ModelConfig,
    qlora_config: QLoRAConfig,
) -> tuple:
    """
    初始化 CausalLM 模型和分词器（注意：不使用分类头）
    """
    print(f"正在加载 CausalLM 模型: {model_config.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 配置量化
    quantization_config = None
    if qlora_config.use_qlora:
        print("启用 QLoRA 4-bit 量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=qlora_config.load_in_4bit,
            bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant,
        )

    # ★ 关键区别：使用 AutoModelForCausalLM，不使用 AutoModelForSequenceClassification
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        quantization_config=quantization_config,
        device_map="auto" if qlora_config.use_qlora else None,
        torch_dtype=torch.bfloat16,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    if qlora_config.use_qlora:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def setup_lora(
    model,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    use_qlora: bool = False,
):
    """
    为 CausalLM 模型添加 LoRA 适配器

    ★ 关键区别：task_type=CAUSAL_LM（而非 SEQ_CLS）
    """
    print(f"配置 LoRA (CAUSAL_LM): r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # ★ 生成式任务
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# ==================== 主训练函数 ====================

def train(args: argparse.Namespace):
    """主训练函数（Label Scoring 方案）"""

    # ==================== 1. 配置初始化 ====================
    local_model_path = get_local_model_path(args.model_name)

    model_config = ModelConfig(
        model_name_or_path=local_model_path,
        num_labels=2,
    )

    data_config = DataConfig(
        dataset_name=args.dataset,
        max_length=args.max_length,
    )

    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    qlora_config = QLoRAConfig(
        use_qlora=args.use_qlora,
    )

    # ==================== 2. 加载 CausalLM 模型 ====================
    model, tokenizer = setup_model_and_tokenizer(model_config, qlora_config)

    # ==================== 3. 应用 LoRA (CAUSAL_LM) ====================
    model = setup_lora(
        model=model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_qlora=args.use_qlora,
    )

    # ==================== 4. 查看标签 token 映射 ====================
    print("\n标签 token 映射：")
    label_token_ids = get_label_token_ids(tokenizer)

    # ==================== 5. 加载和处理数据 ====================
    print(f"\n正在加载数据集: {data_config.dataset_name}")

    use_local, dataset_path = get_local_dataset_path("lansinuote/ChnSentiCorp")
    dataset = load_sentiment_dataset(
        data_config.dataset_name,
        local_path=dataset_path if use_local else None,
    )

    # 保存原始验证集和测试集（用于 logprob 评估）
    raw_eval_dataset = dataset.get("validation", None)
    raw_test_dataset = dataset.get("test", None)

    # 转换为 Label Scoring 格式
    print("正在构建 Label Scoring 训练数据...")
    tokenized_dataset = create_label_scoring_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=data_config.max_length,
    )

    print(f"训练集大小: {len(tokenized_dataset['train'])}")
    if "validation" in tokenized_dataset:
        print(f"验证集大小: {len(tokenized_dataset['validation'])}")

    # ==================== 6. 初始化 wandb ====================
    wandb.init(
        project="qwen-sentiment-analysis",
        name=f"label-scoring-lora-r{args.lora_r}-{args.dataset}",
        config={
            "method": "label_scoring",
            "model_name": args.model_name,
            "dataset": args.dataset,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "use_qlora": args.use_qlora,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "label_texts": LABEL_TEXTS,
            "prompt_template": PROMPT_TEMPLATE,
        },
        tags=["LoRA", "label-scoring", "CausalLM",
              "QLoRA" if args.use_qlora else "LoRA", "sentiment-analysis"],
    )

    # ==================== 7. 配置训练参数 ====================
    training_args_dict = get_training_args(training_config)

    # Label Scoring 方案的特殊修改
    training_args_dict["metric_for_best_model"] = "eval_accuracy"
    training_args_dict["greater_is_better"] = True
    # CausalLM 不需要 remove_unused_columns=False（因为自定义了数据格式）
    training_args_dict["remove_unused_columns"] = False

    training_args = TrainingArguments(**training_args_dict)

    # ==================== 8. 初始化自定义 Trainer ====================
    # DataCollator：处理 label shifting 的 DataCollatorForSeq2Seq 不适用
    # 我们已经在 build_label_scoring_examples 中手动处理了 labels
    # 因此使用 default_data_collator
    from transformers import default_data_collator

    trainer = LabelScoringTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation", None),
        data_collator=default_data_collator,
        # Label Scoring 专用参数
        label_token_ids=label_token_ids,
        eval_dataset_raw=raw_eval_dataset,
        eval_tokenizer=tokenizer,
        eval_max_length=data_config.max_length,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ] if args.early_stopping else [],
    )

    # ==================== 9. 检测断点 ====================
    checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            checkpoint = get_last_checkpoint(args.output_dir)
            if checkpoint:
                print(f"\n检测到断点: {checkpoint}")
                print("将从断点恢复训练...\n")
            else:
                print("\n未检测到可用的 checkpoint，将从头开始训练...\n")
        else:
            checkpoint = args.resume_from_checkpoint
            if os.path.isdir(checkpoint):
                print(f"\n从指定断点恢复: {checkpoint}\n")
            else:
                print(f"\n警告: 指定的 checkpoint 不存在: {checkpoint}")
                print("将从头开始训练...\n")
                checkpoint = None

    # ==================== 10. 开始训练 ====================
    print("\n" + "=" * 50)
    print("开始训练（Label Scoring 方案）...")
    print("=" * 50 + "\n")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # ==================== 11. 保存模型 ====================
    print("\n保存 LoRA 适配器...")

    lora_save_path = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)

    # 保存方案元信息（推理时需要知道 prompt 模板和标签映射）
    meta_info = {
        "method": "label_scoring",
        "prompt_template": PROMPT_TEMPLATE,
        "label_texts": {str(k): v for k, v in LABEL_TEXTS.items()},
        "label_token_ids": {str(k): v for k, v in label_token_ids.items()},
        "model_type": "CausalLM",
        "base_model": args.model_name,
    }
    meta_path = os.path.join(lora_save_path, "label_scoring_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)
    print(f"已保存 Label Scoring 元信息: {meta_path}")

    trainer.save_metrics("train", train_result.metrics)

    # ==================== 12. 最终评估（测试集） ====================
    print("\n在测试集上进行最终评估（Label Scoring）...")
    if raw_test_dataset is not None:
        test_results = trainer.evaluate(
            eval_dataset=raw_test_dataset,
            metric_key_prefix="test",
        )
        print(f"\n测试集结果：")
        for key, value in test_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        trainer.save_metrics("test", test_results)

        # 记录测试结果到 wandb
        wandb.log({f"test/{key}": value for key, value in test_results.items()
                    if isinstance(value, (int, float))})

    print(f"\n训练完成！LoRA 适配器已保存到: {lora_save_path}")

    wandb.finish()

    return trainer


# ==================== 命令行参数 ====================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""

    parser = argparse.ArgumentParser(
        description="使用 LoRA + Label Scoring 微调 Qwen 进行中文情感分析"
    )

    # 模型参数
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
        help="模型名称或路径",
    )

    # 数据参数
    parser.add_argument(
        "--dataset", type=str, default="ChnSentiCorp",
        choices=["ChnSentiCorp", "IMDB_Chinese"],
        help="数据集名称",
    )
    parser.add_argument(
        "--max_length", type=int, default=256,
        help="最大序列长度",
    )

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--use_qlora", action="store_true", help="使用 QLoRA（4-bit 量化）")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./experiments/label_scoring",
                        help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="启用梯度检查点（节省显存）")
    parser.add_argument("--early_stopping", action="store_true", default=True,
                        help="启用早停")
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default="auto",
        help="从 checkpoint 恢复训练。'auto' / 具体路径 / 'none'",
    )

    args = parser.parse_args()

    if args.resume_from_checkpoint.lower() == "none":
        args.resume_from_checkpoint = None

    return args


# ==================== 入口 ====================

if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("中文情感分析 - LoRA + Label Scoring（生成式分类）")
    print("=" * 60)
    print(f"\n配置信息：")
    print(f"  模型: {args.model_name}")
    print(f"  模型类型: CausalLM（复用 LM Head，不使用分类头）")
    print(f"  分类方式: Label Scoring（比较标签 logprob）")
    print(f"  标签映射: {LABEL_TEXTS}")
    print(f"  数据集: {args.dataset}")
    print(f"  LoRA r: {args.lora_r}")
    print(f"  QLoRA: {'是' if args.use_qlora else '否'}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  输出目录: {args.output_dir}")
    print()

    train(args)
