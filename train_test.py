"""
测试训练脚本
用于快速测试训练流程是否正常，只使用50条数据

使用方法:
    python train_test.py
    python train_test.py --use_qlora  # 测试 QLoRA
"""

import os
import argparse
import torch
import numpy as np
from typing import Dict

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, prepare_model_for_kbit_training

# 导入项目配置
from configs import (
    get_lora_config,
    ModelConfig,
    DataConfig,
    QLoRAConfig,
)
from data import load_sentiment_dataset, create_tokenized_dataset, get_data_collator

# 评估指标
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred) -> Dict[str, float]:
    """计算评估指标"""
    predictions, labels = eval_pred
    
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def setup_model_and_tokenizer(
    model_config: ModelConfig,
    qlora_config: QLoRAConfig,
) -> tuple:
    """初始化模型和分词器"""
    
    print(f"正在加载模型: {model_config.model_name_or_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置量化（QLoRA）
    quantization_config = None
    if qlora_config.use_qlora:
        print("启用 QLoRA 4-bit 量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=qlora_config.load_in_4bit,
            bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant,
        )
    
    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=model_config.num_labels,
        trust_remote_code=model_config.trust_remote_code,
        quantization_config=quantization_config,
        device_map="auto" if qlora_config.use_qlora else None,
        torch_dtype=torch.bfloat16,
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if qlora_config.use_qlora:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


def setup_lora(model, lora_r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1):
    """为模型添加 LoRA 适配器"""
    
    print(f"配置 LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    lora_config = get_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def train_test(args: argparse.Namespace):
    """测试训练函数"""
    
    print("=" * 60)
    print("测试训练模式 - 仅使用50条数据")
    print("=" * 60)
    
    # ==================== 1. 配置初始化 ====================
    model_config = ModelConfig(
        model_name_or_path=args.model_name,
        num_labels=2,
    )
    
    data_config = DataConfig(
        dataset_name=args.dataset,
        max_length=256,
    )
    
    qlora_config = QLoRAConfig(
        use_qlora=args.use_qlora,
    )
    
    # ==================== 2. 加载模型和分词器 ====================
    model, tokenizer = setup_model_and_tokenizer(model_config, qlora_config)
    
    # ==================== 3. 应用 LoRA ====================
    model = setup_lora(
        model=model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # ==================== 4. 加载和处理数据 ====================
    print(f"\n正在加载数据集: {data_config.dataset_name}")
    dataset = load_sentiment_dataset(data_config.dataset_name)
    
    print("正在进行分词处理...")
    tokenized_dataset = create_tokenized_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=data_config.max_length,
    )
    
    # ==================== 5. 只选择前50条数据 ====================
    print("\n⚠️  测试模式：只使用50条训练数据，20条验证数据")
    
    train_dataset = tokenized_dataset["train"].select(range(min(50, len(tokenized_dataset["train"]))))
    eval_dataset = tokenized_dataset["validation"].select(range(min(20, len(tokenized_dataset["validation"]))))
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    
    # ==================== 6. 配置训练参数（简化版，无预热） ====================
    # 自动检测精度支持
    use_bf16 = False
    use_fp16 = False
    
    if torch.cuda.is_available():
        # 检测是否支持 bf16
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("✅ 使用 BF16 混合精度训练")
        else:
            use_fp16 = True
            print("✅ 使用 FP16 混合精度训练 (GPU 不支持 BF16)")
    else:
        print("⚠️  未检测到 GPU，使用 CPU 训练 (速度较慢)")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        
        # 学习率配置 - 无预热
        learning_rate=args.learning_rate,
        warmup_ratio=0.0,  # 不使用预热
        lr_scheduler_type="constant",  # 使用恒定学习率
        
        # 评估与保存
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        
        # 日志
        logging_steps=5,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",  # 不使用 tensorboard
        
        # 其他
        seed=42,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=0,  # 测试时不使用多线程
        remove_unused_columns=False,
        load_best_model_at_end=False,  # 测试时不需要
    )
    
    # ==================== 7. 初始化 Trainer ====================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=get_data_collator(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # ==================== 8. 开始训练 ====================
    print("\n" + "=" * 50)
    print("开始测试训练...")
    print("=" * 50 + "\n")
    
    train_result = trainer.train()
    
    # ==================== 9. 保存模型 ====================
    print("\n保存测试模型...")
    
    lora_save_path = os.path.join(args.output_dir, "lora_adapter_test")
    model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)
    
    print(f"\n✅ 测试训练完成！模型已保存到: {lora_save_path}")
    print(f"训练步数: {train_result.global_step}")
    print(f"最终损失: {train_result.training_loss:.4f}")
    
    # ==================== 10. 快速评估 ====================
    print("\n快速评估...")
    eval_results = trainer.evaluate(eval_dataset)
    print(f"\n验证集结果：")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    return trainer


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    
    parser = argparse.ArgumentParser(
        description="测试 LoRA 微调流程（50条数据）"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="模型名称或路径",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ChnSentiCorp",
        help="数据集名称",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA 秩",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="使用 QLoRA（4-bit 量化）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_test",
        help="输出目录",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批次大小",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("=" * 60)
    print("🧪 测试训练模式")
    print("=" * 60)
    print(f"\n配置信息：")
    print(f"  模型: {args.model_name}")
    print(f"  数据集: {args.dataset}")
    print(f"  训练数据: 50 条")
    print(f"  验证数据: 20 条")
    print(f"  LoRA r: {args.lora_r}")
    print(f"  QLoRA: {'是' if args.use_qlora else '否'}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate} (无预热)")
    print()
    
    train_test(args)
