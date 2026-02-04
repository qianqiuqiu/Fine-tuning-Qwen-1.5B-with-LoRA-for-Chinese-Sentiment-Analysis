"""
训练配置文件
定义训练超参数和路径配置
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import torch


@dataclass
class ModelConfig:
    """模型配置"""
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"  # Qwen 1.5B 模型
    num_labels: int = 2  # 二分类：正面/负面
    trust_remote_code: bool = True
    use_cache: bool = False  # 训练时关闭 KV cache


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = "ChnSentiCorp"  # 数据集名称
    max_length: int = 256  # 最大序列长度
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    preprocessing_num_workers: int = 4


@dataclass
class TrainingConfig:
    """训练配置"""
    # 输出路径
    output_dir: str = "./outputs"
    logging_dir: str = "./logs"
    
    # 训练参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4  # 有效批次 = 8 * 4 = 32
    
    # 学习率配置
    learning_rate: float = 2e-4  # LoRA 通常使用较大学习率
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # 优化器
    optim: str = "adamw_torch"
    
    # 评估与保存
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True
    
    # 日志
    logging_steps: int = 10
    report_to: str = "wandb"  # 使用 wandb 进行可视化
    
    # 其他
    seed: int = 42
    bf16: bool = True  # 使用 BF16 混合精度
    fp16: bool = False
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    
    # 梯度检查点（节省显存）
    gradient_checkpointing: bool = True


@dataclass
class QLoRAConfig:
    """QLoRA 量化配置"""
    use_qlora: bool = False  # 是否启用 QLoRA
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # 量化类型
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True  # 双重量化进一步节省显存


# 数据集路径映射
DATASET_PATHS = {
    "ChnSentiCorp": {
        "name": "seamew/ChnSentiCorp",  # HuggingFace Hub 上的数据集
        "text_column": "text",
        "label_column": "label",
    },
    "IMDB_Chinese": {
        "name": "imdb_chinese",  # 自定义路径或 HuggingFace 数据集
        "text_column": "text", 
        "label_column": "label",
    },
}


def get_training_args(config: TrainingConfig):
    """将 TrainingConfig 转换为 TrainingArguments 需要的字典"""
    
    # 自动检测混合精度支持
    use_bf16 = config.bf16
    use_fp16 = config.fp16
    
    if torch.cuda.is_available() and use_bf16:
        # 如果配置要求使用 bf16，检测 GPU 是否支持
        if not torch.cuda.is_bf16_supported():
            print("⚠️  GPU 不支持 BF16，自动切换到 FP16")
            use_bf16 = False
            use_fp16 = True
    
    return {
        "output_dir": config.output_dir,
        "logging_dir": config.logging_dir,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "optim": config.optim,
        "eval_strategy": config.eval_strategy,
        "eval_steps": config.eval_steps,
        "save_strategy": config.save_strategy,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": config.load_best_model_at_end,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": config.greater_is_better,
        "logging_steps": config.logging_steps,
        "report_to": config.report_to,
        "seed": config.seed,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "dataloader_num_workers": config.dataloader_num_workers,
        "remove_unused_columns": config.remove_unused_columns,
        "gradient_checkpointing": config.gradient_checkpointing,
    }
