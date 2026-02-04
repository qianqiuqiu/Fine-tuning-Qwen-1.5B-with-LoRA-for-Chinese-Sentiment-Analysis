"""
数据加载模块
负责从 HuggingFace Hub 或本地加载中文情感分析数据集
"""

from datasets import load_dataset, DatasetDict, Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from typing import Optional, Dict, Any
import os


def load_sentiment_dataset(
    dataset_name: str = "ChnSentiCorp",
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    加载情感分析数据集
    
    Args:
        dataset_name: 数据集名称，支持 "ChnSentiCorp" 或 "IMDB_Chinese"
        cache_dir: 缓存目录
    
    Returns:
        DatasetDict: 包含 train/validation/test 的数据集
    
    数据集格式：
        - text: 评论文本
        - label: 0（负面）或 1（正面）
    """
    
    if dataset_name == "ChnSentiCorp":
        # ChnSentiCorp 是一个中文酒店评论情感分析数据集
        # 约 9600 条训练数据，1200 条验证/测试数据
        # 注意：新版 datasets 库不再支持 trust_remote_code，使用标准 Parquet 格式
        dataset = load_dataset(
            "lansinuote/ChnSentiCorp",  # 使用已转换为 Parquet 格式的版本
            cache_dir=cache_dir,
        )
        
    elif dataset_name == "IMDB_Chinese":
        # 如果使用 IMDB 中文翻译版本
        # 需要自行准备或从其他来源获取
        dataset = load_dataset(
            "imdb",  # 原始 IMDB，需要翻译处理
            cache_dir=cache_dir,
        )
        # 注意：实际使用时需要进行中文翻译处理
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return dataset


def load_local_dataset(
    data_dir: str,
    train_file: str = "train.json",
    eval_file: str = "eval.json",
    test_file: str = "test.json",
) -> DatasetDict:
    """
    从本地文件加载数据集
    
    Args:
        data_dir: 数据目录
        train_file: 训练集文件名
        eval_file: 验证集文件名
        test_file: 测试集文件名
    
    Returns:
        DatasetDict: 数据集字典
    
    文件格式（JSON Lines）：
        {"text": "这个酒店很好", "label": 1}
        {"text": "服务太差了", "label": 0}
    """
    
    data_files = {}
    
    train_path = os.path.join(data_dir, train_file)
    if os.path.exists(train_path):
        data_files["train"] = train_path
        
    eval_path = os.path.join(data_dir, eval_file)
    if os.path.exists(eval_path):
        data_files["validation"] = eval_path
        
    test_path = os.path.join(data_dir, test_file)
    if os.path.exists(test_path):
        data_files["test"] = test_path
    
    if not data_files:
        raise ValueError(f"在 {data_dir} 中未找到任何数据文件")
    
    dataset = load_dataset("json", data_files=data_files)
    
    return dataset


def get_data_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorWithPadding:
    """
    获取数据整理器
    
    用于将不同长度的样本填充到相同长度，形成批次
    
    Args:
        tokenizer: 分词器
    
    Returns:
        DataCollatorWithPadding: 数据整理器
    """
    return DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=None,  # 动态填充到批次最大长度
        return_tensors="pt",
    )


def inspect_dataset(dataset: DatasetDict) -> Dict[str, Any]:
    """
    检查数据集的基本信息
    
    Args:
        dataset: 数据集
    
    Returns:
        包含数据集统计信息的字典
    """
    info = {
        "splits": list(dataset.keys()),
        "features": str(dataset["train"].features) if "train" in dataset else None,
    }
    
    for split in dataset.keys():
        info[f"{split}_size"] = len(dataset[split])
        
        # 标签分布
        if "label" in dataset[split].features:
            labels = dataset[split]["label"]
            info[f"{split}_label_distribution"] = {
                "negative (0)": labels.count(0),
                "positive (1)": labels.count(1),
            }
    
    return info


if __name__ == "__main__":
    # 测试数据加载
    print("加载 ChnSentiCorp 数据集...")
    dataset = load_sentiment_dataset("ChnSentiCorp")
    
    info = inspect_dataset(dataset)
    print("\n数据集信息：")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n样本示例：")
    for i in range(3):
        sample = dataset["train"][i]
        print(f"  文本: {sample['text'][:50]}...")
        print(f"  标签: {'正面' if sample['label'] == 1 else '负面'}")
        print()
