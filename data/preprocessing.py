"""
数据预处理模块
负责文本清洗、分词等预处理工作
"""

import re
from typing import Dict, List, Any, Optional
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer


def clean_text(text: str) -> str:
    """
    清洗文本
    
    Args:
        text: 原始文本
    
    Returns:
        清洗后的文本
    """
    if not isinstance(text, str):
        return ""
    
    # 移除多余空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊控制字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # 移除连续的标点符号（保留一个）
    text = re.sub(r'([。！？，、；：])\1+', r'\1', text)
    
    # 去除首尾空白
    text = text.strip()
    
    return text


def preprocess_function(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    text_column: str = "text",
    label_column: str = "label",
    padding: str = "max_length",
    truncation: bool = True,
) -> Dict[str, List]:
    """
    预处理函数 - 对批量样本进行分词处理
    
    Args:
        examples: 批量样本字典
        tokenizer: 分词器
        max_length: 最大序列长度
        text_column: 文本列名
        label_column: 标签列名
        padding: 填充策略
        truncation: 是否截断
    
    Returns:
        处理后的样本字典，包含 input_ids, attention_mask, labels
    """
    
    # 获取文本并清洗
    texts = examples[text_column]
    texts = [clean_text(text) for text in texts]
    
    # 分词
    tokenized = tokenizer(
        texts,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=None,  # 返回 Python 列表
    )
    
    # 添加标签
    if label_column in examples:
        tokenized["labels"] = examples[label_column]
    
    return tokenized


def create_tokenized_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    text_column: str = "text",
    label_column: str = "label",
    num_proc: int = 4,
    remove_original_columns: bool = True,
) -> DatasetDict:
    """
    创建分词后的数据集
    
    Args:
        dataset: 原始数据集
        tokenizer: 分词器
        max_length: 最大序列长度
        text_column: 文本列名
        label_column: 标签列名
        num_proc: 并行处理进程数
        remove_original_columns: 是否移除原始列
    
    Returns:
        分词后的数据集
    """
    
    # 定义处理函数（使用闭包捕获参数）
    def tokenize_fn(examples):
        return preprocess_function(
            examples=examples,
            tokenizer=tokenizer,
            max_length=max_length,
            text_column=text_column,
            label_column=label_column,
            padding="max_length",
            truncation=True,
        )
    
    # 获取需要移除的列
    columns_to_remove = []
    if remove_original_columns:
        # 保留模型需要的列
        keep_columns = {"input_ids", "attention_mask", "labels", "token_type_ids"}
        sample_split = list(dataset.keys())[0]
        columns_to_remove = [
            col for col in dataset[sample_split].column_names 
            if col not in keep_columns
        ]
    
    # 对每个分割进行处理
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=columns_to_remove,
        desc="Tokenizing dataset",
    )
    
    return tokenized_dataset


def get_dataset_statistics(dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """
    获取数据集的统计信息
    
    Args:
        dataset: 数据集（单个分割）
        tokenizer: 分词器
    
    Returns:
        统计信息字典
    """
    
    lengths = []
    for example in dataset:
        if "text" in example:
            tokens = tokenizer(example["text"], truncation=False)
            lengths.append(len(tokens["input_ids"]))
        elif "input_ids" in example:
            lengths.append(len(example["input_ids"]))
    
    if not lengths:
        return {}
    
    return {
        "num_samples": len(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "min_length": min(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2],
    }


def balance_dataset(dataset: Dataset, label_column: str = "label") -> Dataset:
    """
    平衡数据集（过采样少数类）
    
    Args:
        dataset: 原始数据集
        label_column: 标签列名
    
    Returns:
        平衡后的数据集
    """
    from collections import Counter
    
    labels = dataset[label_column]
    label_counts = Counter(labels)
    
    # 找出多数类数量
    max_count = max(label_counts.values())
    
    # 对每个类别进行过采样
    balanced_indices = []
    for label in label_counts.keys():
        label_indices = [i for i, l in enumerate(labels) if l == label]
        
        # 重复采样直到达到多数类数量
        while len(label_indices) < max_count:
            import random
            label_indices.extend(
                random.choices(label_indices, k=min(len(label_indices), max_count - len(label_indices)))
            )
        
        balanced_indices.extend(label_indices[:max_count])
    
    # 打乱顺序
    import random
    random.shuffle(balanced_indices)
    
    return dataset.select(balanced_indices)


if __name__ == "__main__":
    # 测试预处理
    from transformers import AutoTokenizer
    from data_loader import load_sentiment_dataset
    
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        trust_remote_code=True,
    )
    
    print("加载数据集...")
    dataset = load_sentiment_dataset("ChnSentiCorp")
    
    print("进行分词处理...")
    tokenized_dataset = create_tokenized_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=256,
    )
    
    print("\n处理后的数据集：")
    print(tokenized_dataset)
    
    print("\n样本示例：")
    sample = tokenized_dataset["train"][0]
    print(f"  input_ids 长度: {len(sample['input_ids'])}")
    print(f"  标签: {sample['labels']}")
