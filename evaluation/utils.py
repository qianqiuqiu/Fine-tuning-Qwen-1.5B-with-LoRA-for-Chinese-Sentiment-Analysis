"""
共用工具函数
Common utility functions for model evaluation
"""

import os

# 修复 torch 导入卡死问题 (Intel MKL 库冲突)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_torch_dtype():
    """
    根据设备自动选择合适的数据类型
    
    Returns:
        torch.dtype: 推荐的数据类型
    """
    if not torch.cuda.is_available():
        # CPU不支持float16，使用float32
        return torch.float32
    
    # 检查CUDA设备是否支持BFloat16
    try:
        device = torch.device("cuda")
        # 尝试创建一个BFloat16张量
        test_tensor = torch.zeros(1, dtype=torch.bfloat16, device=device)
        del test_tensor
        torch.cuda.empty_cache()
        return torch.bfloat16
    except:
        # 如果BFloat16不支持，使用float16
        try:
            test_tensor = torch.zeros(1, dtype=torch.float16, device=device)
            del test_tensor
            torch.cuda.empty_cache()
            return torch.float16
        except:
            # 如果float16也不支持，使用float32
            return torch.float32


def ensure_output_dir(subdir: Optional[str] = None) -> Path:
    """
    确保输出目录存在

    Args:
        subdir: 子目录名（如 "plots"）

    Returns:
        输出目录路径
    """
    output_dir = Path(__file__).parent / "outputs"
    if subdir:
        output_dir = output_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_model_and_tokenizer(
    base_model_name: str = "Qwen/Qwen2.5-1.5B",
    lora_path: Optional[str] = None,
    device: str = "auto",
    merge_lora: bool = True,
) -> Tuple[Any, Any]:
    """
    加载模型和分词器

    Args:
        base_model_name: 基础模型名称
        lora_path: LoRA 适配器路径，为 None 时不加载 LoRA
        device: 设备 ("cuda", "cpu", "auto")
        merge_lora: 是否合并 LoRA 权重

    Returns:
        (model, tokenizer) 元组
    """
    device_map = "auto" if device == "auto" else {"": device}
    
    # 获取合适的数据类型
    torch_dtype = get_torch_dtype()
    print(f"使用数据类型: {torch_dtype}")

    print(f"加载分词器: {lora_path if lora_path else base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path if lora_path else base_model_name,
        trust_remote_code=True,
    )

    # 确保padding token正确设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"加载基础模型: {base_model_name}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    if lora_path:
        print(f"加载 LoRA 适配器: {lora_path}")
        model = PeftModel.from_pretrained(base_model, lora_path)
        if merge_lora:
            print("合并 LoRA 权重...")
            model = model.merge_and_unload()
    else:
        model = base_model

    # 确保模型配置中有 pad_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return model, tokenizer


def load_base_model_with_lora_classifier(
    base_model_name: str = "Qwen/Qwen2.5-1.5B",
    lora_path: str = None,
    device: str = "auto",
) -> Tuple[Any, Any]:
    """
    加载冻结的基础模型 + LoRA训练后的分类头
    
    这个函数用于公平对比：加载基础Qwen模型（参数冻结），
    但使用LoRA训练后的分类头。这样可以量化LoRA微调对基础模型的影响。

    Args:
        base_model_name: 基础模型名称
        lora_path: LoRA 适配器路径（用于加载训练后的分类头）
        device: 设备 ("cuda", "cpu", "auto")

    Returns:
        (model, tokenizer) 元组
    """
    device_map = "auto" if device == "auto" else {"": device}
    
    # 获取合适的数据类型
    torch_dtype = get_torch_dtype()
    print(f"使用数据类型: {torch_dtype}")

    print(f"\n加载冻结的基础模型 + LoRA训练后的分类头")
    print(f"  基础模型: {base_model_name}")
    print(f"  分类头来源: {lora_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path if lora_path else base_model_name,
        trust_remote_code=True,
    )

    # 确保padding token正确设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载基础模型
    print(f"  步骤1: 加载基础模型（参数将被冻结）")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    
    # 如果提供了lora_path，加载LoRA适配器并提取分类头
    if lora_path:
        print(f"  步骤2: 加载LoRA适配器以获取训练后的分类头")
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        
        print(f"  步骤3: 合并权重以获得完整的训练后分类头")
        merged_model = lora_model.merge_and_unload()
        
        # 提取分类头的参数
        print(f"  步骤4: 将训练后的分类头复制到冻结的基础模型")
        # 重新加载一个干净的基础模型
        frozen_base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        
        # 复制分类头参数（通常是score/classifier层）
        if hasattr(merged_model, 'score'):
            frozen_base_model.score.load_state_dict(merged_model.score.state_dict())
            print(f"    ✓ 分类头 'score' 已复制")
        elif hasattr(merged_model, 'classifier'):
            frozen_base_model.classifier.load_state_dict(merged_model.classifier.state_dict())
            print(f"    ✓ 分类头 'classifier' 已复制")
        
        # 清理内存
        del base_model, lora_model, merged_model
        torch.cuda.empty_cache()
        
        model = frozen_base_model
        print(f"  ✓ 完成：基础模型（冻结）+ LoRA训练后的分类头")
    else:
        model = base_model
        print(f"  警告：未提供lora_path，使用随机初始化的分类头")

    # 确保模型配置中有 pad_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return model, tokenizer


def get_predictions_with_confidence(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 256,
    return_all_probs: bool = False,
) -> Dict[str, Any]:
    """
    获取模型预测和置信度

    Args:
        model: 模型
        tokenizer: 分词器
        texts: 文本列表
        batch_size: 批次大小
        max_length: 最大序列长度
        return_all_probs: 是否返回所有类别的概率

    Returns:
        {
            "predictions": 预测标签列表,
            "confidences": 正类置信度列表,
            "all_probs": 所有类别概率 (可选),
            "logits": 原始 logits (可选)
        }
    """
    all_predictions = []
    all_confidences = []
    all_probs = []
    all_logits = []

    model.eval()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_predictions.extend(preds.cpu().numpy().tolist())
            all_confidences.extend(probs[:, 1].cpu().float().numpy().tolist())
            all_probs.extend(probs.cpu().float().numpy().tolist())
            all_logits.extend(logits.cpu().float().numpy().tolist())

    result = {
        "predictions": all_predictions,
        "confidences": all_confidences,
    }

    if return_all_probs:
        result["all_probs"] = all_probs
        result["logits"] = all_logits

    return result


def compute_ece(
    predictions: List[int],
    labels: List[int],
    confidences: List[float],
    n_bins: int = 10,
) -> float:
    """
    计算期望校准误差 (Expected Calibration Error, ECE)

    ECE 衡量模型预测置信度与实际准确率之间的校准程度。
    ECE 越小表示模型校准越好。

    Args:
        predictions: 预测标签列表
        labels: 真实标签列表
        confidences: 置信度列表
        n_bins: 分箱数量

    Returns:
        ECE 值
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    confidences = np.array(confidences)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    return ece, {
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries.tolist(),
    }


def save_json(data: Any, filename: str, subdir: Optional[str] = None):
    """
    保存数据到 JSON 文件

    Args:
        data: 要保存的数据
        filename: 文件名
        subdir: 子目录目录
    """
    output_dir = ensure_output_dir(subdir)
    output_path = output_dir / filename

    # 转换 numpy 类型为 Python 原生类型
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [convert_types(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(x) for x in obj]
        return obj

    converted_data = convert_types(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"保存结果到: {output_path}")
    return output_path


def load_json(filename: str, subdir: Optional[str] = None) -> Any:
    """
    从 JSON 文件加载数据

    Args:
        filename: 文件名
        subdir: 子目录

    Returns:
        加载的数据
    """
    output_dir = ensure_output_dir(subdir)
    input_path = output_dir / filename

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_metrics_table(metrics: Dict[str, float], title: str = "评估指标") -> str:
    """
    格式化指标为表格字符串

    Args:
        metrics: 指标字典
        title: 表格标题

    Returns:
        格式化的表格字符串
    """
    lines = [f"\n{'=' * 50}", f"{title}", "=" * 50]

    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")

    lines.append("=" * 50)
    return "\n".join(lines)


def count_trainable_params(model) -> Tuple[int, int]:
    """
    统计模型的可训练参数和总参数数量

    Args:
        model: PyTorch 模型

    Returns:
        (可训练参数数量, 总参数数量)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_device() -> str:
    """获取当前可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def format_confidence_interval(
    mean: float,
    std: float,
    n_samples: int,
    confidence: float = 0.95,
) -> str:
    """
    格式化置信区间

    Args:
        mean: 均值
        std: 标准差
        n_samples: 样本数
        confidence: 置信水平

    Returns:
        格式化的置信区间字符串
    """
    from scipy import stats
    import math

    se = std / math.sqrt(n_samples)
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha / 2, n_samples - 1)
    margin = t_value * se

    lower = mean - margin
    upper = mean + margin

    return f"{mean:.4f} [{lower:.4f}, {upper:.4f}]"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本到指定长度

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


class ProgressTracker:
    """进度跟踪器，用于多阶段任务"""

    def __init__(self, total_stages: int):
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_names = []

    def add_stage(self, name: str):
        """添加一个阶段"""
        self.stage_names.append(name)

    def next_stage(self, name: Optional[str] = None) -> None:
        """进入下一个阶段"""
        self.current_stage += 1
        if name:
            self.stage_names.append(name)

        stage_name = self.stage_names[self.current_stage - 1] if self.current_stage <= len(self.stage_names) else f"阶段 {self.current_stage}"
        print(f"\n{'=' * 50}")
        print(f"[{self.current_stage}/{self.total_stages}] {stage_name}")
        print("=" * 50)

    def progress(self) -> float:
        """获取当前进度 (0-1)"""
        return self.current_stage / self.total_stages
