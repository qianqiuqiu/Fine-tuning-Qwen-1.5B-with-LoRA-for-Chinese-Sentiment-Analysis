"""
LoRA 配置文件
定义 LoRA 微调的超参数
"""

from peft import LoraConfig, TaskType

def get_lora_config(
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    use_qlora: bool = False
) -> LoraConfig:
    """
    获取 LoRA 配置
    
    Args:
        r: LoRA 秩，控制低秩矩阵的维度，越大模型容量越高但训练成本增加
        lora_alpha: LoRA 缩放因子，通常设为 r 的 2-4 倍
        lora_dropout: Dropout 概率，防止过拟合
        use_qlora: 是否使用 QLoRA（4-bit 量化）
    
    Returns:
        LoraConfig 对象
    """
    
    # 目标模块 - Qwen 模型的注意力层
    # 对于 Qwen 系列模型，主要对 query、key、value 投影层进行 LoRA
    target_modules = [
        "q_proj",   # Query 投影
        "k_proj",   # Key 投影
        "v_proj",   # Value 投影
        "o_proj",   # Output 投影
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # 序列分类任务
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",  # 不训练 bias
        inference_mode=False,
    )
    
    return lora_config


# 预定义的配置方案
LORA_CONFIGS = {
    # 轻量级配置 - 适合快速实验
    "light": {
        "r": 4,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
    
    # 标准配置 - 平衡效果与效率
    "standard": {
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    },
    
    # 增强配置 - 追求更好效果
    "enhanced": {
        "r": 16,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
    },
    
    # 完整配置 - 最大容量
    "full": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
    },
}
