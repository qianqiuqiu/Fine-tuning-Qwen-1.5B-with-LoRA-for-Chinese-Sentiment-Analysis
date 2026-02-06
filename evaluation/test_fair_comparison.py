"""
测试公平对比功能
验证：冻结的Qwen基础模型+LoRA分类头 vs LoRA微调的完整模型
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluation.run_full_eval import FullEvaluator


def main():
    """运行公平对比测试"""
    parser = argparse.ArgumentParser(description="测试LoRA微调效果的公平对比")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./experiments/classifier_head/lora_adapter",
        help="LoRA适配器路径",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="批次大小",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LoRA微调效果公平对比测试")
    print("=" * 80)
    print(f"\n对比说明：")
    print(f"  1. 冻结的Qwen基础模型 + LoRA训练后的分类头")
    print(f"     → 测试分类头本身的能力")
    print(f"  2. LoRA微调的Qwen模型 + LoRA训练后的分类头")
    print(f"     → 测试完整微调后的能力")
    print(f"  3. 性能差异 = LoRA对基础模型特征提取的提升")
    print(f"\n这个对比保证了分类头完全相同，只对比基础模型的差异！\n")
    
    # 创建评估器
    evaluator = FullEvaluator(
        model_path=args.model_path,
        base_model="Qwen/Qwen2.5-1.5B",
        dataset="ChnSentiCorp",
        batch_size=args.batch_size,
        max_length=256,
    )
    
    # 只运行基准对比部分
    print("\n加载数据集...")
    texts, labels = evaluator.load_dataset()
    
    print("\n运行公平对比测试...")
    evaluator.run_benchmark(texts, labels, run_bert=False)
    
    print("\n✓ 测试完成！")
    print(f"\n结果已保存到: {project_root / 'outputs' / 'baseline_comparison.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
