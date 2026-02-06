"""
重新生成包含基准对比的评估报告
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluation.run_full_eval import FullEvaluator

def main():
    print("=" * 70)
    print("重新生成包含基准对比的完整评估报告")
    print("=" * 70)
    
    # 配置
    model_path = str(project_root / "outputs" / "lora_adapter")
    
    print(f"\nLoRA 模型路径: {model_path}")
    print(f"基础模型: Qwen/Qwen2.5-1.5B")
    print(f"\n将进行以下评估：")
    print("  1. 基础评估指标")
    print("  2. 置信度分析")
    print("  3. 鲁棒性测试")
    print("  4. 基准对比 (LoRA vs 基础模型)")
    print("  5. 生成可视化报告")
    
    # 创建评估器
    evaluator = FullEvaluator(
        model_path=model_path,
        base_model="Qwen/Qwen2.5-1.5B",
        dataset="ChnSentiCorp",
        batch_size=8,  # 减小批次大小以节省内存
        max_length=256,
    )
    
    # 运行完整评估（不包括 BERT，以节省时间）
    print(f"\n开始评估...")
    results = evaluator.run_full_evaluation(run_bert=False)
    
    print(f"\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)
    
    # 输出文件位置
    outputs_dir = project_root / "evaluation" / "outputs"
    print(f"\n生成的文件：")
    print(f"  - 评估报告: {outputs_dir / 'report.html'}")
    print(f"  - 基础指标: {outputs_dir / 'metrics.json'}")
    print(f"  - 置信度分析: {outputs_dir / 'confidence_analysis.json'}")
    print(f"  - 鲁棒性测试: {outputs_dir / 'robustness_result.json'}")
    print(f"  - 基准对比: {outputs_dir / 'baseline_comparison.json'}")
    print(f"\n请在浏览器中打开 report.html 查看完整报告")

if __name__ == "__main__":
    main()
