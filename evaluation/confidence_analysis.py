"""
置信度分析模块
Confidence Analysis Module

分析模型预测的置信度分布和校准情况
"""

import os

# 修复 torch 导入卡死问题 (Intel MKL 库冲突)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from pathlib import Path

import torch
from datasets import load_dataset

from .utils import (
    load_model_and_tokenizer,
    get_predictions_with_confidence,
    compute_ece,
    save_json,
    ensure_output_dir,
    get_device,
    truncate_text,
)


# 默认配置
DEFAULT_CONFIG = {
    "low_confidence_threshold": 0.6,
    "high_confidence_threshold": 0.9,
    "n_bins": 10,
    "batch_size": 16,
    "max_length": 256,
}


class ConfidenceAnalyzer:
    """置信度分析器"""

    def __init__(
        self,
        model,
        tokenizer,
        low_confidence_threshold: float = 0.6,
        high_confidence_threshold: float = 0.9,
        n_bins: int = 10,
    ):
        """
        初始化分析器

        Args:
            model: 模型
            tokenizer: 分词器
            low_confidence_threshold: 低置信度阈值
            high_confidence_threshold: 高置信度阈值
            n_bins: ECE 分箱数量
        """
        self.model = model
        self.tokenizer = tokenizer
        self.low_threshold = low_confidence_threshold
        self.high_threshold = high_confidence_threshold
        self.n_bins = n_bins

    def analyze(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 16,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """
        执行完整的置信度分析

        Args:
            texts: 文本列表
            labels: 标签列表
            batch_size: 批次大小
            max_length: 最大序列长度

        Returns:
            分析结果字典
        """
        print(f"\n获取预测结果...")
        pred_result = get_predictions_with_confidence(
            self.model, self.tokenizer, texts,
            batch_size=batch_size,
            max_length=max_length,
            return_all_probs=True,
        )

        predictions = pred_result["predictions"]
        confidences = pred_result["confidences"]

        # 计算各项指标
        results = {}

        # 基本统计
        results["basic_stats"] = self._compute_basic_stats(
            labels, predictions, confidences
        )

        # ECE 和可靠性图数据
        ece, reliability_data = compute_ece(
            predictions, labels, confidences, self.n_bins
        )
        results["ece"] = ece
        results["reliability_diagram"] = reliability_data

        # 低置信度样本
        results["low_confidence"] = self._analyze_low_confidence(
            texts, labels, predictions, confidences
        )

        # 高置信度错误样本（最危险）
        results["high_confidence_errors"] = self._analyze_high_confidence_errors(
            texts, labels, predictions, confidences
        )

        # 置信度分布数据
        results["confidence_distribution"] = self._compute_confidence_distribution(
            labels, predictions, confidences
        )

        return results

    def _compute_basic_stats(
        self,
        labels: List[int],
        predictions: List[int],
        confidences: List[float],
    ) -> Dict[str, Any]:
        """计算基本统计信息"""
        labels = np.array(labels)
        predictions = np.array(predictions)
        confidences = np.array(confidences)

        correct = (predictions == labels)
        wrong = ~correct

        return {
            "total_samples": len(labels),
            "correct": int(correct.sum()),
            "wrong": int(wrong.sum()),
            "accuracy": float(correct.mean()),
            "mean_confidence": float(confidences.mean()),
            "mean_confidence_correct": float(confidences[correct].mean()) if correct.sum() > 0 else 0,
            "mean_confidence_wrong": float(confidences[wrong].mean()) if wrong.sum() > 0 else 0,
            "std_confidence": float(confidences.std()),
            "min_confidence": float(confidences.min()),
            "max_confidence": float(confidences.max()),
            "median_confidence": float(np.median(confidences)),
        }

    def _analyze_low_confidence(
        self,
        texts: List[str],
        labels: List[int],
        predictions: List[int],
        confidences: List[float],
        max_examples: int = 20,
    ) -> Dict[str, Any]:
        """分析低置信度样本"""
        low_conf_idx = [
            i for i, c in enumerate(confidences)
            if c < self.low_threshold
        ]

        low_conf_samples = []
        for idx in low_conf_idx[:max_examples]:
            low_conf_samples.append({
                "text": truncate_text(texts[idx], 100),
                "true_label": int(labels[idx]),
                "predicted": int(predictions[idx]),
                "confidence": float(confidences[idx]),
                "correct": (predictions[idx] == labels[idx]),
            })

        # 按置信度排序
        low_conf_idx.sort(key=lambda i: confidences[i])

        return {
            "count": len(low_conf_idx),
            "percentage": len(low_conf_idx) / len(texts) if texts else 0,
            "accuracy": sum(
                predictions[i] == labels[i] for i in low_conf_idx
            ) / len(low_conf_idx) if low_conf_idx else 0,
            "samples": low_conf_samples,
        }

    def _analyze_high_confidence_errors(
        self,
        texts: List[str],
        labels: List[int],
        predictions: List[int],
        confidences: List[float],
        max_examples: int = 20,
    ) -> Dict[str, Any]:
        """分析高置信度但预测错误的样本（最危险的情况）"""
        high_conf_error_idx = [
            i for i, (p, l, c) in enumerate(zip(predictions, labels, confidences))
            if p != l and c > self.high_threshold
        ]

        high_conf_error_samples = []
        for idx in high_conf_error_idx[:max_examples]:
            high_conf_error_samples.append({
                "text": truncate_text(texts[idx], 100),
                "true_label": int(labels[idx]),
                "predicted": int(predictions[idx]),
                "confidence": float(confidences[idx]),
            })

        # 按置信度降序排序（置信度越高越危险）
        high_conf_error_idx.sort(key=lambda i: confidences[i], reverse=True)

        return {
            "count": len(high_conf_error_idx),
            "percentage": len(high_conf_error_idx) / len(texts) if texts else 0,
            "samples": high_conf_error_samples,
        }

    def _compute_confidence_distribution(
        self,
        labels: List[int],
        predictions: List[int],
        confidences: List[float],
        n_bins: int = 20,
    ) -> Dict[str, Any]:
        """计算置信度分布数据"""
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        labels = np.array(labels)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 正确和错误的分布
        correct = (predictions == labels)
        wrong = ~correct

        correct_hist, _ = np.histogram(confidences[correct], bins=bin_edges, density=True)
        wrong_hist, _ = np.histogram(confidences[wrong], bins=bin_edges, density=True)

        return {
            "bin_edges": bin_edges.tolist(),
            "bin_centers": bin_centers.tolist(),
            "correct_distribution": correct_hist.tolist(),
            "wrong_distribution": wrong_hist.tolist(),
        }


def plot_confidence_distribution(
    results: Dict[str, Any],
    output_path: Path,
    dpi: int = 150,
):
    """
    绘制置信度分布图

    Args:
        results: 分析结果
        output_path: 输出路径
        dpi: 图像 DPI
    """
    dist = results["confidence_distribution"]
    stats = results["basic_stats"]

    bin_centers = np.array(dist["bin_centers"])
    correct_dist = np.array(dist["correct_distribution"])
    wrong_dist = np.array(dist["wrong_distribution"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: 重叠直方图
    ax1 = axes[0]
    width = bin_centers[1] - bin_centers[0]
    ax1.bar(
        bin_centers, correct_dist, width,
        alpha=0.7, label='正确预测', color='green', edgecolor='black'
    )
    ax1.bar(
        bin_centers, wrong_dist, width,
        alpha=0.7, label='错误预测', color='red', edgecolor='black'
    )
    ax1.set_xlabel('置信度', fontsize=12)
    ax1.set_ylabel('密度', fontsize=12)
    ax1.set_title('置信度分布（正确 vs 错误）', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 子图2: 箱线图
    ax2 = axes[1]
    correct_confs = np.random.beta(10, 3, 1000) * 0.4 + 0.5  # 模拟数据（仅示例）
    wrong_confs = np.random.beta(3, 10, 1000) * 0.4 + 0.1  # 模拟数据（仅示例）

    box_data = []
    labels_box = []
    if stats["correct"] > 0:
        labels_box.append("正确预测")
        labels_box.append("错误预测")

    bp = ax2.boxplot(
        [[stats["mean_confidence_correct"]] if stats["correct"] > 0 else [],
         [stats["mean_confidence_wrong"]] if stats["wrong"] > 0 else []],
        labels=labels_box,
        patch_artist=True,
    )

    colors = ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_ylabel('置信度', fontsize=12)
    ax2.set_title('置信度统计', fontsize=14)
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.1)

    # 添加均值标记
    y_pos = 1.05
    ax2.text(1, y_pos, f'均值: {stats["mean_confidence_correct"]:.3f}',
             ha='center', fontsize=10, color='darkgreen')
    ax2.text(2, y_pos, f'均值: {stats["mean_confidence_wrong"]:.3f}',
             ha='center', fontsize=10, color='darkred')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"置信度分布图已保存: {output_path}")


def plot_reliability_diagram(
    results: Dict[str, Any],
    output_path: Path,
    dpi: int = 150,
):
    """
    绘制可靠性图

    Args:
        results: 分析结果
        output_path: 输出路径
        dpi: 图像 DPI
    """
    reliability = results["reliability_diagram"]
    ece = results["ece"]

    bin_confidences = np.array(reliability["bin_confidences"])
    bin_accuracies = np.array(reliability["bin_accuracies"])
    bin_counts = np.array(reliability["bin_counts"])
    bin_boundaries = np.array(reliability["bin_boundaries"])

    # 只绘制有样本的箱
    valid_bins = bin_counts > 0
    bin_confidences = bin_confidences[valid_bins]
    bin_accuracies = bin[valid_bins]
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_centers = bin_centers[valid_bins]

    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制完美校准线
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='完美校准')

    # 绘制各箱的点
    scatter = ax.scatter(
        bin_confidences, bin_accuracies,
        s=bin_counts / bin_counts.sum() * 1000 if bin_counts.sum() > 0 else 100,
        c=bin_centers, cmap='viridis',
        alpha=0.6, edgecolors='black',
    )

    # 绘制分段曲线
    sorted_idx = np.argsort(bin_confidences)
    ax.plot(
        bin_confidences[sorted_idx],
        bin_accuracies[sorted_idx],
        'o-', linewidth=2, markersize=8,
        label='模型校准曲线'
    )

    ax.set_xlabel('平均置信度', fontsize=12)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title(f'可靠性图 (ECE = {ece:.4f})', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('置信度区间中心', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"可靠性图已保存: {output_path}")


def print_analysis_report(results: Dict[str, Any]):
    """打印分析报告"""
    print("\n" + "=" * 60)
    print("置信度分析报告")
    print("=" * 60)

    # 基本统计
    stats = results["basic_stats"]
    print(f"\n【基本统计】")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  正确预测: {stats['correct']} ({stats['accuracy']:.2%})")
    print(f"  错误预测: {stats['wrong']} ({1 - stats['accuracy']:.2%})")
    print(f"\n  置信度统计:")
    print(f"    均值: {stats['mean_confidence']:.4f}")
    print(f"    标准差: {stats['std_confidence']:.4f}")
    print(f"    中位数: {stats['median_confidence']:.4f}")
    print(f"    最小值: {stats['min_confidence']:.4f}")
    print(f"    最大值: {stats['max_confidence']:.4f}")
    print(f"\n  分类别置信度:")
    print(f"    正确预测均值: {stats['mean_confidence_correct']:.4f}")
    print(f"    错误预测均值: {stats['mean_confidence_wrong']:.4f}")

    # ECE
    print(f"\n【模型校准】")
    print(f"  ECE (期望校准误差): {results['ece']:.4f}")
    print(f"    - ECE 越小表示模型校准越好")
    print(f"    - 完美校准时 ECE = 0")

    # 低置信度分析
    low = results["low_confidence"]
    print(f"\n【低置信度样本 (< {DEFAULT_CONFIG['low_confidence_threshold']})】")
    print(f"  数量: {low['count']} ({low['percentage']:.2%})")
    print(f"  其中正确: {low['accuracy']:.2%}")
    if low['samples']:
        print(f"\n  示例:")
        for i, sample in enumerate(low['samples'][:5], 1):
            print(f"    {i}. \"{sample['text']}\"")
            print(f"       真实: {sample['true_label']}, 预测: {sample['predicted']}, "
                  f"置信度: {sample['confidence']:.3f}")

    # 高置信度错误
    high_err = results["high_confidence_errors"]
    print(f"\n【高置信度错误样本 (> {DEFAULT_CONFIG['high_confidence_threshold']})】")
    print(f"  数量: {high_err['count']} ({high_err['percentage']:.2%})")
    print(f"    - 这是模型过度自信但预测错误的危险情况")
    if high_err['samples']:
        print(f"\n  示例（按置信度降序）:")
        for i, sample in enumerate(high_err['samples'][:5], 1):
            print(f"    {i}. \"{sample['text']}\"")
            print(f"       真实: {sample['true_label']}, 预测: {sample['predicted']}, "
                  f"置信度: {sample['confidence']:.3f}")

    print("\n" + "=" * 60)


def main(args):
    """主函数"""
    device = get_device()
    print(f"使用设备: {device}")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        base_model_name=args.base_model,
        lora_path=args.model_path,
        device=device,
    )

    # 加载数据集
    print(f"\n加载数据集: {args.dataset}")
    dataset = load_dataset("lansinuote/ChnSentiCorp")

    eval_split = "test" if "test" in dataset else "validation"
    eval_data = dataset[eval_split]

    texts = eval_data["text"]
    labels = eval_data["label"]

    print(f"评估数据量: {len(texts)}")

    # 创建分析器
    analyzer = ConfidenceAnalyzer(
        model=model,
        tokenizer=tokenizer,
        low_confidence_threshold=args.low_confidence_threshold,
        high_confidence_threshold=args.high_confidence_threshold,
        n_bins=args.n_bins,
    )

    # 执行分析
    results = analyzer.analyze(
        texts=texts,
        labels=labels,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # 打印报告
    print_analysis_report(results)

    # 保存结果
    plots_dir = ensure_output_dir("plots")

    # 保存 JSON 结果
    save_json(results, "confidence_analysis.json")

    # 绘制图表
    if args.plot:
        plot_confidence_distribution(
            results,
            plots_dir / "confidence_distribution.png",
        )
        plot_reliability_diagram(
            results,
            plots_dir / "reliability_diagram.png",
        )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="模型置信度分析",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="LoRA 适配器路径",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="基础模型名称",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ChnSentiCorp",
        help="评估数据集",
    )
    parser.add_argument(
        "--low_confidence_threshold",
        type=float,
        default=0.6,
        help="低置信度阈值",
    )
    parser.add_argument(
        "--high_confidence_threshold",
        type=float,
        default=0.9,
        help="高置信度阈值",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="ECE 分箱数量",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="批次大小",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="最大序列长度",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="是否生成图表",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
