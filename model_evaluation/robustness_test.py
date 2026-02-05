"""
鲁棒性测试模块
Robustness Test Module

测试模型在各种边缘情况下的表现
"""

import os

# 修复 torch 导入卡死问题 (Intel MKL 库冲突)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path

import torch

from .utils import (
    load_model_and_tokenizer,
    get_predictions_with_confidence,
    save_json,
    ensure_output_dir,
    get_device,
    truncate_text,
)


# 测试类别权重
CATEGORY_WEIGHTS = {
    "short_positive": 1.0,
    "short_negative": 1.0,
    "irony": 1.5,       # 反语更难，权重更高
    "mixed_emotion": 0, # 混合情感不评分，仅分析
    "negative_negation": 1.2,  # 否定词堆较难
    "slang_positive": 1.0,
    "slang_negative": 1.0,
    "emoji_positive": 0.8,
    "emoji_negative": 0.8,
    "typo": 1.2,       # 错别字测试
    "long_text": 1.0,
    "edge_case": 0,     # 边缘情况不评分
}

# 类别中文名称
CATEGORY_NAMES = {
    "short_positive": "短评正面",
    "short_negative": "短评负面",
    "irony": "反语",
    "mixed_emotion": "混合情感",
    "negative_negation": "否定词堆叠",
    "slang_positive": "网络用语正面",
    "slang_negative": "网络用语负面",
    "emoji_positive": "表情符号正面",
    "emoji_negative": "表情符号负面",
    "typo": "错别字",
    "long_text": "长评",
    "edge_case": "边缘情况",
}


class RobustnessTester:
    """鲁棒性测试器"""

    def __init__(self, model, tokenizer):
        """
        初始化测试器

        Args:
            model: 模型
            tokenizer: 分词器
        """
        self.model = model
        self.tokenizer = tokenizer

    def load_test_samples(self, samples_path: str) -> Dict[str, List[Dict]]:
        """
        加载测试样本

        Args:
            samples_path: 测试样本 JSON 文件路径

        Returns:
            测试样本字典
        """
        with open(samples_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_test(
        self,
        samples: Dict[str, List[Dict]],
        batch_size: int = 16,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """
        运行所有鲁棒性测试

        Args:
            samples: 测试样本
            batch_size: 批次大小
            max_length: 最大序列长度

        Returns:
            测试结果字典
        """
        results = {
            "categories": {},
            "overall": {
                "total_samples": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "weighted_score": 0.0,
            }
        }

        for category, test_cases in samples.items():
            if not test_cases:
                continue

            print(f"\n测试类别: {CATEGORY_NAMES.get(category, category)} ({len(test_cases)} 个样本)")

            category_result = self._test_category(
                category, test_cases, batch_size, max_length
            )
            results["categories"][category] = category_result

            # 更新总体统计（只统计有标签的样本）
            valid_samples = [s for s in test_cases if s.get("label") is not None]
            if valid_samples:
                results["overall"]["total_samples"] += len(valid_samples)

                correct_count = category_result.get("correct", 0)
                wrong_count = category_result.get("wrong", 0)

                results["overall"]["correct"] += correct_count
                results["overall"]["wrong"] += wrong_count

                # 加权分数
                weight = CATEGORY_WEIGHTS.get(category, 1.0)
                if category_result.get("accuracy") is not None:
                    results["overall"]["weighted_score"] += (
                        category_result["accuracy"] * weight
                    )

        # 计算总体准确率
        if results["overall"]["total_samples"] > 0:
            results["overall"]["accuracy"] = (
                results["overall"]["correct"] / results["overall"]["total_samples"]
            )

        # 计算加权平均分数
        total_weight = sum(w for c, w in CATEGORY_WEIGHTS.items()
                         if c in results["categories"] and w > 0)
        if total_weight > 0:
            results["overall"]["weighted_score"] /= total_weight

        # 计算鲁棒性评分（综合指标）
        results["overall"]["robustness_score"] = self._compute_robustness_score(results)

        return results

    def _test_category(
        self,
        category: str,
        test_cases: List[Dict],
        batch_size: int,
        max_length: int,
    ) -> Dict[str, Any]:
        """测试单个类别"""
        # 分离有标签和无标签的样本
        labeled = [s for s in test_cases if s.get("label") is not None]
        unlabeled = [s for s in test_cases if s.get("label") is None]

        result = {
            "name": CATEGORY_NAMES.get(category, category),
            "total": len(test_cases),
            "labeled_samples": len(labeled),
            "samples": [],
        }

        # 测试有标签的样本
        if labeled:
            texts = [s["text"] for s in labeled]
            labels = [s["label"] for s in labeled]

            pred_result = get_predictions_with_confidence(
                self.model, self.tokenizer, texts,
                batch_size=batch_size,
                max_length=max_length,
            )

            predictions = pred_result["predictions"]
            confidences = pred_result["confidences"]

            correct = sum(1 for p, l in zip(predictions, labels) if p == l)
            wrong = len(labeled) - correct

            result["correct"] = correct
            result["wrong"] = wrong
            result["accuracy"] = correct / len(labeled)

            # 记录每个样本的预测结果
            for i, (sample, pred, conf) in enumerate(zip(labeled, predictions, confidences)):
                result["samples"].append({
                    "text": truncate_text(sample["text"], 100),
                    "true_label": sample["label"],
                    "predicted": int(pred),
                    "confidence": float(conf),
                    "correct": (pred == sample["label"]),
                    "description": sample.get("description", ""),
                })
        else:
            result["accuracy"] = None

        # 测试无标签的样本（仅记录预测结果）
        if unlabeled:
            texts = [s["text"] for s in unlabeled]

            pred_result = get_predictions_with_confidence(
                self.model, self.tokenizer, texts,
                batch_size=batch_size,
                max_length=max_length,
            )

            predictions = pred_result["predictions"]
            confidences = pred_result["confidences"]

            for i, (sample, pred, conf) in enumerate(zip(unlabeled, predictions, confidences)):
                result["samples"].append({
                    "text": truncate_text(sample["text"], 100),
                    "true_label": None,
                    "predicted": int(pred),
                    "confidence": float(conf),
                    "correct": None,
                    "description": sample.get("description", ""),
                })

        return result

    def _compute_robustness_score(self, results: Dict[str, Any]) -> float:
        """
        计算综合鲁棒性评分

        考虑因素：
        1. 各类别准确率的加权平均
        2. 对困难类别（反语、否定词堆叠、错别字）给予更高关注
        3. 综合评估模型的鲁棒性

        Args:
            results: 测试结果

        Returns:
            鲁棒性评分 (0-100)
        """
        # 使用加权平均分数作为基础
        base_score = results["overall"].get("weighted_score", 0)

        # 关注困难类别的表现
        difficult_categories = ["irony", "negative_negation", "typo"]
        difficult_scores = []

        for cat in difficult_categories:
            if cat in results["categories"]:
                cat_result = results["categories"][cat]
                if cat_result.get("accuracy") is not None:
                    difficult_scores.append(cat_result["accuracy"])

        # 如果有困难类别的测试，则给予额外关注
        if difficult_scores:
            # 困难类别的平均表现
            difficult_avg = np.mean(difficult_scores)

            # 综合评分：70% 基础分数 + 30% 困难类别分数
            final_score = base_score * 0.7 + difficult_avg * 0.3
        else:
            final_score = base_score

        return final_score * 100


def print_robustness_report(results: Dict[str, Any]):
    """打印鲁棒性测试报告"""
    print("\n" + "=" * 60)
    print("鲁棒性测试报告")
    print("=" * 60)

    # 打印各类别结果
    print(f"\n【各测试类别结果】")
    print(f"{'类别':<20} {'样本数':<10} {'正确':<10} {'错误':<10} {'准确率':<10}")
    print("-" * 60)

    for category, cat_result in results["categories"].items():
        name = cat_result.get("name", category)
        total = cat_result["total"]
        correct = cat_result.get("correct", "-")
        wrong = cat_result.get("wrong", "-")

        if cat_result.get("accuracy") is not None:
            acc = f"{cat_result['accuracy']:.2%}"
        else:
            acc = "N/A"

        print(f"{name:<20} {total:<10} {str(correct):<10} {str(wrong):<10} {acc:<10}")

    # 总体统计
    overall = results["overall"]
    print(f"\n【总体统计】")
    print(f"  总样本数: {overall['total_samples']}")
    print(f"  正确预测: {overall['correct']}")
    print(f"  错误预测: {overall['wrong']}")
    print(f"  整体准确率: {overall['accuracy']:.2%}")
    print(f"  加权平均分: {overall['weighted_score']:.2%}")
    print(f"  鲁棒性评分: {overall['robustness_score']:.1f}/100")

    # 评分说明
    print(f"\n【鲁棒性评分说明】")
    score = overall["robustness_score"]
    if score >= 90:
        level = "优秀"
        desc = "模型在各类场景下表现稳定，鲁棒性极佳"
    elif score >= 80:
        level = "良好"
        desc = "模型在大多数场景下表现良好，有少量弱点"
    elif score >= 70:
        level = "中等"
        desc = "模型在常规场景下尚可，在困难场景下表现一般"
    elif score >= 60:
        level = "及格"
        desc = "模型勉强可用，在多种场景下存在明显不足"
    else:
        level = "不合格"
        desc = "模型鲁棒性较差，需要改进"

    print(f"  评级: {level}")
    print(f"  说明: {desc}")

    # 困难类别分析
    print(f"\n【困难类别分析】")
    difficult = ["irony", "negative_negation", "typo"]
    for cat in difficult:
        if cat in results["categories"]:
            cat_result = results["categories"][cat]
            name = cat_result["name"]
            acc = cat_result.get("accuracy")
            if acc is not None:
                print(f"  {name}: {acc:.2%}")
                if acc < 0.5:
                    print(f"    该类别准确率较低，建议针对性优化")

    # 失败样本分析
    print(f"\n【失败样本分析】")
    failure_count = 0
    for cat, cat_result in results["categories"].items():
        for sample in cat_result["samples"]:
            if sample.get("correct") is False:
                failure_count += 1

    print(f"  总失败样本数: {failure_count}")

    if failure_count > 0:
        print(f"\n  典型失败示例:")
        shown = 0
        for cat, cat_result in results["categories"].items():
            for sample in cat_result["samples"]:
                if sample.get("correct") is False and shown < 10:
                    print(f"    [{cat_result.get('name', cat)}] \"{sample['text']}\"")
                    print(f"      真实: {sample['true_label']}, "
                          f"预测: {sample['predicted']}, "
                          f"置信度: {sample['confidence']:.3f}")
                    shown += 1

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

    # 创建测试器
    tester = RobustnessTester(model, tokenizer)

    # 加载测试样本
    print(f"\n加载测试样本: {args.samples_path}")
    samples = tester.load_test_samples(args.samples_path)

    total_samples = sum(len(v) for v in samples.values())
    print(f"测试样本总数: {total_samples}")
    print(f"测试类别数: {len(samples)}")

    # 运行测试
    results = tester.run_test(
        samples=samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # 打印报告
    print_robustness_report(results)

    # 保存结果
    save_json(results, "robustness_result.json")

    return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="模型鲁棒性测试",
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
        "--samples_path",
        type=str,
        default=None,
        help="测试样本 JSON 文件路径（默认使用内置）",
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 默认使用内置测试样本
    if args.samples_path is None:
        args.samples_path = str(
            Path(__file__).parent / "test_samples.json"
        )

    main(args)
