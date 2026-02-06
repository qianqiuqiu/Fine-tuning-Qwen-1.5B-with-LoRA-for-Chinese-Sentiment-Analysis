"""
一键运行所有评估
Run Full Evaluation

执行完整的模型评估流程：
1. 基础评估
2. 置信度分析
3. 鲁棒性测试
4. 基准对比
5. 生成可视化报告
"""

import os

# 修复 torch 导入卡死问题 (Intel MKL 库冲突)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import json
from typing import Dict, Any, Optional
from pathlib import Path

# 处理直接运行时的相对导入问题
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "evaluation"

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# 条件导入：支持直接运行和作为模块导入
try:
    from .utils import (
        load_model_and_tokenizer,
        get_predictions_with_confidence,
        save_json,
        ensure_output_dir,
        get_device,
    )
except ImportError:
    from evaluation.utils import (
        load_model_and_tokenizer,
        get_predictions_with_confidence,
        save_json,
        ensure_output_dir,
        get_device,
    )

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


class FullEvaluator:
    """完整评估器"""

    def __init__(
        self,
        model_path: str,
        base_model: str = "Qwen/Qwen2.5-1.5B",
        dataset: str = "ChnSentiCorp",
        batch_size: int = 16,
        max_length: int = 256,
    ):
        """
        初始化评估器

        Args:
            model_path: LoRA 适配器路径
            base_model: 基础模型名称
            dataset: 数据集名称
            batch_size: 批次大小
            max_length: 最大序列长度
        """
        self.model_path = model_path
        self.base_model = base_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = get_device()

        self.model = None
        self.tokenizer = None

        self.results = {
            "basic_metrics": {},
            "confidence_analysis": None,
            "robustness_result": None,
            "benchmark_result": None,
        }

    def load_model(self):
        """加载模型"""
        print(f"\n{'=' * 60}")
        print("加载模型")
        print('=' * 60)

        print(f"使用设备: {self.device}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            base_model_name=self.base_model,
            lora_path=self.model_path,
            device=self.device,
        )

    def load_dataset(self) -> tuple:
        """加载数据集"""
        print(f"\n{'=' * 60}")
        print("加载数据集")
        print('=' * 60)

        print(f"数据集: {self.dataset}")
        dataset = load_dataset("lansinuote/ChnSentiCorp")

        eval_split = "test" if "test" in dataset else "validation"
        eval_data = dataset[eval_split]

        texts = eval_data["text"]
        labels = eval_data["label"]

        print(f"评估数据集: {eval_split}")
        print(f"样本数量: {len(texts)}")

        return texts, labels

    def run_basic_evaluation(self, texts, labels):
        """运行基础评估"""
        print(f"\n{'=' * 60}")
        print("1. 基础评估")
        print('=' * 60)

        pred_result = get_predictions_with_confidence(
            self.model, self.tokenizer, texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_all_probs=True,
        )

        predictions = pred_result["predictions"]
        confidences = pred_result["confidences"]

        # 计算指标
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )

        try:
            auc = roc_auc_score(labels, confidences)
        except:
            auc = 0.0

        # 混淆矩阵
        cm = confusion_matrix(labels, predictions).tolist()

        # 分类报告
        report = classification_report(
            labels, predictions,
            target_names=['负面', '正面'],
            output_dict=True,
        )

        self.results["basic_metrics"] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(auc),
            "confusion_matrix": cm,
            "classification_report": report,
        }

        # 打印结果
        print(f"\n基础评估结果：")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1 分数: {f1:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")

        print(f"\n混淆矩阵：")
        print(f"  {'':>10} 预测负面  预测正面")
        print(f"  实际负面 {cm[0][0]:>8} {cm[0][1]:>8}")
        print(f"  实际正面 {cm[1][0]:>8} {cm[1][1]:>8}")

        # 保存结果
        save_json(self.results["basic_metrics"], "metrics.json")

        return predictions, confidences

    def run_confidence_analysis(self, texts, labels, predictions, confidences):
        """运行置信度分析"""
        print(f"\n{'=' * 60}")
        print("2. 置信度分析")
        print('=' * 60)

        low_threshold = 0.6
        high_threshold = 0.9
        n_bins = 10

        # 基本统计
        import numpy as np

        correct = (np.array(predictions) == np.array(labels))
        wrong = ~correct

        basic_stats = {
            "total_samples": len(labels),
            "correct": int(correct.sum()),
            "wrong": int(wrong.sum()),
            "accuracy": float(correct.mean()),
            "mean_confidence": float(np.mean(confidences)),
            "mean_confidence_correct": float(np.mean(np.array(confidences)[correct])) if correct.sum() > 0 else 0,
            "mean_confidence_wrong": float(np.mean(np.array(confidences)[wrong])) if wrong.sum() > 0 else 0,
            "std_confidence": float(np.std(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
        }

        # ECE 计算
        from .utils import compute_ece
        ece, reliability_data = compute_ece(predictions, labels, confidences, n_bins)

        # 低置信度样本
        low_conf_idx = [i for i, c in enumerate(confidences) if c < low_threshold]
        low_conf_samples = []
        for i in low_conf_idx[:10]:
            low_conf_samples.append({
                "text": texts[i][:100] if len(texts[i]) > 100 else texts[i],
                "true_label": int(labels[i]),
                "predicted": int(predictions[i]),
                "confidence": float(confidences[i]),
            })

        # 高置信度错误
        high_conf_error_idx = [
            i for i, (p, l, c) in enumerate(zip(predictions, labels, confidences))
            if p != l and c > high_threshold
        ]
        high_conf_error_samples = []
        for i in high_conf_error_idx[:10]:
            high_conf_error_samples.append({
                "text": texts[i][:100] if len(texts[i]) > 100 else texts[i],
                "true_label": int(labels[i]),
                "predicted": int(predictions[i]),
                "confidence": float(confidences[i]),
            })

        # 置信度分布
        bin_edges = np.linspace(0, 1, 20 + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        correct_hist, _ = np.histogram(np.array(confidences)[correct], bins=bin_edges, density=True)
        wrong_hist, _ = np.histogram(np.array(confidences)[wrong], bins=bin_edges, density=True)

        confidence_analysis = {
            "basic_stats": basic_stats,
            "ece": float(ece),
            "reliability_diagram": reliability_data,
            "low_confidence": {
                "count": len(low_conf_idx),
                "samples": low_conf_samples,
            },
            "high_confidence_errors": {
                "count": len(high_conf_error_idx),
                "samples": high_conf_error_samples,
            },
            "confidence_distribution": {
                "bin_edges": bin_edges.tolist(),
                "bin_centers": bin_centers.tolist(),
                "correct_distribution": correct_hist.tolist(),
                "wrong_distribution": wrong_hist.tolist(),
            },
        }

        self.results["confidence_analysis"] = confidence_analysis

        print(f"\n置信度分析结果：")
        print(f"  平均置信度: {basic_stats['mean_confidence']:.4f}")
        print(f"  正确预测平均置信度: {basic_stats['mean_confidence_correct']:.4f}")
        print(f"  错误预测平均置信度: {basic_stats['mean_confidence_wrong']:.4f}")
        print(f"  ECE (期望校准误差): {ece:.4f}")
        print(f"  低置信度样本数 (<{low_threshold}): {len(low_conf_idx)}")
        print(f"  高置信度错误样本数 (>{high_threshold}): {len(high_conf_error_idx)}")

        # 保存结果
        save_json(confidence_analysis, "confidence_analysis.json")

    def run_robustness_test(self):
        """运行鲁棒性测试"""
        print(f"\n{'=' * 60}")
        print("3. 鲁棒性测试")
        print('=' * 60)

        # 导入 robustness_test 模块
        try:
            try:
                from . import robustness_test
            except ImportError:
                from evaluation import robustness_test

            tester = robustness_test.RobustnessTester(self.model, self.tokenizer)
            samples = tester.load_test_samples(str(
                Path(__file__).parent / "test_samples.json"
            ))

            result = tester.run_test(samples, self.batch_size, self.max_length)
            self.results["robustness_result"] = result

            # 打印摘要
            overall = result["overall"]
            print(f"\n鲁棒性测试摘要：")
            print(f"  鲁棒性评分: {overall['robustness_score']:.1f}/100")
            print(f"  整体准确率: {overall['accuracy']:.2%}")

            # 保存结果
            save_json(result, "robustness_result.json")

        except Exception as e:
            print(f"鲁棒性测试出错: {e}")

    def run_benchmark(self, texts, labels, run_bert: bool = False):
        """运行基准对比 - 公平对比LoRA微调效果"""
        print(f"\n{'=' * 60}")
        print("4. 基准对比 - 量化LoRA微调效果")
        print('=' * 60)
        
        print(f"\n对比说明：")
        print(f"  模型A: 冻结的Qwen基础模型 + LoRA训练后的分类头")
        print(f"         (测试分类头本身的能力)")
        print(f"  模型B: LoRA微调的Qwen + LoRA训练后的分类头")
        print(f"         (测试完整微调后的能力)")
        print(f"  差异 = 模型B - 模型A = LoRA对基础模型的提升")

        try:
            try:
                from . import benchmark
            except ImportError:
                from evaluation import benchmark

            # 限制样本数以加快测试速度
            test_texts = texts[:500]
            test_labels = labels[:500]
            
            print(f"\n使用 {len(test_texts)} 个样本进行基准对比")

            evaluator = benchmark.BenchmarkEvaluator(
                texts=test_texts,
                labels=test_labels,
                batch_size=self.batch_size,
                max_length=self.max_length,
            )

            # 初始化结果
            self.results["benchmark_result"] = {}
            
            # 测试1: 冻结的基础模型 + LoRA分类头
            print(f"\n{'=' * 60}")
            print(f"测试1: 冻结的Qwen基础模型 + LoRA训练后的分类头")
            print(f"{'=' * 60}")
            try:
                frozen_result = evaluator.run_benchmark("qwen_base_frozen", self.model_path)
                self.results["benchmark_result"]["qwen_base_frozen"] = frozen_result
            except Exception as e:
                import traceback
                print(f"冻结模型评估失败: {e}")
                traceback.print_exc()
            
            # 测试2: LoRA 微调模型
            print(f"\n{'=' * 60}")
            print(f"测试2: LoRA微调的完整Qwen模型")
            print(f"{'=' * 60}")
            try:
                lora_result = evaluator.run_benchmark("qwen_lora", self.model_path)
                self.results["benchmark_result"]["qwen_lora"] = lora_result
            except Exception as e:
                import traceback
                print(f"LoRA模型评估失败: {e}")
                traceback.print_exc()
            
            # 如果启用 BERT 对比
            if run_bert:
                print(f"\n{'=' * 60}")
                print(f"测试3: BERT基线模型（可选）")
                print(f"{'=' * 60}")
                try:
                    bert_result = evaluator.run_benchmark("bert_chinese", None)
                    self.results["benchmark_result"]["bert_chinese"] = bert_result
                except Exception as e:
                    print(f"BERT 模型评估失败: {e}")

            # 打印详细对比摘要
            print(f"\n{'=' * 80}")
            print("基准对比摘要 - LoRA微调效果量化分析")
            print('=' * 80)
            
            print(f"\n{'模型':<45} {'准确率':<10} {'F1':<10} {'精确率':<10} {'召回率':<10}")
            print("-" * 85)
            
            # 按顺序显示
            model_order = ["qwen_base_frozen", "qwen_lora", "bert_chinese"]
            model_results = {}
            
            for key in model_order:
                if key in self.results["benchmark_result"]:
                    result = self.results["benchmark_result"][key]
                    if "metrics" in result:
                        metrics = result["metrics"]
                        model_results[key] = metrics
                        print(f"{result['name']:<45} "
                              f"{metrics['accuracy']:.4f}    "
                              f"{metrics['f1']:.4f}    "
                              f"{metrics['precision']:.4f}    "
                              f"{metrics['recall']:.4f}")
            
            # 计算LoRA微调的实际效果（关键对比）
            if "qwen_base_frozen" in model_results and "qwen_lora" in model_results:
                frozen_metrics = model_results["qwen_base_frozen"]
                lora_metrics = model_results["qwen_lora"]
                
                print(f"\n{'=' * 80}")
                print("LoRA微调效果量化分析")
                print('=' * 80)
                
                print(f"\n指标对比 (LoRA微调模型 vs 冻结基础模型+分类头):")
                print(f"\n{'指标':<15} {'冻结+分类头':<15} {'LoRA微调':<15} {'绝对提升':<15} {'相对提升':<15}")
                print("-" * 75)
                
                metrics_to_compare = [
                    ('准确率', 'accuracy'),
                    ('F1分数', 'f1'),
                    ('精确率', 'precision'),
                    ('召回率', 'recall'),
                    ('AUC-ROC', 'auc_roc'),
                ]
                
                for metric_name, metric_key in metrics_to_compare:
                    frozen_val = frozen_metrics.get(metric_key, 0)
                    lora_val = lora_metrics.get(metric_key, 0)
                    abs_improvement = lora_val - frozen_val
                    rel_improvement = (abs_improvement / frozen_val * 100) if frozen_val > 0 else 0
                    
                    print(f"{metric_name:<15} "
                          f"{frozen_val:<15.4f} "
                          f"{lora_val:<15.4f} "
                          f"{abs_improvement:+<15.4f} "
                          f"{rel_improvement:+<14.2f}%")
                
                print(f"\n结论：")
                acc_improvement = (lora_metrics['accuracy'] - frozen_metrics['accuracy']) * 100
                f1_improvement = (lora_metrics['f1'] - frozen_metrics['f1']) * 100
                
                if acc_improvement > 0:
                    print(f"  ✓ LoRA微调使模型准确率提升了 {acc_improvement:.2f} 个百分点")
                    print(f"  ✓ LoRA微调使模型F1分数提升了 {f1_improvement:.2f} 个百分点")
                    print(f"  ✓ 这表明LoRA成功地提升了基础模型的特征提取能力")
                else:
                    print(f"  ⚠ LoRA微调效果不明显，可能需要调整训练参数")
                
                print(f"\n速度对比:")
                frozen_qps = frozen_metrics.get('qps', 0)
                lora_qps = lora_metrics.get('qps', 0)
                print(f"  冻结模型 QPS: {frozen_qps:.2f}")
                print(f"  LoRA模型 QPS: {lora_qps:.2f}")
                if frozen_qps > 0:
                    speed_diff = ((lora_qps - frozen_qps) / frozen_qps) * 100
                    print(f"  速度差异: {speed_diff:+.2f}% (负值表示LoRA模型稍慢，这是正常的)")

            save_json(self.results["benchmark_result"], "baseline_comparison.json")

        except Exception as e:
            import traceback
            print(f"基准对比出错: {e}")
            traceback.print_exc()
            
            # 降级方案：只记录当前模型信息
            print(f"\n使用降级方案：只记录 LoRA 模型信息")
            self.results["benchmark_result"] = {
                "qwen_lora": {
                    "name": "Qwen-1.5B + LoRA微调",
                    "metrics": self.results["basic_metrics"],
                }
            }

    def generate_report(self, texts, labels, predictions, confidences):
        """生成可视化报告"""
        print(f"\n{'=' * 60}")
        print("5. 生成可视化报告")
        print('=' * 60)

        try:
            try:
                from . import report_generator
            except ImportError:
                from evaluation import report_generator

            generator = report_generator.ReportGenerator()

            # 加载数据
            generator.load_basic_metrics(self.results["basic_metrics"])
            generator.load_predictions(predictions, labels, confidences)

            if self.results["confidence_analysis"]:
                generator.load_confidence_analysis(self.results["confidence_analysis"])

            if self.results["robustness_result"]:
                generator.load_robustness_result(self.results["robustness_result"])

            if self.results["benchmark_result"]:
                generator.load_benchmark_result(self.results["benchmark_result"])

            # 生成报告
            report_path = generator.generate_html_report()
            print(f"\n可视化报告已生成: {report_path}")

        except Exception as e:
            print(f"生成报告出错: {e}")

    def print_summary(self):
        """打印评估摘要"""
        print(f"\n{'=' * 60}")
        print("评估摘要")
        print('=' * 60)

        if self.results["basic_metrics"]:
            metrics = self.results["basic_metrics"]
            print(f"\n基础指标：")
            print(f"  准确率: {metrics['accuracy']:.4f}")
            print(f"  F1 分数: {metrics['f1']:.4f}")
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")

        if self.results["confidence_analysis"]:
            ece = self.results["confidence_analysis"]["ece"]
            print(f"\n校准指标：")
            print(f"  ECE: {ece:.4f}")

        if self.results["robustness_result"]:
            score = self.results["robustness_result"]["overall"]["robustness_score"]
            print(f"\n鲁棒性：")
            print(f"  评分: {score:.1f}/100")

        print(f"\n输出文件位置:")
        output_dir = ensure_output_dir()
        print(f"  {output_dir}")

        print("\n" + "=" * 60)

    def run_full_evaluation(self, run_bert: bool = False):
        """运行完整评估流程"""
        # 加载模型
        self.load_model()

        # 加载数据集
        texts, labels = self.load_dataset()

        # 1. 基础评估
        predictions, confidences = self.run_basic_evaluation(texts, labels)

        # 2.2. 置信度分析
        self.run_confidence_analysis(texts, labels, predictions, confidences)

        # 3. 鲁棒性测试
        self.run_robustness_test()

        # 4. 基准对比
        self.run_benchmark(texts, labels, run_bert)

        # 5. 生成报告
        self.generate_report(texts, labels, predictions, confidences)

        # 打印摘要
        self.print_summary()

        return self.results


def main(args):
    """主函数"""
    evaluator = FullEvaluator(
        model_path=args.model_path,
        base_model=args.base_model,
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    results = evaluator.run_full_evaluation(run_bert=args.run_bert)

    return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="运行完整的模型评估流程",
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
        "--run_bert",
        action="store_true",
        help="是否运行完整的基准对比（包括 BERT）",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
