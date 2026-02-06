"""
基准对比模块
Benchmark Module

对比 LoRA 微调模型与基线模型的性能
"""

import os

# 修复 torch 导入卡死问题 (Intel MKL 库冲突)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import sys
import time
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

# 处理直接运行时的相对导入问题
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "evaluation"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel

try:
    from .utils import (
        get_predictions_with_confidence,
        save_json,
        ensure_output_dir,
        get_device,
        get_torch_dtype,
        load_base_model_with_lora_classifier,
    )
except ImportError:
    from evaluation.utils import (
        get_predictions_with_confidence,
        save_json,
        ensure_output_dir,
        get_device,
        get_torch_dtype,
        load_base_model_with_lora_classifier,
    )

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


# 基线模型配置
BASELINE_MODELS = {
    "qwen_base_frozen": {
        "name": "Qwen-1.5B 冻结 + LoRA分类头",
        "model_name": "Qwen/Qwen2.5-1.5B",
        "type": "qwen_frozen",
        "description": "冻结的基础Qwen模型 + LoRA训练后的分类头（测试分类头能力）",
        "use_lora": False,
        "use_lora_classifier": True,
    },
    "qwen_lora": {
        "name": "Qwen-1.5B + LoRA微调",
        "model_name": "Qwen/Qwen2.5-1.5B",
        "type": "qwen",
        "description": "LoRA微调的完整模型（基础模型+分类头都经过微调）",
        "use_lora": True,
        "use_lora_classifier": False,
    },
    "bert_chinese": {
        "name": "BERT-base-chinese",
        "model_name": "bert-base-chinese",
        "type": "bert",
        "description": "经典的中文 BERT 基线模型",
        "use_lora": False,
    },
    "roberta_wwm": {
        "name": "RoBERTa-wwm-ext",
        "model_name": "hfl/chinese-roberta-wwm-ext",
        "type": "bert",
        "description": "中文 RoBERTa 全词掩码模型",
        "use_lora": False,
    },
}


class BenchmarkEvaluator:
    """基准对比评估器"""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 16,
        max_length: int = 256,
    ):
        """
        初始化评估器

        Args:
            texts: 测试文本列表
            labels: 测试标签列表
            batch_size: 批次大小
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = get_device()
        self.results = {}

    def load_qwen_model(
        self,
        model_name: str,
        lora_path: Optional[str] = None,
    ):
        """
        加载 Qwen 模型

        Args:
            model_name: 模型名称
            lora_path: LoRA 适配器路径（可选）

        Returns:
            (model, tokenizer) 元组
        """
        # 获取合适的数据类型
        torch_dtype = get_torch_dtype()
        print(f"使用数据类型: {torch_dtype}")
        
        print(f"\n加载 Qwen 分词器: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            lora_path if lora_path else model_name,
            trust_remote_code=True,
        )

        # 确保padding token正确设置
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        print(f"加载 Qwen 模型: {model_name}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

        if lora_path:
            print(f"加载 LoRA 适配器: {lora_path}")
            model = PeftModel.from_pretrained(base_model, lora_path)
            model = model.merge_and_unload()
        else:
            model = base_model

        # 确保模型配置中有 pad_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id

        model.eval()
        return model, tokenizer

    def load_bert_model(self, model_name: str):
        """
        加载 BERT 类模型

        Args:
            model_name: 模型名称

        Returns:
            (model, tokenizer) 元组
        """
        print(f"\n加载 BERT 分词器: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"加载 BERT 模型: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )

        model.eval()
        return model, tokenizer

    def evaluate_qwen(
        self,
        model_name: str,
        lora_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        评估 Qwen 模型

        Args:
            model_name: 模型名称
            lora_path: LoRA 适配器路径（可选）

        Returns:
            评估结果
        """
        model, tokenizer = self.load_qwen_model(model_name, lora_path)

        # 性能测试
        inference_times = []
        all_predictions = []
        all_probabilities = []

        model.eval()

        with torch.no_grad():
            for i in range(0, len(self.texts), self.batch_size):
                batch_texts = self.texts[i:i + self.batch_size]

                start_time = time.time()

                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                outputs = model(**inputs)
                logits = outputs.logits

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                all_predictions.extend(preds.cpu().numpy().tolist())
                all_probabilities.extend(probs[:, 1].cpu().float().numpy().tolist())

        # 清理显存
        del model
        torch.cuda.empty_cache()

        return self._compute_metrics(all_predictions, all_probabilities, inference_times)

    def evaluate_bert(self, model_name: str) -> Dict[str, Any]:
        """
        评估 BERT 类模型

        Args:
            model_name: 模型名称

        Returns:
            评估结果
        """
        model, tokenizer = self.load_bert_model(model_name)

        inference_times = []
        all_predictions = []
        all_probabilities = []

        model.eval()

        with torch.no_grad():
            for i in range(0, len(self.texts), self.batch_size):
                batch_texts = self.texts[i:i + self.batch_size]

                start_time = time.time()

                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                outputs = model(**inputs)
                logits = outputs.logits

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                all_predictions.extend(preds.cpu().numpy().tolist())
                all_probabilities.extend(probs[:, 1].cpu().float().numpy().tolist())

        # 清理显存
        del model
        torch.cuda.empty_cache()

        return self._compute_metrics(all_predictions, all_probabilities, inference_times)

    def _compute_metrics(
        self,
        predictions: List[int],
        probabilities: List[float],
        inference_times: List[float],
    ) -> Dict[str, Any]:
        """计算指标"""
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)

        # 基础指标
        accuracy = accuracy_score(self.labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.labels, predictions, average='binary'
        )

        # 各类别召回率
        _, recalls, _, _ = precision_recall_fscore_support(
            self.labels, predictions, average=None
        )
        negative_recall = recalls[0]
        positive_recall = recalls[1]

        # AUC-ROC
        try:
            auc = roc_auc_score(self.labels, probabilities)
        except:
            auc = 0.0

        # 推理速度
        total_time = sum(inference_times)
        total_samples = len(self.texts)
        avg_time_per_sample = total_time / total_samples
        qps = total_samples / total_time  # Queries Per Second

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "negative_recall": float(negative_recall),
            "positive_recall": float(positive_recall),
            "auc_roc": float(auc),
            "avg_inference_time_ms": avg_time_per_sample * 1000,
            "qps": float(qps),
        }

    def evaluate_qwen_frozen_with_lora_classifier(
        self,
        model_name: str,
        lora_path: str,
    ) -> Dict[str, Any]:
        """
        评估冻结的Qwen模型 + LoRA训练后的分类头

        Args:
            model_name: 基础模型名称
            lora_path: LoRA 适配器路径（用于获取训练后的分类头）

        Returns:
            评估结果
        """
        model, tokenizer = load_base_model_with_lora_classifier(
            base_model_name=model_name,
            lora_path=lora_path,
            device=self.device,
        )

        # 性能测试
        inference_times = []
        all_predictions = []
        all_probabilities = []

        model.eval()

        with torch.no_grad():
            for i in range(0, len(self.texts), self.batch_size):
                batch_texts = self.texts[i:i + self.batch_size]

                start_time = time.time()

                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                outputs = model(**inputs)
                logits = outputs.logits

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                all_predictions.extend(preds.cpu().numpy().tolist())
                all_probabilities.extend(probs[:, 1].cpu().float().numpy().tolist())

        # 清理显存
        del model
        torch.cuda.empty_cache()

        return self._compute_metrics(all_predictions, all_probabilities, inference_times)

    def run_benchmark(
        self,
        model_key: str,
        lora_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行基准测试

        Args:
            model_key: 模型键（在 BASELINE_MODELS 中定义）
            lora_path: LoRA 适配器路径

        Returns:
            评估结果
        """
        config = BASELINE_MODELS[model_key]
        print(f"\n{'=' * 60}")
        print(f"评估模型: {config['name']}")
        print(f"描述: {config['description']}")
        print('=' * 60)

        try:
            if config["type"] == "qwen":
                metrics = self.evaluate_qwen(config["model_name"], lora_path)
            elif config["type"] == "qwen_frozen":
                # 冻结模型 + LoRA分类头
                metrics = self.evaluate_qwen_frozen_with_lora_classifier(
                    config["model_name"], lora_path
                )
            else:  # bert
                metrics = self.evaluate_bert(config["model_name"])

            # 添加模型信息
            result = {
                "key": model_key,
                "name": config["name"],
                "description": config["description"],
                "metrics": metrics,
            }

            self.results[model_key] = result
            return result

        except Exception as e:
            import traceback
            print(f"评估 {config['name']} 时出错: {e}")
            traceback.print_exc()
            return {
                "key": model_key,
                "name": config["name"],
                "error": str(e),
            }

    def run_all_benchmarks(
        self,
        lora_path: Optional[str] = None,
        skip_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        运行所有基准测试

        Args:
            lora_path: LoRA 适配器路径
            skip_models: 要跳过的模型键列表

        Returns:
            所有评估结果
        """
        skip_models = skip_models or []

        for key, config in BASELINE_MODELS.items():
            if key in skip_models:
                print(f"\n跳过模型: {config['name']}")
                continue

            self.run_benchmark(key, lora_path if config["use_lora"] else None)

        # 生成对比报告
        comparison = self._generate_comparison()
        self.results["comparison"] = comparison

        return self.results

    def _generate_comparison(self) -> Dict[str, Any]:
        """生成对比报告"""
        comparison = {
            "baseline": "qwen_lora",  # 以 LoRA 模型为基准
            "metrics": {},
            "summary": {},
        }

        baseline_result = self.results.get("qwen_lora")
        if not baseline_result or "metrics" not in baseline_result:
            return comparison

        baseline_metrics = baseline_result["metrics"]

        for key, result in self.results.items():
            if key == "comparison" or "metrics" not in result:
                continue

            metrics = result["metrics"]
            comparison["metrics"][key] = {}

            for metric_name in ["accuracy", "f1", "positive_recall", "negative_recall", "qps"]:
                baseline_value = baseline_metrics.get(metric_name, 0)
                current_value = metrics.get(metric_name, 0)

                if baseline_value > 0:
                    improvement = ((current_value - baseline_value) / baseline_value) * 100
                else:
                    improvement = 0

                comparison["metrics"][key][metric_name] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "improvement_pct": improvement,
                }

        return comparison


def print_benchmark_report(results: Dict[str, Any]):
    """打印基准对比报告"""
    print("\n" + "=" * 60)
    print("基准对比报告")
    print("=" * 60)

    # 表头
    print(f"\n{'模型':<25} {'准确率':<10} {'F1':<10} {'召回率(正)':<12} {'召回率(负)':<12} {'QPS':<10}")
    print("-" * 80)

    # 模型结果
    for key, result in results.items():
        if key == "comparison" or "metrics" not in result:
            continue

        name = result["name"]
        metrics = result["metrics"]

        print(f"{name:<25} ", end="")
        print(f"{metrics['accuracy']:.4f}   ", end="")
        print(f"{metrics['f1']:.4f}   ", end="")
        print(f"{metrics['positive_recall']:.4f}     ", end="")
        print(f"{metrics['negative_recall']:.4f}     ", end="")
        print(f"{metrics['qps']:.2f}")

    # 对比分析
    if "comparison" in results:
        comparison = results["comparison"]
        print(f"\n【性能提升对比（以 LoRA 模型为基准）】")
        print("-" * 60)

        for key, metrics in comparison["metrics"].items():
            if key == "qwen_lora":
                continue

            name = results[key]["name"]
            print(f"\n{name}:")
            for metric_name, data in metrics.items():
                improvement = data["improvement_pct"]
                if improvement > 0:
                    print(f"    {metric_name}: +{improvement:.2f}%")
                else:
                    print(f"    {metric_name}: {improvement:.2f}%")

    print("\n" + "=" * 60)


def main(args):
    """主函数"""
    print(f"使用设备: {get_device()}")

    # 加载测试数据
    print(f"\n加载数据集: {args.dataset}")
    dataset = load_dataset("lansinuote/ChnSentiCorp")

    eval_split = "test" if "test" in dataset else "validation"
    eval_data = dataset[eval_split]

    texts = eval_data["text"]
    labels = eval_data["label"]

    # 限制测试样本数（基准测试可能耗时较长）
    if args.max_samples > 0 and len(texts) > args.max_samples:
        print(f"限制测试样本数: {len(texts)} -> {args.max_samples}")
        texts = texts[:args.max_samples]
        labels = labels[:args.max_samples]

    print(f"测试数据量: {len(texts)}")

    # 创建评估器
    evaluator = BenchmarkEvaluator(
        texts=texts,
        labels=labels,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # 解析要测试的模型
    models_to_test = args.models.split(",")
    skip_models = [k for k in BASELINE_MODELS.keys() if k not in models_to_test]

    # 运行基准测试
    results = evaluator.run_all_benchmarks(
        lora_path=args.model_path if "qwen_lora" in models_to_test else None,
        skip_models=skip_models,
    )

    # 打印报告
    print_benchmark_report(results)

    # 保存结果
    save_json(results, "baseline_comparison.json")

    return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="模型基准对比测试",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="./experiments/classifier_head/lora_adapter",
        help="LoRA 适配器路径",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ChnSentiCorp",
        help="评估数据集",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen_lora,qwen_base",
        help="要测试的模型（逗号分隔）: qwen_lora,qwen_base,bert_chinese,roberta_wwm",
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
        "--max_samples",
        type=int,
        default=1000,
        help="最大测试样本数（0 表示使用全部）",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
