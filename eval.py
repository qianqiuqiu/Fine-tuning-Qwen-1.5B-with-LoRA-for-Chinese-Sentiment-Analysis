"""
评估脚本
对训练好的模型进行全面评估

使用方法:
    python eval.py --model_path ./outputs/lora_adapter
    python eval.py --model_path ./outputs/lora_adapter --dataset ChnSentiCorp
"""

import os
import argparse
import torch
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from data import load_sentiment_dataset, create_tokenized_dataset


def load_model_for_eval(
    base_model_name: str,
    lora_path: str,
    device: str = "cuda",
) -> tuple:
    """
    加载模型用于评估
    
    Args:
        base_model_name: 基础模型名称
        lora_path: LoRA 适配器路径
        device: 设备
    
    Returns:
        (model, tokenizer) 元组
    """
    
    print(f"加载基础模型: {base_model_name}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path,  # 从 LoRA 路径加载（包含相同的 tokenizer）
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 加载 LoRA 适配器
    print(f"加载 LoRA 适配器: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 合并 LoRA 权重（推理时可选，提高速度）
    model = model.merge_and_unload()
    
    model.eval()
    
    return model, tokenizer


def evaluate_batch(
    model,
    tokenizer,
    texts: List[str],
    labels: List[int],
    batch_size: int = 16,
    max_length: int = 256,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    批量评估
    
    Args:
        model: 模型
        tokenizer: 分词器
        texts: 文本列表
        labels: 标签列表
        batch_size: 批次大小
        max_length: 最大长度
        device: 设备
    
    Returns:
        评估结果字典
    """
    
    all_predictions = []
    all_probabilities = []
    
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="评估中"):
            batch_texts = texts[i:i + batch_size]
            
            # 分词
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            
            # 推理
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 获取预测和概率
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(preds.cpu().numpy().tolist())
            all_probabilities.extend(probs[:, 1].cpu().numpy().tolist())  # 正类概率
    
    # 计算指标
    results = compute_all_metrics(labels, all_predictions, all_probabilities)
    
    return results


def compute_all_metrics(
    labels: List[int],
    predictions: List[int],
    probabilities: List[float],
) -> Dict[str, Any]:
    """
    计算所有评估指标
    
    Args:
        labels: 真实标签
        predictions: 预测标签
        probabilities: 预测概率
    
    Returns:
        指标字典
    """
    
    labels = np.array(labels)
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # 基础指标
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    # AUC-ROC
    try:
        auc = roc_auc_score(labels, probabilities)
    except:
        auc = 0.0
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    
    # 分类报告
    report = classification_report(
        labels, predictions,
        target_names=['负面', '正面'],
        output_dict=True,
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def print_evaluation_results(results: Dict[str, Any]):
    """打印评估结果"""
    
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    
    print(f"\n整体指标：")
    print(f"  准确率 (Accuracy): {results['accuracy']:.4f}")
    print(f"  精确率 (Precision): {results['precision']:.4f}")
    print(f"  召回率 (Recall): {results['recall']:.4f}")
    print(f"  F1 分数: {results['f1']:.4f}")
    print(f"  AUC-ROC: {results['auc_roc']:.4f}")
    
    print(f"\n混淆矩阵：")
    cm = results['confusion_matrix']
    print(f"  {'':>10} 预测负面  预测正面")
    print(f"  实际负面 {cm[0][0]:>8} {cm[0][1]:>8}")
    print(f"  实际正面 {cm[1][0]:>8} {cm[1][1]:>8}")
    
    print(f"\n分类报告：")
    report = results['classification_report']
    for label in ['负面', '正面']:
        print(f"  {label}:")
        print(f"    精确率: {report[label]['precision']:.4f}")
        print(f"    召回率: {report[label]['recall']:.4f}")
        print(f"    F1: {report[label]['f1-score']:.4f}")


def analyze_errors(
    texts: List[str],
    labels: List[int],
    predictions: List[int],
    num_examples: int = 10,
) -> Dict[str, List]:
    """
    分析错误案例
    
    Args:
        texts: 文本列表
        labels: 真实标签
        predictions: 预测标签
        num_examples: 显示的错误案例数量
    
    Returns:
        错误分析结果
    """
    
    errors = {
        "false_positives": [],  # 误判为正面
        "false_negatives": [],  # 误判为负面
    }
    
    for text, label, pred in zip(texts, labels, predictions):
        if label != pred:
            if pred == 1:  # 实际负面，预测正面
                errors["false_positives"].append({
                    "text": text,
                    "true_label": "负面",
                    "predicted": "正面",
                })
            else:  # 实际正面，预测负面
                errors["false_negatives"].append({
                    "text": text,
                    "true_label": "正面",
                    "predicted": "负面",
                })
    
    print(f"\n错误分析：")
    print(f"  误判为正面（假阳性）: {len(errors['false_positives'])} 例")
    print(f"  误判为负面（假阴性）: {len(errors['false_negatives'])} 例")
    
    print(f"\n假阳性示例（实际负面，预测正面）：")
    for i, error in enumerate(errors["false_positives"][:num_examples // 2]):
        print(f"  {i+1}. {error['text'][:100]}...")
    
    print(f"\n假阴性示例（实际正面，预测负面）：")
    for i, error in enumerate(errors["false_negatives"][:num_examples // 2]):
        print(f"  {i+1}. {error['text'][:100]}...")
    
    return errors


def main(args):
    """主函数"""
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    model, tokenizer = load_model_for_eval(
        base_model_name=args.base_model,
        lora_path=args.model_path,
        device=device,
    )
    
    # 加载测试数据
    print(f"\n加载数据集: {args.dataset}")
    dataset = load_sentiment_dataset(args.dataset)
    
    # 使用测试集（如果有），否则使用验证集
    eval_split = "test" if "test" in dataset else "validation"
    eval_data = dataset[eval_split]
    
    texts = eval_data["text"]
    labels = eval_data["label"]
    
    print(f"评估数据量: {len(texts)}")
    
    # 执行评估
    results = evaluate_batch(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        labels=labels,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    
    # 打印结果
    print_evaluation_results(results)
    
    # 错误分析
    if args.analyze_errors:
        # 重新获取预测结果用于错误分析
        predictions = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), args.batch_size), desc="获取预测"):
                batch_texts = texts[i:i + args.batch_size]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(device)
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy().tolist())
        
        analyze_errors(texts, labels, predictions)
    
    # 保存结果
    if args.save_results:
        import json
        results_to_save = {
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
            "auc_roc": results["auc_roc"],
        }
        
        output_file = os.path.join(args.model_path, "eval_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")


def parse_args():
    """解析命令行参数"""
    
    parser = argparse.ArgumentParser(description="评估情感分析模型")
    
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
        "--analyze_errors",
        action="store_true",
        default=True,
        help="是否进行错误分析",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        default=True,
        help="是否保存结果",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
