"""
推理脚本
使用训练好的模型进行情感预测

使用方法:
    python inference.py --model_path ./outputs/lora_adapter --text "这个产品非常好用！"
    python inference.py --model_path ./outputs/lora_adapter --interactive
"""

import os

# 修复 torch 导入卡死问题 (Intel MKL 库冲突)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
from typing import List, Dict, Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel


class SentimentPredictor:
    """情感分析预测器"""
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B",
        lora_path: str = None,
        device: str = None,
        merge_lora: bool = True,
    ):
        """
        初始化预测器
        
        Args:
            base_model_name: 基础模型名称
            lora_path: LoRA 适配器路径（如果为 None，使用基础模型）
            device: 设备（cuda/cpu）
            merge_lora: 是否合并 LoRA 权重
        """
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = ["负面", "正面"]
        
        print(f"使用设备: {self.device}")
        
        # 加载分词器
        tokenizer_path = lora_path if lora_path else base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        
        # 确保设置 padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型
        print(f"加载模型: {base_model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # 确保模型配置中也设置了 pad_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # 加载 LoRA
        if lora_path:
            print(f"加载 LoRA 适配器: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            
            if merge_lora:
                print("合并 LoRA 权重...")
                self.model = self.model.merge_and_unload()
        
        self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成！\n")
    
    def predict(
        self,
        text: str,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """
        对单条文本进行预测
        
        Args:
            text: 输入文本
            max_length: 最大长度
        
        Returns:
            预测结果字典
        """
        
        # 分词
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(logits, dim=-1).item()
        
        return {
            "text": text,
            "label": self.labels[pred_id],
            "label_id": pred_id,
            "confidence": probs[0][pred_id].item(),
            "probabilities": {
                "负面": probs[0][0].item(),
                "正面": probs[0][1].item(),
            }
        }
    
    def predict_batch(
        self,
        texts: List[str],
        max_length: int = 256,
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            texts: 文本列表
            max_length: 最大长度
            batch_size: 批次大小
        
        Returns:
            预测结果列表
        """
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 分词
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_ids = torch.argmax(logits, dim=-1)
            
            # 整理结果
            for j, text in enumerate(batch_texts):
                pred_id = pred_ids[j].item()
                results.append({
                    "text": text,
                    "label": self.labels[pred_id],
                    "label_id": pred_id,
                    "confidence": probs[j][pred_id].item(),
                    "probabilities": {
                        "负面": probs[j][0].item(),
                        "正面": probs[j][1].item(),
                    }
                })
        
        return results


def interactive_mode(predictor: SentimentPredictor):
    """交互式预测模式"""
    
    print("=" * 50)
    print("交互式情感分析")
    print("输入文本进行预测，输入 'quit' 或 'exit' 退出")
    print("=" * 50)
    
    while True:
        print()
        text = input("请输入文本: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        
        if not text:
            print("请输入有效文本")
            continue
        
        result = predictor.predict(text)
        
        print(f"\n预测结果：")
        print(f"  情感: {result['label']}")
        print(f"  置信度: {result['confidence']:.2%}")
        print(f"  概率分布: 负面 {result['probabilities']['负面']:.2%}, "
              f"正面 {result['probabilities']['正面']:.2%}")


def demo_examples(predictor: SentimentPredictor):
    """演示示例"""
    
    examples = [
        # 正面示例
        "这家酒店环境非常好，服务态度也很棒，下次还会来！",
        "产品质量很好，物流也快，非常满意的一次购物体验。",
        "电影剧情精彩，演员演技在线，强烈推荐！",
        "这本书写得太好了，让我受益匪浅。",
        
        # 负面示例
        "服务态度太差了，再也不会来了。",
        "产品质量很差，用了两天就坏了，非常失望。",
        "等了半个小时还没上菜，体验极差。",
        "这个电影太无聊了，浪费了两个小时。",
        
        # 边界/中性示例
        "还行吧，一般般。",
        "产品还可以，但是价格有点贵。",
    ]
    
    print("=" * 60)
    print("示例预测")
    print("=" * 60)
    
    results = predictor.predict_batch(examples)
    
    for result in results:
        emoji = "😊" if result["label"] == "正面" else "😞"
        print(f"\n{emoji} [{result['label']}] (置信度: {result['confidence']:.2%})")
        print(f"   \"{result['text'][:50]}{'...' if len(result['text']) > 50 else ''}\"")


def main(args):
    """主函数"""
    
    # 初始化预测器
    predictor = SentimentPredictor(
        base_model_name=args.base_model,
        lora_path=args.model_path,
        merge_lora=True,
    )
    
    # 运行示例
    if args.demo:
        demo_examples(predictor)
    
    # 单条预测
    if args.text:
        result = predictor.predict(args.text)
        print(f"\n输入: {result['text']}")
        print(f"预测: {result['label']}")
        print(f"置信度: {result['confidence']:.2%}")
    
    # 交互模式
    if args.interactive:
        interactive_mode(predictor)


def parse_args():
    """解析命令行参数"""
    
    parser = argparse.ArgumentParser(description="情感分析推理")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/lora_adapter",
        help="LoRA 适配器路径",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="基础模型名称",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="要预测的文本",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式模式",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="运行示例演示",
    )
    parser.add_argument(
        "--no-demo",
        dest="demo",
        action="store_false",
        help="不运行示例演示",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
