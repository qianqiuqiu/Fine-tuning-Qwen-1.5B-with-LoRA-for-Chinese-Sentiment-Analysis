"""
报告生成器模块
Report Generator Module

生成 HTML 格式的可视化评估报告
"""

import os
import json
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from .utils import ensure_output_dir


# 设置绘图样式
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ReportGenerator:
    """报告生成器"""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir or ensure_output_dir()
        self.plots_dir = ensure_output_dir("plots")

        self.data = {
            "basic_metrics": {},
            "confusion_matrix": None,
            "predictions": None,
            "labels": None,
            "probabilities": None,
            "confidence_analysis": None,
            "robustness_result": None,
            "benchmark_result": None,
        }

        self.plots = {}

    def load_basic_metrics(self, metrics: Dict[str, Any]):
        """加载基础评估指标"""
        self.data["basic_metrics"] = metrics

    def load_predictions(
        self,
        predictions: List[int],
        labels: List[int],
        probabilities: Optional[List[float]] = None,
    ):
        """加载预测结果"""
        self.data["predictions"] = predictions
        self.data["labels"] = labels
        self.data["probabilities"] = probabilities or []

    def load_confidence_analysis(self, analysis: Dict[str, Any]):
        """加载置信度分析结果"""
        self.data["confidence_analysis"] = analysis

    def load_robustness_result(self, result: Dict[str, Any]):
        """加载鲁棒性测试结果"""
        self.data["robustness_result"] = result

    def load_benchmark_result(self, result: Dict[str, Any]):
        """加载基准对比结果"""
        self.data["benchmark_result"] = result

    def generate_all_plots(self):
        """生成所有图表"""
        print("\n生成可视化图表...")

        # 混淆矩阵
        if self.data["predictions"] is not None and self.data["labels"] is not None:
            self.plot_confusion_matrix()

        # ROC 曲线和 PR 曲线
        if self.data["probabilities"] and len(self.data["probabilities"]) > 0:
            self.plot_roc_curve()
            self.plot_pr_curve()

        # 置信度分布
        if self.data["confidence_analysis"]:
            self.plot_confidence_distribution_report()
            self.plot_reliability_diagram_report()

    def plot_confusion_matrix(self):
        """绘制混淆矩阵热图"""
        predictions = self.data["predictions"]
        labels = self.data["labels"]

        cm = confusion_matrix(labels, predictions)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['负面', '正面'],
            yticklabels=['负面', '正面'],
            ax=ax,
        )
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        ax.set_title('混淆矩阵', fontsize=14)

        output_path = self.plots_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.plots["confusion_matrix"] = str(output_path)
        print(f"  混淆矩阵: {output_path}")

        return output_path

    def plot_roc_curve(self):
        """绘制 ROC 曲线"""
        labels = self.data["labels"]
        probabilities = self.data["probabilities"]

        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(fpr, tpr, linewidth=2, label=f'ROC 曲线 (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')

        ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=12)
        ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=12)
        ax.set_title('ROC 曲线', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)

        output_path = self.plots_dir / "roc_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.plots["roc_curve"] = str(output_path)
        print(f"  ROC 曲线: {output_path}")

        return output_path

    def plot_pr_curve(self):
        """绘制精确率-召回率曲线"""
        labels = self.data["labels"]
        probabilities = self.data["probabilities"]

        precision, recall, _ = precision_recall_curve(labels, probabilities)
        avg_precision = average_precision_score(labels, probabilities)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(recall, precision, linewidth=2,
                label=f'PR 曲线 (AP = {avg_precision:.4f})')

        ax.set_xlabel('召回率 (Recall)', fontsize=12)
        ax.set_ylabel('精确率 (Precision)', fontsize=12)
        ax.set_title('精确率-召回率曲线', fontsize=14)
        ax.legend(loc='lower left')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)

        output_path = self.plots_dir / "pr_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.plots["pr_curve"] = str(output_path)
        print(f"  PR 曲线: {output_path}")

        return output_path

    def plot_confidence_distribution_report(self):
        """绘制置信度分布图（用于报告）"""
        analysis = self.data["confidence_analysis"]
        dist = analysis["confidence_distribution"]
        stats = analysis["basic_stats"]

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
        box_data = [
            [stats["mean_confidence_correct"]],
            [stats["mean_confidence_wrong"]],
        ]
        bp = ax2.boxplot(
            box_data,
            labels=["正确预测", "错误预测"],
            patch_artist=True,
        )
        colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax2.set_ylabel('置信度', fontsize=12)
        ax2.set_title('置信度统计', fontsize=14)
        ax2.grid(alpha=0.3, axis='y')
        ax2.set_ylim(0, 1.1)

        plt.tight_layout()
        output_path = self.plots_dir / "confidence_distribution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.plots["confidence_distribution"] = str(output_path)
        print(f"  置信度分布: {output_path}")

        return output_path

    def plot_reliability_diagram_report(self):
        """绘制可靠性图（用于报告）"""
        analysis = self.data["confidence_analysis"]
        reliability = analysis["reliability_diagram"]
        ece = analysis["ece"]

        bin_confidences = np.array(reliability["bin_confidences"])
        bin_accuracies = np.array(reliability["bin_accuracies"])
        bin_counts = np.array(reliability["bin_counts"])
        bin_boundaries = np.array(reliability["bin_boundaries"])

        valid_bins = bin_counts > 0
        bin_confidences = bin_confidences[valid_bins]
        bin_accuracies = bin_accuracies[valid_bins]
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_centers = bin_centers[valid_bins]

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='完美校准')

        scatter = ax.scatter(
            bin_confidences, bin_accuracies,
            s=bin_counts / bin_counts.sum() * 1000 if bin_counts.sum() > 0 else 100,
            c=bin_centers, cmap='viridis',
            alpha=0.6, edgecolors='black',
        )

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

        output_path = self.plots_dir / "reliability_diagram.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.plots["reliability_diagram"] = str(output_path)
        print(f"  可靠性图: {output_path}")

        return output_path

    def image_to_base64(self, image_path: Path) -> str:
        """将图片转换为 base64 编码"""
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        return base64.b64encode(img_bytes).decode('utf-8')

    def generate_html_report(self) -> Path:
        """生成 HTML 报告"""
        print("\n生成 HTML 报告...")

        # 生成图表
        self.generate_all_plots()

        # 转换图片为 base64
        images = {}
        for name, path in self.plots.items():
            if path and os.path.exists(path):
                images[name] = self.image_to_base64(Path(path))

        # 构建 HTML
        html = self._build_html_template(images)

        # 保存报告
        output_path = self.output_dir / "report.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"HTML 报告已生成: {output_path}")
        return output_path

    def _build_html_template(self, images: Dict[str, str]) -> str:
        """构建 HTML 模板"""
        # 基础指标
        metrics = self.data["basic_metrics"]
        metrics_html = self._build_metrics_section(metrics)

        # 置信度分析
        confidence_html = ""
        if self.data["confidence_analysis"]:
            confidence_html = self._build_confidence_section()

        # 鲁棒性测试
        robustness_html = ""
        if self.data["robustness_result"]:
            robustness_html = self._build_robustness_section()

        # 基准对比
        benchmark_html = ""
        if self.data["benchmark_result"]:
            benchmark_html = self._build_benchmark_section()

        # HTML 模板
        template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型评估报告 - Qwen-1.5B LoRA 情感分析</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        .card h3 {{
            color: #555;
            margin: 20px 0 10px;
            font-size: 1.3em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }}
        .metric-box .label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .metric-box .value {{
            color: #28a745;
            font-size: 1.8em;
            font-weight: bold;
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .table th, .table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .table th {{
            background: #f8f9fa;
            font-weight: bold;
        }}
        .table tr:hover {{
            background: #ff0;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .badge-success {{ background: #28a745; color: white; }}
        .badge-warning {{ background: #ffc107; color: #333; }}
        .badge-danger {{ background: #dc3545; color: white; }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>模型评估报告</h1>
            <p>Qwen-1.5B LoRA 情感分析模型综合评估</p>
        </div>

        <!-- 基础指标 -->
        <div class="card">
            <h2>基础评估指标</h2>
            {metrics_html}
        </div>

        <!-- 可视化图表 -->
        <div class="card">
            <h2>可视化分析</h2>
            <div class="plot-grid">
                {self._build_plot_html('confusion_matrix', '混淆矩阵', images)}
                {self._build_plot_html('roc_curve', 'ROC 曲线', images)}
                {self._build_plot_html('pr_curve', '精确率-召回率曲线', images)}
            </div>
        </div>

        <!-- 置信度分析 -->
        {confidence_html}

        <!-- 鲁棒性测试 -->
        {robustness_html}

        <!-- 基准对比 -->
        {benchmark_html}

        <div class="footer">
            <p>报告生成时间: {self._get_timestamp()}</p>
        </div>
    </div>
</body>
</html>"""
        return template

    def _build_metrics_section(self, metrics: Dict[str, Any]) -> str:
        """构建指标部分"""
        if not metrics:
            return "<p>暂无指标数据</p>"

        metric_items = [
            ("准确率", metrics.get("accuracy", 0)),
            ("精确率", metrics.get("precision", 0)),
            ("召回率", metrics.get("recall", 0)),
            ("F1 分数", metrics.get("f1", 0)),
            ("AUC-ROC", metrics.get("auc_roc", 0)),
        ]

        html = '<div class="metrics-grid">'
        for label, value in metric_items:
            html += f"""
                <div class="metric-box">
                    <div class="label">{label}</div>
                    <div class="value">{value:.4f}</div>
                </div>
            """
        html += "</div>"
        return html

    def _build_confidence_section(self) -> str:
        """构建置信度分析部分"""
        analysis = self.data["confidence_analysis"]
        if not analysis:
            return ""

        stats = analysis["basic_stats"]
        ece = analysis["ece"]
        low_conf = analysis["low_confidence"]
        high_err = analysis["high_confidence_errors"]

        html = """
        <div class="card">
            <h2>置信度分析</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="label">ECE (校准误差)</div>
                    <div class="value">{ece:.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="label">平均置信度</div>
                    <div class="value">{stats['mean_confidence']:.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="label">低置信度样本</div>
                    <div class="value">{low_conf['count']}</div>
                </div>
                <div class="metric-box">
                    <div class="label">高置信度错误</div>
                    <div class="value">{high_err['count']}</div>
                </div>
            </div>
            <div class="plot-grid">
                {self._build_plot_html('confidence_distribution', '置信度分布', self.plots)}
                {self._build_plot_html('reliability_diagram', '可靠性图', self.plots)}
            </div>
        </div>
        """
        return html.format(
            ece=ece,
            stats=stats,
            low_conf=low_conf,
            high_err=high_err,
        )

    def _build_robustness_section(self) -> str:
        """构建鲁棒性测试部分"""
        result = self.data["robustness_result"]
        if not result:
            return ""

        overall = result["overall"]
        categories = result["categories"]

        html = """
        <div class="card">
            <h2>鲁棒性测试</h2>
            <h3>总体评分: {robustness_score:.1f}/100</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>测试类别</th>
                        <th>样本数</th>
                        <th>正确</th>
                        <th>错误</th>
                        <th>准确率</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

        rows = ""
        for cat_key, cat_result in categories.items():
            name = cat_result["name"]
            total = cat_result["total"]
            correct = cat_result.get("correct", "-")
            wrong = cat_result.get("wrong", "-")
            acc = cat_result.get("accuracy")
            acc_str = f"{acc:.2%}" if acc is not None else "N/A"

            rows += f"""
                <tr>
                    <td>{name}</td>
                    <td>{total}</td>
                    <td>{correct}</td>
                    <td>{wrong}</td>
                    <td>{acc_str}</td>
                </tr>
            """

        return html.format(
            robustness_score=overall["robustness_score"],
            rows=rows,
        )

    def _build_benchmark_section(self) -> str:
        """构建基准对比部分"""
        result = self.data["benchmark_result"]
        if not result:
            return ""

        html = """
        <div class="card">
            <h2>基准模型对比</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>模型</th>
                        <th>准确率</th>
                        <th>F1</th>
                        <th>召回率(正)</th>
                        <th>召回率(负)</th>
                        <th>QPS</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

        rows = ""
        for key, model_result in result.items():
            if key == "comparison" or "metrics" not in model_result:
                continue

            name = model_result["name"]
            metrics = model_result["metrics"]

            rows += f"""
                <tr>
                    <td>{name}</td>
                    <td>{metrics['accuracy']:.4f}</td>
                    <td>{metrics['f1']:.4f}</td>
                    <td>{metrics['positive_recall']:.4f}</td>
                    <td>{metrics['negative_recall']:.4f}</td>
                    <td>{metrics['qps']:.2f}</td>
                </tr>
            """

        return html.format(rows=rows)

    def _build_plot_html(self, plot_name: str, title: str, images: Dict[str, str]) -> str:
        """构建图表 HTML"""
        if plot_name not in images or not images[plot_name]:
            return ""

        return f"""
            <div class="plot-container">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{images[plot_name]}" alt="{title}">
            </div>
        """

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main(args):
    """主函数"""
    generator = ReportGenerator()

    # 加载数据
    outputs_dir = Path(__file__).parent / "outputs"

    if args.metrics:
        metrics_path = outputs_dir / args.metrics
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            generator.load_basic_metrics(metrics)
            print(f"加载指标: {metrics_path}")

    if args.confidence_analysis:
        analysis_path = outputs_dir / args.confidence_analysis
        if analysis_path.exists():
            with open(analysis_path, "r", encoding="utf-8") as f:
                analysis = json.load(f)
            generator.load_confidence_analysis(analysis)
            print(f"加载置信度分析: {analysis_path}")

    if args.robustness_result:
        robustness_path = outputs_dir / args.robustness_result
        if robustness_path.exists():
            with open(robustness_path, "r", encoding="utf-8") as f:
                robustness = json.load(f)
            generator.load_robustness_result(robustness)
            print(f"加载鲁棒性测试: {robustness_path}")

    if args.benchmark_result:
        benchmark_path = outputs_dir / args.benchmark_result
        if benchmark_path.exists():
            with open(benchmark_path, "r", encoding="utf-8") as f:
                benchmark = json.load(f)
            generator.load_benchmark_result(benchmark)
            print(f"加载基准对比: {benchmark_path}")

    # 生成报告
    generator.generate_html_report()


def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="生成模型评估 HTML 报告",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="metrics.json",
        help="基础指标文件名",
    )
    parser.add_argument(
        "--confidence_analysis",
        type=str,
        default="confidence_analysis.json",
        help="置信度分析文件名",
    )
    parser.add_argument(
        "--robustness_result",
        type=str,
        default="robustness_result.json",
        help="鲁棒性测试结果文件名",
    )
    parser.add_argument(
        "--benchmark_result",
        type=str,
        default="baseline_comparison.json",
        help="基准对比结果文件名",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
