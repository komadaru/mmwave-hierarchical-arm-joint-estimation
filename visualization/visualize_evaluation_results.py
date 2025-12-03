#!/usr/bin/env python3
"""
評価結果の可視化スクリプト
検出領域とGT関節の位置関係、誤差の分布を可視化
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import List, Dict
import seaborn as sns

sns.set_style("whitegrid")


def load_evaluation_results(results_path: str) -> Dict:
    """評価結果を読み込む"""
    with open(results_path, 'r') as f:
        return json.load(f)


def visualize_error_distribution(metrics: Dict, output_path: str):
    """誤差の分布を可視化"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1D誤差と3D誤差の比較（利用可能な場合）
    if 'mean_error_1d' in metrics and metrics.get('mean_error_1d') is not None:
        errors_1d = [metrics.get('mean_error_1d', 0)]
        errors_3d = [metrics.get('mean_error_3d', metrics.get('spatial_mean_error', 0))]
        
        axes[0].bar(['1D Error', '3D Error'], [errors_1d[0], errors_3d[0]], 
                   color=['skyblue', 'coral'], alpha=0.7)
        axes[0].set_ylabel('Mean Error (m)')
        axes[0].set_title('Mean Error Comparison (1D vs 3D)')
        axes[0].grid(True, alpha=0.3)
    
    # 閾値以内の検出率
    within_threshold = metrics.get('spatial_within_threshold', 0)
    axes[1].bar(['Within 0.25m'], [within_threshold], color='lightgreen', alpha=0.7)
    axes[1].set_ylabel('Detection Rate')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Detection Rate Within Threshold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error distribution plot saved to: {output_path}")


def visualize_metrics_comparison(metrics: Dict, output_path: str):
    """メトリクスの比較を可視化"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ピクセルレベルメトリクス
    pixel_metrics = {
        'Accuracy': metrics.get('pixel_accuracy', 0),
        'IoU': metrics.get('iou', 0),
        'Precision': metrics.get('precision', 0),
        'Recall': metrics.get('recall', 0),
        'F1-Score': metrics.get('f1_score', 0)
    }
    
    axes[0].bar(pixel_metrics.keys(), pixel_metrics.values(), 
               color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Pixel-level Metrics')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 空間メトリクス
    if 'spatial_mean_error' in metrics:
        spatial_metrics = {
            'Mean Error': metrics.get('spatial_mean_error', 0),
            'Median Error': metrics.get('spatial_median_error', 0),
            'Within Threshold': metrics.get('spatial_within_threshold', 0)
        }
        
        # 誤差はメートル単位、検出率は0-1の範囲
        error_values = [spatial_metrics['Mean Error'], spatial_metrics['Median Error']]
        rate_value = spatial_metrics['Within Threshold']
        
        # 2つのY軸を使用
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar(['Mean Error', 'Median Error'], error_values, 
                       color='coral', alpha=0.7, label='Error (m)')
        bar2 = ax2_twin.bar(['Within Threshold'], [rate_value], 
                           color='lightgreen', alpha=0.7, label='Rate')
        
        ax2.set_ylabel('Error (m)', color='coral')
        ax2_twin.set_ylabel('Detection Rate', color='lightgreen')
        ax2.set_title('Spatial Metrics')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 凡例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison plot saved to: {output_path}")


def visualize_sample_predictions(
    results_path: str,
    output_dir: str,
    num_samples: int = 10
):
    """サンプル予測結果を可視化（確率マップと検出領域）"""
    # 注意: 確率マップは通常保存されていないため、この機能は将来の拡張用
    print(f"Sample prediction visualization is not yet implemented.")
    print(f"  (Probability maps are not saved to reduce file size)")


def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to evaluation_results.json')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save visualization plots')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of sample predictions to visualize')
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 評価結果を読み込む
    print(f"Loading evaluation results from {args.results_path}...")
    results = load_evaluation_results(args.results_path)
    metrics = results['metrics']
    
    print(f"\nGenerating visualizations...")
    
    # 誤差分布の可視化
    visualize_error_distribution(
        metrics,
        str(output_dir / 'error_distribution.png')
    )
    
    # メトリクス比較の可視化
    visualize_metrics_comparison(
        metrics,
        str(output_dir / 'metrics_comparison.png')
    )
    
    # サンプル予測の可視化（将来の拡張）
    # visualize_sample_predictions(
    #     args.results_path,
    #     str(output_dir),
    #     num_samples=args.num_samples
    # )
    
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

