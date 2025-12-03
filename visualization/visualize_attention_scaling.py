#!/usr/bin/env python3
"""
Figure 4: 関節ごとのAttentionスケーリング値の可視化スクリプト

Attentionマップが空間的に一様なため、関節ごとのスケーリング値を可視化します。
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import os
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heatmap_distal_detection.dataset_arm_joints import ArmJointDataset, collate_fn
from heatmap_distal_detection.models.arm_joint_regressor_hierarchical import (
    create_hierarchical_arm_joint_regressor
)
from torch.utils.data import DataLoader


def load_model(model_path: str, device: str = 'cuda') -> nn.Module:
    """モデルを読み込む"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model = create_hierarchical_arm_joint_regressor(
        heatmap_size=config.get('heatmap_size', 50),
        num_joints_per_arm=config.get('num_joints_per_arm', 3),
        base_channels=config.get('base_channels', 32),
        dropout=config.get('dropout', 0.5),
        use_attention=config.get('use_attention', True),
        use_improved_attention=config.get('use_improved_attention', False)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def collect_attention_scaling_values(
    model: nn.Module,
    dataset: ArmJointDataset,
    num_samples: int = 100,
    device: str = 'cuda'
) -> Dict[str, List[float]]:
    """
    Attentionスケーリング値を収集
    
    Returns:
        Dict with keys: 'left_shoulder', 'left_elbow', 'left_wrist',
                       'right_shoulder', 'right_elbow', 'right_wrist'
    """
    if not model.use_attention:
        print("⚠️  Warning: The loaded model does not use attention mechanism.")
        return {}
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    scaling_values = {
        'left_shoulder': [],
        'left_elbow': [],
        'left_wrist': [],
        'right_shoulder': [],
        'right_elbow': [],
        'right_wrist': []
    }
    
    sample_count = 0
    
    for batch in tqdm(dataloader, desc="Collecting attention scaling values"):
        if sample_count >= num_samples:
            break
        
        heatmap = batch['heatmap'].to(device)
        
        with torch.no_grad():
            features = model.encoder(heatmap)
            attn_left = model.attention_left(features)  # (1, 3, 6, 6)
            attn_right = model.attention_right(features)  # (1, 3, 6, 6)
        
        # Attentionマップは空間的に一様なので、平均値を取得
        attn_left_mean = attn_left.mean(dim=[2, 3]).cpu().numpy()[0]  # (3,)
        attn_right_mean = attn_right.mean(dim=[2, 3]).cpu().numpy()[0]  # (3,)
        
        scaling_values['left_shoulder'].append(float(attn_left_mean[0]))
        scaling_values['left_elbow'].append(float(attn_left_mean[1]))
        scaling_values['left_wrist'].append(float(attn_left_mean[2]))
        scaling_values['right_shoulder'].append(float(attn_right_mean[0]))
        scaling_values['right_elbow'].append(float(attn_right_mean[1]))
        scaling_values['right_wrist'].append(float(attn_right_mean[2]))
        
        sample_count += 1
    
    return scaling_values


def visualize_attention_scaling(
    scaling_values: Dict[str, List[float]],
    output_dir: Path = Path("outputs/paper_figures"),
    model_name: str = "Model"
):
    """
    関節ごとのAttentionスケーリング値を可視化
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 関節名
    joint_names = ['Shoulder', 'Elbow', 'Wrist']
    left_keys = ['left_shoulder', 'left_elbow', 'left_wrist']
    right_keys = ['right_shoulder', 'right_elbow', 'right_wrist']
    
    # 統計値を計算
    left_means = [np.mean(scaling_values[key]) for key in left_keys]
    left_stds = [np.std(scaling_values[key]) for key in left_keys]
    right_means = [np.mean(scaling_values[key]) for key in right_keys]
    right_stds = [np.std(scaling_values[key]) for key in right_keys]
    
    # 図を作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(joint_names))
    width = 0.35
    
    # 左腕
    axes[0].bar(x, left_means, width, yerr=left_stds, capsize=5, 
                label='Left Arm', color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Joint', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Attention Scaling Value', fontsize=12, fontweight='bold')
    axes[0].set_title('Left Arm: Joint-Specific Attention Scaling', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(joint_names)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # 数値を表示
    for i, (mean, std) in enumerate(zip(left_means, left_stds)):
        axes[0].text(i, mean + std + 0.01, f'{mean:.3f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 右腕
    axes[1].bar(x, right_means, width, yerr=right_stds, capsize=5,
                label='Right Arm', color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Joint', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Attention Scaling Value', fontsize=12, fontweight='bold')
    axes[1].set_title('Right Arm: Joint-Specific Attention Scaling', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(joint_names)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend(fontsize=10)
    
    # 数値を表示
    for i, (mean, std) in enumerate(zip(right_means, right_stds)):
        axes[1].text(i, mean + std + 0.01, f'{mean:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'{model_name}: Joint-Specific Attention Scaling Values', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / 'figure_4_attention_scaling.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_4_attention_scaling.pdf', bbox_inches='tight')
    plt.close()
    
    # データをJSON形式で保存
    figures_data_dir = output_dir / 'figures_data'
    figures_data_dir.mkdir(parents=True, exist_ok=True)
    
    figure_4_data = {
        'title': f'{model_name}: Joint-Specific Attention Scaling Values',
        'type': 'bar_plot',
        'layout': '1x2',
        'joints': joint_names,
        'left_arm': {
            'joints': joint_names,
            'means': left_means,
            'stds': left_stds
        },
        'right_arm': {
            'joints': joint_names,
            'means': right_means,
            'stds': right_stds
        },
        'raw_data': {
            'left_shoulder': scaling_values['left_shoulder'],
            'left_elbow': scaling_values['left_elbow'],
            'left_wrist': scaling_values['left_wrist'],
            'right_shoulder': scaling_values['right_shoulder'],
            'right_elbow': scaling_values['right_elbow'],
            'right_wrist': scaling_values['right_wrist']
        },
        'created_at': datetime.now().isoformat(),
        'figure_name': 'figure_4_attention_scaling'
    }
    
    with open(figures_data_dir / 'figure_4_attention_scaling_data.json', 'w', encoding='utf-8') as f:
        json.dump(figure_4_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created Figure 4: {output_path}")
    print(f"  → Saved figure data: {figures_data_dir / 'figure_4_attention_scaling_data.json'}")
    
    # 統計情報を出力
    print(f"\n[Statistics]")
    print(f"  Left Arm:")
    for i, joint_name in enumerate(joint_names):
        mean = left_means[i]
        std = left_stds[i]
        print(f"    {joint_name}: {mean:.4f} ± {std:.4f}")
    print(f"  Right Arm:")
    for i, joint_name in enumerate(joint_names):
        mean = right_means[i]
        std = right_stds[i]
        print(f"    {joint_name}: {mean:.4f} ± {std:.4f}")


def compare_attention_versions(
    scaling_values_v1: Dict[str, List[float]],
    scaling_values_v2: Dict[str, List[float]],
    output_dir: Path = Path("outputs/paper_figures"),
    v1_name: str = "Conventional",
    v2_name: str = "Improved"
):
    """
    従来版と改善版のAttentionスケーリング値を比較
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joint_names = ['Shoulder', 'Elbow', 'Wrist']
    left_keys = ['left_shoulder', 'left_elbow', 'left_wrist']
    right_keys = ['right_shoulder', 'right_elbow', 'right_wrist']
    
    # 統計値を計算
    v1_left_means = [np.mean(scaling_values_v1[key]) for key in left_keys]
    v1_left_stds = [np.std(scaling_values_v1[key]) for key in left_keys]
    v1_right_means = [np.mean(scaling_values_v1[key]) for key in right_keys]
    v1_right_stds = [np.std(scaling_values_v1[key]) for key in right_keys]
    
    v2_left_means = [np.mean(scaling_values_v2[key]) for key in left_keys]
    v2_left_stds = [np.std(scaling_values_v2[key]) for key in left_keys]
    v2_right_means = [np.mean(scaling_values_v2[key]) for key in right_keys]
    v2_right_stds = [np.std(scaling_values_v2[key]) for key in right_keys]
    
    # 図を作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(joint_names))
    width = 0.35
    
    # 左腕
    axes[0].bar(x - width/2, v1_left_means, width, yerr=v1_left_stds, capsize=5,
                label=v1_name, color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].bar(x + width/2, v2_left_means, width, yerr=v2_left_stds, capsize=5,
                label=v2_name, color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Joint', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Attention Scaling Value', fontsize=12, fontweight='bold')
    axes[0].set_title('Left Arm: Attention Scaling Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(joint_names)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # 右腕
    axes[1].bar(x - width/2, v1_right_means, width, yerr=v1_right_stds, capsize=5,
                label=v1_name, color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].bar(x + width/2, v2_right_means, width, yerr=v2_right_stds, capsize=5,
                label=v2_name, color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Joint', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Attention Scaling Value', fontsize=12, fontweight='bold')
    axes[1].set_title('Right Arm: Attention Scaling Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(joint_names)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.suptitle('Comparison of Joint-Specific Attention Scaling Values', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / 'figure_4_attention_scaling_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_4_attention_scaling_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created comparison figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize joint-specific attention scaling values')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to the test data JSONL file')
    parser.add_argument('--plane', type=str, choices=['xz', 'xy'], required=True,
                       help='Plane (xz or xy)')
    parser.add_argument('--x_range', type=float, nargs=2, default=[-1.0, 1.0],
                       metavar=('X_MIN', 'X_MAX'), help='X-axis range')
    parser.add_argument('--z_range', type=float, nargs=2, default=[-1.0, 0.75],
                       metavar=('Z_MIN', 'Z_MAX'), help='Z-axis range (for xz plane)')
    parser.add_argument('--y_range', type=float, nargs=2, default=[2.0, 5.0],
                       metavar=('Y_MIN', 'Y_MAX'), help='Y-axis range (for xy plane)')
    parser.add_argument('--bins', type=int, default=50, help='Heatmap size (bins)')
    parser.add_argument('--feature', type=str, choices=['velocity', 'energy_power'], 
                       default='energy_power', help='Feature type')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to collect')
    parser.add_argument('--output_dir', type=str, default='outputs/paper_figures',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # デバイス設定
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    # モデルを読み込み
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, device)
    
    # データセットを読み込み
    print(f"Loading dataset from {args.test_data}...")
    if args.plane == 'xz':
        dataset = ArmJointDataset(
            data_path=args.test_data,
            x_range=tuple(args.x_range),
            z_range=tuple(args.z_range),
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=True
        )
    else:  # xy
        from heatmap_distal_detection.dataset_arm_joints_xy import ArmJointDatasetXY
        dataset = ArmJointDatasetXY(
            data_path=args.test_data,
            x_range=tuple(args.x_range),
            y_range=tuple(args.y_range),
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=True
        )
    
    print(f"Loaded {len(dataset)} samples")
    
    # Attentionスケーリング値を収集
    scaling_values = collect_attention_scaling_values(
        model, dataset, num_samples=args.num_samples, device=device
    )
    
    if not scaling_values:
        print("⚠️  No attention scaling values collected. Exiting.")
        return
    
    # 可視化
    model_name = Path(args.model_path).stem
    output_dir = Path(args.output_dir)
    visualize_attention_scaling(scaling_values, output_dir, model_name)


if __name__ == '__main__':
    main()

