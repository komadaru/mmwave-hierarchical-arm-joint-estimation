#!/usr/bin/env python3
"""
Figure 4: Attentionマップの可視化スクリプト

モデルからAttentionマップを抽出して可視化します。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import sys
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

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


def visualize_attention_maps(
    model: nn.Module,
    dataset: ArmJointDataset,
    num_samples: int = 5,
    output_dir: Path = Path("outputs/paper_figures"),
    device: str = 'cuda'
):
    """
    Attentionマップを可視化
    
    Args:
        model: 学習済みモデル
        dataset: データセット
        num_samples: 可視化するサンプル数
        output_dir: 出力ディレクトリ
        device: デバイス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # モデルがAttentionを使用しているか確認
    if not model.use_attention:
        print("=" * 60)
        print("⚠️  Warning: The loaded model does not use attention mechanism.")
        print("   Attention map visualization requires a model trained with --use_attention flag.")
        print("   Please use a model that was trained with attention enabled.")
        print("=" * 60)
        return
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    sample_count = 0
    skipped_count = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Visualizing attention maps")):
        if sample_count >= num_samples:
            break
        
        heatmap = batch['heatmap'].to(device)  # (1, 1, H, W)
        
        # Attentionマップを取得
        with torch.no_grad():
            # Encoder
            features = model.encoder(heatmap)  # (1, C, H', W')
            
            if model.use_attention:
                # Attentionマップを生成
                attn_left = model.attention_left(features)  # (1, 3, H', W')
                attn_right = model.attention_right(features)  # (1, 3, H', W')
                
                # 予測も取得
                pred_coords, attention_dict = model(heatmap, return_attention=True)
            else:
                skipped_count += 1
                if skipped_count == 1:
                    print(f"\n⚠️  Model does not use attention. Skipping samples...")
                continue
        
        # ヒートマップをCPUに移動
        heatmap_cpu = heatmap[0, 0].cpu().numpy()  # (H, W) = (50, 50)
        attn_left_cpu = attn_left[0].cpu().numpy()  # (3, H', W') = (3, 6, 6)
        attn_right_cpu = attn_right[0].cpu().numpy()  # (3, H', W') = (3, 6, 6)
        
        # デバッグ: Attentionマップの統計情報を出力（最初のサンプルのみ）
        if sample_count == 0:
            print(f"\n[Debug] Attention map statistics:")
            print(f"  Original heatmap size: {heatmap_cpu.shape}")
            print(f"  Attention map size: {attn_left_cpu.shape}")
            print(f"  Left attention - min: {attn_left_cpu.min():.4f}, max: {attn_left_cpu.max():.4f}, mean: {attn_left_cpu.mean():.4f}, std: {attn_left_cpu.std():.4f}")
            print(f"  Right attention - min: {attn_right_cpu.min():.4f}, max: {attn_right_cpu.max():.4f}, mean: {attn_right_cpu.mean():.4f}, std: {attn_right_cpu.std():.4f}")
        
        # Attentionマップを元のヒートマップサイズにリサイズ
        heatmap_h, heatmap_w = heatmap_cpu.shape
        attn_h, attn_w = attn_left_cpu.shape[1], attn_left_cpu.shape[2]
        
        # デバッグ: リサイズ前の6×6マップの詳細情報（最初のサンプルのみ）
        if sample_count == 0:
            print(f"\n[Debug] Original 6×6 Attention Maps (before resize):")
            joint_names = ['Shoulder', 'Elbow', 'Wrist']
            for i, joint_name in enumerate(joint_names):
                attn_left_6x6 = attn_left_cpu[i]
                attn_right_6x6 = attn_right_cpu[i]
                print(f"\n  {joint_name}:")
                print(f"    Left 6×6 map:")
                print(f"      Values: {attn_left_6x6.flatten()}")
                print(f"      Min: {attn_left_6x6.min():.6f}, Max: {attn_left_6x6.max():.6f}, Mean: {attn_left_6x6.mean():.6f}, Std: {attn_left_6x6.std():.6f}")
                print(f"      Unique values: {len(np.unique(attn_left_6x6))}")
                print(f"    Right 6×6 map:")
                print(f"      Values: {attn_right_6x6.flatten()}")
                print(f"      Min: {attn_right_6x6.min():.6f}, Max: {attn_right_6x6.max():.6f}, Mean: {attn_right_6x6.mean():.6f}, Std: {attn_right_6x6.std():.6f}")
                print(f"      Unique values: {len(np.unique(attn_right_6x6))}")
        
        # torch.nn.functional.interpolateを使用してリサイズ
        # 修正: nearest補間を使用して各ピクセルが明確に見えるように
        attn_left_tensor = torch.from_numpy(attn_left_cpu).unsqueeze(0).float()  # (1, 3, 6, 6)
        attn_right_tensor = torch.from_numpy(attn_right_cpu).unsqueeze(0).float()  # (1, 3, 6, 6)
        
        # nearest補間でリサイズ（各ピクセルが明確に見える）
        attn_left_resized = F.interpolate(
            attn_left_tensor,
            size=(heatmap_h, heatmap_w),
            mode='nearest'  # bilinearからnearestに変更（align_cornersは不要）
        )[0].numpy()  # (3, 50, 50)
        
        attn_right_resized = F.interpolate(
            attn_right_tensor,
            size=(heatmap_h, heatmap_w),
            mode='nearest'  # bilinearからnearestに変更（align_cornersは不要）
        )[0].numpy()  # (3, 50, 50)
        
        # リサイズ後の統計情報（最初のサンプルのみ）
        if sample_count == 0:
            print(f"\n  Resized attention map size: {attn_left_resized.shape}")
            print(f"  Resized left attention - min: {attn_left_resized.min():.4f}, max: {attn_left_resized.max():.4f}, mean: {attn_left_resized.mean():.4f}, std: {attn_left_resized.std():.4f}")
        
        # 可視化: リサイズ前（6×6）とリサイズ後（50×50）の両方を表示
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        
        # 行1: 元のヒートマップと左腕のAttention（リサイズ後50×50）
        # 元のヒートマップ
        im0 = axes[0, 0].imshow(heatmap_cpu, cmap='hot', origin='lower')
        axes[0, 0].set_title('Original Heatmap', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # 左腕のAttention（肩、肘、手首）- リサイズ後のマップを使用
        joint_names = ['Shoulder', 'Elbow', 'Wrist']
        for i, joint_name in enumerate(joint_names):
            # Attentionマップの範囲を動的に設定（一様でない場合）
            attn_map = attn_left_resized[i]
            vmin = attn_map.min()
            vmax = attn_map.max()
            # 一様でない場合のみ範囲を調整
            if vmax - vmin > 0.01:
                im = axes[0, i+1].imshow(attn_map, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
            else:
                # 一様な場合はデフォルト範囲を使用
                im = axes[0, i+1].imshow(attn_map, cmap='viridis', origin='lower', vmin=0, vmax=1, interpolation='nearest')
            axes[0, i+1].set_title(f'Left {joint_name} Attention (50×50)', fontsize=12, fontweight='bold')
            axes[0, i+1].axis('off')
            plt.colorbar(im, ax=axes[0, i+1])
        
        # 行2: 右腕のAttention（リサイズ後50×50）
        axes[1, 0].axis('off')  # 空欄
        for i, joint_name in enumerate(joint_names):
            # Attentionマップの範囲を動的に設定（一様でない場合）
            attn_map = attn_right_resized[i]
            vmin = attn_map.min()
            vmax = attn_map.max()
            # 一様でない場合のみ範囲を調整
            if vmax - vmin > 0.01:
                im = axes[1, i+1].imshow(attn_map, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
            else:
                # 一様な場合はデフォルト範囲を使用
                im = axes[1, i+1].imshow(attn_map, cmap='viridis', origin='lower', vmin=0, vmax=1, interpolation='nearest')
            axes[1, i+1].set_title(f'Right {joint_name} Attention (50×50)', fontsize=12, fontweight='bold')
            axes[1, i+1].axis('off')
            plt.colorbar(im, ax=axes[1, i+1])
        
        # 行3: 左腕のAttention（リサイズ前6×6）- 直接可視化
        axes[2, 0].axis('off')  # 空欄
        for i, joint_name in enumerate(joint_names):
            attn_map_6x6 = attn_left_cpu[i]  # (6, 6)
            vmin = attn_map_6x6.min()
            vmax = attn_map_6x6.max()
            # 6×6マップを直接可視化
            im = axes[2, i+1].imshow(attn_map_6x6, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
            axes[2, i+1].set_title(f'Left {joint_name} Attention (6×6)', fontsize=12, fontweight='bold')
            axes[2, i+1].axis('off')
            # 6×6マップの各ピクセル値を数値で表示
            for y in range(6):
                for x in range(6):
                    text = axes[2, i+1].text(x, y, f'{attn_map_6x6[y, x]:.3f}',
                                            ha="center", va="center", color="white", fontsize=8, fontweight='bold')
            plt.colorbar(im, ax=axes[2, i+1])
        
        # 行4: 右腕のAttention（リサイズ前6×6）- 直接可視化
        axes[3, 0].axis('off')  # 空欄
        for i, joint_name in enumerate(joint_names):
            attn_map_6x6 = attn_right_cpu[i]  # (6, 6)
            vmin = attn_map_6x6.min()
            vmax = attn_map_6x6.max()
            # 6×6マップを直接可視化
            im = axes[3, i+1].imshow(attn_map_6x6, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
            axes[3, i+1].set_title(f'Right {joint_name} Attention (6×6)', fontsize=12, fontweight='bold')
            axes[3, i+1].axis('off')
            # 6×6マップの各ピクセル値を数値で表示
            for y in range(6):
                for x in range(6):
                    text = axes[3, i+1].text(x, y, f'{attn_map_6x6[y, x]:.3f}',
                                            ha="center", va="center", color="white", fontsize=8, fontweight='bold')
            plt.colorbar(im, ax=axes[3, i+1])
        
        # 行3: Attention適用後の特徴量（左腕のみ、例として）
        # 特徴量もリサイズが必要（featuresは6×6なので、50×50にリサイズ）
        features_resized = F.interpolate(
            features[0:1, 0:1, :, :].float(),  # (1, 1, 6, 6)
            size=(heatmap_h, heatmap_w),
            mode='bilinear',
            align_corners=False
        )[0, 0].cpu().numpy()  # (50, 50)
        
        # 元の特徴量（リサイズ後）
        im_feat = axes[2, 0].imshow(features_resized, cmap='hot', origin='lower')
        axes[2, 0].set_title('Feature Map (Channel 0)', fontsize=12, fontweight='bold')
        axes[2, 0].axis('off')
        plt.colorbar(im_feat, ax=axes[2, 0])
        
        # Attention適用後の特徴量（左腕の肩、肘、手首）
        for i, joint_name in enumerate(joint_names):
            attn_applied = features_resized * attn_left_resized[i]  # (50, 50) * (50, 50)
            im = axes[2, i+1].imshow(attn_applied, cmap='hot', origin='lower')
            axes[2, i+1].set_title(f'Left {joint_name} Features', fontsize=12, fontweight='bold')
            axes[2, i+1].axis('off')
            plt.colorbar(im, ax=axes[2, i+1])
        
        plt.suptitle(f'Sample {sample_count + 1}: Joint-Specific Attention Maps', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = output_dir / f'figure_4_attention_maps_sample_{sample_count + 1}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created attention visualization for sample {sample_count + 1}: {output_path}")
        sample_count += 1
    
    # 複数サンプルをまとめた図も作成
    if sample_count > 0:
        print(f"\n✓ Created {sample_count} attention map visualizations")
        print(f"  Output directory: {output_dir}")
    else:
        print("\n⚠️  No samples were visualized. Please check if the model uses attention.")


def main():
    parser = argparse.ArgumentParser(description='Visualize attention maps from hierarchical model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data JSONL')
    parser.add_argument('--plane', type=str, default='xz', choices=['xz', 'xy'],
                       help='Plane: xz or xy')
    parser.add_argument('--x_range', type=float, nargs=2, default=[-1.0, 1.0],
                       help='X range')
    parser.add_argument('--z_range', type=float, nargs=2, default=[-1.0, 0.75],
                       help='Z range (for xz plane)')
    parser.add_argument('--y_range', type=float, nargs=2, default=[2.0, 5.0],
                       help='Y range (for xy plane)')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of bins for heatmap')
    parser.add_argument('--feature', type=str, default='energy_power',
                       choices=['energy_power', 'velocity'],
                       help='Feature type')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/paper_figures',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # モデルを読み込む
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, device=args.device)
    
    # データセットを読み込む
    print(f"Loading dataset from {args.test_data}...")
    if args.plane == 'xz':
        dataset = ArmJointDataset(
            args.test_data,
            x_range=tuple(args.x_range),
            z_range=tuple(args.z_range),
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=True
        )
    else:
        from heatmap_distal_detection.dataset_arm_joints_xy import ArmJointDatasetXY
        dataset = ArmJointDatasetXY(
            args.test_data,
            x_range=tuple(args.x_range),
            y_range=tuple(args.y_range),
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=True
        )
    
    print(f"Loaded {len(dataset)} samples")
    
    # Attentionマップを可視化
    output_dir = Path(args.output_dir)
    visualize_attention_maps(
        model=model,
        dataset=dataset,
        num_samples=args.num_samples,
        output_dir=output_dir,
        device=args.device
    )
    
    print("\n✓ Attention map visualization completed!")


if __name__ == '__main__':
    main()

