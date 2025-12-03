#!/usr/bin/env python3
"""
Vision Transformer (ViT) を使用した腕関節回帰モデルの評価スクリプト
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heatmap_distal_detection.dataset_arm_joints import ArmJointDataset, collate_fn
from heatmap_distal_detection.dataset_arm_joints_xy import ArmJointDatasetXY
from heatmap_distal_detection.models.vit_arm_joint_regressor import create_vit_arm_joint_regressor


def compute_mpjpe(
    pred_coords: np.ndarray,  # (N, num_joints*2)
    gt_coords: np.ndarray,  # (N, num_joints*2)
    mask: np.ndarray  # (N, num_joints*2) - True if valid
) -> Dict[str, float]:
    """
    Mean Per Joint Position Error (MPJPE)を計算
    
    Args:
        pred_coords: 予測関節座標（正規化済み、0-1範囲）
        gt_coords: GT関節座標（正規化済み、0-1範囲）
        mask: 有効な関節のマスク
    
    Returns:
        metrics: 評価指標の辞書
    """
    # NaNをマスク
    valid_mask = mask & ~np.isnan(gt_coords) & ~np.isnan(pred_coords)
    
    if valid_mask.sum() == 0:
        return {
            'mpjpe': np.nan,
            'mpjpe_per_joint': [np.nan] * 6,
            'valid_joints': 0
        }
    
    # 各関節の誤差を計算（正規化座標での誤差）
    num_joints = 6
    joint_errors = []
    joint_names = ['L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']
    
    for i in range(num_joints):
        joint_mask = valid_mask[:, i*2:(i+1)*2]  # (N, 2)
        if joint_mask.sum() > 0:
            pred_joint = pred_coords[:, i*2:(i+1)*2]  # (N, 2)
            gt_joint = gt_coords[:, i*2:(i+1)*2]  # (N, 2)
            
            # 正規化座標での誤差（0-1範囲）
            error = np.sqrt(np.sum((pred_joint - gt_joint) ** 2, axis=1))  # (N,)
            error = error[joint_mask[:, 0]]  # 有効なサンプルのみ
            
            if len(error) > 0:
                joint_errors.append(np.mean(error))
            else:
                joint_errors.append(np.nan)
        else:
            joint_errors.append(np.nan)
    
    # 全関節の平均誤差
    valid_errors = [e for e in joint_errors if not np.isnan(e)]
    mpjpe = np.mean(valid_errors) if len(valid_errors) > 0 else np.nan
    
    return {
        'mpjpe': mpjpe,
        'mpjpe_per_joint': joint_errors,
        'valid_joints': valid_mask.sum()
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    coord_ranges: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    """
    モデルを評価
    
    Args:
        model: 評価するモデル
        dataloader: データローダー
        device: デバイス
        coord_ranges: 座標範囲の辞書（正規化解除用）
    
    Returns:
        metrics: 評価指標の辞書
    """
    model.eval()
    
    all_pred = []
    all_gt = []
    all_mask = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            heatmaps = batch['heatmap'].to(device)  # (B, 1, H, W)
            gt_coords = batch['joint_coords'].numpy()  # (B, num_joints*2)
            
            # 予測
            pred_coords = model(heatmaps).cpu().numpy()  # (B, num_joints*2)
            
            # マスク
            mask = ~np.isnan(gt_coords)  # (B, num_joints*2)
            
            all_pred.append(pred_coords)
            all_gt.append(gt_coords)
            all_mask.append(mask)
    
    # 結合
    all_pred = np.concatenate(all_pred, axis=0)  # (N, num_joints*2)
    all_gt = np.concatenate(all_gt, axis=0)  # (N, num_joints*2)
    all_mask = np.concatenate(all_mask, axis=0)  # (N, num_joints*2)
    
    # MPJPEを計算（正規化座標での誤差）
    metrics = compute_mpjpe(all_pred, all_gt, all_mask)
    
    # 正規化解除して空間座標での誤差も計算
    joint_names = ['L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']
    
    # 座標範囲を取得
    x_range = coord_ranges.get('x_range', (-1.0, 1.0))
    z_or_y_range = coord_ranges.get('z_range') or coord_ranges.get('y_range', (2.0, 5.0))
    
    # 正規化解除
    def denormalize_coord(normalized: float, coord_range: Tuple[float, float]) -> float:
        min_val, max_val = coord_range
        return normalized * (max_val - min_val) + min_val
    
    # 空間座標に変換
    pred_spatial = np.zeros_like(all_pred)
    gt_spatial = np.zeros_like(all_gt)
    
    for i in range(6):  # 6関節
        # x座標
        pred_spatial[:, i*2] = denormalize_coord(all_pred[:, i*2], x_range)
        gt_spatial[:, i*2] = denormalize_coord(all_gt[:, i*2], x_range)
        # z/y座標
        pred_spatial[:, i*2+1] = denormalize_coord(all_pred[:, i*2+1], z_or_y_range)
        gt_spatial[:, i*2+1] = denormalize_coord(all_gt[:, i*2+1], z_or_y_range)
    
    # 空間座標でのMPJPE（メートル単位）
    spatial_errors = []
    for i in range(6):
        joint_mask = all_mask[:, i*2:(i+1)*2]
        if joint_mask.sum() > 0:
            pred_joint = pred_spatial[:, i*2:(i+1)*2]
            gt_joint = gt_spatial[:, i*2:(i+1)*2]
            error = np.sqrt(np.sum((pred_joint - gt_joint) ** 2, axis=1))
            error = error[joint_mask[:, 0]]
            if len(error) > 0:
                spatial_errors.append(np.mean(error) * 100)  # メートル→センチメートル
            else:
                spatial_errors.append(np.nan)
        else:
            spatial_errors.append(np.nan)
    
    metrics['mpjpe_spatial_cm'] = np.nanmean(spatial_errors)
    metrics['mpjpe_per_joint_spatial_cm'] = spatial_errors
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate ViT arm joint regressor')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data JSONL file')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save evaluation results (JSON)')
    parser.add_argument('--plane', type=str, default='xz',
                       choices=['xz', 'xy'],
                       help='Plane: xz or xy')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--x_range', type=float, nargs=2, default=[-1.0, 1.0],
                       metavar=('X_MIN', 'X_MAX'),
                       help='Fixed x-axis range')
    parser.add_argument('--z_range', type=float, nargs=2, default=[-1.0, 0.75],
                       metavar=('Z_MIN', 'Z_MAX'),
                       help='Fixed z-axis range (for xz plane)')
    parser.add_argument('--y_range', type=float, nargs=2, default=[2.0, 5.0],
                       metavar=('Y_MIN', 'Y_MAX'),
                       help='Fixed y-axis range (for xy plane)')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of bins for heatmap')
    parser.add_argument('--feature', type=str, default='energy_power',
                       choices=['velocity', 'energy_power'],
                       help='Feature type')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading model from {args.model_path}...")
    
    # モデルを読み込み
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model_size = config.get('model_size', 'small')
    print(f"Model size: {model_size}")
    
    # モデルを作成
    model = create_vit_arm_joint_regressor(
        img_size=config.get('bins', args.bins),
        patch_size=5,
        num_joints=6,
        model_size=model_size,
        drop_rate=config.get('drop_rate', 0.0),
        attn_drop_rate=config.get('attn_drop_rate', 0.0),
        drop_path_rate=config.get('drop_path_rate', 0.0)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # データセットを読み込み
    print(f"Loading dataset from {args.test_data}...")
    x_range = tuple(args.x_range)
    
    if args.plane == 'xz':
        z_range = tuple(args.z_range)
        dataset = ArmJointDataset(
            data_path=args.test_data,
            x_range=x_range,
            z_range=z_range,
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=config.get('normalize_heatmap', False)
        )
        coord_ranges = {'x_range': x_range, 'z_range': z_range}
    else:  # xy
        y_range = tuple(args.y_range)
        dataset = ArmJointDatasetXY(
            data_path=args.test_data,
            x_range=x_range,
            y_range=y_range,
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=config.get('normalize_heatmap', False)
        )
        coord_ranges = {'x_range': x_range, 'y_range': y_range}
    
    print(f"Loaded {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 評価
    print("Evaluating model...")
    metrics = evaluate_model(model, dataloader, device, coord_ranges)
    
    # 結果を表示
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Overall MPJPE (normalized): {metrics['mpjpe']:.6f}")
    print(f"Overall MPJPE (spatial): {metrics['mpjpe_spatial_cm']:.2f} cm")
    print(f"Valid joints: {metrics['valid_joints']}")
    print("\nPer-joint MPJPE (normalized):")
    joint_names = ['L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']
    for name, error in zip(joint_names, metrics['mpjpe_per_joint']):
        print(f"  {name}: {error:.6f}")
    print("\nPer-joint MPJPE (spatial, cm):")
    for name, error in zip(joint_names, metrics['mpjpe_per_joint_spatial_cm']):
        print(f"  {name}: {error:.2f} cm")
    
    # 結果を保存
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # NumPy型をPython型に変換
        results = {
            'overall_mpjpe': float(metrics['mpjpe']),
            'overall_mpjpe_spatial_cm': float(metrics['mpjpe_spatial_cm']),
            'valid_joints': int(metrics['valid_joints']),
            'per_joint_mpjpe': [float(e) if not np.isnan(e) else None for e in metrics['mpjpe_per_joint']],
            'per_joint_mpjpe_spatial_cm': [float(e) if not np.isnan(e) else None for e in metrics['mpjpe_per_joint_spatial_cm']],
            'joint_names': joint_names,
            'model_size': model_size,
            'plane': args.plane
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

