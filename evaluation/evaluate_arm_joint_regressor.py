#!/usr/bin/env python3
"""
xzヒートマップから腕の関節座標を回帰するモデルの評価スクリプト
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
from heatmap_distal_detection.models.arm_joint_regressor import create_arm_joint_regressor
from heatmap_distal_detection.models.arm_joint_regressor_xy import create_arm_joint_regressor_xy
from heatmap_distal_detection.models.arm_joint_regressor_hierarchical import (
    create_hierarchical_arm_joint_regressor,
    create_hierarchical_arm_joint_regressor_xy
)


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


def denormalize_coords(
    normalized_coords: np.ndarray,  # (N, num_joints*2)
    x_range: Tuple[float, float],
    second_range: Tuple[float, float],
    plane: str = 'xz'
) -> np.ndarray:
    """
    正規化された座標を元の範囲に戻す
    
    Args:
        normalized_coords: 正規化された座標（0-1範囲）
        x_range: x軸の固定範囲
        second_range: 第2軸の固定範囲（z_range for xz, y_range for xy）
        plane: 'xz' or 'xy'
    
    Returns:
        coords: 元の範囲の座標（メートル単位）
    """
    coords = normalized_coords.copy()
    num_joints = coords.shape[1] // 2
    
    x_min, x_max = x_range
    second_min, second_max = second_range
    
    for i in range(num_joints):
        # x座標
        coords[:, i*2] = coords[:, i*2] * (x_max - x_min) + x_min
        # 第2軸座標（z or y）
        coords[:, i*2+1] = coords[:, i*2+1] * (second_max - second_min) + second_min
    
    return coords


def compute_spatial_mpjpe(
    pred_coords: np.ndarray,  # (N, num_joints*2) - 正規化済み
    gt_coords: np.ndarray,  # (N, num_joints*2) - 正規化済み
    mask: np.ndarray,  # (N, num_joints*2)
    x_range: Tuple[float, float],
    second_range: Tuple[float, float],
    plane: str = 'xz'
) -> Dict[str, float]:
    """
    空間座標（メートル単位）でのMPJPEを計算
    
    Args:
        pred_coords: 予測関節座標（正規化済み）
        gt_coords: GT関節座標（正規化済み）
        mask: 有効な関節のマスク
        x_range: x軸の固定範囲
        second_range: 第2軸の固定範囲（z_range for xz, y_range for xy）
        plane: 'xz' or 'xy'
    
    Returns:
        metrics: 評価指標の辞書（メートル単位）
    """
    # 正規化座標を元の範囲に戻す
    pred_coords_m = denormalize_coords(pred_coords, x_range, second_range, plane)
    gt_coords_m = denormalize_coords(gt_coords, x_range, second_range, plane)
    
    # NaNをマスク
    valid_mask = mask & ~np.isnan(gt_coords) & ~np.isnan(pred_coords)
    
    if valid_mask.sum() == 0:
        return {
            'spatial_mpjpe_cm': np.nan,
            'spatial_mpjpe_per_joint_cm': [np.nan] * 6,
            'valid_joints': 0
        }
    
    # 各関節の誤差を計算（メートル単位）
    num_joints = 6
    joint_errors = []
    
    for i in range(num_joints):
        joint_mask = valid_mask[:, i*2:(i+1)*2]  # (N, 2)
        if joint_mask.sum() > 0:
            pred_joint = pred_coords_m[:, i*2:(i+1)*2]  # (N, 2)
            gt_joint = gt_coords_m[:, i*2:(i+1)*2]  # (N, 2)
            
            # 空間座標での誤差（メートル）
            error = np.sqrt(np.sum((pred_joint - gt_joint) ** 2, axis=1))  # (N,)
            error = error[joint_mask[:, 0]]  # 有効なサンプルのみ
            
            if len(error) > 0:
                joint_errors.append(np.mean(error) * 100)  # センチメートルに変換
            else:
                joint_errors.append(np.nan)
        else:
            joint_errors.append(np.nan)
    
    # 全関節の平均誤差
    valid_errors = [e for e in joint_errors if not np.isnan(e)]
    spatial_mpjpe_cm = np.mean(valid_errors) if len(valid_errors) > 0 else np.nan
    
    return {
        'spatial_mpjpe_cm': spatial_mpjpe_cm,
        'spatial_mpjpe_per_joint_cm': joint_errors,
        'valid_joints': valid_mask.sum()
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    x_range: Tuple[float, float],
    second_range: Tuple[float, float],
    joint_names: List[str],
    plane: str = 'xz'
) -> Dict[str, float]:
    """
    モデルを評価
    
    Args:
        model: 評価するモデル
        data_loader: データローダー
        device: デバイス
        x_range: x軸の固定範囲
        second_range: 第2軸の固定範囲（z_range for xz, y_range for xy）
        joint_names: 関節名のリスト
        plane: 'xz' or 'xy'
    
    Returns:
        metrics: 評価指標の辞書
    """
    model.eval()
    
    all_pred_coords = []
    all_gt_coords = []
    all_masks = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Evaluating')
        for batch in pbar:
            # データをデバイスに移動
            heatmaps = batch['heatmap'].to(device)  # (B, 1, H, W)
            joint_coords = batch['joint_coords'].to(device)  # (B, num_joints*2)
            
            # 予測
            pred_coords = model(heatmaps)  # (B, num_joints*2)
            
            # CPUに移動してNumPy配列に変換
            pred_coords = pred_coords.cpu().numpy()
            gt_coords = joint_coords.cpu().numpy()
            mask = ~np.isnan(gt_coords)  # (B, num_joints*2)
            
            all_pred_coords.append(pred_coords)
            all_gt_coords.append(gt_coords)
            all_masks.append(mask)
    
    # 全データを結合
    all_pred_coords = np.concatenate(all_pred_coords, axis=0)  # (N, num_joints*2)
    all_gt_coords = np.concatenate(all_gt_coords, axis=0)  # (N, num_joints*2)
    all_masks = np.concatenate(all_masks, axis=0)  # (N, num_joints*2)
    
    # デバッグ: モデルの出力範囲を確認
    print("\n" + "="*60)
    print("[Debug] Model Output Analysis")
    print("="*60)
    
    # 有効な値のみで範囲を計算
    valid_pred = all_pred_coords[~np.isnan(all_pred_coords)]
    valid_gt = all_gt_coords[~np.isnan(all_gt_coords)]
    
    if len(valid_pred) > 0:
        print(f"Predicted coordinates (normalized, raw output):")
        print(f"  Shape: {all_pred_coords.shape}")
        print(f"  Min: {valid_pred.min():.6f}, Max: {valid_pred.max():.6f}")
        print(f"  Mean: {valid_pred.mean():.6f}, Std: {valid_pred.std():.6f}")
        
        # 範囲外の値の割合を確認
        out_of_range = (all_pred_coords < 0.0) | (all_pred_coords > 1.0)
        out_of_range_ratio = out_of_range.sum() / (~np.isnan(all_pred_coords)).sum() if (~np.isnan(all_pred_coords)).sum() > 0 else 0.0
        print(f"  Values outside [0, 1] range: {out_of_range_ratio*100:.2f}%")
        
        if out_of_range_ratio > 0.1:  # 10%以上が範囲外の場合
            print(f"  ⚠️  Warning: {out_of_range_ratio*100:.2f}% of values are outside [0, 1] range")
            print(f"  This is expected if the model's output layer has no sigmoid activation")
    else:
        print("  No valid predicted coordinates found")
    
    if len(valid_gt) > 0:
        print(f"\nGround truth coordinates (normalized):")
        print(f"  Shape: {all_gt_coords.shape}")
        print(f"  Min: {valid_gt.min():.6f}, Max: {valid_gt.max():.6f}")
        print(f"  Mean: {valid_gt.mean():.6f}, Std: {valid_gt.std():.6f}")
        
        # GTが正規化されているか確認
        gt_out_of_range = (all_gt_coords < 0.0) | (all_gt_coords > 1.0)
        gt_out_of_range_ratio = gt_out_of_range.sum() / (~np.isnan(all_gt_coords)).sum() if (~np.isnan(all_gt_coords)).sum() > 0 else 0.0
        if gt_out_of_range_ratio > 0.01:  # 1%以上が範囲外の場合
            print(f"  ⚠️  Warning: {gt_out_of_range_ratio*100:.2f}% of GT values are outside [0, 1] range")
        else:
            print(f"  ✓ GT coordinates are properly normalized (within [0, 1] range)")
    else:
        print("  No valid ground truth coordinates found")
    
    # 正規化解除後の範囲を確認（サンプル数が多い場合は最初の1000サンプルのみ）
    sample_size = min(1000, len(all_pred_coords))
    sample_pred = all_pred_coords[:sample_size]
    sample_gt = all_gt_coords[:sample_size]
    
    pred_denorm = denormalize_coords(sample_pred, x_range, second_range, plane)
    gt_denorm = denormalize_coords(sample_gt, x_range, second_range, plane)
    
    valid_pred_denorm = pred_denorm[~np.isnan(pred_denorm)]
    valid_gt_denorm = gt_denorm[~np.isnan(gt_denorm)]
    
    if len(valid_pred_denorm) > 0:
        print(f"\nPredicted coordinates (denormalized, first {sample_size} samples):")
        print(f"  Min: {valid_pred_denorm.min():.3f}m, Max: {valid_pred_denorm.max():.3f}m")
        print(f"  Mean: {valid_pred_denorm.mean():.3f}m, Std: {valid_pred_denorm.std():.3f}m")
    
    if len(valid_gt_denorm) > 0:
        print(f"\nGround truth coordinates (denormalized, first {sample_size} samples):")
        print(f"  Min: {valid_gt_denorm.min():.3f}m, Max: {valid_gt_denorm.max():.3f}m")
        print(f"  Mean: {valid_gt_denorm.mean():.3f}m, Std: {valid_gt_denorm.std():.3f}m")
    
    print("="*60 + "\n")
    
    # 正規化座標でのMPJPE
    norm_metrics = compute_mpjpe(all_pred_coords, all_gt_coords, all_masks)
    
    # 空間座標でのMPJPE
    spatial_metrics = compute_spatial_mpjpe(
        all_pred_coords, all_gt_coords, all_masks,
        x_range, second_range, plane
    )
    
    # 結果をまとめる
    metrics = {
        'normalized_mpjpe': norm_metrics['mpjpe'],
        'spatial_mpjpe_cm': spatial_metrics['spatial_mpjpe_cm'],
        'valid_joints': norm_metrics['valid_joints'],
        'total_samples': len(all_pred_coords)
    }
    
    # 関節ごとの誤差
    for i, joint_name in enumerate(joint_names):
        metrics[f'{joint_name}_normalized_mpjpe'] = norm_metrics['mpjpe_per_joint'][i]
        metrics[f'{joint_name}_spatial_mpjpe_cm'] = spatial_metrics['spatial_mpjpe_per_joint_cm'][i]
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate arm joint regressor from heatmap')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Test data path (JSONL)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for evaluation results (JSON)')
    parser.add_argument('--plane', type=str, default=None,
                       choices=['xz', 'xy'],
                       help='Plane to use: xz or xy (default: auto-detect from checkpoint)')
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
    parser.add_argument('--y_range', type=float, nargs=2, default=[3.5, 5.0],
                       metavar=('Y_MIN', 'Y_MAX'),
                       help='Fixed y-axis range (for xy plane, default: 3.5 5.0)')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of bins for heatmap')
    parser.add_argument('--feature', type=str, default='energy_power',
                       choices=['velocity', 'energy_power'],
                       help='Feature to use for heatmap')
    parser.add_argument('--base_channels', type=int, default=32,
                       help='Base number of channels')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    args = parser.parse_args()
    
    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # モデルを読み込み
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 設定を取得
    config = checkpoint.get('config', {})
    
    # 平面を決定（チェックポイントから取得、なければ引数から、なければデフォルトでxz）
    plane = args.plane or config.get('plane', 'xz')
    print(f"Using plane: {plane}")
    
    x_range = tuple(config.get('x_range', args.x_range))
    if plane == 'xz':
        second_range = tuple(config.get('z_range', args.z_range))
        range_key = 'z_range'
    else:  # xy
        second_range = tuple(config.get('y_range', args.y_range))
        range_key = 'y_range'
    
    bins = config.get('bins', args.bins)
    feature = config.get('feature', args.feature)
    base_channels = config.get('base_channels', args.base_channels)
    dropout = config.get('dropout', args.dropout)
    model_type = config.get('model_type', 'standard')
    use_attention = config.get('use_attention', False)
    normalize_heatmap = config.get('normalize_heatmap', False)  # チェックポイントから読み込み
    
    print(f"  x_range: {x_range}")
    print(f"  {range_key}: {second_range}")
    print(f"  bins: {bins}")
    print(f"  feature: {feature}")
    print(f"  base_channels: {base_channels}")
    print(f"  dropout: {dropout}")
    print(f"  model_type: {model_type}")
    print(f"  use_attention: {use_attention}")
    print(f"  normalize_heatmap: {normalize_heatmap}")
    
    # モデルを作成（階層的モデルはxz/xy両方でサポート）
    if model_type == 'hierarchical':
        if plane == 'xz':
            model = create_hierarchical_arm_joint_regressor(
                heatmap_size=bins,
                num_joints_per_arm=3,
                base_channels=base_channels,
                dropout=dropout,
                use_attention=use_attention
            )
        else:  # xy
            model = create_hierarchical_arm_joint_regressor_xy(
                heatmap_size=bins,
                num_joints_per_arm=3,
                base_channels=base_channels,
                dropout=dropout,
                use_attention=use_attention
            )
    else:
        if plane == 'xz':
            create_model_fn = create_arm_joint_regressor
        else:  # xy
            create_model_fn = create_arm_joint_regressor_xy
        model = create_model_fn(
            heatmap_size=bins,
            num_joints=6,
            base_channels=base_channels,
            dropout=dropout
        )
    
    # モデルの重みを読み込み
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # データセット
    if plane == 'xz':
        test_dataset = ArmJointDataset(
            data_path=args.test_data,
            x_range=x_range,
            z_range=second_range,
            bins=bins,
            feature=feature,
            normalize_heatmap=normalize_heatmap
        )
    else:  # xy
        test_dataset = ArmJointDatasetXY(
            data_path=args.test_data,
            x_range=x_range,
            y_range=second_range,
            bins=bins,
            feature=feature,
            normalize_heatmap=normalize_heatmap
        )
    
    # データローダー
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 関節名
    joint_names = test_dataset.arm_joint_names
    
    # 評価
    print("\nEvaluating model...")
    metrics = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        x_range=x_range,
        second_range=second_range,
        joint_names=joint_names,
        plane=plane
    )
    
    # 結果を表示
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid joints: {metrics['valid_joints']}")
    print(f"\nNormalized MPJPE: {metrics['normalized_mpjpe']:.6f}")
    print(f"Spatial MPJPE: {metrics['spatial_mpjpe_cm']:.2f} cm")
    print(f"\nPer-joint errors (spatial, cm):")
    for joint_name in joint_names:
        key = f'{joint_name}_spatial_mpjpe_cm'
        if key in metrics:
            error = metrics[key]
            if not np.isnan(error):
                print(f"  {joint_name:15s}: {error:6.2f} cm")
            else:
                print(f"  {joint_name:15s}: N/A")
    print("="*60)
    
    # 結果を保存
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # NumPy型をPython型に変換（JSONシリアライズのため）
        def convert_to_json_serializable(obj):
            """NumPy型をPython型に変換"""
            if isinstance(obj, (np.float32, np.float64)):
                if np.isnan(obj):
                    return None
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            elif isinstance(obj, np.ndarray):
                return [convert_to_json_serializable(x) for x in obj.tolist()]
            elif isinstance(obj, list):
                return [convert_to_json_serializable(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            else:
                return obj
        
        metrics_json = {k: convert_to_json_serializable(v) for k, v in metrics.items()}
        
        with open(output_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

