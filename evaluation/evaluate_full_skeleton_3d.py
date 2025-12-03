#!/usr/bin/env python3
"""
xz平面モデルとxy平面モデルを組み合わせて全骨格（22関節）の3次元評価を行うスクリプト

xz平面モデルから (x, z) を取得
xy平面モデルから (x, y) を取得（y座標のみ使用）
これらを組み合わせて (x, y, z) の3次元座標を作成し、3次元MPJPEを計算
"""

import torch
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

from heatmap_distal_detection.dataset_full_skeleton import FullSkeletonDataset, collate_fn
from heatmap_distal_detection.dataset_full_skeleton_xy import FullSkeletonDatasetXY
from heatmap_distal_detection.models.full_skeleton_regressor import create_full_skeleton_regressor
from heatmap_distal_detection.models.full_skeleton_regressor_xy import create_full_skeleton_regressor_xy


def load_3d_gt_joints(
    data_path: str,
    x_range_xz: Tuple[float, float],
    z_range: Tuple[float, float],
    x_range_xy: Tuple[float, float],
    y_range: Tuple[float, float],
    joint_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JSONLファイルから3次元GT関節座標を読み込む（全22関節）
    
    Args:
        data_path: JSONLファイルのパス
        x_range_xz: xz平面のx軸範囲
        z_range: z軸範囲
        x_range_xy: xy平面のx軸範囲
        y_range: y軸範囲
        joint_names: 関節名のリスト（22関節）
    
    Returns:
        gt_joints_3d: (N, 22, 3) - 3次元GT関節座標
        valid_mask: (N, 22) - 有効な関節のマスク
    """
    gt_joints_3d_list = []
    valid_mask_list = []
    
    num_joints = len(joint_names)  # 22
    
    num_samples_with_gt = 0
    
    with open(data_path, 'r') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                sample = json.loads(line)
                
                # gt_jointsから全22関節を取得
                gt_joints = sample.get('gt_joints', None)
                
                if gt_joints is not None and len(gt_joints) > 0:
                    gt_joints = np.array(gt_joints)  # (22, 3)
                    if len(gt_joints) >= num_joints:
                        joints_3d = gt_joints[:num_joints]  # (22, 3)
                        valid = ~np.isnan(joints_3d).any(axis=1)  # (22,)
                        num_samples_with_gt += 1
                    else:
                        # 関節数が不足している場合はNaNで埋める
                        joints_3d = np.full((num_joints, 3), np.nan, dtype=np.float32)
                        joints_3d[:len(gt_joints)] = gt_joints
                        valid = ~np.isnan(joints_3d).any(axis=1)  # (22,)
                        num_samples_with_gt += 1
                else:
                    # GT関節がない場合はNaN
                    joints_3d = np.full((num_joints, 3), np.nan, dtype=np.float32)
                    valid = np.zeros(num_joints, dtype=bool)
                
                gt_joints_3d_list.append(joints_3d)
                valid_mask_list.append(valid)
                
                # デバッグ: 最初の数サンプルで確認
                if line_idx < 3:
                    print(f"[Debug] Sample {line_idx}:")
                    print(f"  gt_joints present: {'gt_joints' in sample}")
                    print(f"  joints_3d shape: {joints_3d.shape}")
                    print(f"  valid joints: {valid.sum()}/{len(valid)}")
                    if valid.sum() > 0:
                        print(f"  First valid joint: {joints_3d[valid][0] if valid.any() else 'N/A'}")
    
    gt_joints_3d = np.array(gt_joints_3d_list)  # (N, 22, 3)
    valid_mask = np.array(valid_mask_list)  # (N, 22)
    
    print(f"\n[Debug] Loaded {len(gt_joints_3d)} samples")
    print(f"  Samples with gt_joints: {num_samples_with_gt}")
    print(f"  Total valid joints: {valid_mask.sum()}/{valid_mask.size}")
    
    return gt_joints_3d, valid_mask


def denormalize_coords_2d(
    normalized_coords: np.ndarray,  # (N, num_joints*2)
    x_range: Tuple[float, float],
    second_range: Tuple[float, float]
) -> np.ndarray:
    """
    正規化された2次元座標を元の範囲に戻す
    
    Args:
        normalized_coords: 正規化された座標（0-1範囲）
        x_range: x軸の固定範囲
        second_range: 第2軸の固定範囲
    
    Returns:
        coords: 元の範囲の座標 (N, num_joints*2)
    """
    coords = normalized_coords.copy()
    num_joints = coords.shape[1] // 2
    
    x_min, x_max = x_range
    second_min, second_max = second_range
    
    for i in range(num_joints):
        # x座標
        coords[:, i*2] = coords[:, i*2] * (x_max - x_min) + x_min
        # 第2軸座標
        coords[:, i*2+1] = coords[:, i*2+1] * (second_max - second_min) + second_min
    
    return coords


def combine_2d_predictions_to_3d(
    pred_xz: np.ndarray,  # (N, num_joints*2) - xz平面モデルの予測 [x, z, x, z, ...]
    pred_xy: np.ndarray,  # (N, num_joints*2) - xy平面モデルの予測 [x, y, x, y, ...]
    x_range_xz: Tuple[float, float],
    z_range: Tuple[float, float],
    x_range_xy: Tuple[float, float],
    y_range: Tuple[float, float]
) -> np.ndarray:
    """
    xz平面モデルとxy平面モデルの予測を組み合わせて3次元座標を作成
    
    Args:
        pred_xz: xz平面モデルの予測（正規化済み）
        pred_xy: xy平面モデルの予測（正規化済み）
        x_range_xz: xz平面のx軸範囲
        z_range: z軸範囲
        x_range_xy: xy平面のx軸範囲
        y_range: y軸範囲
    
    Returns:
        pred_3d: (N, num_joints, 3) - 3次元予測座標
    """
    num_samples = pred_xz.shape[0]
    num_joints = pred_xz.shape[1] // 2
    
    # xzモデル: 正規化済み（0-1範囲）と仮定 → クリップして正規化解除
    pred_xz_clipped = np.clip(pred_xz, 0.0, 1.0)
    pred_xz_denorm = denormalize_coords_2d(pred_xz_clipped, x_range_xz, z_range)  # (N, num_joints*2)
    
    # xyモデル: 正規化済み（0-1範囲）と仮定 → クリップせずに正規化解除
    pred_xy_denorm = denormalize_coords_2d(pred_xy, x_range_xy, y_range)  # (N, num_joints*2)
    
    # 3次元座標を作成
    pred_3d = np.zeros((num_samples, num_joints, 3), dtype=np.float32)
    
    for i in range(num_joints):
        # x座標: xzモデルから取得（xzモデルのx座標を使用）
        pred_3d[:, i, 0] = pred_xz_denorm[:, i*2]
        
        # y座標: xyモデルから取得
        pred_3d[:, i, 1] = pred_xy_denorm[:, i*2+1]
        
        # z座標: xzモデルから取得
        pred_3d[:, i, 2] = pred_xz_denorm[:, i*2+1]
    
    return pred_3d


def compute_3d_mpjpe(
    pred_3d: np.ndarray,  # (N, num_joints, 3)
    gt_3d: np.ndarray,  # (N, num_joints, 3)
    valid_mask: np.ndarray,  # (N, num_joints)
    joint_names: List[str]
) -> Dict[str, float]:
    """
    3次元MPJPEを計算（全22関節）
    
    Args:
        pred_3d: 予測3次元座標
        gt_3d: GT 3次元座標
        valid_mask: 有効な関節のマスク
        joint_names: 関節名のリスト
    
    Returns:
        metrics: 評価指標の辞書
    """
    num_joints = pred_3d.shape[1]
    
    # 各関節の誤差を計算（cm単位）
    joint_errors = []
    
    for i in range(num_joints):
        joint_mask = valid_mask[:, i]  # (N,)
        
        if joint_mask.sum() > 0:
            pred_joint = pred_3d[joint_mask, i, :]  # (M, 3)
            gt_joint = gt_3d[joint_mask, i, :]  # (M, 3)
            
            # 3次元ユークリッド距離（cm）
            error = np.sqrt(np.sum((pred_joint - gt_joint) ** 2, axis=1))  # (M,)
            error_cm = error * 100  # メートルからcmに変換
            
            joint_errors.append(np.mean(error_cm))
        else:
            joint_errors.append(np.nan)
    
    # 全関節の平均誤差
    valid_errors = [e for e in joint_errors if not np.isnan(e)]
    mpjpe_3d_cm = np.mean(valid_errors) if len(valid_errors) > 0 else np.nan
    
    # 結果をまとめる
    metrics = {
        'mpjpe_3d_cm': mpjpe_3d_cm,
        'valid_joints': valid_mask.sum(),
        'total_samples': pred_3d.shape[0]
    }
    
    # 関節ごとの誤差
    for i, joint_name in enumerate(joint_names):
        if i < len(joint_errors):
            metrics[f'{joint_name}_mpjpe_3d_cm'] = joint_errors[i]
    
    return metrics


def evaluate_models_3d(
    model_xz,
    model_xy,
    data_loader_xz,
    data_loader_xy,
    device,
    x_range_xz: Tuple[float, float],
    z_range: Tuple[float, float],
    x_range_xy: Tuple[float, float],
    y_range: Tuple[float, float],
    joint_names: List[str]
) -> Dict[str, float]:
    """
    両方のモデルで予測を取得し、3次元評価を実行（全22関節）
    
    Args:
        model_xz: xz平面モデル
        model_xy: xy平面モデル
        data_loader_xz: xz平面データローダー
        data_loader_xy: xy平面データローダー
        device: デバイス
        x_range_xz: xz平面のx軸範囲
        z_range: z軸範囲
        x_range_xy: xy平面のx軸範囲
        y_range: y軸範囲
        joint_names: 関節名のリスト（22関節）
    
    Returns:
        metrics: 評価指標の辞書
    """
    model_xz.eval()
    model_xy.eval()
    
    num_joints = len(joint_names)  # 22
    
    all_pred_xz = []
    all_pred_xy = []
    all_gt_3d = []
    all_valid_mask = []
    
    # xzモデルで予測とGT座標を取得
    with torch.no_grad():
        pbar = tqdm(data_loader_xz, desc='Evaluating xz model')
        for batch in pbar:
            heatmaps = batch['heatmap'].to(device)  # (B, 1, H, W)
            pred_coords = model_xz(heatmaps)  # (B, num_joints*2)
            pred_coords = pred_coords.cpu().numpy()
            all_pred_xz.append(pred_coords)
            
            # GT 3次元座標を取得（metadataから）
            metadata_list = batch.get('metadata', [])
            for metadata in metadata_list:
                gt_joints_3d_sample = None
                
                # gt_jointsから全22関節を取得
                if 'gt_joints' in metadata and metadata['gt_joints'] is not None:
                    gt_joints = np.array(metadata['gt_joints'])  # (22, 3)
                    if len(gt_joints) >= num_joints:
                        gt_joints_3d_sample = gt_joints[:num_joints]  # (22, 3)
                
                if gt_joints_3d_sample is not None and gt_joints_3d_sample.shape == (num_joints, 3):
                    all_gt_3d.append(gt_joints_3d_sample)
                    valid = ~np.isnan(gt_joints_3d_sample).any(axis=1)  # (22,)
                    all_valid_mask.append(valid)
                else:
                    # GTがない場合はNaN
                    all_gt_3d.append(np.full((num_joints, 3), np.nan, dtype=np.float32))
                    all_valid_mask.append(np.zeros(num_joints, dtype=bool))
    
    # xyモデルで予測を取得
    with torch.no_grad():
        pbar = tqdm(data_loader_xy, desc='Evaluating xy model')
        for batch in pbar:
            heatmaps = batch['heatmap'].to(device)  # (B, 1, H, W)
            pred_coords = model_xy(heatmaps)  # (B, num_joints*2)
            pred_coords = pred_coords.cpu().numpy()
            all_pred_xy.append(pred_coords)
    
    # 全データを結合
    all_pred_xz = np.concatenate(all_pred_xz, axis=0)  # (N, num_joints*2)
    all_pred_xy = np.concatenate(all_pred_xy, axis=0)  # (N, num_joints*2)
    gt_3d = np.array(all_gt_3d)  # (N, 22, 3)
    valid_mask = np.array(all_valid_mask)  # (N, 22)
    
    # データ数の一致を確認
    if len(all_pred_xz) != len(all_pred_xy) or len(all_pred_xz) != len(gt_3d):
        print(f"\n⚠️  Warning: Data size mismatch!")
        print(f"  pred_xz: {len(all_pred_xz)}")
        print(f"  pred_xy: {len(all_pred_xy)}")
        print(f"  gt_3d: {len(gt_3d)}")
        min_size = min(len(all_pred_xz), len(all_pred_xy), len(gt_3d))
        print(f"  Using minimum size: {min_size}")
        all_pred_xz = all_pred_xz[:min_size]
        all_pred_xy = all_pred_xy[:min_size]
        gt_3d = gt_3d[:min_size]
        valid_mask = valid_mask[:min_size]
    
    # デバッグ: モデルの出力範囲を確認
    print("\n" + "="*60)
    print("[Debug] Model Output Analysis (before combining)")
    print("="*60)
    
    # xzモデルの出力範囲
    valid_pred_xz = all_pred_xz[~np.isnan(all_pred_xz)]
    if len(valid_pred_xz) > 0:
        print(f"xz model predictions (normalized, raw output):")
        print(f"  Shape: {all_pred_xz.shape}")
        print(f"  Min: {valid_pred_xz.min():.6f}, Max: {valid_pred_xz.max():.6f}")
        print(f"  Mean: {valid_pred_xz.mean():.6f}, Std: {valid_pred_xz.std():.6f}")
        
        # 範囲外の値の割合を確認
        out_of_range_xz = (all_pred_xz < 0.0) | (all_pred_xz > 1.0)
        out_of_range_ratio_xz = out_of_range_xz.sum() / (~np.isnan(all_pred_xz)).sum() if (~np.isnan(all_pred_xz)).sum() > 0 else 0.0
        print(f"  Values outside [0, 1] range: {out_of_range_ratio_xz*100:.2f}%")
    
    # xyモデルの出力範囲
    valid_pred_xy = all_pred_xy[~np.isnan(all_pred_xy)]
    if len(valid_pred_xy) > 0:
        print(f"\nxy model predictions (normalized, raw output):")
        print(f"  Shape: {all_pred_xy.shape}")
        print(f"  Min: {valid_pred_xy.min():.6f}, Max: {valid_pred_xy.max():.6f}")
        print(f"  Mean: {valid_pred_xy.mean():.6f}, Std: {valid_pred_xy.std():.6f}")
        
        # 範囲外の値の割合を確認
        out_of_range_xy = (all_pred_xy < 0.0) | (all_pred_xy > 1.0)
        out_of_range_ratio_xy = out_of_range_xy.sum() / (~np.isnan(all_pred_xy)).sum() if (~np.isnan(all_pred_xy)).sum() > 0 else 0.0
        print(f"  Values outside [0, 1] range: {out_of_range_ratio_xy*100:.2f}%")
    
    print("="*60 + "\n")
    
    # 3次元予測座標を作成
    pred_3d = combine_2d_predictions_to_3d(
        all_pred_xz, all_pred_xy,
        x_range_xz, z_range,
        x_range_xy, y_range
    )  # (N, 22, 3)
    
    # 3次元MPJPEを計算
    metrics = compute_3d_mpjpe(pred_3d, gt_3d, valid_mask, joint_names)
    
    # 予測とGTの3D座標を保存（可視化用）
    metrics['pred_3d'] = pred_3d.tolist()  # (N, 22, 3)
    metrics['gt_3d'] = gt_3d.tolist()  # (N, 22, 3)
    metrics['valid_mask'] = valid_mask.tolist()  # (N, 22)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate full skeleton (22 joints) regressor in 3D by combining xz and xy models')
    parser.add_argument('--model_path_xz', type=str, required=True,
                       help='Path to xz plane model checkpoint')
    parser.add_argument('--model_path_xy', type=str, required=True,
                       help='Path to xy plane model checkpoint')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Test data path (JSONL) - used for both xz and xy if --test_data_xz and --test_data_xy are not specified')
    parser.add_argument('--test_data_xz', type=str, default=None,
                       help='Test data path for xz plane (JSONL) - overrides --test_data for xz')
    parser.add_argument('--test_data_xy', type=str, default=None,
                       help='Test data path for xy plane (JSONL) - overrides --test_data for xy')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for evaluation results (JSON)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # xz平面の範囲
    parser.add_argument('--x_range_xz', type=float, nargs=2, default=[-1.0, 1.0],
                       metavar=('X_MIN', 'X_MAX'),
                       help='Fixed x-axis range for xz plane')
    parser.add_argument('--z_range', type=float, nargs=2, default=[-1.0, 0.75],
                       metavar=('Z_MIN', 'Z_MAX'),
                       help='Fixed z-axis range')
    
    # xy平面の範囲
    parser.add_argument('--x_range_xy', type=float, nargs=2, default=[-1.0, 1.0],
                       metavar=('X_MIN', 'X_MAX'),
                       help='Fixed x-axis range for xy plane')
    parser.add_argument('--y_range', type=float, nargs=2, default=[2.0, 5.0],
                       metavar=('Y_MIN', 'Y_MAX'),
                       help='Fixed y-axis range')
    
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of bins for heatmap')
    parser.add_argument('--feature', type=str, default='energy_power',
                       choices=['velocity', 'energy_power'],
                       help='Feature to use for heatmap')
    parser.add_argument('--base_channels', type=int, default=32,
                       help='Base number of channels')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--normalize_heatmap', action='store_true',
                       help='Normalize heatmap to 0-1 range')
    
    args = parser.parse_args()
    
    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # xz平面モデルを読み込み
    print(f"\nLoading xz plane model from {args.model_path_xz}")
    checkpoint_xz = torch.load(args.model_path_xz, map_location=device)
    config_xz = checkpoint_xz.get('config', {})
    
    normalize_heatmap_xz = config_xz.get('normalize_heatmap', args.normalize_heatmap)
    bins_xz = config_xz.get('bins', args.bins)
    base_channels_xz = config_xz.get('base_channels', args.base_channels)
    dropout_xz = config_xz.get('dropout', args.dropout)
    num_joints_xz = config_xz.get('num_joints', 22)
    
    model_xz = create_full_skeleton_regressor(
        heatmap_size=bins_xz,
        num_joints=num_joints_xz,
        base_channels=base_channels_xz,
        dropout=dropout_xz
    )
    
    model_xz.load_state_dict(checkpoint_xz['model_state_dict'])
    model_xz = model_xz.to(device)
    
    # xy平面モデルを読み込み
    print(f"\nLoading xy plane model from {args.model_path_xy}")
    checkpoint_xy = torch.load(args.model_path_xy, map_location=device)
    config_xy = checkpoint_xy.get('config', {})
    
    normalize_heatmap_xy = config_xy.get('normalize_heatmap', args.normalize_heatmap)
    bins_xy = config_xy.get('bins', args.bins)
    base_channels_xy = config_xy.get('base_channels', args.base_channels)
    dropout_xy = config_xy.get('dropout', args.dropout)
    num_joints_xy = config_xy.get('num_joints', 22)
    
    model_xy = create_full_skeleton_regressor_xy(
        heatmap_size=bins_xy,
        num_joints=num_joints_xy,
        base_channels=base_channels_xy,
        dropout=dropout_xy
    )
    
    model_xy.load_state_dict(checkpoint_xy['model_state_dict'])
    model_xy = model_xy.to(device)
    
    # テストデータパスを決定
    if args.test_data_xz is not None:
        test_data_xz = args.test_data_xz
    elif args.test_data is not None:
        test_data_xz = args.test_data
    else:
        raise ValueError("Either --test_data or --test_data_xz must be specified")
    
    if args.test_data_xy is not None:
        test_data_xy = args.test_data_xy
    elif args.test_data is not None:
        test_data_xy = args.test_data
    else:
        raise ValueError("Either --test_data or --test_data_xy must be specified")
    
    print(f"\nUsing test data:")
    print(f"  xz plane: {test_data_xz}")
    print(f"  xy plane: {test_data_xy}")
    print(f"\nNote: Datasets will load point clouds from 'file_path' in JSONL files")
    print(f"      and generate heatmaps dynamically (xz/xy planes respectively).")
    
    # データセットを作成
    x_range_xz = tuple(args.x_range_xz)
    z_range = tuple(args.z_range)
    x_range_xy = tuple(args.x_range_xy)
    y_range = tuple(args.y_range)
    
    dataset_xz = FullSkeletonDataset(
        data_path=test_data_xz,
        x_range=x_range_xz,
        z_range=z_range,
        bins=args.bins,
        feature=args.feature,
        normalize_heatmap=normalize_heatmap_xz
    )
    
    dataset_xy = FullSkeletonDatasetXY(
        data_path=test_data_xy,
        x_range=x_range_xy,
        y_range=y_range,
        bins=args.bins,
        feature=args.feature,
        normalize_heatmap=normalize_heatmap_xy
    )
    
    # 関節名を取得
    joint_names = dataset_xz.joint_names
    
    # データセットのサイズが一致するか確認
    if len(dataset_xz) != len(dataset_xy):
        print(f"\nWarning: Dataset sizes do not match!")
        print(f"  xz dataset: {len(dataset_xz)} samples")
        print(f"  xy dataset: {len(dataset_xy)} samples")
        print(f"  Using minimum size: {min(len(dataset_xz), len(dataset_xy))} samples")
    
    data_loader_xz = DataLoader(
        dataset_xz,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    data_loader_xy = DataLoader(
        dataset_xy,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 3次元評価を実行
    print("\nEvaluating in 3D...")
    metrics = evaluate_models_3d(
        model_xz, model_xy,
        data_loader_xz, data_loader_xy,
        device,
        x_range_xz, z_range,
        x_range_xy, y_range,
        joint_names
    )
    
    # 結果を表示
    print("\n" + "="*60)
    print("3D Evaluation Results (Full Skeleton - 22 Joints)")
    print("="*60)
    print(f"MPJPE (3D): {metrics['mpjpe_3d_cm']:.2f} cm")
    print(f"Valid joints: {metrics['valid_joints']}")
    print(f"Total samples: {metrics['total_samples']}")
    print("\nPer-joint MPJPE (3D):")
    for joint_name in joint_names:
        key = f'{joint_name}_mpjpe_3d_cm'
        if key in metrics:
            value = metrics[key]
            if not np.isnan(value):
                print(f"  {joint_name:15s}: {value:6.2f} cm")
            else:
                print(f"  {joint_name:15s}: N/A")
    
    # 結果を保存
    if args.output_path:
        # NumPy型をJSON serializableに変換
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            return obj
        
        metrics_json = convert_to_json_serializable(metrics)
        
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"\nResults saved to {args.output_path}")


if __name__ == '__main__':
    main()

