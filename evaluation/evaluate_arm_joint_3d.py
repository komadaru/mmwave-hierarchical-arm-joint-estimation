#!/usr/bin/env python3
"""
Vision Transformer (ViT) を使用した腕関節回帰モデルの3次元評価スクリプト

xz平面モデルとxy平面モデルを組み合わせて3次元評価を行うスクリプト

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

from heatmap_distal_detection.dataset_arm_joints import ArmJointDataset, collate_fn
from heatmap_distal_detection.dataset_arm_joints_xy import ArmJointDatasetXY
from heatmap_distal_detection.dataset_full_skeleton import FullSkeletonDataset, collate_fn as full_collate_fn
from heatmap_distal_detection.dataset_full_skeleton_xy import FullSkeletonDatasetXY
from heatmap_distal_detection.models.arm_joint_regressor import create_arm_joint_regressor
from heatmap_distal_detection.models.arm_joint_regressor_xy import create_arm_joint_regressor_xy
from heatmap_distal_detection.models.arm_joint_regressor_hierarchical import (
    create_hierarchical_arm_joint_regressor,
    create_hierarchical_arm_joint_regressor_xy
)
from heatmap_distal_detection.models.vit_arm_joint_regressor import create_vit_arm_joint_regressor
from heatmap_distal_detection.models.full_skeleton_regressor import create_full_skeleton_regressor
from heatmap_distal_detection.models.full_skeleton_regressor_xy import create_full_skeleton_regressor_xy


def load_3d_gt_joints(
    data_path: str,
    x_range_xz: Tuple[float, float],
    z_range: Tuple[float, float],
    x_range_xy: Tuple[float, float],
    y_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JSONLファイルから3次元GT関節座標を読み込む
    
    Args:
        data_path: JSONLファイルのパス
        x_range_xz: xz平面のx軸範囲
        z_range: z軸範囲
        x_range_xy: xy平面のx軸範囲
        y_range: y軸範囲
    
    Returns:
        gt_joints_3d: (N, num_joints, 3) - 3次元GT関節座標
        valid_mask: (N, num_joints) - 有効な関節のマスク
    """
    gt_joints_3d_list = []
    valid_mask_list = []
    
    # 関節インデックス（joints_def_22.jsonから）
    # L_Shoulder (16), L_Elbow (18), L_Wrist (20)
    # R_Shoulder (17), R_Elbow (19), R_Wrist (21)
    arm_joint_indices = [16, 18, 20, 17, 19, 21]
    arm_joint_names = [
        'L_Shoulder', 'L_Elbow', 'L_Wrist',
        'R_Shoulder', 'R_Elbow', 'R_Wrist'
    ]
    
    num_samples_with_gt = 0
    num_samples_with_arm_coords = 0
    
    def denormalize_coord(normalized_coord: float, coord_range: Tuple[float, float]) -> float:
        """正規化された座標を元の範囲に戻す"""
        min_val, max_val = coord_range
        return normalized_coord * (max_val - min_val) + min_val
    
    with open(data_path, 'r') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                sample = json.loads(line)
                
                # まずarm_joint_coords_3dを確認（データ収集スクリプトで保存されている）
                arm_joint_coords_3d = sample.get('arm_joint_coords_3d', None)
                
                if arm_joint_coords_3d is not None and len(arm_joint_coords_3d) > 0:
                    # arm_joint_coords_3dから直接3次元座標を取得
                    joints_3d = np.array(arm_joint_coords_3d)  # (6, 3)
                    if joints_3d.shape == (len(arm_joint_indices), 3):
                        valid = ~np.isnan(joints_3d).any(axis=1)  # (6,)
                        num_samples_with_gt += 1
                    else:
                        joints_3d = np.full((len(arm_joint_indices), 3), np.nan, dtype=np.float32)
                        valid = np.zeros(len(arm_joint_indices), dtype=bool)
                else:
                    # arm_joint_coords_3dがない場合は、gt_jointsから取得を試みる
                    gt_joints = sample.get('gt_joints', None)
                    if gt_joints is not None and len(gt_joints) > 0:
                        gt_joints = np.array(gt_joints)  # (22, 3)
                        if len(gt_joints) >= 22:
                            # 腕の関節のみ抽出
                            joints_3d = gt_joints[arm_joint_indices]  # (6, 3)
                            valid = ~np.isnan(joints_3d).any(axis=1)  # (6,)
                            num_samples_with_gt += 1
                        else:
                            joints_3d = np.full((len(arm_joint_indices), 3), np.nan, dtype=np.float32)
                            valid = np.zeros(len(arm_joint_indices), dtype=bool)
                    else:
                        # どちらもない場合はNaN
                        joints_3d = np.full((len(arm_joint_indices), 3), np.nan, dtype=np.float32)
                        valid = np.zeros(len(arm_joint_indices), dtype=bool)
                
                gt_joints_3d_list.append(joints_3d)
                valid_mask_list.append(valid)
                
                # デバッグ: 最初の数サンプルで確認
                if line_idx < 3:
                    print(f"[Debug] Sample {line_idx}:")
                    print(f"  arm_joint_coords_3d present: {'arm_joint_coords_3d' in sample}")
                    print(f"  gt_joints present: {'gt_joints' in sample}")
                    print(f"  joints_3d shape: {joints_3d.shape}")
                    print(f"  valid joints: {valid.sum()}/{len(valid)}")
                    if valid.sum() > 0:
                        print(f"  First valid joint: {joints_3d[valid][0] if valid.any() else 'N/A'}")
    
    gt_joints_3d = np.array(gt_joints_3d_list)  # (N, 6, 3)
    valid_mask = np.array(valid_mask_list)  # (N, 6)
    
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
    
    # デバッグ: 正規化前の予測値を確認（コメントアウト）
    # print(f"\n[Debug] Normalized predictions (before denormalization):")
    # print(f"  pred_xz shape: {pred_xz.shape}")
    # print(f"  pred_xy shape: {pred_xy.shape}")
    # if num_samples > 0:
    #     print(f"  Sample 0 - xz (first joint, normalized): x={pred_xz[0, 0]:.6f}, z={pred_xz[0, 1]:.6f}")
    #     print(f"  Sample 0 - xy (first joint, normalized): x={pred_xy[0, 0]:.6f}, y={pred_xy[0, 1]:.6f}")
    #     print(f"  pred_xz range: min={pred_xz.min():.6f}, max={pred_xz.max():.6f}")
    #     print(f"  pred_xy range: min={pred_xy.min():.6f}, max={pred_xy.max():.6f}")
    #     
    #     # x座標の不一致を確認（最初の10サンプル）
    #     print(f"\n[Debug] X-coordinate comparison (xz vs xy, first 10 samples, first joint):")
    #     for i in range(min(10, num_samples)):
    #         x_xz_norm = pred_xz[i, 0]
    #         x_xy_norm = pred_xy[i, 0]
    #         diff_norm = abs(x_xz_norm - x_xy_norm)
    #         print(f"  Sample {i}: xz_x={x_xz_norm:.6f}, xy_x={x_xy_norm:.6f}, diff={diff_norm:.6f}")
    
    # xzモデル: 正規化済み（0-1範囲）と仮定 → クリップして正規化解除
    pred_xz_clipped = np.clip(pred_xz, 0.0, 1.0)
    pred_xz_denorm = denormalize_coords_2d(pred_xz_clipped, x_range_xz, z_range)  # (N, num_joints*2)
    
    # xyモデル: 正規化済み（0-1範囲）と仮定 → クリップせずに正規化解除
    # 注意: モデルの出力層にsigmoidがないため、0-1範囲を超える値が出る可能性がある
    # evaluate_arm_joint_regressor_xy.pyと同じ処理（クリップなし）
    # デバッグ: xyモデルの出力範囲を確認（コメントアウト）
    # print(f"\n[Debug] xy model output format:")
    # print(f"  Assumed: Normalized (0-1 range), but may exceed due to no sigmoid in output layer")
    # print(f"  Raw range: min={pred_xy.min():.6f}, max={pred_xy.max():.6f}")
    # 
    # # 範囲外の値の割合を確認
    # out_of_range = (pred_xy < 0.0) | (pred_xy > 1.0)
    # out_of_range_ratio = out_of_range.sum() / pred_xy.size
    # if out_of_range_ratio > 0.1:  # 10%以上が範囲外の場合
    #     print(f"  ⚠️  Warning: {out_of_range_ratio*100:.2f}% of values are outside [0, 1] range")
    #     print(f"  Using values as-is (no clipping) for denormalization, same as evaluate_arm_joint_regressor_xy.py")
    
    # クリップせずに正規化解除（evaluate_arm_joint_regressor_xy.pyと同じ処理）
    pred_xy_denorm = denormalize_coords_2d(pred_xy, x_range_xy, y_range)  # (N, num_joints*2)
    
    # デバッグ: 最終座標を確認（コメントアウト）
    # print(f"\n[Debug] Final coordinates (both denormalized):")
    # print(f"  pred_xz_denorm shape: {pred_xz_denorm.shape}")
    # print(f"  pred_xy_denorm shape: {pred_xy_denorm.shape}")
    # print(f"  x_range_xz: {x_range_xz}, z_range: {z_range}")
    # print(f"  x_range_xy: {x_range_xy}, y_range: {y_range}")
    # if num_samples > 0:
    #     print(f"  Sample 0 - xz (first joint, denormalized): x={pred_xz_denorm[0, 0]:.3f}, z={pred_xz_denorm[0, 1]:.3f}")
    #     print(f"  Sample 0 - xy (first joint, denormalized): x={pred_xy_denorm[0, 0]:.3f}, y={pred_xy_denorm[0, 1]:.3f}")
    #     
    #     # x座標の不一致を確認（両方とも正規化解除後、最初の10サンプル）
    #     print(f"\n[Debug] X-coordinate comparison (xz vs xy, both denormalized, first 10 samples, first joint):")
    #     for i in range(min(10, num_samples)):
    #         x_xz_denorm = pred_xz_denorm[i, 0]
    #         x_xy_denorm = pred_xy_denorm[i, 0]
    #         diff = abs(x_xz_denorm - x_xy_denorm)
    #         diff_cm = diff * 100  # メートルからcmに変換
    #         print(f"  Sample {i}: xz_x={x_xz_denorm:.3f}m, xy_x={x_xy_denorm:.3f}m, diff={diff:.3f}m ({diff_cm:.2f}cm)")
    
    # 3次元座標を作成
    pred_3d = np.zeros((num_samples, num_joints, 3), dtype=np.float32)
    
    for i in range(num_joints):
        # x座標: xzモデルから取得（xzモデルのx座標を使用）
        # 注意: xzモデルとxyモデルでx座標が一致しているか確認が必要
        pred_3d[:, i, 0] = pred_xz_denorm[:, i*2]
        
        # y座標: xyモデルから取得
        pred_3d[:, i, 1] = pred_xy_denorm[:, i*2+1]
        
        # z座標: xzモデルから取得
        pred_3d[:, i, 2] = pred_xz_denorm[:, i*2+1]
    
    return pred_3d


def compute_3d_mpjpe(
    pred_3d: np.ndarray,  # (N, num_joints, 3)
    gt_3d: np.ndarray,  # (N, num_joints, 3)
    valid_mask: np.ndarray  # (N, num_joints)
) -> Dict[str, float]:
    """
    3次元MPJPEを計算
    
    Args:
        pred_3d: 予測3次元座標
        gt_3d: GT 3次元座標
        valid_mask: 有効な関節のマスク
    
    Returns:
        metrics: 評価指標の辞書
    """
    num_joints = pred_3d.shape[1]
    joint_names = [
        'L_Shoulder', 'L_Elbow', 'L_Wrist',
        'R_Shoulder', 'R_Elbow', 'R_Wrist'
    ]
    
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
        metrics[f'{joint_name}_mpjpe_3d_cm'] = joint_errors[i]
    
    return metrics


def load_full_skeleton_gt_dict(full_skeleton_data_path: str) -> Dict:
    """
    full_skeleton_data_xyのJSONLからGT骨格を読み込み、file_pathをキーとした辞書に格納
    
    Args:
        full_skeleton_data_path: full_skeleton_data_xyのJSONLファイルパス
    
    Returns:
        gt_dict: {file_path: gt_joints} の辞書
    """
    gt_dict = {}
    
    if full_skeleton_data_path is None or not os.path.exists(full_skeleton_data_path):
        print(f"Warning: Full skeleton data path not found: {full_skeleton_data_path}")
        return gt_dict
    
    print(f"Loading full skeleton GT from: {full_skeleton_data_path}")
    count = 0
    file_path_count = 0
    with open(full_skeleton_data_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    sample = json.loads(line)
                    gt_joints = sample.get('gt_joints', None)
                    if gt_joints is not None:
                        # file_pathでマッチング（最も確実な方法）
                        file_path = sample.get('file_path', '')
                        if file_path:
                            gt_dict[file_path] = np.array(gt_joints)
                            file_path_count += 1
                            count += 1
                except Exception as e:
                    continue
    
    print(f"  Loaded {count} samples with full skeleton GT")
    print(f"  Entries with file_path key: {file_path_count}")
    return gt_dict


def extract_shoulder_coords_from_full_skeleton(
    full_pred_2d: np.ndarray,  # (N, 44) - Full modelの2D予測 [x, z/y, x, z/y, ...]
    joint_names: List[str]  # 22関節の名前
) -> np.ndarray:
    """
    全身モデルの予測から肩の座標を抽出（正規化済み）
    
    Args:
        full_pred_2d: 全身モデルの2D予測（正規化済み、0-1範囲）
        joint_names: 22関節の名前
    
    Returns:
        shoulder_coords: (N, 4) - 肩の座標（正規化済み）
            [L_Shoulder_x, L_Shoulder_z/y, R_Shoulder_x, R_Shoulder_z/y]
    """
    num_samples = full_pred_2d.shape[0]
    shoulder_coords = np.zeros((num_samples, 4), dtype=np.float32)
    
    # L_ShoulderとR_Shoulderのインデックスを取得
    l_shoulder_idx = joint_names.index('L_Shoulder')
    r_shoulder_idx = joint_names.index('R_Shoulder')
    
    # 肩の座標を抽出
    shoulder_coords[:, 0] = full_pred_2d[:, l_shoulder_idx * 2]  # L_Shoulder_x
    shoulder_coords[:, 1] = full_pred_2d[:, l_shoulder_idx * 2 + 1]  # L_Shoulder_z/y
    shoulder_coords[:, 2] = full_pred_2d[:, r_shoulder_idx * 2]  # R_Shoulder_x
    shoulder_coords[:, 3] = full_pred_2d[:, r_shoulder_idx * 2 + 1]  # R_Shoulder_z/y
    
    return shoulder_coords


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
    full_skeleton_data_path: str = None,
    full_model_xz = None,
    full_model_xy = None,
    data_loader_xz_full = None,
    data_loader_xy_full = None,
    joint_names: List[str] = None
) -> Dict[str, float]:
    """
    両方のモデルで予測を取得し、3次元評価を実行
    
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
        full_skeleton_data_path: full_skeleton_data_xyのJSONLファイルパス（オプション）
        full_model_xz: 全身モデル（xz平面、オプション）
        full_model_xy: 全身モデル（xy平面、オプション）
        data_loader_xz_full: 全身モデル用データローダー（xz平面、オプション）
        data_loader_xy_full: 全身モデル用データローダー（xy平面、オプション）
        joint_names: 22関節の名前（オプション）
    
    Returns:
        metrics: 評価指標の辞書
    """
    model_xz.eval()
    model_xy.eval()
    
    # 全身モデルから肩の座標を取得（提供されている場合）
    use_full_model_shoulders = (full_model_xz is not None and full_model_xy is not None and 
                                data_loader_xz_full is not None and data_loader_xy_full is not None and
                                joint_names is not None)
    
    if use_full_model_shoulders:
        print("\nUsing shoulder coordinates from full skeleton model")
        full_model_xz.eval()
        full_model_xy.eval()
        
        all_full_pred_xz = []
        all_full_pred_xy = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader_xz_full, desc='Getting shoulder coords from full model XZ'):
                heatmaps = batch['heatmap'].to(device)
                pred = full_model_xz(heatmaps).cpu().numpy()
                all_full_pred_xz.append(pred)
            
            for batch in tqdm(data_loader_xy_full, desc='Getting shoulder coords from full model XY'):
                heatmaps = batch['heatmap'].to(device)
                pred = full_model_xy(heatmaps).cpu().numpy()
                all_full_pred_xy.append(pred)
        
        all_full_pred_xz = np.concatenate(all_full_pred_xz, axis=0)  # (N, 44)
        all_full_pred_xy = np.concatenate(all_full_pred_xy, axis=0)  # (N, 44)
        
        # 肩の座標を抽出（正規化済み）
        shoulder_coords_xz = extract_shoulder_coords_from_full_skeleton(all_full_pred_xz, joint_names)  # (N, 4)
        shoulder_coords_xy = extract_shoulder_coords_from_full_skeleton(all_full_pred_xy, joint_names)  # (N, 4)
    else:
        print("\nUsing shoulder coordinates from arm model (full model not provided)")
        shoulder_coords_xz = None
        shoulder_coords_xy = None
    
    # full_skeleton_data_xyからGT骨格を読み込む
    full_skeleton_gt_dict = load_full_skeleton_gt_dict(full_skeleton_data_path)
    
    all_pred_xz = []
    all_pred_xy = []
    all_gt_3d = []
    all_gt_full_3d = []  # 全骨格のGT（アニメーション用）
    all_valid_mask = []
    all_frame_ids_xz = []  # フレームIDを保存（デバッグ用）
    all_frame_ids_xy = []  # フレームIDを保存（デバッグ用）
    
    # 関節インデックス（joints_def_22.jsonから）
    # L_Shoulder (16), L_Elbow (18), L_Wrist (20)
    # R_Shoulder (17), R_Elbow (19), R_Wrist (21)
    arm_joint_indices = [16, 18, 20, 17, 19, 21]
    
    # xzモデルで予測とGT座標を取得
    with torch.no_grad():
        pbar = tqdm(data_loader_xz, desc='Evaluating xz model')
        sample_idx = 0
        for batch in pbar:
            heatmaps = batch['heatmap'].to(device)  # (B, 1, H, W)
            batch_size = heatmaps.shape[0]
            
            # 肩の座標を使用（提供されている場合）
            if use_full_model_shoulders:
                end_idx = sample_idx + batch_size
                batch_shoulder_coords = torch.from_numpy(shoulder_coords_xz[sample_idx:end_idx]).to(device)  # (B, 4)
                # 階層的モデルの場合、shoulder_coordsを渡す
                try:
                    pred_coords = model_xz(heatmaps, return_attention=False, shoulder_coords=batch_shoulder_coords)  # (B, num_joints*2)
                except TypeError:
                    # shoulder_coordsパラメータがサポートされていない場合は従来通り
                    pred_coords = model_xz(heatmaps)  # (B, num_joints*2)
            else:
                pred_coords = model_xz(heatmaps)  # (B, num_joints*2)
            
            pred_coords = pred_coords.cpu().numpy()
            all_pred_xz.append(pred_coords)
            sample_idx += batch_size
            
            # GT 3次元座標を取得（metadataから）
            metadata_list = batch.get('metadata', [])
            for metadata in metadata_list:
                # フレームIDを保存
                frame_id = metadata.get('frame_id', '')
                sequence_id = metadata.get('sequence_id', '')
                frame_num = metadata.get('frame_num', -1)
                file_path = metadata.get('file_path', '')
                all_frame_ids_xz.append({
                    'frame_id': frame_id,
                    'sequence_id': sequence_id,
                    'frame_num': frame_num,
                    'file_path': file_path
                })
                
                gt_joints_3d_sample = None
                gt_joints_full = None
                
                # まずarm_joint_coords_3dを確認
                if 'arm_joint_coords_3d' in metadata and metadata['arm_joint_coords_3d'] is not None:
                    gt_joints_3d_sample = np.array(metadata['arm_joint_coords_3d'])  # (6, 3)
                elif 'gt_joints' in metadata and metadata['gt_joints'] is not None:
                    # gt_jointsから腕の関節のみ抽出
                    gt_joints = np.array(metadata['gt_joints'])  # (22, 3)
                    if len(gt_joints) >= 22:
                        gt_joints_3d_sample = gt_joints[arm_joint_indices]  # (6, 3)
                        gt_joints_full = gt_joints  # 全骨格のGTを保存
                
                # full_skeleton_data_xyからGT骨格を取得（metadataにない場合）
                if gt_joints_full is None and len(full_skeleton_gt_dict) > 0:
                    # file_pathでマッチングを試みる（最も確実な方法）
                    matched_gt = None
                    file_path = metadata.get('file_path', '')
                    if file_path and file_path in full_skeleton_gt_dict:
                        matched_gt = full_skeleton_gt_dict[file_path]
                    
                    if matched_gt is not None and len(matched_gt) >= 22:
                        gt_joints_full = matched_gt
                        # 腕の関節も抽出
                        if gt_joints_3d_sample is None:
                            gt_joints_3d_sample = matched_gt[arm_joint_indices]  # (6, 3)
                    
                    # デバッグ: 最初の数サンプルでマッチング状況を確認
                    if len(all_gt_full_3d) < 5:
                        if matched_gt is not None:
                            print(f"  [Debug] Frame {len(all_gt_full_3d)}: Matched via file_path (file_path={file_path[:50]}...)")
                        else:
                            print(f"  [Debug] Frame {len(all_gt_full_3d)}: No match (file_path={file_path[:50] if file_path else 'N/A'}...)")
                
                if gt_joints_3d_sample is not None and gt_joints_3d_sample.shape == (len(arm_joint_indices), 3):
                    all_gt_3d.append(gt_joints_3d_sample)
                    valid = ~np.isnan(gt_joints_3d_sample).any(axis=1)  # (6,)
                    all_valid_mask.append(valid)
                else:
                    # GTがない場合はNaN
                    all_gt_3d.append(np.full((len(arm_joint_indices), 3), np.nan, dtype=np.float32))
                    all_valid_mask.append(np.zeros(len(arm_joint_indices), dtype=bool))
                
                # 全骨格のGTを保存（アニメーション用）
                if gt_joints_full is not None and len(gt_joints_full) >= 22:
                    all_gt_full_3d.append(gt_joints_full)
                else:
                    # 全骨格のGTがない場合はNaN
                    all_gt_full_3d.append(np.full((22, 3), np.nan, dtype=np.float32))
    
    # デバッグ: gt_full_3dの統計情報
    print(f"\n[Debug] GT Full Skeleton Statistics:")
    print(f"  Total samples: {len(all_gt_full_3d)}")
    if len(all_gt_full_3d) > 0:
        gt_full_3d_array = np.array(all_gt_full_3d)
        valid_frames = ~np.isnan(gt_full_3d_array).all(axis=(1, 2))
        valid_joints_per_frame = (~np.isnan(gt_full_3d_array).all(axis=2)).sum(axis=1)
        print(f"  Frames with valid GT: {valid_frames.sum()}/{len(valid_frames)} ({valid_frames.sum()/len(valid_frames)*100:.1f}%)")
        print(f"  Average valid joints per frame: {valid_joints_per_frame.mean():.2f}/22")
        print(f"  Frames with all 22 joints valid: {(valid_joints_per_frame == 22).sum()}/{len(valid_joints_per_frame)}")
        if valid_frames.sum() == 0:
            print(f"  ⚠️  Warning: No valid GT full skeleton found!")
            print(f"      This means 'gt_joints' (22 joints) is not in the JSONL metadata.")
            print(f"      Only 'arm_joint_coords_3d' (6 joints) may be available.")
    
    # xyモデルで予測を取得
    with torch.no_grad():
        pbar = tqdm(data_loader_xy, desc='Evaluating xy model')
        sample_idx = 0
        for batch in pbar:
            heatmaps = batch['heatmap'].to(device)  # (B, 1, H, W)
            batch_size = heatmaps.shape[0]
            
            # 肩の座標を使用（提供されている場合）
            if use_full_model_shoulders:
                end_idx = sample_idx + batch_size
                batch_shoulder_coords = torch.from_numpy(shoulder_coords_xy[sample_idx:end_idx]).to(device)  # (B, 4)
                # 階層的モデルの場合、shoulder_coordsを渡す
                try:
                    pred_coords = model_xy(heatmaps, return_attention=False, shoulder_coords=batch_shoulder_coords)  # (B, num_joints*2)
                except TypeError:
                    # shoulder_coordsパラメータがサポートされていない場合は従来通り
                    pred_coords = model_xy(heatmaps)  # (B, num_joints*2)
            else:
                pred_coords = model_xy(heatmaps)  # (B, num_joints*2)
            
            pred_coords = pred_coords.cpu().numpy()
            all_pred_xy.append(pred_coords)
            sample_idx += batch_size
            
            # フレームIDを保存
            metadata_list = batch.get('metadata', [])
            for metadata in metadata_list:
                frame_id = metadata.get('frame_id', '')
                sequence_id = metadata.get('sequence_id', '')
                frame_num = metadata.get('frame_num', -1)
                file_path = metadata.get('file_path', '')
                all_frame_ids_xy.append({
                    'frame_id': frame_id,
                    'sequence_id': sequence_id,
                    'frame_num': frame_num,
                    'file_path': file_path
                })
    
    # 全データを結合
    all_pred_xz = np.concatenate(all_pred_xz, axis=0)  # (N, num_joints*2)
    all_pred_xy = np.concatenate(all_pred_xy, axis=0)  # (N, num_joints*2)
    gt_3d = np.array(all_gt_3d)  # (N, 6, 3)
    gt_full_3d = np.array(all_gt_full_3d)  # (N, 22, 3) - 全骨格のGT
    valid_mask = np.array(all_valid_mask)  # (N, 6)
    
    # file_pathでマッチング（データセットサイズが異なる場合）
    if len(all_pred_xz) != len(all_pred_xy):
        print(f"\n⚠️  Warning: Data size mismatch!")
        print(f"  pred_xz: {len(all_pred_xz)}")
        print(f"  pred_xy: {len(all_pred_xy)}")
        print(f"  Matching by file_path...")
        
        # xyデータのfile_pathをキーとした辞書を作成
        xy_dict = {}
        for i, frame_info in enumerate(all_frame_ids_xy):
            file_path = frame_info.get('file_path', '')
            if file_path:
                if file_path not in xy_dict:
                    xy_dict[file_path] = []
                xy_dict[file_path].append(i)
        
        # xzデータに対応するxyデータを検索
        matched_pred_xz = []
        matched_pred_xy = []
        matched_gt_3d = []
        matched_gt_full_3d = []
        matched_valid_mask = []
        matched_count = 0
        unmatched_count = 0
        
        for i, frame_info in enumerate(all_frame_ids_xz):
            file_path = frame_info.get('file_path', '')
            if file_path and file_path in xy_dict:
                # 最初のマッチを使用（同じfile_pathが複数ある場合）
                xy_idx = xy_dict[file_path][0]
                matched_pred_xz.append(all_pred_xz[i])
                matched_pred_xy.append(all_pred_xy[xy_idx])
                matched_gt_3d.append(gt_3d[i])
                matched_gt_full_3d.append(gt_full_3d[i])
                matched_valid_mask.append(valid_mask[i])
                matched_count += 1
                # 使用済みのインデックスを削除（重複を避ける）
                xy_dict[file_path].pop(0)
                if len(xy_dict[file_path]) == 0:
                    del xy_dict[file_path]
            else:
                unmatched_count += 1
        
        print(f"  Matched: {matched_count} samples")
        print(f"  Unmatched: {unmatched_count} samples")
        
        if matched_count == 0:
            raise ValueError("No matching frames found by file_path! Check if file_path is consistent between xz and xy datasets.")
        
        all_pred_xz = np.array(matched_pred_xz)
        all_pred_xy = np.array(matched_pred_xy)
        gt_3d = np.array(matched_gt_3d)
        gt_full_3d = np.array(matched_gt_full_3d)
        valid_mask = np.array(matched_valid_mask)
        
        print(f"  Using {matched_count} matched samples for evaluation")
    elif len(all_pred_xz) != len(gt_3d):
        print(f"\n⚠️  Warning: Data size mismatch!")
        print(f"  pred_xz: {len(all_pred_xz)}")
        print(f"  pred_xy: {len(all_pred_xy)}")
        print(f"  gt_3d: {len(gt_3d)}")
        min_size = min(len(all_pred_xz), len(all_pred_xy), len(gt_3d))
        print(f"  Using minimum size: {min_size}")
        all_pred_xz = all_pred_xz[:min_size]
        all_pred_xy = all_pred_xy[:min_size]
        gt_3d = gt_3d[:min_size]
        gt_full_3d = gt_full_3d[:min_size]
        valid_mask = valid_mask[:min_size]
    
    # デバッグ: モデルの出力範囲を確認（combine_2d_predictions_to_3d呼び出し前）
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
        if out_of_range_ratio_xz > 0.1:
            print(f"  ⚠️  Warning: {out_of_range_ratio_xz*100:.2f}% of values are outside [0, 1] range")
    
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
        if out_of_range_ratio_xy > 0.1:
            print(f"  ⚠️  Warning: {out_of_range_ratio_xy*100:.2f}% of values are outside [0, 1] range")
            print(f"  This is expected if the model's output layer has no sigmoid activation")
        else:
            print(f"  ✓ All values are within [0, 1] range (normalized)")
    
    print("="*60 + "\n")
    
    # 3次元予測座標を作成
    pred_3d = combine_2d_predictions_to_3d(
        all_pred_xz, all_pred_xy,
        x_range_xz, z_range,
        x_range_xy, y_range
    )  # (N, 6, 3)
    
    # デバッグ: xzとxyのx座標の差を確認
    print(f"\n[Debug] X coordinate comparison (xz vs xy):")
    num_joints = all_pred_xz.shape[1] // 2
    
    # xzモデル: 正規化済み → クリップして正規化解除
    x_xz_normalized = all_pred_xz[:, 0::2]  # (N, num_joints) - xzモデルのx座標（正規化済み）
    x_xz_normalized_clipped = np.clip(x_xz_normalized, 0.0, 1.0)
    x_xz_denorm = x_xz_normalized_clipped * (x_range_xz[1] - x_range_xz[0]) + x_range_xz[0]
    
    # xyモデル: 正規化済み → クリップして正規化解除
    x_xy_normalized = all_pred_xy[:, 0::2]  # (N, num_joints) - xyモデルのx座標（正規化済み）
    x_xy_normalized_clipped = np.clip(x_xy_normalized, 0.0, 1.0)
    x_xy_denorm = x_xy_normalized_clipped * (x_range_xy[1] - x_range_xy[0]) + x_range_xy[0]
    
    x_diff = np.abs(x_xz_denorm - x_xy_denorm)  # (N, num_joints)
    print(f"  X coordinate difference (xz - xy) in meters:")
    print(f"    Mean: {x_diff.mean():.4f} m")
    print(f"    Max: {x_diff.max():.4f} m")
    print(f"    Median: {np.median(x_diff):.4f} m")
    print(f"    Std: {x_diff.std():.4f} m")
    if x_diff.max() > 0.1:  # 10cm以上の差がある場合
        print(f"  ⚠️  Large x-coordinate differences detected (max: {x_diff.max()*100:.2f} cm)")
        # 差が大きいサンプルを表示
        large_diff_mask = x_diff > 0.1
        large_diff_indices = np.where(large_diff_mask)
        if len(large_diff_indices[0]) > 0:
            print(f"    Found {len(large_diff_indices[0])} samples with >10cm difference")
            for idx in range(min(5, len(large_diff_indices[0]))):
                sample_idx = large_diff_indices[0][idx]
                joint_idx = large_diff_indices[1][idx]
                joint_names = ['L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']
                print(f"      Sample {sample_idx}, {joint_names[joint_idx]}: "
                      f"xz_x={x_xz_denorm[sample_idx, joint_idx]:.3f}, "
                      f"xy_x={x_xy_denorm[sample_idx, joint_idx]:.3f}, "
                      f"diff={x_diff[sample_idx, joint_idx]*100:.2f} cm")
    
    # デバッグ: 各軸ごとの誤差を計算
    print(f"\n[Debug] Per-axis error analysis:")
    axis_errors = {'x': [], 'y': [], 'z': []}
    for i in range(len(pred_3d)):
        for j in range(num_joints):
            if valid_mask[i, j]:
                pred_joint = pred_3d[i, j, :]  # (3,)
                gt_joint = gt_3d[i, j, :]  # (3,)
                axis_errors['x'].append(abs(pred_joint[0] - gt_joint[0]) * 100)  # cm
                axis_errors['y'].append(abs(pred_joint[1] - gt_joint[1]) * 100)  # cm
                axis_errors['z'].append(abs(pred_joint[2] - gt_joint[2]) * 100)  # cm
    
    for axis in ['x', 'y', 'z']:
        if axis_errors[axis]:
            errors = np.array(axis_errors[axis])
            print(f"  {axis.upper()}-axis error: mean={errors.mean():.2f} cm, "
                  f"median={np.median(errors):.2f} cm, max={errors.max():.2f} cm, std={errors.std():.2f} cm")
    
    # デバッグ: 最初の数サンプルで予測とGTを比較（コメントアウト）
    # print(f"\n[Debug] Comparing predictions and GT (first 5 samples):")
    # for i in range(min(5, len(pred_3d))):
    #     if valid_mask[i].any():
    #         frame_id = all_frame_ids_xz[i].get('frame_id', 'N/A') if i < len(all_frame_ids_xz) else 'N/A'
    #         print(f"\nSample {i} (frame_id: {frame_id}):")
    #         joint_names = ['L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']
    #         for j, joint_name in enumerate(joint_names):
    #             if valid_mask[i, j]:
    #                 pred_joint = pred_3d[i, j, :]  # (3,)
    #                 gt_joint = gt_3d[i, j, :]  # (3,)
    #                 error_3d = np.sqrt(np.sum((pred_joint - gt_joint) ** 2)) * 100  # cm
    #                 error_x = abs(pred_joint[0] - gt_joint[0]) * 100  # cm
    #                 error_y = abs(pred_joint[1] - gt_joint[1]) * 100  # cm
    #                 error_z = abs(pred_joint[2] - gt_joint[2]) * 100  # cm
    #                 print(f"  {joint_name}:")
    #                 print(f"    Pred 3D: [{pred_joint[0]:.3f}, {pred_joint[1]:.3f}, {pred_joint[2]:.3f}] m")
    #                 print(f"    GT 3D:   [{gt_joint[0]:.3f}, {gt_joint[1]:.3f}, {gt_joint[2]:.3f}] m")
    #                 print(f"    Error 3D: {error_3d:.2f} cm (x: {error_x:.2f}, y: {error_y:.2f}, z: {error_z:.2f} cm)")
    #                 
    #                 # xz/xyモデルの個別予測も表示
    #                 if i < len(all_pred_xz) and i < len(all_pred_xy):
    #                     # xzモデル: 正規化解除
    #                     xz_pred_norm = all_pred_xz[i, j*2:(j+1)*2]  # (x, z) 正規化済み
    #                     xz_pred_norm_clipped = np.clip(xz_pred_norm, 0.0, 1.0)
    #                     xz_pred_denorm = np.array([
    #                         xz_pred_norm_clipped[0] * (x_range_xz[1] - x_range_xz[0]) + x_range_xz[0],
    #                         xz_pred_norm_clipped[1] * (z_range[1] - z_range[0]) + z_range[0]
    #                     ])
    #                     
    #                     # xyモデル: 正規化解除
    #                     xy_pred_norm = all_pred_xy[i, j*2:(j+1)*2]  # (x, y) 正規化済み
    #                     xy_pred_norm_clipped = np.clip(xy_pred_norm, 0.0, 1.0)
    #                     xy_pred_denorm = np.array([
    #                         xy_pred_norm_clipped[0] * (x_range_xy[1] - x_range_xy[0]) + x_range_xy[0],
    #                         xy_pred_norm_clipped[1] * (y_range[1] - y_range[0]) + y_range[0]
    #                     ])
    #                     
    #                     print(f"    xz model (denorm): x={xz_pred_denorm[0]:.3f}m, z={xz_pred_denorm[1]:.3f}m")
    #                     print(f"    xy model (denorm): x={xy_pred_denorm[0]:.3f}m, y={xy_pred_denorm[1]:.3f}m")
    #                     
    #                     # x座標の差を確認
    #                     x_diff = abs(xz_pred_denorm[0] - xy_pred_denorm[0]) * 100  # cm
    #                     if x_diff > 5.0:  # 5cm以上の差がある場合
    #                         print(f"    ⚠️  Large x-coordinate difference: {x_diff:.2f} cm")
    # 
    # # デバッグ: 座標の統計情報（コメントアウト）
    # print(f"\n[Debug] Coordinate statistics:")
    # print(f"  Pred 3D - X: min={pred_3d[:, :, 0].min():.3f}, max={pred_3d[:, :, 0].max():.3f}, mean={pred_3d[:, :, 0].mean():.3f}, std={pred_3d[:, :, 0].std():.3f}")
    # print(f"  Pred 3D - Y: min={pred_3d[:, :, 1].min():.3f}, max={pred_3d[:, :, 1].max():.3f}, mean={pred_3d[:, :, 1].mean():.3f}, std={pred_3d[:, :, 1].std():.3f}")
    # print(f"  Pred 3D - Z: min={pred_3d[:, :, 2].min():.3f}, max={pred_3d[:, :, 2].max():.3f}, mean={pred_3d[:, :, 2].mean():.3f}, std={pred_3d[:, :, 2].std():.3f}")
    # print(f"  GT 3D - X: min={gt_3d[:, :, 0].min():.3f}, max={gt_3d[:, :, 0].max():.3f}, mean={gt_3d[:, :, 0].mean():.3f}, std={gt_3d[:, :, 0].std():.3f}")
    # print(f"  GT 3D - Y: min={gt_3d[:, :, 1].min():.3f}, max={gt_3d[:, :, 1].max():.3f}, mean={gt_3d[:, :, 1].mean():.3f}, std={gt_3d[:, :, 1].std():.3f}")
    # print(f"  GT 3D - Z: min={gt_3d[:, :, 2].min():.3f}, max={gt_3d[:, :, 2].max():.3f}, mean={gt_3d[:, :, 2].mean():.3f}, std={gt_3d[:, :, 2].std():.3f}")
    
    # 3次元MPJPEを計算
    metrics = compute_3d_mpjpe(pred_3d, gt_3d, valid_mask)
    
    # 予測とGTの3D座標を保存（アニメーション用）
    metrics['pred_3d'] = pred_3d.tolist()  # (N, 6, 3)
    metrics['gt_3d'] = gt_3d.tolist()  # (N, 6, 3)
    # gt_full_3dをJSON serializableに変換（NaNをNoneに変換）
    gt_full_3d_list = []
    for frame in gt_full_3d:
        frame_list = []
        for joint in frame:
            if np.isnan(joint).any():
                frame_list.append([None, None, None])
            else:
                frame_list.append(joint.tolist())
        gt_full_3d_list.append(frame_list)
    metrics['gt_full_3d'] = gt_full_3d_list  # (N, 22, 3) - 全骨格のGT
    metrics['valid_mask'] = valid_mask.tolist()  # (N, 6)
    metrics['frame_ids'] = all_frame_ids_xz  # フレームID情報
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate arm joint regressor in 3D by combining xz and xy models')
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
    parser.add_argument('--full_skeleton_data', type=str, default=None,
                       help='Path to full_skeleton_data_xy JSONL file (for GT full skeleton)')
    parser.add_argument('--full_model_xz', type=str, default=None,
                       help='Path to full skeleton model XZ checkpoint (optional, for using full model shoulders)')
    parser.add_argument('--full_model_xy', type=str, default=None,
                       help='Path to full skeleton model XY checkpoint (optional, for using full model shoulders)')
    
    args = parser.parse_args()
    
    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # xz平面モデルを読み込み
    print(f"\nLoading xz plane model from {args.model_path_xz}")
    checkpoint_xz = torch.load(args.model_path_xz, map_location=device)
    config_xz = checkpoint_xz.get('config', {})
    
    model_type_xz = config_xz.get('model_type', 'standard')
    use_attention_xz = config_xz.get('use_attention', False)
    normalize_heatmap_xz = config_xz.get('normalize_heatmap', False)
    bins_xz = config_xz.get('bins', args.bins)
    base_channels_xz = config_xz.get('base_channels', args.base_channels)
    dropout_xz = config_xz.get('dropout', args.dropout)
    model_size_xz = config_xz.get('model_size', 'small')
    
    if model_type_xz == 'vit':
        model_xz = create_vit_arm_joint_regressor(
            img_size=bins_xz,
            patch_size=5,
            num_joints=6,
            model_size=model_size_xz,
            drop_rate=config_xz.get('drop_rate', 0.0),
            attn_drop_rate=config_xz.get('attn_drop_rate', 0.0),
            drop_path_rate=config_xz.get('drop_path_rate', 0.0)
        )
    elif model_type_xz == 'hierarchical':
        model_xz = create_hierarchical_arm_joint_regressor(
            heatmap_size=bins_xz,
            num_joints_per_arm=3,
            base_channels=base_channels_xz,
            dropout=dropout_xz,
            use_attention=use_attention_xz,
            use_improved_attention=config_xz.get('use_improved_attention', False)
        )
    else:
        model_xz = create_arm_joint_regressor(
            heatmap_size=bins_xz,
            num_joints=6,
            base_channels=base_channels_xz,
            dropout=dropout_xz
        )
    
    model_xz.load_state_dict(checkpoint_xz['model_state_dict'])
    model_xz = model_xz.to(device)
    
    # xy平面モデルを読み込み
    print(f"\nLoading xy plane model from {args.model_path_xy}")
    checkpoint_xy = torch.load(args.model_path_xy, map_location=device)
    config_xy = checkpoint_xy.get('config', {})
    
    model_type_xy = config_xy.get('model_type', 'standard')
    use_attention_xy = config_xy.get('use_attention', False)
    normalize_heatmap_xy = config_xy.get('normalize_heatmap', False)
    bins_xy = config_xy.get('bins', args.bins)
    base_channels_xy = config_xy.get('base_channels', args.base_channels)
    dropout_xy = config_xy.get('dropout', args.dropout)
    model_size_xy = config_xy.get('model_size', 'small')
    
    if model_type_xy == 'vit':
        model_xy = create_vit_arm_joint_regressor(
            img_size=bins_xy,
            patch_size=5,
            num_joints=6,
            model_size=model_size_xy,
            drop_rate=config_xy.get('drop_rate', 0.0),
            attn_drop_rate=config_xy.get('attn_drop_rate', 0.0),
            drop_path_rate=config_xy.get('drop_path_rate', 0.0)
        )
    elif model_type_xy == 'hierarchical':
        model_xy = create_hierarchical_arm_joint_regressor_xy(
            heatmap_size=bins_xy,
            num_joints_per_arm=3,
            base_channels=base_channels_xy,
            dropout=dropout_xy,
            use_attention=use_attention_xy,
            use_improved_attention=config_xy.get('use_improved_attention', False)
        )
    else:
        model_xy = create_arm_joint_regressor_xy(
            heatmap_size=bins_xy,
            num_joints=6,
            base_channels=base_channels_xy,
            dropout=dropout_xy
        )
    
    model_xy.load_state_dict(checkpoint_xy['model_state_dict'])
    model_xy = model_xy.to(device)
    
    # テストデータパスを決定
    # 注意: データセットクラスはJSONLファイルに保存されているヒートマップではなく、
    # file_pathから点群を読み込んで動的にヒートマップを生成します。
    # したがって、xz平面とxy平面で異なるJSONLファイルを使用しても、
    # 同じfile_pathが含まれていれば、それぞれの平面のヒートマップを正しく生成できます。
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
    print(f"      The saved heatmaps in JSONL files are not used.")
    
    # データセットを作成
    x_range_xz = tuple(args.x_range_xz)
    z_range = tuple(args.z_range)
    x_range_xy = tuple(args.x_range_xy)
    y_range = tuple(args.y_range)
    
    dataset_xz = ArmJointDataset(
        data_path=test_data_xz,
        x_range=x_range_xz,
        z_range=z_range,
        bins=args.bins,
        feature=args.feature,
        normalize_heatmap=normalize_heatmap_xz  # チェックポイントから読み込んだ値を使用
    )
    
    dataset_xy = ArmJointDatasetXY(
        data_path=test_data_xy,
        x_range=x_range_xy,
        y_range=y_range,
        bins=args.bins,
        feature=args.feature,
        normalize_heatmap=normalize_heatmap_xy  # チェックポイントから読み込んだ値を使用
    )
    
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
    
    # full_skeleton_data_xyのパスを決定
    full_skeleton_data_path = args.full_skeleton_data
    if full_skeleton_data_path is None:
        # デフォルトパスを試す（test_dataと同じディレクトリ構造を想定）
        # test_dataが data/arm_joint_data_xy/val.jsonl の場合、
        # full_skeleton_dataは data/full_skeleton_data_xy/val.jsonl になる
        if args.test_data:
            test_data_dir = os.path.dirname(args.test_data)
            test_data_filename = os.path.basename(args.test_data)
            # arm_joint_data_xy -> full_skeleton_data_xy
            if 'arm_joint_data_xy' in test_data_dir:
                full_skeleton_data_dir = test_data_dir.replace('arm_joint_data_xy', 'full_skeleton_data_xy')
                full_skeleton_data_path = os.path.join(full_skeleton_data_dir, test_data_filename)
                if not os.path.exists(full_skeleton_data_path):
                    full_skeleton_data_path = None
    
    if full_skeleton_data_path:
        print(f"\nUsing full skeleton data: {full_skeleton_data_path}")
    else:
        print(f"\nWarning: Full skeleton data path not specified. GT full skeleton will not be loaded.")
        print(f"  Use --full_skeleton_data to specify the path to full_skeleton_data_xy JSONL file.")
    
    # 全身モデルを読み込み（オプション）
    full_model_xz = None
    full_model_xy = None
    data_loader_xz_full = None
    data_loader_xy_full = None
    joint_names = None
    
    if args.full_model_xz and args.full_model_xy:
        print(f"\nLoading full skeleton models for shoulder coordinates...")
        
        # 関節名を読み込み
        joints_def_path = Path(__file__).parent.parent.parent / 'data_specs' / 'joints_def_22.json'
        with open(joints_def_path, 'r') as f:
            joints_def = json.load(f)
        joint_names = joints_def.get('joint_names', [])
        
        # 全身モデル（xz平面）を読み込み
        print(f"  Loading full model XZ from {args.full_model_xz}")
        full_checkpoint_xz = torch.load(args.full_model_xz, map_location=device)
        full_config_xz = full_checkpoint_xz.get('config', {})
        full_normalize_heatmap_xz = full_config_xz.get('normalize_heatmap', False)
        full_bins_xz = full_config_xz.get('bins', args.bins)
        full_base_channels_xz = full_config_xz.get('base_channels', args.base_channels)
        full_dropout_xz = full_config_xz.get('dropout', args.dropout)
        
        full_model_xz = create_full_skeleton_regressor(
            heatmap_size=full_bins_xz,
            num_joints=22,
            base_channels=full_base_channels_xz,
            dropout=full_dropout_xz
        )
        if 'model_state_dict' in full_checkpoint_xz:
            full_model_xz.load_state_dict(full_checkpoint_xz['model_state_dict'])
        else:
            full_model_xz.load_state_dict(full_checkpoint_xz)
        full_model_xz = full_model_xz.to(device)
        
        # 全身モデル（xy平面）を読み込み
        print(f"  Loading full model XY from {args.full_model_xy}")
        full_checkpoint_xy = torch.load(args.full_model_xy, map_location=device)
        full_config_xy = full_checkpoint_xy.get('config', {})
        full_normalize_heatmap_xy = full_config_xy.get('normalize_heatmap', False)
        full_bins_xy = full_config_xy.get('bins', args.bins)
        full_base_channels_xy = full_config_xy.get('base_channels', args.base_channels)
        full_dropout_xy = full_config_xy.get('dropout', args.dropout)
        
        full_model_xy = create_full_skeleton_regressor_xy(
            heatmap_size=full_bins_xy,
            num_joints=22,
            base_channels=full_base_channels_xy,
            dropout=full_dropout_xy
        )
        if 'model_state_dict' in full_checkpoint_xy:
            full_model_xy.load_state_dict(full_checkpoint_xy['model_state_dict'])
        else:
            full_model_xy.load_state_dict(full_checkpoint_xy)
        full_model_xy = full_model_xy.to(device)
        
        # 全身モデル用のデータセットを作成
        full_dataset_xz = FullSkeletonDataset(
            data_path=test_data_xz,
            x_range=x_range_xz,
            z_range=z_range,
            bins=full_bins_xz,
            feature=args.feature,
            normalize_heatmap=full_normalize_heatmap_xz
        )
        
        full_dataset_xy = FullSkeletonDatasetXY(
            data_path=test_data_xy,
            x_range=x_range_xy,
            y_range=y_range,
            bins=full_bins_xy,
            feature=args.feature,
            normalize_heatmap=full_normalize_heatmap_xy
        )
        
        data_loader_xz_full = DataLoader(
            full_dataset_xz,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=full_collate_fn,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        data_loader_xy_full = DataLoader(
            full_dataset_xy,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=full_collate_fn,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        print(f"  Full model datasets created (XZ: {len(full_dataset_xz)}, XY: {len(full_dataset_xy)})")
    else:
        print(f"\nFull skeleton models not provided. Using arm model's own shoulder predictions.")
    
    # 3次元評価を実行
    print("\nEvaluating in 3D...")
    metrics = evaluate_models_3d(
        model_xz, model_xy,
        data_loader_xz, data_loader_xy,
        device,
        x_range_xz, z_range,
        x_range_xy, y_range,
        full_skeleton_data_path=full_skeleton_data_path,
        full_model_xz=full_model_xz,
        full_model_xy=full_model_xy,
        data_loader_xz_full=data_loader_xz_full,
        data_loader_xy_full=data_loader_xy_full,
        joint_names=joint_names
    )
    
    # 結果を表示
    print("\n" + "="*60)
    print("3D Evaluation Results")
    print("="*60)
    print(f"MPJPE (3D): {metrics['mpjpe_3d_cm']:.2f} cm")
    print(f"Valid joints: {metrics['valid_joints']}")
    print(f"Total samples: {metrics['total_samples']}")
    print("\nPer-joint MPJPE (3D):")
    joint_names = [
        'L_Shoulder', 'L_Elbow', 'L_Wrist',
        'R_Shoulder', 'R_Elbow', 'R_Wrist'
    ]
    for joint_name in joint_names:
        key = f'{joint_name}_mpjpe_3d_cm'
        if key in metrics:
            value = metrics[key]
            if not np.isnan(value):
                print(f"  {joint_name}: {value:.2f} cm")
            else:
                print(f"  {joint_name}: N/A")
    
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

