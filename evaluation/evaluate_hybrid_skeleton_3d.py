#!/usr/bin/env python3
"""
ハイブリッド評価スクリプト: 肩はfull model、肘と手首はarm modelを使用

xz平面モデルとxy平面モデルを組み合わせて全骨格（22関節）の3次元評価を行う
- 肩（L_Shoulder, R_Shoulder）: full modelから取得し、arm modelの肘・手首予測に使用
- 肘と手首（L_Elbow, R_Elbow, L_Wrist, R_Wrist）: arm modelから取得（肩はfull modelの予測を使用）
- 他の関節: full modelから取得
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

from heatmap_distal_detection.dataset_arm_joints import ArmJointDataset, collate_fn as arm_collate_fn
from heatmap_distal_detection.dataset_arm_joints_xy import ArmJointDatasetXY
from heatmap_distal_detection.dataset_full_skeleton import FullSkeletonDataset, collate_fn as full_collate_fn
from heatmap_distal_detection.dataset_full_skeleton_xy import FullSkeletonDatasetXY
from heatmap_distal_detection.models.arm_joint_regressor_hierarchical import (
    create_hierarchical_arm_joint_regressor,
    create_hierarchical_arm_joint_regressor_xy
)
from heatmap_distal_detection.models.vit_arm_joint_regressor import create_vit_arm_joint_regressor
from heatmap_distal_detection.models.full_skeleton_regressor import create_full_skeleton_regressor
from heatmap_distal_detection.models.full_skeleton_regressor_xy import create_full_skeleton_regressor_xy


# 関節インデックス（22関節中のインデックス）
# L_Shoulder: 16, R_Shoulder: 17, L_Elbow: 18, R_Elbow: 19, L_Wrist: 20, R_Wrist: 21
SHOULDER_INDICES = [16, 17]  # 肩のインデックス
SHOULDER_NAMES = ['L_Shoulder', 'R_Shoulder']
ELBOW_WRIST_INDICES = [18, 19, 20, 21]  # 肘と手首のインデックス
ELBOW_WRIST_NAMES = ['L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']

# Arm modelの出力順序: [L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist]
ARM_JOINT_ORDER = ['L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']
ARM_ELBOW_WRIST_INDICES_IN_ARM_OUTPUT = [1, 2, 4, 5]  # Arm model出力中の肘と手首のインデックス


def denormalize_coords_2d(
    normalized_coords: np.ndarray,  # (N, num_joints*2)
    x_range: Tuple[float, float],
    second_range: Tuple[float, float]
) -> np.ndarray:
    """
    正規化された2次元座標を元の範囲に戻す
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
    # 形状の確認と修正
    if len(full_pred_2d.shape) == 1:
        raise ValueError(
            f"extract_shoulder_coords_from_full_skeleton: full_pred_2dが1次元配列です。"
            f"shape={full_pred_2d.shape}, 期待される形状=(N, 44)"
        )
    if len(full_pred_2d.shape) != 2:
        raise ValueError(
            f"extract_shoulder_coords_from_full_skeleton: full_pred_2dが2次元配列ではありません。"
            f"shape={full_pred_2d.shape}, 期待される形状=(N, 44)"
        )
    if full_pred_2d.shape[1] != 44:
        raise ValueError(
            f"extract_shoulder_coords_from_full_skeleton: full_pred_2dの2次元目のサイズが44ではありません。"
            f"shape={full_pred_2d.shape}, 期待される形状=(N, 44)"
        )
    
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


def extract_elbow_wrist_from_arm_pred(
    arm_pred_3d: np.ndarray  # (N, 6, 3) - [L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist]
) -> np.ndarray:
    """
    Arm modelの予測から肘と手首のみを抽出
    """
    # Arm modelの出力順序: [L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist]
    # 肘と手首のインデックス: [1, 2, 4, 5] = [L_Elbow, L_Wrist, R_Elbow, R_Wrist]
    return arm_pred_3d[:, ARM_ELBOW_WRIST_INDICES_IN_ARM_OUTPUT, :]  # (N, 4, 3)


def combine_hybrid_predictions(
    full_pred_3d: np.ndarray,  # (N, 22, 3) - Full modelの予測
    arm_pred_3d: np.ndarray,  # (N, 6, 3) - Arm modelの予測
    joint_names: List[str]  # 22関節の名前
) -> np.ndarray:
    """
    Full modelとArm modelの予測を組み合わせる
    - 肘と手首: Arm modelから取得
    - 他の関節: Full modelから取得
    """
    hybrid_pred_3d = full_pred_3d.copy()  # (N, 22, 3)
    
    # Arm modelから肘と手首を抽出
    # arm_elbow_wristの順序: [L_Elbow, L_Wrist, R_Elbow, R_Wrist] (インデックス [0, 1, 2, 3])
    arm_elbow_wrist = extract_elbow_wrist_from_arm_pred(arm_pred_3d)  # (N, 4, 3)
    
    # 肘と手首をhybrid_pred_3dに代入
    # ELBOW_WRIST_NAMESの順序: ['L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'] (22関節中の順序)
    # arm_elbow_wristの順序: [L_Elbow, L_Wrist, R_Elbow, R_Wrist] (インデックス [0, 1, 2, 3])
    # マッピング: L_Elbow -> 0, R_Elbow -> 2, L_Wrist -> 1, R_Wrist -> 3
    elbow_wrist_mapping = {
        'L_Elbow': 0,   # arm_elbow_wrist[:, 0, :]
        'L_Wrist': 1,  # arm_elbow_wrist[:, 1, :]
        'R_Elbow': 2,  # arm_elbow_wrist[:, 2, :]
        'R_Wrist': 3   # arm_elbow_wrist[:, 3, :]
    }
    
    for i, joint_name in enumerate(joint_names):
        if joint_name in elbow_wrist_mapping:
            idx_in_elbow_wrist = elbow_wrist_mapping[joint_name]
            hybrid_pred_3d[:, i, :] = arm_elbow_wrist[:, idx_in_elbow_wrist, :]
    
    return hybrid_pred_3d


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
    """
    gt_joints_3d_list = []
    valid_mask_list = []
    
    num_joints = len(joint_names)  # 22
    
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                gt_joints = sample.get('gt_joints', None)
                
                if gt_joints is not None and len(gt_joints) > 0:
                    gt_joints = np.array(gt_joints)  # (22, 3)
                    if len(gt_joints) >= num_joints:
                        joints_3d = gt_joints[:num_joints]  # (22, 3)
                        valid = ~np.isnan(joints_3d).any(axis=1)  # (22,)
                    else:
                        joints_3d = np.full((num_joints, 3), np.nan, dtype=np.float32)
                        joints_3d[:len(gt_joints)] = gt_joints
                        valid = ~np.isnan(joints_3d).any(axis=1)
                else:
                    joints_3d = np.full((num_joints, 3), np.nan, dtype=np.float32)
                    valid = np.zeros(num_joints, dtype=bool)
                
                gt_joints_3d_list.append(joints_3d)
                valid_mask_list.append(valid)
    
    gt_joints_3d = np.array(gt_joints_3d_list)  # (N, 22, 3)
    valid_mask = np.array(valid_mask_list)  # (N, 22)
    
    return gt_joints_3d, valid_mask


def compute_3d_mpjpe(
    pred_3d: np.ndarray,  # (N, num_joints, 3)
    gt_3d: np.ndarray,  # (N, num_joints, 3)
    valid_mask: np.ndarray,  # (N, num_joints)
    joint_names: List[str]
) -> Dict[str, float]:
    """
    3次元MPJPEを計算
    全22関節と6関節（腕関節）の両方のOverall MPJPEを計算
    """
    num_joints = pred_3d.shape[1]
    
    # 腕関節のインデックス（22関節中のインデックス）
    # L_Shoulder: 16, R_Shoulder: 17, L_Elbow: 18, R_Elbow: 19, L_Wrist: 20, R_Wrist: 21
    arm_joint_indices = [16, 17, 18, 19, 20, 21]
    arm_joint_names = ['L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']
    
    # 各関節の誤差を計算（cm単位）
    joint_errors = []
    
    for i in range(num_joints):
        joint_mask = valid_mask[:, i]  # (N,)
        
        if joint_mask.sum() > 0:
            pred_joint = pred_3d[joint_mask, i, :]  # (M, 3)
            gt_joint = gt_3d[joint_mask, i, :]  # (M, 3)
            
            # 3次元ユークリッド距離（メートル単位）
            error_3d = np.sqrt(np.sum((pred_joint - gt_joint) ** 2, axis=1))  # (M,)
            
            if len(error_3d) > 0:
                joint_errors.append(np.mean(error_3d) * 100)  # cmに変換
            else:
                joint_errors.append(np.nan)
        else:
            joint_errors.append(np.nan)
    
    # 全22関節の平均誤差
    valid_errors = [e for e in joint_errors if not np.isnan(e)]
    mpjpe_3d_all = np.mean(valid_errors) if len(valid_errors) > 0 else np.nan
    
    # 6関節（腕関節）の平均誤差
    arm_joint_errors = []
    for idx in arm_joint_indices:
        if idx < len(joint_errors) and not np.isnan(joint_errors[idx]):
            arm_joint_errors.append(joint_errors[idx])
    mpjpe_3d_arm = np.mean(arm_joint_errors) if len(arm_joint_errors) > 0 else np.nan
    
    # JSONシリアライズ可能な型に変換
    metrics = {
        'mpjpe_3d_cm_all_joints': float(mpjpe_3d_all) if not np.isnan(mpjpe_3d_all) else None,
        'mpjpe_3d_cm_arm_joints': float(mpjpe_3d_arm) if not np.isnan(mpjpe_3d_arm) else None,
        'mpjpe_3d_cm': float(mpjpe_3d_all) if not np.isnan(mpjpe_3d_all) else None,  # 後方互換性のため
        'total_samples': int(pred_3d.shape[0]),
        'valid_joints': int(valid_mask.sum())
    }
    
    # 関節ごとの誤差
    for i, joint_name in enumerate(joint_names):
        error_value = joint_errors[i]
        metrics[f'{joint_name}_mpjpe_3d_cm'] = float(error_value) if not np.isnan(error_value) else None
    
    return metrics


def evaluate_hybrid_models_3d(
    arm_model_xz,
    arm_model_xy,
    full_model_xz,
    full_model_xy,
    data_loader_xz_arm,
    data_loader_xy_arm,
    data_loader_xz_full,
    data_loader_xy_full,
    device,
    x_range_xz: Tuple[float, float],
    z_range: Tuple[float, float],
    x_range_xy: Tuple[float, float],
    y_range: Tuple[float, float],
    joint_names: List[str],
    gt_3d: np.ndarray,
    valid_mask: np.ndarray,
    arm_model_type_xz: str = 'hierarchical',
    arm_model_type_xy: str = 'hierarchical'
) -> Dict[str, float]:
    """
    ハイブリッド評価を実行
    
    肩は全身モデルから取得し、それを腕特化モデルの肘・手首予測に使用
    """
    print("\nEvaluating hybrid models:")
    print("  - Shoulders: from full model (used for arm model prediction)")
    print("  - Elbows and wrists: from arm model (using full model's shoulder predictions)")
    print("  - Other joints: from full model")
    
    arm_model_xz.eval()
    arm_model_xy.eval()
    full_model_xz.eval()
    full_model_xy.eval()
    
    all_arm_pred_xz = []
    all_arm_pred_xy = []
    all_full_pred_xz = []
    all_full_pred_xy = []
    
    # file_pathを保存（サンプル数の不一致を解決するため）
    file_paths_xz_arm = []
    file_paths_xy_arm = []
    file_paths_xz_full = []
    file_paths_xy_full = []
    
    # Full modelの予測を先に取得（肩の座標を取得するため）
    print("  Getting predictions from full models...")
    with torch.no_grad():
        for batch in tqdm(data_loader_xz_full, desc="  Full model XZ"):
            heatmaps = batch['heatmap'].to(device)
            pred = full_model_xz(heatmaps).cpu().numpy()
            all_full_pred_xz.append(pred)
            
            # file_pathを保存
            metadata_list = batch.get('metadata', [])
            for metadata in metadata_list:
                file_paths_xz_full.append(metadata.get('file_path', ''))
        
        for batch in tqdm(data_loader_xy_full, desc="  Full model XY"):
            heatmaps = batch['heatmap'].to(device)
            pred = full_model_xy(heatmaps).cpu().numpy()
            all_full_pred_xy.append(pred)
            
            # file_pathを保存
            metadata_list = batch.get('metadata', [])
            for metadata in metadata_list:
                file_paths_xy_full.append(metadata.get('file_path', ''))
    
    # 結合
    all_full_pred_xz = np.concatenate(all_full_pred_xz, axis=0)  # (N, 44)
    all_full_pred_xy = np.concatenate(all_full_pred_xy, axis=0)  # (N, 44)
    
    # 形状の確認（デバッグ用）
    print(f"  [DEBUG] all_full_pred_xz.shape={all_full_pred_xz.shape}")
    print(f"  [DEBUG] all_full_pred_xy.shape={all_full_pred_xy.shape}")
    
    # 2次元配列であることを保証
    if len(all_full_pred_xz.shape) == 1:
        # 1次元配列の場合、2次元に変換
        if all_full_pred_xz.shape[0] % 44 == 0:
            num_samples_xz = all_full_pred_xz.shape[0] // 44
            all_full_pred_xz = all_full_pred_xz.reshape(num_samples_xz, 44)
            print(f"  [DEBUG] all_full_pred_xzを2次元配列に変換: {all_full_pred_xz.shape}")
        else:
            raise ValueError(
                f"all_full_pred_xzの要素数が44の倍数ではありません。"
                f"shape={all_full_pred_xz.shape}, 要素数={all_full_pred_xz.shape[0]}"
            )
    if len(all_full_pred_xy.shape) == 1:
        # 1次元配列の場合、2次元に変換
        if all_full_pred_xy.shape[0] % 44 == 0:
            num_samples_xy = all_full_pred_xy.shape[0] // 44
            all_full_pred_xy = all_full_pred_xy.reshape(num_samples_xy, 44)
            print(f"  [DEBUG] all_full_pred_xyを2次元配列に変換: {all_full_pred_xy.shape}")
        else:
            raise ValueError(
                f"all_full_pred_xyの要素数が44の倍数ではありません。"
                f"shape={all_full_pred_xy.shape}, 要素数={all_full_pred_xy.shape[0]}"
            )
    
    # 全身モデルから肩の座標を抽出（正規化済み）
    shoulder_coords_xz = extract_shoulder_coords_from_full_skeleton(all_full_pred_xz, joint_names)  # (N, 4)
    shoulder_coords_xy = extract_shoulder_coords_from_full_skeleton(all_full_pred_xy, joint_names)  # (N, 4)
    
    # Arm modelの予測を取得
    if arm_model_type_xz == 'vit' or arm_model_type_xy == 'vit':
        print("  Getting predictions from arm models (ViT: predicting all 6 joints, extracting elbow/wrist)...")
    else:
        print("  Getting predictions from arm models (using shoulder coords from full model)...")
    
    with torch.no_grad():
        sample_idx = 0
        for batch in tqdm(data_loader_xz_arm, desc="  Arm model XZ"):
            heatmaps = batch['heatmap'].to(device)
            batch_size = heatmaps.shape[0]
            
            # file_pathを保存
            metadata_list = batch.get('metadata', [])
            for metadata in metadata_list:
                file_paths_xz_arm.append(metadata.get('file_path', ''))
            
            if arm_model_type_xz == 'vit':
                # ViTモデル: 6関節すべてを予測（shoulder_coordsは不要）
                pred_all = arm_model_xz(heatmaps).cpu().numpy()  # (B, 12) - 6関節×2座標
                # 肘・手首のみを抽出: [L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist]
                # → [L_Elbow, L_Wrist, R_Elbow, R_Wrist] = indices [2, 4, 8, 10] (x座標) と [3, 5, 9, 11] (z座標)
                pred = np.zeros((batch_size, 8), dtype=np.float32)  # 4関節×2座標
                pred[:, 0] = pred_all[:, 4]   # L_Elbow_x
                pred[:, 1] = pred_all[:, 5]     # L_Elbow_z
                pred[:, 2] = pred_all[:, 6]   # L_Wrist_x
                pred[:, 3] = pred_all[:, 7]   # L_Wrist_z
                pred[:, 4] = pred_all[:, 8]   # R_Elbow_x
                pred[:, 5] = pred_all[:, 9]   # R_Elbow_z
                pred[:, 6] = pred_all[:, 10]  # R_Wrist_x
                pred[:, 7] = pred_all[:, 11]  # R_Wrist_z
            else:
                # Hierarchical CNNモデル: 肩の座標を使用して肘・手首を予測
                end_idx = sample_idx + batch_size
                batch_shoulder_coords = torch.from_numpy(shoulder_coords_xz[sample_idx:end_idx]).to(device)  # (B, 4)
                pred = arm_model_xz(heatmaps, return_attention=False, shoulder_coords=batch_shoulder_coords).cpu().numpy()
            
            all_arm_pred_xz.append(pred)
            sample_idx += batch_size
        
        sample_idx = 0
        for batch in tqdm(data_loader_xy_arm, desc="  Arm model XY"):
            heatmaps = batch['heatmap'].to(device)
            batch_size = heatmaps.shape[0]
            
            # file_pathを保存
            metadata_list = batch.get('metadata', [])
            for metadata in metadata_list:
                file_paths_xy_arm.append(metadata.get('file_path', ''))
            
            if arm_model_type_xy == 'vit':
                # ViTモデル: 6関節すべてを予測（shoulder_coordsは不要）
                pred_all = arm_model_xy(heatmaps).cpu().numpy()  # (B, 12) - 6関節×2座標
                # 肘・手首のみを抽出: [L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist]
                # → [L_Elbow, L_Wrist, R_Elbow, R_Wrist] = indices [2, 4, 8, 10] (x座標) と [3, 5, 9, 11] (y座標)
                pred = np.zeros((batch_size, 8), dtype=np.float32)  # 4関節×2座標
                pred[:, 0] = pred_all[:, 4]   # L_Elbow_x
                pred[:, 1] = pred_all[:, 5]   # L_Elbow_y
                pred[:, 2] = pred_all[:, 6]   # L_Wrist_x
                pred[:, 3] = pred_all[:, 7]   # L_Wrist_y
                pred[:, 4] = pred_all[:, 8]   # R_Elbow_x
                pred[:, 5] = pred_all[:, 9]   # R_Elbow_y
                pred[:, 6] = pred_all[:, 10]  # R_Wrist_x
                pred[:, 7] = pred_all[:, 11]  # R_Wrist_y
            else:
                # Hierarchical CNNモデル: 肩の座標を使用して肘・手首を予測
                end_idx = sample_idx + batch_size
                batch_shoulder_coords = torch.from_numpy(shoulder_coords_xy[sample_idx:end_idx]).to(device)  # (B, 4)
                pred = arm_model_xy(heatmaps, return_attention=False, shoulder_coords=batch_shoulder_coords).cpu().numpy()
            
            all_arm_pred_xy.append(pred)
            sample_idx += batch_size
    
    # 結合
    all_arm_pred_xz = np.concatenate(all_arm_pred_xz, axis=0)  # (N, 8) for ViT or (N, 12) for Hierarchical
    all_arm_pred_xy = np.concatenate(all_arm_pred_xy, axis=0)  # (N, 8) for ViT or (N, 12) for Hierarchical
    all_full_pred_xz = np.concatenate(all_full_pred_xz, axis=0)  # (N, 44)
    all_full_pred_xy = np.concatenate(all_full_pred_xy, axis=0)  # (N, 44)
    
    # サンプル数の不一致をチェック
    num_samples_xz_arm = all_arm_pred_xz.shape[0]
    num_samples_xy_arm = all_arm_pred_xy.shape[0]
    num_samples_xz_full = all_full_pred_xz.shape[0]
    num_samples_xy_full = all_full_pred_xy.shape[0]
    
    if num_samples_xz_arm != num_samples_xy_arm:
        print(f"\n⚠️  サンプル数の不一致が検出されました:")
        print(f"  Arm model XZ: {num_samples_xz_arm}サンプル")
        print(f"  Arm model XY: {num_samples_xy_arm}サンプル")
        print(f"  file_pathでマッチングしてフィルタリングします...")
        
        # file_pathでマッチング
        # XZ平面のfile_pathをセットに変換
        xz_file_paths_set = set(file_paths_xz_arm)
        
        # XY平面のデータをXZ平面のfile_pathに基づいてフィルタリング
        xy_indices_to_keep = []
        for i, file_path in enumerate(file_paths_xy_arm):
            if file_path in xz_file_paths_set:
                xy_indices_to_keep.append(i)
        
        if len(xy_indices_to_keep) != num_samples_xz_arm:
            print(f"  ⚠️  マッチング後もサンプル数が一致しません:")
            print(f"    XZ平面: {num_samples_xz_arm}サンプル")
            print(f"    XY平面（マッチング後）: {len(xy_indices_to_keep)}サンプル")
            print(f"    共通のfile_path: {len(xz_file_paths_set & set(file_paths_xy_arm))}")
            raise ValueError(
                f"file_pathでマッチングしてもサンプル数が一致しません。"
                f"データセットのfile_pathが一致しているか確認してください。"
            )
        
        # XY平面のデータをフィルタリング
        all_arm_pred_xy = all_arm_pred_xy[xy_indices_to_keep]
        file_paths_xy_arm = [file_paths_xy_arm[i] for i in xy_indices_to_keep]
        
        print(f"  ✓ フィルタリング完了: {len(xy_indices_to_keep)}サンプルに一致")
        
        # Full modelのデータも同様にフィルタリング
        if num_samples_xz_full != num_samples_xy_full:
            print(f"  Full modelもフィルタリングします...")
            xz_full_file_paths_set = set(file_paths_xz_full)
            xy_full_indices_to_keep = []
            for i, file_path in enumerate(file_paths_xy_full):
                if file_path in xz_full_file_paths_set:
                    xy_full_indices_to_keep.append(i)
            
            if len(xy_full_indices_to_keep) != num_samples_xz_full:
                # XZ平面のfile_pathに基づいてフィルタリング
                xy_full_indices_to_keep = []
                for i, file_path in enumerate(file_paths_xy_full):
                    if file_path in xz_full_file_paths_set:
                        xy_full_indices_to_keep.append(i)
            
            # フィルタリング前の形状を確認
            print(f"    フィルタリング前: all_full_pred_xy.shape={all_full_pred_xy.shape}, xy_full_indices_to_keep={len(xy_full_indices_to_keep)}")
            
            # フィルタリング前に2次元配列であることを確認
            if len(all_full_pred_xy.shape) == 1:
                # 1次元配列の場合、2次元に変換
                if all_full_pred_xy.shape[0] % 44 == 0:
                    num_samples_xy_before = all_full_pred_xy.shape[0] // 44
                    all_full_pred_xy = all_full_pred_xy.reshape(num_samples_xy_before, 44)
                    print(f"    ⚠️  警告: フィルタリング前にall_full_pred_xyを2次元配列に変換: {all_full_pred_xy.shape}")
                else:
                    raise ValueError(
                        f"all_full_pred_xyの要素数が44の倍数ではありません。"
                        f"shape={all_full_pred_xy.shape}, 要素数={all_full_pred_xy.shape[0]}"
                    )
            
            # フィルタリング（2次元配列を保証）
            if len(xy_full_indices_to_keep) > 0:
                # インデックスをNumPy配列に変換
                xy_full_indices_to_keep = np.array(xy_full_indices_to_keep, dtype=np.int64)
                
                # フィルタリング（2次元配列として）
                all_full_pred_xy = all_full_pred_xy[xy_full_indices_to_keep]
                
                # 形状の確認（フィルタリング後も2次元であることを確認）
                if len(all_full_pred_xy.shape) != 2:
                    raise ValueError(
                        f"フィルタリング後、all_full_pred_xyが2次元配列ではありません。"
                        f"shape={all_full_pred_xy.shape}, 期待される形状=(N, 44)"
                    )
                if all_full_pred_xy.shape[1] != 44:
                    raise ValueError(
                        f"フィルタリング後、all_full_pred_xyの2次元目のサイズが44ではありません。"
                        f"shape={all_full_pred_xy.shape}, 期待される形状=(N, 44)"
                    )
                
                file_paths_xy_full = [file_paths_xy_full[i] for i in xy_full_indices_to_keep]
                
                print(f"    フィルタリング後: all_full_pred_xy.shape={all_full_pred_xy.shape}")
                
                # 肩の座標も再計算
                shoulder_coords_xy = extract_shoulder_coords_from_full_skeleton(all_full_pred_xy, joint_names)  # (N, 4)
            else:
                raise ValueError(
                    f"Full model XYのフィルタリングで一致するサンプルが見つかりませんでした。"
                    f"XZ平面のfile_pathと一致するXY平面のサンプルが存在しません。"
                )
        
        # GTデータもフィルタリング（GTデータはXZ平面のデータから読み込まれるため、通常は一致している）
        if gt_3d.shape[0] != num_samples_xz_arm:
            print(f"  ⚠️  GTデータのサンプル数が一致しません: {gt_3d.shape[0]} -> {num_samples_xz_arm}")
            if gt_3d.shape[0] > num_samples_xz_arm:
                # GTデータが多すぎる場合、最初のnum_samples_xz_armサンプルを使用
                print(f"    最初の{num_samples_xz_arm}サンプルを使用します")
                gt_3d = gt_3d[:num_samples_xz_arm]
                valid_mask = valid_mask[:num_samples_xz_arm]
            else:
                raise ValueError(
                    f"GTデータのサンプル数が不足しています: "
                    f"GT={gt_3d.shape[0]}, 必要={num_samples_xz_arm}"
                )
    
    # 最終的なサンプル数の確認
    num_samples = all_arm_pred_xz.shape[0]
    assert all_arm_pred_xy.shape[0] == num_samples, f"フィルタリング後もサンプル数が一致しません: XZ={num_samples}, XY={all_arm_pred_xy.shape[0]}"
    assert gt_3d.shape[0] == num_samples, f"GTデータのサンプル数が一致しません: GT={gt_3d.shape[0]}, 必要={num_samples}"
    assert valid_mask.shape[0] == num_samples, f"Valid maskのサンプル数が一致しません: Valid={valid_mask.shape[0]}, 必要={num_samples}"
    
    # 3次元座標に変換
    arm_pred_3d_elbow_wrist = combine_2d_predictions_to_3d(
        all_arm_pred_xz, all_arm_pred_xy,
        x_range_xz, z_range, x_range_xy, y_range
    )  # (N, 4, 3) for ViT or (N, 6, 3) for Hierarchical - [L_Elbow, L_Wrist, R_Elbow, R_Wrist] or [L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist]
    
    # ViTモデルの場合、4関節（肘・手首のみ）を6関節形式に変換
    if arm_model_type_xz == 'vit' or arm_model_type_xy == 'vit':
        # 4関節（肘・手首のみ）を6関節形式に変換
        # 肩の座標はFull Skeleton Modelから取得（後でcombine_hybrid_predictionsで使用）
        arm_pred_3d = np.zeros((arm_pred_3d_elbow_wrist.shape[0], 6, 3), dtype=np.float32)
        # [L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist]
        # 肩は後でFull Skeleton Modelから取得するため、ここではNaNまたは0を設定
        arm_pred_3d[:, 0, :] = np.nan  # L_Shoulder (後でFull Skeleton Modelから取得)
        arm_pred_3d[:, 1, :] = arm_pred_3d_elbow_wrist[:, 0, :]  # L_Elbow
        arm_pred_3d[:, 2, :] = arm_pred_3d_elbow_wrist[:, 1, :]  # L_Wrist
        arm_pred_3d[:, 3, :] = np.nan  # R_Shoulder (後でFull Skeleton Modelから取得)
        arm_pred_3d[:, 4, :] = arm_pred_3d_elbow_wrist[:, 2, :]  # R_Elbow
        arm_pred_3d[:, 5, :] = arm_pred_3d_elbow_wrist[:, 3, :]  # R_Wrist
    else:
        # Hierarchical CNNモデルの場合、そのまま使用
        arm_pred_3d = arm_pred_3d_elbow_wrist  # (N, 6, 3)
    
    # Full modelの予測もサンプル数が一致していることを確認
    # まず、形状を確認して2次元配列であることを保証
    if len(all_full_pred_xz.shape) == 1:
        if all_full_pred_xz.shape[0] % 44 == 0:
            num_samples_xz_before = all_full_pred_xz.shape[0] // 44
            all_full_pred_xz = all_full_pred_xz.reshape(num_samples_xz_before, 44)
            print(f"  [DEBUG] all_full_pred_xzを2次元配列に変換: {all_full_pred_xz.shape}")
        else:
            raise ValueError(
                f"all_full_pred_xzの要素数が44の倍数ではありません。"
                f"shape={all_full_pred_xz.shape}, 要素数={all_full_pred_xz.shape[0]}"
            )
    if len(all_full_pred_xy.shape) == 1:
        if all_full_pred_xy.shape[0] % 44 == 0:
            num_samples_xy_before = all_full_pred_xy.shape[0] // 44
            all_full_pred_xy = all_full_pred_xy.reshape(num_samples_xy_before, 44)
            print(f"  [DEBUG] all_full_pred_xyを2次元配列に変換: {all_full_pred_xy.shape}")
        else:
            raise ValueError(
                f"all_full_pred_xyの要素数が44の倍数ではありません。"
                f"shape={all_full_pred_xy.shape}, 要素数={all_full_pred_xy.shape[0]}"
            )
    
    if all_full_pred_xz.shape[0] != num_samples or all_full_pred_xy.shape[0] != num_samples:
        # Full modelのデータもフィルタリングが必要な場合
        if all_full_pred_xz.shape[0] > num_samples:
            all_full_pred_xz = all_full_pred_xz[:num_samples]
            # スライス後も2次元配列であることを確認
            if len(all_full_pred_xz.shape) == 1:
                if all_full_pred_xz.shape[0] % 44 == 0:
                    num_samples_xz_after = all_full_pred_xz.shape[0] // 44
                    all_full_pred_xz = all_full_pred_xz.reshape(num_samples_xz_after, 44)
                else:
                    raise ValueError(
                        f"スライス後、all_full_pred_xzの要素数が44の倍数ではありません。"
                        f"shape={all_full_pred_xz.shape}, 要素数={all_full_pred_xz.shape[0]}"
                    )
        if all_full_pred_xy.shape[0] > num_samples:
            all_full_pred_xy = all_full_pred_xy[:num_samples]
            # スライス後も2次元配列であることを確認
            if len(all_full_pred_xy.shape) == 1:
                if all_full_pred_xy.shape[0] % 44 == 0:
                    num_samples_xy_after = all_full_pred_xy.shape[0] // 44
                    all_full_pred_xy = all_full_pred_xy.reshape(num_samples_xy_after, 44)
                else:
                    raise ValueError(
                        f"スライス後、all_full_pred_xyの要素数が44の倍数ではありません。"
                        f"shape={all_full_pred_xy.shape}, 要素数={all_full_pred_xy.shape[0]}"
                    )
        # 肩の座標も再計算
        shoulder_coords_xz = extract_shoulder_coords_from_full_skeleton(all_full_pred_xz, joint_names)  # (N, 4)
        shoulder_coords_xy = extract_shoulder_coords_from_full_skeleton(all_full_pred_xy, joint_names)  # (N, 4)
    
    full_pred_3d = combine_2d_predictions_to_3d(
        all_full_pred_xz, all_full_pred_xy,
        x_range_xz, z_range, x_range_xy, y_range
    )  # (N, 22, 3)
    
    # ハイブリッド予測を作成
    hybrid_pred_3d = combine_hybrid_predictions(
        full_pred_3d, arm_pred_3d, joint_names
    )  # (N, 22, 3)
    
    # 3次元MPJPEを計算
    metrics = compute_3d_mpjpe(hybrid_pred_3d, gt_3d, valid_mask, joint_names)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate hybrid skeleton model (elbow and wrist from arm model, others from full model)'
    )
    parser.add_argument('--arm_model_xz', type=str, required=True,
                        help='Path to arm model XZ checkpoint')
    parser.add_argument('--arm_model_xy', type=str, required=True,
                        help='Path to arm model XY checkpoint')
    parser.add_argument('--full_model_xz', type=str, required=True,
                        help='Path to full model XZ checkpoint')
    parser.add_argument('--full_model_xy', type=str, required=True,
                        help='Path to full model XY checkpoint')
    parser.add_argument('--data_xz', type=str, required=True,
                        help='Path to XZ plane data (JSONL)')
    parser.add_argument('--data_xy', type=str, required=True,
                        help='Path to XY plane data (JSONL)')
    parser.add_argument('--x_range_xz', type=float, nargs=2, default=[-1.0, 1.0],
                        help='X range for XZ plane')
    parser.add_argument('--z_range', type=float, nargs=2, default=[-1.0, 0.75],
                        help='Z range')
    parser.add_argument('--x_range_xy', type=float, nargs=2, default=[-1.0, 1.0],
                        help='X range for XY plane')
    parser.add_argument('--y_range', type=float, nargs=2, default=[2.0, 5.0],
                        help='Y range')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of bins')
    parser.add_argument('--feature', type=str, default='energy_power',
                        help='Feature name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output', type=str, default='outputs/evaluation_hybrid_skeleton_3d.json',
                        help='Output JSON path')
    parser.add_argument('--normalize_heatmap_arm_xz', action='store_true',
                        help='Normalize heatmap for arm model XZ')
    parser.add_argument('--normalize_heatmap_arm_xy', action='store_true',
                        help='Normalize heatmap for arm model XY')
    parser.add_argument('--normalize_heatmap_full_xz', action='store_true',
                        help='Normalize heatmap for full model XZ')
    parser.add_argument('--normalize_heatmap_full_xy', action='store_true',
                        help='Normalize heatmap for full model XY')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 関節名を読み込み
    joints_def_path = Path(__file__).parent.parent.parent / 'data_specs' / 'joints_def_22.json'
    with open(joints_def_path, 'r') as f:
        joints_def = json.load(f)
    joint_names = joints_def.get('joint_names', [])  # 22関節
    
    print(f"\nJoint names: {joint_names}")
    print(f"Elbow and wrist joints (using arm model): {ELBOW_WRIST_NAMES}")
    
    # モデルを読み込み
    print("\nLoading models...")
    
    # Arm models
    arm_checkpoint_xz = torch.load(args.arm_model_xz, map_location=device)
    arm_config_xz = arm_checkpoint_xz.get('config', {})
    arm_model_type_xz = arm_config_xz.get('model_type', 'hierarchical')
    
    if arm_model_type_xz == 'vit':
        print("  Loading ViT model for XZ plane...")
        arm_model_xz = create_vit_arm_joint_regressor(
            img_size=args.bins,
            patch_size=5,
            num_joints=6,
            model_size=arm_config_xz.get('model_size', 'small'),
            drop_rate=arm_config_xz.get('drop_rate', 0.0),
            attn_drop_rate=arm_config_xz.get('attn_drop_rate', 0.0),
            drop_path_rate=arm_config_xz.get('drop_path_rate', 0.0)
        )
    else:
        print("  Loading Hierarchical CNN model for XZ plane...")
        arm_model_xz = create_hierarchical_arm_joint_regressor(
            heatmap_size=args.bins,
            num_joints_per_arm=3,
            base_channels=arm_config_xz.get('base_channels', 32),
            dropout=arm_config_xz.get('dropout', 0.5),
            use_attention=arm_config_xz.get('use_attention', True),
            use_improved_attention=arm_config_xz.get('use_improved_attention', False)
        )
    
    if 'model_state_dict' in arm_checkpoint_xz:
        arm_model_xz.load_state_dict(arm_checkpoint_xz['model_state_dict'])
    else:
        arm_model_xz.load_state_dict(arm_checkpoint_xz)
    arm_model_xz.to(device)
    arm_model_xz.eval()
    
    arm_checkpoint_xy = torch.load(args.arm_model_xy, map_location=device)
    arm_config_xy = arm_checkpoint_xy.get('config', {})
    arm_model_type_xy = arm_config_xy.get('model_type', 'hierarchical')
    
    if arm_model_type_xy == 'vit':
        print("  Loading ViT model for XY plane...")
        arm_model_xy = create_vit_arm_joint_regressor(
            img_size=args.bins,
            patch_size=5,
            num_joints=6,
            model_size=arm_config_xy.get('model_size', 'small'),
            drop_rate=arm_config_xy.get('drop_rate', 0.0),
            attn_drop_rate=arm_config_xy.get('attn_drop_rate', 0.0),
            drop_path_rate=arm_config_xy.get('drop_path_rate', 0.0)
        )
    else:
        print("  Loading Hierarchical CNN model for XY plane...")
        arm_model_xy = create_hierarchical_arm_joint_regressor_xy(
            heatmap_size=args.bins,
            num_joints_per_arm=3,
            base_channels=arm_config_xy.get('base_channels', 32),
            dropout=arm_config_xy.get('dropout', 0.5),
            use_attention=arm_config_xy.get('use_attention', True),
            use_improved_attention=arm_config_xy.get('use_improved_attention', False)
        )
    
    if 'model_state_dict' in arm_checkpoint_xy:
        arm_model_xy.load_state_dict(arm_checkpoint_xy['model_state_dict'])
    else:
        arm_model_xy.load_state_dict(arm_checkpoint_xy)
    arm_model_xy.to(device)
    arm_model_xy.eval()
    
    # Full models
    full_model_xz = create_full_skeleton_regressor(
        heatmap_size=args.bins,
        num_joints=22,
        base_channels=32,
        dropout=0.5
    )
    full_checkpoint_xz = torch.load(args.full_model_xz, map_location=device)
    if 'model_state_dict' in full_checkpoint_xz:
        full_model_xz.load_state_dict(full_checkpoint_xz['model_state_dict'])
    else:
        full_model_xz.load_state_dict(full_checkpoint_xz)
    full_model_xz.to(device)
    full_model_xz.eval()
    
    full_model_xy = create_full_skeleton_regressor_xy(
        heatmap_size=args.bins,
        num_joints=22,
        base_channels=32,
        dropout=0.5
    )
    full_checkpoint_xy = torch.load(args.full_model_xy, map_location=device)
    if 'model_state_dict' in full_checkpoint_xy:
        full_model_xy.load_state_dict(full_checkpoint_xy['model_state_dict'])
    else:
        full_model_xy.load_state_dict(full_checkpoint_xy)
    full_model_xy.to(device)
    full_model_xy.eval()
    
    print("  All models loaded")
    
    # データセットを読み込み
    print("\nLoading datasets...")
    
    # Arm model用のデータセット
    arm_dataset_xz = ArmJointDataset(
        data_path=args.data_xz,
        x_range=tuple(args.x_range_xz),
        z_range=tuple(args.z_range),
        bins=args.bins,
        feature=args.feature,
        normalize_heatmap=args.normalize_heatmap_arm_xz
    )
    arm_dataset_xy = ArmJointDatasetXY(
        data_path=args.data_xy,
        x_range=tuple(args.x_range_xy),
        y_range=tuple(args.y_range),
        bins=args.bins,
        feature=args.feature,
        normalize_heatmap=args.normalize_heatmap_arm_xy
    )
    
    # Full model用のデータセット
    full_dataset_xz = FullSkeletonDataset(
        data_path=args.data_xz,
        x_range=tuple(args.x_range_xz),
        z_range=tuple(args.z_range),
        bins=args.bins,
        feature=args.feature,
        normalize_heatmap=args.normalize_heatmap_full_xz
    )
    full_dataset_xy = FullSkeletonDatasetXY(
        data_path=args.data_xy,
        x_range=tuple(args.x_range_xy),
        y_range=tuple(args.y_range),
        bins=args.bins,
        feature=args.feature,
        normalize_heatmap=args.normalize_heatmap_full_xy
    )
    
    arm_loader_xz = DataLoader(arm_dataset_xz, batch_size=args.batch_size, shuffle=False,
                               collate_fn=arm_collate_fn, num_workers=0)
    arm_loader_xy = DataLoader(arm_dataset_xy, batch_size=args.batch_size, shuffle=False,
                               collate_fn=arm_collate_fn, num_workers=0)
    full_loader_xz = DataLoader(full_dataset_xz, batch_size=args.batch_size, shuffle=False,
                                collate_fn=full_collate_fn, num_workers=0)
    full_loader_xy = DataLoader(full_dataset_xy, batch_size=args.batch_size, shuffle=False,
                                collate_fn=full_collate_fn, num_workers=0)
    
    print(f"  Arm dataset XZ: {len(arm_dataset_xz)} samples")
    print(f"  Arm dataset XY: {len(arm_dataset_xy)} samples")
    print(f"  Full dataset XZ: {len(full_dataset_xz)} samples")
    print(f"  Full dataset XY: {len(full_dataset_xy)} samples")
    
    # GT関節を読み込み
    print("\nLoading GT joints...")
    gt_3d, valid_mask = load_3d_gt_joints(
        args.data_xz,
        tuple(args.x_range_xz), tuple(args.z_range),
        tuple(args.x_range_xy), tuple(args.y_range),
        joint_names
    )
    print(f"  GT joints shape: {gt_3d.shape}")
    print(f"  Valid joints: {valid_mask.sum()}/{valid_mask.size}")
    
    # 評価を実行
    metrics = evaluate_hybrid_models_3d(
        arm_model_xz, arm_model_xy,
        full_model_xz, full_model_xy,
        arm_loader_xz, arm_loader_xy,
        full_loader_xz, full_loader_xy,
        device,
        tuple(args.x_range_xz), tuple(args.z_range),
        tuple(args.x_range_xy), tuple(args.y_range),
        joint_names,
        gt_3d, valid_mask,
        arm_model_type_xz=arm_model_type_xz,
        arm_model_type_xy=arm_model_type_xy
    )
    
    # 結果を保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 結果を表示
    print("\n" + "="*60)
    print("Hybrid Model Evaluation Results (3D MPJPE)")
    print("="*60)
    mpjpe_value_all = metrics.get('mpjpe_3d_cm_all_joints')
    mpjpe_value_arm = metrics.get('mpjpe_3d_cm_arm_joints')
    if mpjpe_value_all is not None:
        print(f"Overall MPJPE (全22関節平均): {mpjpe_value_all:.2f} cm")
    else:
        print(f"Overall MPJPE (全22関節平均): N/A")
    if mpjpe_value_arm is not None:
        print(f"Overall MPJPE (6関節平均): {mpjpe_value_arm:.2f} cm")
    else:
        print(f"Overall MPJPE (6関節平均): N/A")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid joints: {metrics['valid_joints']}")
    print("\nPer-joint MPJPE (cm):")
    for joint_name in joint_names:
        key = f'{joint_name}_mpjpe_3d_cm'
        if key in metrics:
            value = metrics[key]
            if value is not None:
                if joint_name in ELBOW_WRIST_NAMES:
                    marker = " [ARM, using FULL shoulder]"
                elif joint_name in SHOULDER_NAMES:
                    marker = " [FULL, used for ARM]"
                else:
                    marker = " [FULL]"
                print(f"  {joint_name:15s}: {value:6.2f} cm{marker}")
            else:
                print(f"  {joint_name:15s}: N/A")
    
    print(f"\nResults saved to: {output_path}")
    print("="*60)
    
    # 肘と手首の統計
    print("\nElbow and Wrist Statistics (from Arm Model):")
    elbow_wrist_errors = []
    for joint_name in ELBOW_WRIST_NAMES:
        key = f'{joint_name}_mpjpe_3d_cm'
        if key in metrics:
            value = metrics[key]
            if value is not None:
                elbow_wrist_errors.append(value)
                print(f"  {joint_name}: {value:.2f} cm")
    if elbow_wrist_errors:
        print(f"  Average: {np.mean(elbow_wrist_errors):.2f} cm")
    
    print("="*60)


if __name__ == '__main__':
    main()

