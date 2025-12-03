#!/usr/bin/env python3
"""
xzヒートマップと腕の関節座標のデータセット
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heatmap_distal_detection.visualize_xz_heatmap import create_xz_heatmap


class ArmJointDataset(Dataset):
    """
    xzヒートマップと腕の関節座標のデータセット
    
    入力: xzヒートマップ（固定範囲、固定解像度）
    ラベル: 腕の関節座標（xz平面に投影、固定範囲内で正規化）
    """
    
    def __init__(
        self,
        data_path: str,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        z_range: Tuple[float, float] = (-1.0, 0.75),
        bins: int = 50,
        feature: str = 'energy_power',  # 'velocity' or 'energy_power'
        transform: Optional[callable] = None,
        normalize_heatmap: bool = False  # ヒートマップを0-1に正規化するか
    ):
        """
        Args:
            data_path: JSONL形式のデータファイルパス
            x_range: x軸の固定範囲 (min, max)
            z_range: z軸の固定範囲 (min, max)
            bins: ヒートマップのビン数
            feature: ヒートマップに使用する特徴量
            transform: データ拡張（オプション）
            normalize_heatmap: ヒートマップを0-1に正規化するか（デフォルト: False）
        """
        self.data_path = data_path
        self.x_range = x_range
        self.z_range = z_range
        self.bins = bins
        self.feature = feature
        self.transform = transform
        self.normalize_heatmap = normalize_heatmap
        
        # 関節インデックス（joints_def_22.jsonから）
        # L_Shoulder (16), L_Elbow (18), L_Wrist (20)
        # R_Shoulder (17), R_Elbow (19), R_Wrist (21)
        self.arm_joint_indices = [16, 18, 20, 17, 19, 21]
        self.arm_joint_names = [
            'L_Shoulder', 'L_Elbow', 'L_Wrist',
            'R_Shoulder', 'R_Elbow', 'R_Wrist'
        ]
        
        # データを読み込み
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
        print(f"Fixed ranges: x={x_range}, z={z_range}")
        print(f"Heatmap size: {bins}×{bins}, feature: {feature}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def normalize_coord(
        self,
        coord: float,
        coord_range: Tuple[float, float]
    ) -> float:
        """
        座標を固定範囲内で正規化（0-1範囲にマッピング）
        
        Args:
            coord: 座標値
            coord_range: 固定範囲 (min, max)
        
        Returns:
            normalized_coord: 正規化された座標（0-1範囲）
        """
        min_val, max_val = coord_range
        normalized = (coord - min_val) / (max_val - min_val)
        # 範囲外の値はクリップ
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized
    
    def denormalize_coord(
        self,
        normalized_coord: float,
        coord_range: Tuple[float, float]
    ) -> float:
        """
        正規化された座標を元の範囲に戻す
        
        Args:
            normalized_coord: 正規化された座標（0-1範囲）
            coord_range: 固定範囲 (min, max)
        
        Returns:
            coord: 元の範囲の座標
        """
        min_val, max_val = coord_range
        coord = normalized_coord * (max_val - min_val) + min_val
        return coord
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        データサンプルを取得
        
        Returns:
            sample: Dict
                - heatmap: (1, H, W) - xzヒートマップ
                - joint_coords: (num_joints * 2,) - 関節座標（正規化済み）
                - metadata: Dict - メタデータ
        """
        sample = self.data[idx]
        
        # 点群を読み込み
        file_path = sample.get('file_path', '')
        if not file_path or not os.path.exists(file_path):
            # 点群がない場合は空のヒートマップとNaN座標を返す
            heatmap = np.zeros((self.bins, self.bins), dtype=np.float32)
            heatmap[:] = np.nan
            joint_coords = np.full((len(self.arm_joint_indices) * 2,), np.nan, dtype=np.float32)
        else:
            try:
                from mph.postprocessing import load_points_from_npz
                points, _ = load_points_from_npz(file_path, normalize=False)
                
                # xzヒートマップを作成（固定範囲）
                heatmap, _, _ = create_xz_heatmap(
                    points,
                    feature=self.feature,
                    bins=self.bins,
                    x_range=self.x_range,
                    z_range=self.z_range
                )
                
                # NaNを0に置き換え（学習のため）
                heatmap = np.nan_to_num(heatmap, nan=0.0)
                
                # ヒートマップを0-1の範囲に正規化（オプション、勾配爆発を防ぐため）
                if self.normalize_heatmap:
                    valid_mask = heatmap > 0
                    if valid_mask.any():
                        heatmap_min = heatmap[valid_mask].min()
                        heatmap_max = heatmap[valid_mask].max()
                        if heatmap_max > heatmap_min:
                            heatmap[valid_mask] = (heatmap[valid_mask] - heatmap_min) / (heatmap_max - heatmap_min)
                        # すべて同じ値の場合は0.5に設定
                        else:
                            heatmap[valid_mask] = 0.5
                
                # float32に変換（モデルの重みと型を一致させるため）
                heatmap = heatmap.astype(np.float32)
                
                # GT関節座標を取得
                gt_joints = None
                if 'gt_joints' in sample and sample['gt_joints'] is not None:
                    gt_joints = np.array(sample['gt_joints'])  # (22, 3)
                
                # 腕の関節座標を抽出して正規化
                joint_coords = []
                if gt_joints is not None:
                    for joint_idx in self.arm_joint_indices:
                        if joint_idx < len(gt_joints):
                            joint_3d = gt_joints[joint_idx]  # (3,)
                            x, z = joint_3d[0], joint_3d[2]  # xz平面に投影
                            
                            # 正規化
                            x_norm = self.normalize_coord(x, self.x_range)
                            z_norm = self.normalize_coord(z, self.z_range)
                            
                            joint_coords.extend([x_norm, z_norm])
                        else:
                            # 関節が存在しない場合はNaN
                            joint_coords.extend([np.nan, np.nan])
                else:
                    # GT関節がない場合はNaN
                    joint_coords = [np.nan] * (len(self.arm_joint_indices) * 2)
                
                joint_coords = np.array(joint_coords, dtype=np.float32)
                
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                heatmap = np.zeros((self.bins, self.bins), dtype=np.float32)
                heatmap[:] = np.nan
                joint_coords = np.full((len(self.arm_joint_indices) * 2,), np.nan, dtype=np.float32)
        
        # NumPy配列をTorchテンソルに変換
        heatmap = torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)
        joint_coords = torch.from_numpy(joint_coords)  # (num_joints * 2,)
        
        # データ拡張（オプション）
        if self.transform is not None:
            heatmap, joint_coords = self.transform(heatmap, joint_coords)
        
        metadata = {
            'frame_id': sample.get('frame_id', ''),
            'sequence_id': sample.get('sequence_id', ''),
            'frame_num': sample.get('frame_num', -1),
            'file_path': file_path,
            'x_range': self.x_range,
            'z_range': self.z_range,
            'bins': self.bins,
            'feature': self.feature,
            'arm_joint_indices': self.arm_joint_indices,
            'arm_joint_names': self.arm_joint_names
        }
        
        # GT関節がある場合は追加
        if 'gt_joints' in sample and sample['gt_joints'] is not None:
            metadata['gt_joints'] = sample['gt_joints']
        
        # arm_joint_coords_3dがある場合は追加（3次元評価用）
        if 'arm_joint_coords_3d' in sample and sample['arm_joint_coords_3d'] is not None:
            metadata['arm_joint_coords_3d'] = sample['arm_joint_coords_3d']
        
        return {
            'heatmap': heatmap,
            'joint_coords': joint_coords,
            'metadata': metadata
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    バッチを結合
    
    Args:
        batch: データサンプルのリスト
    
    Returns:
        batched: Dict
            - heatmap: (B, 1, H, W)
            - joint_coords: (B, num_joints * 2)
            - metadata: List[Dict]
    """
    heatmaps = torch.stack([item['heatmap'] for item in batch])
    joint_coords = torch.stack([item['joint_coords'] for item in batch])
    metadata = [item['metadata'] for item in batch]
    
    return {
        'heatmap': heatmaps,
        'joint_coords': joint_coords,
        'metadata': metadata
    }

