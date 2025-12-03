#!/usr/bin/env python3
"""
xzヒートマップと全骨格（22関節）座標のデータセット
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


class FullSkeletonDataset(Dataset):
    """
    xzヒートマップと全骨格（22関節）座標のデータセット
    
    入力: xzヒートマップ（固定範囲、固定解像度）
    ラベル: 全22関節の座標（xz平面に投影、固定範囲内で正規化）
    """
    
    def __init__(
        self,
        data_path: str,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        z_range: Tuple[float, float] = (-1.0, 0.75),
        bins: int = 50,
        feature: str = 'energy_power',  # 'velocity' or 'energy_power'
        transform: Optional[callable] = None,
        joints_def_path: Optional[str] = None,
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
            joints_def_path: joints_def_22.jsonのパス
            normalize_heatmap: ヒートマップを0-1に正規化するか（デフォルト: False）
        """
        self.data_path = data_path
        self.x_range = x_range
        self.z_range = z_range
        self.bins = bins
        self.feature = feature
        self.normalize_heatmap = normalize_heatmap
        self.transform = transform
        
        # 関節定義を読み込み
        if joints_def_path is None:
            possible_paths = [
                "data_specs/joints_def_22.json",
                "../data_specs/joints_def_22.json",
                "../../data_specs/joints_def_22.json",
                "mmwave_pose_reconstraction_pj/data_specs/joints_def_22.json",
                "../mmwave_pose_reconstraction_pj/data_specs/joints_def_22.json",
                "/home/users/grad/2024/24t0010/joints_def_22.json",
                "/cluster/users/grad/2024/24t0010/mmwave_pose_reconstraction_pj/data_specs/joints_def_22.json"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    joints_def_path = path
                    break
        
        if joints_def_path and os.path.exists(joints_def_path):
            with open(joints_def_path, 'r') as f:
                joints_def = json.load(f)
            self.joint_names = joints_def.get("joint_names", [])
        else:
            # デフォルトの関節名
            self.joint_names = ["Pelvis","L_Hip","R_Hip","Spine1","L_Knee","R_Knee","Spine2","L_Ankle","R_Ankle","Spine3","L_Foot","R_Foot","Neck","L_Collar","R_Collar","Head","L_Shoulder","R_Shoulder","L_Elbow","R_Elbow","L_Wrist","R_Wrist"]
        
        self.num_joints = len(self.joint_names)  # 22
        
        # データを読み込み
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
        print(f"Fixed ranges: x={x_range}, z={z_range}")
        print(f"Heatmap size: {bins}×{bins}, feature: {feature}")
        print(f"Number of joints: {self.num_joints}")
    
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
                - joint_coords: (num_joints * 2,) - 全関節座標（正規化済み）
                - metadata: Dict - メタデータ
        """
        sample = self.data[idx]
        
        # 点群を読み込み
        file_path = sample.get('file_path', '')
        if not file_path or not os.path.exists(file_path):
            # 点群がない場合は空のヒートマップとNaN座標を返す
            heatmap = np.zeros((self.bins, self.bins), dtype=np.float32)
            heatmap[:] = np.nan
            joint_coords = np.full((self.num_joints * 2,), np.nan, dtype=np.float32)
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
                
                # 全関節座標を抽出して正規化
                joint_coords = []
                if gt_joints is not None and len(gt_joints) >= self.num_joints:
                    for joint_idx in range(self.num_joints):
                        joint_3d = gt_joints[joint_idx]  # (3,)
                        x, z = joint_3d[0], joint_3d[2]  # xz平面に投影
                        
                        # 正規化
                        x_norm = self.normalize_coord(x, self.x_range)
                        z_norm = self.normalize_coord(z, self.z_range)
                        
                        joint_coords.extend([x_norm, z_norm])
                else:
                    # GT関節がない場合はNaN
                    joint_coords = [np.nan] * (self.num_joints * 2)
                
                joint_coords = np.array(joint_coords, dtype=np.float32)
                
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                heatmap = np.zeros((self.bins, self.bins), dtype=np.float32)
                heatmap[:] = np.nan
                joint_coords = np.full((self.num_joints * 2,), np.nan, dtype=np.float32)
        
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
            'joint_names': self.joint_names,
            'num_joints': self.num_joints
        }
        
        # GT関節がある場合は追加
        if 'gt_joints' in sample and sample['gt_joints'] is not None:
            metadata['gt_joints'] = sample['gt_joints']
        
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

