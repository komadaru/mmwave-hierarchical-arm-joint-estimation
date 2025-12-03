"""
マルチ仮説＋不確実性を考慮したデータセット
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os
import json


class MultiHypothesisPCMDataset(Dataset):
    """
    マルチ仮説＋不確実性を考慮したPCMデータセット
    
    特徴:
    - 複数仮説のGT生成
    - 不確実性のラベリング
    - 可視性マスクの生成
    - 骨長制約の計算
    """
    
    def __init__(self, data_root: str, split: str = 'train', 
                 num_components: int = 3, bone_pairs: List[List[int]] = None,
                 occlusion_rate: float = 0.2, noise_level: float = 0.01):
        self.data_root = data_root
        self.split = split
        self.num_components = num_components
        self.bone_pairs = bone_pairs or []
        self.occlusion_rate = occlusion_rate
        self.noise_level = noise_level
        
        # データパスの読み込み
        self.data_paths = self._load_data_paths()
        
        # 骨長制約の計算
        self.bone_lengths = self._compute_bone_lengths()
        
    def _load_data_paths(self) -> List[str]:
        """データパスの読み込み"""
        split_file = os.path.join(self.data_root, f'{self.split}_index.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            # ダミーデータの生成
            return [f'sample_{i:06d}' for i in range(100)]
    
    def _compute_bone_lengths(self) -> np.ndarray:
        """骨長制約の計算"""
        if not self.bone_pairs:
            return np.array([])
        
        # ダミーの骨長制約（実際のデータから計算する場合はここを修正）
        num_bones = len(self.bone_pairs)
        return np.random.uniform(0.1, 0.5, num_bones)  # 0.1m - 0.5m
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """データサンプルの取得"""
        sample_id = self.data_paths[idx]
        
        # ダミーデータの生成（実際のデータローダーに置き換え）
        sample = self._generate_dummy_sample(sample_id)
        
        return sample
    
    def _generate_dummy_sample(self, sample_id: str) -> Dict[str, torch.Tensor]:
        """ダミーデータサンプルの生成（学習最適化）"""
        B = 1  # バッチサイズ1
        
        # 基本パラメータ
        num_joints = 22  # SMPL形式の22関節
        num_bones = len(self.bone_pairs)
        
        # ボクセル特徴量の生成（学習最適化のため適切なサイズに調整）
        # 32x32x32: メモリ効率と学習速度のバランス
        # 実際のmmWaveデータに近い形状を想定
        voxel_features = torch.randn(1, 6, 32, 32, 32)  # (B, C, D, H, W)
        
        # 学習最適化のため、より現実的なデータ分布を生成
        # 中心部により多くの情報を集中
        center_mask = torch.zeros_like(voxel_features)
        center_size = 16
        start = 32 // 2 - center_size // 2
        end = start + center_size
        center_mask[:, :, start:end, start:end, start:end] = 1.0
        voxel_features = voxel_features * center_mask + torch.randn_like(voxel_features) * 0.1
        
        # GT関節座標の生成
        gt_joints = torch.randn(num_joints, 3) * 0.5  # (J, 3) 中心化済み
        
        # 複数仮説のGT生成
        mdn_gt = self._generate_multihypothesis_gt(gt_joints)
        
        # 可視性マスクの生成
        visibility = self._generate_visibility_mask(num_joints)
        
        # 骨長制約の生成
        bone_lengths = torch.tensor(self.bone_lengths, dtype=torch.float32)
        
        # 不確実性ラベルの生成
        uncertainty_labels = self._generate_uncertainty_labels(gt_joints)
        
        # PCMヒートマップの生成
        pcm_heatmaps = self._generate_pcm_heatmaps(gt_joints, voxel_features.shape)
        
        return {
            'voxel_features': voxel_features,
            'gt_joints': gt_joints,
            'mdn_gt': mdn_gt,
            'visibility': visibility,
            'bone_lengths': bone_lengths,
            'uncertainty_labels': uncertainty_labels,
            'pcm_heatmaps': pcm_heatmaps,
            'sample_id': sample_id
        }
    
    def _generate_multihypothesis_gt(self, gt_joints: torch.Tensor) -> Dict[str, torch.Tensor]:
        """複数仮説のGT生成"""
        J, _ = gt_joints.shape
        K = self.num_components
        
        # 各仮説の平均
        mu = gt_joints.unsqueeze(0).expand(K, -1, -1)  # (K, J, 3)
        
        # ノイズを追加して仮説を生成
        noise = torch.randn(K, J, 3) * 0.05  # 5cmのノイズ
        mu = mu + noise
        
        # 各仮説の標準偏差
        sigma = torch.ones(K, J, 3) * 0.02  # 2cmの標準偏差
        
        # 各仮説の重み
        pi = torch.softmax(torch.randn(K), dim=0)  # (K,)
        
        return {
            'mu': mu,
            'sigma': sigma,
            'pi': pi
        }
    
    def _generate_visibility_mask(self, num_joints: int) -> torch.Tensor:
        """可視性マスクの生成"""
        # ランダムに一部の関節を非可視化
        visibility = torch.ones(num_joints)
        num_occluded = int(num_joints * self.occlusion_rate)
        if num_occluded > 0:
            occluded_indices = torch.randperm(num_joints)[:num_occluded]
            visibility[occluded_indices] = 0.0
        
        return visibility
    
    def _generate_uncertainty_labels(self, gt_joints: torch.Tensor) -> Dict[str, torch.Tensor]:
        """不確実性ラベルの生成"""
        J, _ = gt_joints.shape
        
        # 不確実性パラメータ
        nu = torch.rand(J) * 0.1 + 0.01  # 0.01 - 0.11
        alpha = torch.rand(J) * 2.0 + 1.0  # 1.0 - 3.0
        beta = torch.rand(J) * 2.0 + 1.0  # 1.0 - 3.0
        
        # 不確実性と信頼度の計算
        uncertainty = nu / (alpha - 1)
        confidence = alpha / (alpha + beta)
        
        return {
            'nu': nu,
            'alpha': alpha,
            'beta': beta,
            'uncertainty': uncertainty,
            'confidence': confidence
        }


class MultiHypothesisPCMDataModule:
    """
    マルチ仮説PCMデータモジュール
    
    データローダーの設定と管理
    """
    
    def __init__(self, data_root: str, batch_size: int = 4, num_workers: int = 4,
                 num_components: int = 3, bone_pairs: List[List[int]] = None,
                 occlusion_rate: float = 0.2, noise_level: float = 0.01):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_components = num_components
        self.bone_pairs = bone_pairs or []
        self.occlusion_rate = occlusion_rate
        self.noise_level = noise_level
        
        # 骨長制約の計算
        self.bone_lengths = self._compute_bone_lengths()
    
    def _compute_bone_lengths(self) -> np.ndarray:
        """骨長制約の計算"""
        if not self.bone_pairs:
            return np.array([])
        
        # ダミーの骨長制約
        num_bones = len(self.bone_pairs)
        return np.random.uniform(0.1, 0.5, num_bones)
    
    def train_dataloader(self) -> DataLoader:
        """訓練データローダー"""
        dataset = MultiHypothesisPCMDataset(
            self.data_root, 'train', self.num_components, self.bone_pairs,
            self.occlusion_rate, self.noise_level
        )
        
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_multihypothesis_batch
        )
    
    def val_dataloader(self) -> DataLoader:
        """検証データローダー"""
        dataset = MultiHypothesisPCMDataset(
            self.data_root, 'val', self.num_components, self.bone_pairs,
            self.occlusion_rate, self.noise_level
        )
        
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_multihypothesis_batch
        )
    
    def test_dataloader(self) -> DataLoader:
        """テストデータローダー"""
        dataset = MultiHypothesisPCMDataset(
            self.data_root, 'test', self.num_components, self.bone_pairs,
            self.occlusion_rate, self.noise_level
        )
        
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_multihypothesis_batch
        )


def collate_multihypothesis_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    マルチ仮説バッチのコレート関数
    
    Args:
        batch: バッチのリスト
        
    Returns:
        Dict containing:
            - voxel_features: (B, C, D, H, W) ボクセル特徴量
            - gt_joints: (B, J, 3) GT関節座標
            - mdn_gt: 複数仮説のGT
            - visibility: (B, J) 可視性マスク
            - bone_lengths: (B, num_bones) 骨長制約
            - uncertainty_labels: 不確実性ラベル
    """
    # バッチサイズ
    B = len(batch)
    
    # ボクセル特徴量のスタック（形状修正）
    voxel_features_list = [item['voxel_features'] for item in batch]
    # 各サンプルは (1, C, D, H, W) なので、squeeze(0)で (C, D, H, W) にしてからstack
    voxel_features_list = [feat.squeeze(0) for feat in voxel_features_list]
    voxel_features = torch.stack(voxel_features_list, dim=0)  # (B, C, D, H, W)
    
    # GT関節座標のスタック
    gt_joints = torch.stack([item['gt_joints'] for item in batch], dim=0)
    
    # 複数仮説のGTのスタック
    mdn_gt = {
        'mu': torch.stack([item['mdn_gt']['mu'] for item in batch], dim=0),
        'sigma': torch.stack([item['mdn_gt']['sigma'] for item in batch], dim=0),
        'pi': torch.stack([item['mdn_gt']['pi'] for item in batch], dim=0)
    }
    
    # 可視性マスクのスタック
    visibility = torch.stack([item['visibility'] for item in batch], dim=0)
    
    # 骨長制約のスタック
    bone_lengths = torch.stack([item['bone_lengths'] for item in batch], dim=0)
    
    # 不確実性ラベルのスタック
    uncertainty_labels = {
        'nu': torch.stack([item['uncertainty_labels']['nu'] for item in batch], dim=0),
        'alpha': torch.stack([item['uncertainty_labels']['alpha'] for item in batch], dim=0),
        'beta': torch.stack([item['uncertainty_labels']['beta'] for item in batch], dim=0),
        'uncertainty': torch.stack([item['uncertainty_labels']['uncertainty'] for item in batch], dim=0),
        'confidence': torch.stack([item['uncertainty_labels']['confidence'] for item in batch], dim=0)
    }
    
    return {
        'voxel_features': voxel_features,
        'gt_joints': gt_joints,
        'mdn_gt': mdn_gt,
        'visibility': visibility,
        'bone_lengths': bone_lengths,
        'uncertainty_labels': uncertainty_labels
    }
    
    def _generate_pcm_heatmaps(self, gt_joints: torch.Tensor, voxel_shape: Tuple[int, ...]) -> torch.Tensor:
        """PCMヒートマップの生成（距離適応スプラット）"""
        B, C, D, H, W = voxel_shape
        J = gt_joints.shape[0]
        
        # ボクセル座標系に変換（-1から1の範囲をD×H×Wにマッピング）
        # 関節座標をボクセルインデックスに変換
        voxel_coords = torch.zeros(J, 3, dtype=torch.long)
        
        for j in range(J):
            # 関節座標をボクセル座標に変換
            x, y, z = gt_joints[j]
            
            # 正規化座標（-1, 1）をボクセルインデックスに変換
            x_idx = int((x + 1) * (W - 1) / 2)
            y_idx = int((y + 1) * (H - 1) / 2)
            z_idx = int((z + 1) * (D - 1) / 2)
            
            # 境界チェック
            x_idx = max(0, min(W - 1, x_idx))
            y_idx = max(0, min(H - 1, y_idx))
            z_idx = max(0, min(D - 1, z_idx))
            
            voxel_coords[j] = torch.tensor([z_idx, y_idx, x_idx])
        
        # 各関節のヒートマップを生成
        heatmaps = torch.zeros(J, D, H, W)
        
        for j in range(J):
            if voxel_coords[j].sum() > 0:  # 有効な関節の場合
                z_idx, y_idx, x_idx = voxel_coords[j]
                
                # 距離適応スプラット（σを距離に応じて調整）
                sigma = 2.0  # 基本σ
                
                # 3Dガウシアン分布を生成
                z_coords = torch.arange(D, dtype=torch.float32)
                y_coords = torch.arange(H, dtype=torch.float32)
                x_coords = torch.arange(W, dtype=torch.float32)
                
                Z, Y, X = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
                
                # ガウシアン分布
                gaussian = torch.exp(-((Z - z_idx) ** 2 + (Y - y_idx) ** 2 + (X - x_idx) ** 2) / (2 * sigma ** 2))
                
                heatmaps[j] = gaussian
        
        return heatmaps
