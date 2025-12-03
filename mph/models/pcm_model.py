"""
統合PCMモデル（距離適応スプラット + Focal Loss + Soft-Argmax）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .voxelize import MultiScaleVoxelization
from ..losses.pcm_losses import PCMLoss, PCMHead, SoftArgmax3D


class IntegratedPCMModel(nn.Module):
    """
    統合PCMモデル
    """
    
    def __init__(self, in_channels: int = 6, num_joints: int = 22,
                 voxel_sizes: list = [(32, 32, 32), (16, 16, 16)],
                 use_group_norm: bool = True, pos_weight: float = 200.0):
        super().__init__()
        self.num_joints = num_joints
        self.voxel_sizes = voxel_sizes
        
        # マルチスケールボクセル化
        self.voxelization = MultiScaleVoxelization(voxel_sizes)
        
        # 各スケールのPCMヘッド
        self.pcm_heads = nn.ModuleList([
            PCMHead(in_channels, num_joints, use_group_norm)
            for _ in voxel_sizes
        ])
        
        # 特徴融合（6次元入力に対応）
        total_channels = in_channels * len(voxel_sizes)
        self.feature_fusion = nn.Sequential(
            nn.Conv3d(total_channels, 128, 3, padding=1),
            nn.GroupNorm(32, 128) if use_group_norm else nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.GroupNorm(32, 64) if use_group_norm else nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        # 最終PCMヘッド（6次元入力に対応）
        self.final_pcm_head = PCMHead(64, num_joints, use_group_norm)
        
        # 損失関数
        self.pcm_loss = PCMLoss(num_joints, pos_weight)
        
        # Soft-Argmax
        self.soft_argmax = SoftArgmax3D()
    
    def forward(self, points: torch.Tensor, features: torch.Tensor,
                snr: Optional[torch.Tensor] = None,
                gt_heatmaps: Optional[torch.Tensor] = None,
                visibility: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: (N, 3) 点群座標
            features: (N, F) 点群特徴量
            snr: (N,) SNR値（オプション）
            gt_heatmaps: (J, D, H, W) GTヒートマップ（学習時）
            visibility: (J,) 可視性マスク（学習時）
        Returns:
            dict containing:
                - heatmaps: (J, D, H, W) PCMヒートマップ
                - coords: (J, 3) 抽出された座標
                - loss: 損失値（学習時）
        """
        # マルチスケールボクセル化
        voxel_features_list = self.voxelization(points, features, snr)
        
        # 各スケールのPCM予測
        scale_outputs = []
        for i, (voxel_features, pcm_head) in enumerate(zip(voxel_features_list, self.pcm_heads)):
            # バッチ次元を追加
            voxel_features = voxel_features.unsqueeze(0)  # (1, F, D, H, W)
            
            # PCM予測
            output = pcm_head(voxel_features)
            scale_outputs.append(output['heatmaps'].squeeze(0))  # (J, D, H, W)
        
        # 特徴融合
        # 各スケールの特徴量をアップサンプリングして結合
        target_size = self.voxel_sizes[0]  # 最大スケールに合わせる
        fused_features = []
        
        for i, voxel_features in enumerate(voxel_features_list):
            if i == 0:
                # 最大スケールはそのまま
                fused_features.append(voxel_features)
            else:
                # 小さいスケールはアップサンプリング
                upsampled = F.interpolate(
                    voxel_features.unsqueeze(0), 
                    size=target_size, 
                    mode='trilinear', 
                    align_corners=False
                ).squeeze(0)
                fused_features.append(upsampled)
        
        # 特徴量を結合
        fused_features = torch.cat(fused_features, dim=0)  # (F*num_scales, D, H, W)
        fused_features = fused_features.unsqueeze(0)  # (1, F*num_scales, D, H, W)
        
        # 特徴融合
        fused_features = self.feature_fusion(fused_features)  # (1, 64, D, H, W)
        
        # 最終PCM予測
        final_output = self.final_pcm_head(fused_features)
        final_heatmaps = final_output['heatmaps'].squeeze(0)  # (J, D, H, W)
        final_coords = final_output['coords'].squeeze(0)  # (J, 3)
        
        result = {
            'heatmaps': final_heatmaps,
            'coords': final_coords
        }
        
        # 損失計算（学習時）
        if gt_heatmaps is not None:
            loss = self.pcm_loss(
                final_heatmaps.unsqueeze(0),  # (1, J, D, H, W)
                gt_heatmaps.unsqueeze(0),     # (1, J, D, H, W)
                visibility.unsqueeze(0) if visibility is not None else None
            )
            result['loss'] = loss
        
        return result


class PCMWithKinematicConstraints(nn.Module):
    """
    運動学制約付きPCMモデル
    """
    
    def __init__(self, in_channels: int = 6, num_joints: int = 22,
                 bone_pairs: list = None, voxel_sizes: list = [(32, 32, 32)]):
        super().__init__()
        self.num_joints = num_joints
        self.bone_pairs = bone_pairs or []
        
        # 基本PCMモデル
        self.pcm_model = IntegratedPCMModel(in_channels, num_joints, voxel_sizes)
        
        # 運動学制約
        if bone_pairs:
            self.kinematic_constraints = KinematicConstraints(bone_pairs)
        else:
            self.kinematic_constraints = None
    
    def forward(self, points: torch.Tensor, features: torch.Tensor,
                snr: Optional[torch.Tensor] = None,
                gt_heatmaps: Optional[torch.Tensor] = None,
                visibility: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: (N, 3) 点群座標
            features: (N, F) 点群特徴量
            snr: (N,) SNR値（オプション）
            gt_heatmaps: (J, D, H, W) GTヒートマップ（学習時）
            visibility: (J,) 可視性マスク（学習時）
        Returns:
            dict containing:
                - heatmaps: (J, D, H, W) PCMヒートマップ
                - coords: (J, 3) 抽出された座標
                - kinematic_loss: 運動学制約損失（学習時）
                - loss: 総損失（学習時）
        """
        # PCM予測
        pcm_output = self.pcm_model(points, features, snr, gt_heatmaps, visibility)
        
        result = {
            'heatmaps': pcm_output['heatmaps'],
            'coords': pcm_output['coords']
        }
        
        # 運動学制約（学習時）
        if self.kinematic_constraints is not None and gt_heatmaps is not None:
            kinematic_loss = self.kinematic_constraints(pcm_output['coords'])
            result['kinematic_loss'] = kinematic_loss
            
            # 総損失
            if 'loss' in pcm_output:
                total_loss = pcm_output['loss'] + 0.1 * kinematic_loss
                result['loss'] = total_loss
        
        return result


class KinematicConstraints(nn.Module):
    """
    運動学制約
    """
    
    def __init__(self, bone_pairs: list):
        super().__init__()
        self.bone_pairs = bone_pairs
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (J, 3) 関節座標
        Returns:
            kinematic_loss: 運動学制約損失
        """
        if not self.bone_pairs:
            return torch.tensor(0.0, device=coords.device)
        
        # 骨長制約
        bone_lengths = []
        for i, j in self.bone_pairs:
            if i < coords.shape[0] and j < coords.shape[0]:
                bone_vec = coords[j] - coords[i]
                bone_length = torch.norm(bone_vec)
                bone_lengths.append(bone_length)
        
        if bone_lengths:
            bone_lengths = torch.stack(bone_lengths)
            # 骨長の一貫性を保つ（標準偏差を最小化）
            length_std = torch.std(bone_lengths)
            return length_std
        else:
            return torch.tensor(0.0, device=coords.device)
