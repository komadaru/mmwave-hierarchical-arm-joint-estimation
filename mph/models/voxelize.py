"""
距離適応スプラット機能付きボクセル化
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class DistanceAdaptiveSplatting(nn.Module):
    """
    距離適応スプラット機能付きボクセル化
    """
    
    def __init__(self, voxel_size: Tuple[int, int, int] = (32, 32, 32),
                 sigma_base: float = 1.0, sigma_scale: float = 0.1):
        super().__init__()
        self.voxel_size = voxel_size
        self.sigma_base = sigma_base
        self.sigma_scale = sigma_scale
        
    def forward(self, points: torch.Tensor, features: torch.Tensor,
                snr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            points: (N, 3) 点群座標
            features: (N, F) 点群特徴量
            snr: (N,) SNR値（オプション）
        Returns:
            voxel_features: (F, D, H, W) ボクセル化された特徴量
        """
        N, F = features.shape
        D, H, W = self.voxel_size
        
        # 座標を正規化（-1から1の範囲）
        points_norm = self._normalize_coordinates(points)
        
        # 距離適応σを計算
        sigma = self._compute_adaptive_sigma(points, snr)
        
        # ボクセル化
        voxel_features = self._splat_features(points_norm, features, sigma, F, D, H, W)
        
        return voxel_features
    
    def _normalize_coordinates(self, points: torch.Tensor) -> torch.Tensor:
        """座標を正規化"""
        # 点群の範囲を取得
        min_coords = points.min(dim=0)[0]
        max_coords = points.max(dim=0)[0]
        
        # 中心化
        center = (min_coords + max_coords) / 2
        points_centered = points - center
        
        # 正規化（最大範囲でスケール）
        max_range = (max_coords - min_coords).max()
        if max_range > 0:
            points_norm = points_centered / (max_range / 2)
        else:
            points_norm = points_centered
            
        return points_norm
    
    def _compute_adaptive_sigma(self, points: torch.Tensor, 
                               snr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """距離適応σを計算"""
        N = points.shape[0]
        
        # 基本σ
        sigma = torch.full((N,), self.sigma_base, device=points.device)
        
        # 距離に基づく調整
        center = points.mean(dim=0)
        distances = torch.norm(points - center, dim=1)
        max_distance = distances.max()
        
        if max_distance > 0:
            # 距離が遠いほどσを大きく
            distance_factor = 1 + self.sigma_scale * (distances / max_distance)
            sigma = sigma * distance_factor
        
        # SNRに基づく調整
        if snr is not None:
            # SNRが低いほどσを大きく
            snr_factor = 1 + self.sigma_scale * (1 / (snr + 1e-6))
            sigma = sigma * snr_factor
        
        return sigma
    
    def _splat_features(self, points_norm: torch.Tensor, features: torch.Tensor,
                       sigma: torch.Tensor, F: int, D: int, H: int, W: int) -> torch.Tensor:
        """特徴量をスプラット"""
        N = points_norm.shape[0]
        
        # ボクセル座標に変換
        voxel_coords = (points_norm + 1) * torch.tensor([W-1, H-1, D-1], device=points_norm.device) / 2
        voxel_coords = voxel_coords.long()
        
        # 境界チェック
        voxel_coords[:, 0] = torch.clamp(voxel_coords[:, 0], 0, W-1)
        voxel_coords[:, 1] = torch.clamp(voxel_coords[:, 1], 0, H-1)
        voxel_coords[:, 2] = torch.clamp(voxel_coords[:, 2], 0, D-1)
        
        # ボクセル特徴量を初期化
        voxel_features = torch.zeros(F, D, H, W, device=features.device)
        voxel_weights = torch.zeros(D, H, W, device=features.device)
        
        # 各点についてスプラット
        for i in range(N):
            x, y, z = voxel_coords[i]
            sigma_i = sigma[i]
            
            # ガウシアン分布の範囲を計算
            radius = int(3 * sigma_i) + 1
            
            # 近傍ボクセルを取得
            x_start = max(0, x - radius)
            x_end = min(W, x + radius + 1)
            y_start = max(0, y - radius)
            y_end = min(H, y + radius + 1)
            z_start = max(0, z - radius)
            z_end = min(D, z + radius + 1)
            
            # ガウシアン重みを計算
            for dx in range(x_start, x_end):
                for dy in range(y_start, y_end):
                    for dz in range(z_start, z_end):
                        dist_sq = (dx - x) ** 2 + (dy - y) ** 2 + (dz - z) ** 2
                        weight = torch.exp(-dist_sq / (2 * sigma_i ** 2))
                        
                        # 特徴量を重み付きで加算
                        voxel_features[:, dz, dy, dx] += features[i] * weight
                        voxel_weights[dz, dy, dx] += weight
        
        # 重みで正規化（occで割る）
        voxel_weights = torch.clamp(voxel_weights, min=1e-6)
        voxel_features = voxel_features / voxel_weights.unsqueeze(0)
        
        return voxel_features


class MultiScaleVoxelization(nn.Module):
    """
    マルチスケールボクセル化
    """
    
    def __init__(self, voxel_sizes: list = [(32, 32, 32), (16, 16, 16), (8, 8, 8)]):
        super().__init__()
        self.voxel_sizes = voxel_sizes
        self.splatters = nn.ModuleList([
            DistanceAdaptiveSplatting(voxel_size) for voxel_size in voxel_sizes
        ])
    
    def forward(self, points: torch.Tensor, features: torch.Tensor,
                snr: Optional[torch.Tensor] = None) -> list:
        """
        Args:
            points: (N, 3) 点群座標
            features: (N, F) 点群特徴量
            snr: (N,) SNR値（オプション）
        Returns:
            voxel_features_list: list of (F, D, H, W) マルチスケールボクセル特徴量
        """
        voxel_features_list = []
        
        for splatter in self.splatters:
            voxel_features = splatter(points, features, snr)
            voxel_features_list.append(voxel_features)
        
        return voxel_features_list
