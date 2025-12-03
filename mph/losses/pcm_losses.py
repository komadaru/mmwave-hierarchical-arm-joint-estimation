"""
PCM用の損失関数（Focal Loss + pos_weight対応）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for PCM training
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, J, D, H, W) 予測ヒートマップ
            targets: (B, J, D, H, W) GTヒートマップ
        """
        # BCE損失を計算
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Focal Lossの重みを計算
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PCMLoss(nn.Module):
    """
    PCM用の統合損失関数
    """
    
    def __init__(self, num_joints: int, pos_weight: float = 200.0, 
                 alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.num_joints = num_joints
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.gamma = gamma
        
        # Focal Loss
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        
        # BCE with pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.full((num_joints,), pos_weight),
            reduction='none'
        )
    
    def forward(self, pred_heatmaps: torch.Tensor, gt_heatmaps: torch.Tensor,
                visibility: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_heatmaps: (B, J, D, H, W) 予測ヒートマップ
            gt_heatmaps: (B, J, D, H, W) GTヒートマップ
            visibility: (B, J) 可視性マスク
        """
        B, J, D, H, W = pred_heatmaps.shape
        
        # 可視性マスクの適用
        if visibility is not None:
            # 可視性が低い関節の損失を減重
            visibility_weight = visibility.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, J, 1, 1, 1)
            visibility_weight = visibility_weight.expand(-1, -1, D, H, W)  # (B, J, D, H, W)
            gt_heatmaps = gt_heatmaps * visibility_weight
        
        # BCE損失を計算
        bce_loss = self.bce_loss(pred_heatmaps, gt_heatmaps)
        
        # Focal Lossの重みを適用
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()


class SoftArgmax3D(nn.Module):
    """
    3D Soft-Argmax for coordinate extraction
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmaps: (B, J, D, H, W) ヒートマップ
        Returns:
            coords: (B, J, 3) 抽出された座標
        """
        B, J, D, H, W = heatmaps.shape
        
        # 3D座標グリッドを生成
        z_coords = torch.arange(D, dtype=torch.float32, device=heatmaps.device)
        y_coords = torch.arange(H, dtype=torch.float32, device=heatmaps.device)
        x_coords = torch.arange(W, dtype=torch.float32, device=heatmaps.device)
        
        Z, Y, X = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        coords_3d = torch.stack([X, Y, Z], dim=-1)  # (D, H, W, 3)
        
        # Soft-Argmax計算
        heatmaps_flat = heatmaps.view(B, J, -1)  # (B, J, D*H*W)
        coords_flat = coords_3d.view(-1, 3)  # (D*H*W, 3)
        
        # Softmaxを適用
        weights = F.softmax(heatmaps_flat / self.temperature, dim=-1)  # (B, J, D*H*W)
        
        # 重み付き平均で座標を計算
        coords = torch.matmul(weights, coords_flat)  # (B, J, 3) - ボクセルインデックス (0, 1, ..., W-1)
        
        # ボクセルインデックス → 正規化座標 [-1, 1] に変換
        # coords の順序は (X, Y, Z) なので、対応するサイズは (W-1, H-1, D-1)
        scale = torch.tensor([W - 1, H - 1, D - 1], dtype=torch.float32, device=coords.device)
        coords_normalized = (coords / scale) * 2.0 - 1.0  # (B, J, 3) - 正規化座標 [-1, 1]
        
        return coords_normalized


class PCMHead(nn.Module):
    """
    PCM予測ヘッド（GroupNorm + Soft-Argmax対応）
    """
    
    def __init__(self, in_channels: int, num_joints: int, 
                 use_group_norm: bool = True, num_groups: int = 32):
        super().__init__()
        self.num_joints = num_joints
        
        # 特徴抽出
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 128, 3, padding=1),
            nn.GroupNorm(num_groups, 128) if use_group_norm else nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.GroupNorm(num_groups, 64) if use_group_norm else nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        # PCM予測ヘッド
        self.pcm_head = nn.Conv3d(64, num_joints, 1)
        
        # Soft-Argmax
        self.soft_argmax = SoftArgmax3D()
    
    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: (B, C, D, H, W) 入力特徴量
        Returns:
            dict containing:
                - heatmaps: (B, J, D, H, W) PCMヒートマップ
                - coords: (B, J, 3) 抽出された座標
        """
        # 特徴抽出
        x = self.feature_extractor(features)
        
        # PCM予測（logits）
        heatmaps = self.pcm_head(x)  # (B, J, D, H, W)
        
        # Soft-Argmaxで座標抽出
        coords = self.soft_argmax(heatmaps)
        
        return {
            'heatmaps': heatmaps,
            'coords': coords
        }
