#!/usr/bin/env python3
"""
損失関数: Focal Loss, Dice Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss（不均衡データ対策）
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, num_classes, H, W) - ロジット
            target: (B, H, W) - ラベル（0 or 1）
        """
        # Cross Entropy Loss
        ce_loss = F.cross_entropy(pred, target, reduction='none')  # (B, H, W)
        
        # 確率を計算
        pt = torch.exp(-ce_loss)  # (B, H, W)
        
        # Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss（領域の重複を重視）
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, num_classes, H, W) - ロジット
            target: (B, H, W) - ラベル（0 or 1）
        """
        # 確率に変換
        pred_prob = pred.softmax(dim=1)  # (B, num_classes, H, W)
        
        # One-hot encoding
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()  # (B, num_classes, H, W)
        
        # Dice係数を計算（各クラスごと）
        intersection = (pred_prob * target_one_hot).sum(dim=(2, 3))  # (B, num_classes)
        union = pred_prob.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # (B, num_classes)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, num_classes)
        
        # Dice Loss（1 - Dice）
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Focal Loss + Dice Loss の組み合わせ
    """
    
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
        dice_lambda: float = 1.0
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.dice_lambda = dice_lambda
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, num_classes, H, W) - ロジット
            target: (B, H, W) - ラベル（0 or 1）
        """
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        total_loss = focal + self.dice_lambda * dice
        
        return total_loss


def create_loss_function(
    loss_type: str = 'focal_dice',
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    dice_smooth: float = 1.0,
    dice_lambda: float = 1.0,
    class_weights: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    損失関数を作成
    
    Args:
        loss_type: 損失関数のタイプ ('focal', 'dice', 'focal_dice', 'ce')
        focal_alpha: Focal Lossのalpha
        focal_gamma: Focal Lossのgamma
        dice_smooth: Dice Lossのsmooth
        dice_lambda: Dice Lossの重み
        class_weights: クラス重み（Cross Entropy用）
    
    Returns:
        criterion: 損失関数
    """
    if loss_type == 'focal':
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == 'dice':
        return DiceLoss(smooth=dice_smooth)
    elif loss_type == 'focal_dice':
        return CombinedLoss(
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            dice_smooth=dice_smooth,
            dice_lambda=dice_lambda
        )
    elif loss_type == 'ce':
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

