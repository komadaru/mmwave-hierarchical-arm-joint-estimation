#!/usr/bin/env python3
"""
改善されたAttention機構を持つ階層的腕関節回帰モデル

改善点:
1. より大きな特徴マップ（12×12）でAttentionを計算
2. SpatialAttentionに3x3 Convを使用して空間的な関係を捉える
3. より深いネットワーク
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class ImprovedSpatialAttention(nn.Module):
    """
    改善された空間的Attention機構
    
    3x3 Convを使用して空間的な関係を捉える
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            # 第1層: 3x3 Convで空間的な関係を捉える
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            # 第2層: さらに空間的な関係を捉える
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            # 第3層: 最終的なAttentionマップを生成
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - 特徴マップ
        
        Returns:
            attention_map: (B, 1, H, W) - Attentionマップ
        """
        attention_map = self.conv(x)  # (B, 1, H, W)
        return attention_map


class ImprovedJointSpecificAttention(nn.Module):
    """
    改善された関節ごとの専用Attention機構
    """
    
    def __init__(self, in_channels: int, num_joints: int = 3):
        super().__init__()
        self.num_joints = num_joints
        # 各関節用の改善されたAttentionモジュール
        self.attention_modules = nn.ModuleList([
            ImprovedSpatialAttention(in_channels) for _ in range(num_joints)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - 特徴マップ
        
        Returns:
            attention_maps: (B, num_joints, H, W) - 関節ごとのAttentionマップ
        """
        attention_maps = []
        for attn_module in self.attention_modules:
            attn_map = attn_module(x)  # (B, 1, H, W)
            attention_maps.append(attn_map)
        attention_maps = torch.cat(attention_maps, dim=1)  # (B, num_joints, H, W)
        return attention_maps


class ImprovedHierarchicalArmJointRegressor(nn.Module):
    """
    改善された階層的腕関節回帰モデル
    
    改善点:
    1. Block 3の出力（12×12）でAttentionを計算
    2. 改善されたSpatialAttention（3x3 Conv使用）
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        heatmap_size: int = 50,
        num_joints_per_arm: int = 3,
        base_channels: int = 32,
        dropout: float = 0.5,
        use_attention: bool = True,
        attention_at_block3: bool = True  # Block 3でAttentionを計算するか
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.heatmap_size = heatmap_size
        self.num_joints_per_arm = num_joints_per_arm
        self.base_channels = base_channels
        self.use_attention = use_attention
        self.attention_at_block3 = attention_at_block3
        
        # Encoder: Block 1-3
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, base_channels, H/2, W/2)
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, base_channels*2, H/4, W/4)
        )
        
        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, base_channels*4, H/8, W/8) = (B, 128, 12, 12)
        )
        
        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            # MaxPoolなし → (B, base_channels*8, H/8, W/8) = (B, 256, 12, 12)
        )
        
        # Attention機構（オプション）
        if use_attention:
            if attention_at_block3:
                # Block 3の出力（12×12）でAttentionを計算
                attention_channels = base_channels * 4  # 128
            else:
                # Block 4の出力（6×6）でAttentionを計算（従来通り）
                attention_channels = base_channels * 8  # 256
                # Block 4の後にMaxPoolを追加
                self.encoder_block4.add_module('maxpool', nn.MaxPool2d(2))
            
            self.attention_left = ImprovedJointSpecificAttention(attention_channels, num_joints_per_arm)
            self.attention_right = ImprovedJointSpecificAttention(attention_channels, num_joints_per_arm)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 回帰ヘッド（既存と同じ）
        # ... (既存のコードと同じ) ...
        
        # 注意: 完全な実装には、既存のHierarchicalArmJointRegressorの
        # 回帰ヘッド部分もコピーする必要があります


def create_improved_hierarchical_arm_joint_regressor(
    heatmap_size: int = 50,
    num_joints_per_arm: int = 3,
    base_channels: int = 32,
    dropout: float = 0.5,
    use_attention: bool = True,
    attention_at_block3: bool = True
) -> ImprovedHierarchicalArmJointRegressor:
    """
    改善された階層的腕関節回帰モデルを作成
    
    Args:
        heatmap_size: ヒートマップのサイズ（50×50）
        num_joints_per_arm: 1腕あたりの関節数（3: 肩、肘、手首）
        base_channels: ベースチャネル数
        dropout: Dropout率
        use_attention: Attention機構を使用するか
        attention_at_block3: Block 3でAttentionを計算するか（True: 12×12, False: 6×6）
    
    Returns:
        ImprovedHierarchicalArmJointRegressor: 改善されたモデル
    """
    # 注意: 完全な実装には、既存のcreate_hierarchical_arm_joint_regressorの
    # 回帰ヘッド部分も実装する必要があります
    pass

