#!/usr/bin/env python3
"""
xyヒートマップから腕の関節座標を回帰するCNNモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ArmJointRegressorXY(nn.Module):
    """
    xyヒートマップから腕の関節座標を回帰するCNNモデル
    
    入力: xyヒートマップ (1, H, W) - 固定範囲、固定解像度
    出力: 腕の関節座標 (12次元)
        [L_Shoulder_x, L_Shoulder_y, L_Elbow_x, L_Elbow_y, L_Wrist_x, L_Wrist_y,
         R_Shoulder_x, R_Shoulder_y, R_Elbow_x, R_Elbow_y, R_Wrist_x, R_Wrist_y]
    """
    
    def __init__(
        self,
        in_channels: int = 1,  # xyヒートマップは1チャネル
        heatmap_size: int = 50,  # ヒートマップの解像度（50×50）
        num_joints: int = 6,  # 腕の関節数（左右各3関節）
        base_channels: int = 32,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.heatmap_size = heatmap_size
        self.num_joints = num_joints
        self.base_channels = base_channels
        
        # Encoder: 特徴抽出（xz版と同じ構造）
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, base_channels, H/2, W/2)
            
            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, base_channels*2, H/4, W/4)
            
            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, base_channels*4, H/8, W/8)
            
            # Block 4
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # エンコーダ後の特徴量サイズを計算
        feature_dim = base_channels * 8
        
        # 回帰ヘッド: 全結合層で関節座標を出力
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 4, num_joints * 2)  # 各関節の(x, y)座標
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) - xyヒートマップ
        
        Returns:
            joint_coords: (B, num_joints * 2) - 関節座標（正規化済み、0-1範囲）
                [L_Shoulder_x, L_Shoulder_y, L_Elbow_x, L_Elbow_y, L_Wrist_x, L_Wrist_y,
                 R_Shoulder_x, R_Shoulder_y, R_Elbow_x, R_Elbow_y, R_Wrist_x, R_Wrist_y]
        """
        # Encoder
        features = self.encoder(x)  # (B, base_channels*8, H/8, W/8)
        
        # Global Average Pooling
        global_features = self.global_pool(features)  # (B, base_channels*8, 1, 1)
        global_features = global_features.squeeze(-1).squeeze(-1)  # (B, base_channels*8)
        
        # 回帰ヘッド
        joint_coords = self.regressor(global_features)  # (B, num_joints * 2)
        
        return joint_coords
    
    def get_num_parameters(self) -> int:
        """モデルのパラメータ数を取得"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_arm_joint_regressor_xy(
    heatmap_size: int = 50,
    num_joints: int = 6,
    base_channels: int = 32,
    dropout: float = 0.5
) -> ArmJointRegressorXY:
    """
    xyヒートマップから腕の関節座標を回帰するモデルを作成
    
    Args:
        heatmap_size: ヒートマップの解像度（デフォルト: 50×50）
        num_joints: 関節数（デフォルト: 6 = 左右各3関節）
        base_channels: ベースチャネル数
        dropout: Dropout率
    
    Returns:
        model: ArmJointRegressorXY
    """
    model = ArmJointRegressorXY(
        in_channels=1,
        heatmap_size=heatmap_size,
        num_joints=num_joints,
        base_channels=base_channels,
        dropout=dropout
    )
    
    return model

