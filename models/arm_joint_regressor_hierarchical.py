#!/usr/bin/env python3
"""
xz/xyヒートマップから腕の関節座標を階層的に回帰するCNNモデル（Attention機構付き）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SpatialAttention(nn.Module):
    """
    空間的Attention機構
    関節の位置に応じてヒートマップの重要領域を強調
    
    改善版: 3x3 Convを使用して空間的な関係を捉える
    """
    
    def __init__(self, in_channels: int, use_improved: bool = True):
        super().__init__()
        if use_improved:
            # 改善版: 3x3 Convを使用
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
        else:
            # 従来版: 1x1 Convのみ
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
                nn.ReLU(inplace=True),
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


class JointSpecificAttention(nn.Module):
    """
    関節ごとの専用Attention機構
    各関節（肩、肘、手首）に異なるAttentionマップを生成
    """
    
    def __init__(self, in_channels: int, num_joints: int = 3, use_improved: bool = True):
        super().__init__()
        self.num_joints = num_joints
        
        # 各関節用のAttentionモジュール
        self.attention_modules = nn.ModuleList([
            SpatialAttention(in_channels, use_improved=use_improved) for _ in range(num_joints)
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


class HierarchicalArmJointRegressor(nn.Module):
    """
    xz/xyヒートマップから腕の関節座標を階層的に回帰するCNNモデル（Attention機構付き）
    
    階層的回帰:
    1. 肩を予測
    2. 肩の情報を使って肘を予測
    3. 肩+肘の情報を使って手首を予測
    
    Attention機構:
    - 各関節ごとに異なる空間的Attentionマップを生成
    - 関節の位置に応じてヒートマップの重要領域を強調
    """
    
    def __init__(
        self,
        in_channels: int = 1,  # xzヒートマップは1チャネル
        heatmap_size: int = 50,  # ヒートマップの解像度（50×50）
        num_joints_per_arm: int = 3,  # 1腕あたりの関節数（肩、肘、手首）
        base_channels: int = 32,
        dropout: float = 0.5,
        use_attention: bool = True,
        use_improved_attention: bool = True  # 改善版Attentionを使用するか
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.heatmap_size = heatmap_size
        self.num_joints_per_arm = num_joints_per_arm
        self.base_channels = base_channels
        self.use_attention = use_attention
        
        # Encoder: 特徴抽出（既存と同じ）
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
        
        # Attention機構（オプション）
        if use_attention:
            # 左腕用と右腕用のAttention
            self.attention_left = JointSpecificAttention(
                base_channels * 8, num_joints_per_arm, use_improved=use_improved_attention
            )
            self.attention_right = JointSpecificAttention(
                base_channels * 8, num_joints_per_arm, use_improved=use_improved_attention
            )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特徴量サイズ
        feature_dim = base_channels * 8
        
        # 階層的回帰ヘッド
        
        # 1. 肩の回帰ヘッド（基本特徴量のみ）
        self.shoulder_regressor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 4, 2)  # (x, z) for 1 shoulder
        )
        
        # 2. 肘の回帰ヘッド（基本特徴量 + 肩の情報）
        elbow_input_dim = feature_dim + 2  # 基本特徴量 + 左肩座標
        self.elbow_regressor_left = nn.Sequential(
            nn.Linear(elbow_input_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 4, 2)  # (x, z) for left elbow
        )
        
        elbow_input_dim = feature_dim + 2  # 基本特徴量 + 右肩座標
        self.elbow_regressor_right = nn.Sequential(
            nn.Linear(elbow_input_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 4, 2)  # (x, z) for right elbow
        )
        
        # 3. 手首の回帰ヘッド（基本特徴量 + 肩 + 肘の情報）
        wrist_input_dim = feature_dim + 4  # 基本特徴量 + 肩座標 + 肘座標
        self.wrist_regressor_left = nn.Sequential(
            nn.Linear(wrist_input_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 4, 2)  # (x, z) for left wrist
        )
        
        self.wrist_regressor_right = nn.Sequential(
            nn.Linear(wrist_input_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 4, 2)  # (x, z) for right wrist
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False,
        shoulder_coords: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: (B, in_channels, H, W) - xz/xyヒートマップ
            return_attention: Attentionマップも返すか（デフォルト: False）
            shoulder_coords: (B, 4) - 肩の座標（正規化済み、0-1範囲）
                [L_Shoulder_x, L_Shoulder_z/y, R_Shoulder_x, R_Shoulder_z/y]
                提供された場合、この座標を使用して肘・手首を予測
                提供されない場合、自分で肩を予測
        
        Returns:
            joint_coords: (B, num_joints * 2) - 関節座標（正規化済み、0-1範囲）
                xz平面の場合: [L_Shoulder_x, L_Shoulder_z, L_Elbow_x, L_Elbow_z, L_Wrist_x, L_Wrist_z,
                              R_Shoulder_x, R_Shoulder_z, R_Elbow_x, R_Elbow_z, R_Wrist_x, R_Wrist_z]
                xy平面の場合: [L_Shoulder_x, L_Shoulder_y, L_Elbow_x, L_Elbow_y, L_Wrist_x, L_Wrist_y,
                              R_Shoulder_x, R_Shoulder_y, R_Elbow_x, R_Elbow_y, R_Wrist_x, R_Wrist_y]
            または
            (joint_coords, attention_dict): return_attention=Trueの場合
                - joint_coords: (B, num_joints * 2)
                - attention_dict: {'attention_left': (B, 3, H/8, W/8), 
                                  'attention_right': (B, 3, H/8, W/8)}
        """
        # Encoder
        features = self.encoder(x)  # (B, base_channels*8, H/8, W/8)
        
        # Attention機構を適用（オプション）
        if self.use_attention:
            # 左腕用と右腕用のAttentionマップを生成
            attn_left = self.attention_left(features)  # (B, 3, H/8, W/8)
            attn_right = self.attention_right(features)  # (B, 3, H/8, W/8)
            
            # 各関節ごとにAttentionを適用
            # 肩用Attention（肩が外部から提供される場合でも、特徴量抽出のため使用）
            features_shoulder_left = features * attn_left[:, 0:1, :, :]  # (B, C, H/8, W/8)
            features_shoulder_right = features * attn_right[:, 0:1, :, :]
            
            # 肘用Attention
            features_elbow_left = features * attn_left[:, 1:2, :, :]
            features_elbow_right = features * attn_right[:, 1:2, :, :]
            
            # 手首用Attention
            features_wrist_left = features * attn_left[:, 2:3, :, :]
            features_wrist_right = features * attn_right[:, 2:3, :, :]
        else:
            # Attentionなしの場合、すべて同じ特徴量を使用
            features_shoulder_left = features_shoulder_right = features
            features_elbow_left = features_elbow_right = features
            features_wrist_left = features_wrist_right = features
            attn_left = None
            attn_right = None
        
        # Global Average Pooling
        def pool_features(feat):
            return self.global_pool(feat).squeeze(-1).squeeze(-1)  # (B, C)
        
        # 1. 肩を予測（外部から提供されない場合のみ）
        if shoulder_coords is None:
            global_features_shoulder_left = pool_features(features_shoulder_left)  # (B, base_channels*8)
            global_features_shoulder_right = pool_features(features_shoulder_right)
            
            shoulder_left = self.shoulder_regressor(global_features_shoulder_left)  # (B, 2)
            shoulder_right = self.shoulder_regressor(global_features_shoulder_right)  # (B, 2)
        else:
            # 外部から提供された肩の座標を使用
            # shoulder_coords: (B, 4) = [L_Shoulder_x, L_Shoulder_z/y, R_Shoulder_x, R_Shoulder_z/y]
            shoulder_left = shoulder_coords[:, :2]  # (B, 2)
            shoulder_right = shoulder_coords[:, 2:4]  # (B, 2)
        
        # 2. 肘を予測（肩の情報を使用）
        global_features_elbow_left = pool_features(features_elbow_left)  # (B, base_channels*8)
        global_features_elbow_right = pool_features(features_elbow_right)
        
        elbow_input_left = torch.cat([global_features_elbow_left, shoulder_left], dim=1)  # (B, C+2)
        elbow_input_right = torch.cat([global_features_elbow_right, shoulder_right], dim=1)
        
        elbow_left = self.elbow_regressor_left(elbow_input_left)  # (B, 2)
        elbow_right = self.elbow_regressor_right(elbow_input_right)  # (B, 2)
        
        # 3. 手首を予測（肩 + 肘の情報を使用）
        global_features_wrist_left = pool_features(features_wrist_left)  # (B, base_channels*8)
        global_features_wrist_right = pool_features(features_wrist_right)
        
        wrist_input_left = torch.cat([global_features_wrist_left, shoulder_left, elbow_left], dim=1)  # (B, C+4)
        wrist_input_right = torch.cat([global_features_wrist_right, shoulder_right, elbow_right], dim=1)
        
        wrist_left = self.wrist_regressor_left(wrist_input_left)  # (B, 2)
        wrist_right = self.wrist_regressor_right(wrist_input_right)  # (B, 2)
        
        # 結果を結合: [L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist]
        joint_coords = torch.cat([
            shoulder_left, elbow_left, wrist_left,
            shoulder_right, elbow_right, wrist_right
        ], dim=1)  # (B, 12)
        
        # Attentionマップを返す場合
        if return_attention and self.use_attention:
            attention_dict = {
                'attention_left': attn_left,
                'attention_right': attn_right
            }
            return joint_coords, attention_dict
        
        return joint_coords
    
    def get_num_parameters(self) -> int:
        """モデルのパラメータ数を取得"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_hierarchical_arm_joint_regressor(
    heatmap_size: int = 50,
    num_joints_per_arm: int = 3,
    base_channels: int = 32,
    dropout: float = 0.5,
    use_attention: bool = True,
    use_improved_attention: bool = True
) -> HierarchicalArmJointRegressor:
    """
    xzヒートマップから腕の関節座標を階層的に回帰するモデルを作成（Attention機構付き）
    
    Args:
        heatmap_size: ヒートマップの解像度（デフォルト: 50×50）
        num_joints_per_arm: 1腕あたりの関節数（デフォルト: 3 = 肩、肘、手首）
        base_channels: ベースチャネル数
        dropout: Dropout率
        use_attention: Attention機構を使用するか
        use_improved_attention: 改善版Attentionを使用するか（3x3 Conv使用）
    
    Returns:
        model: HierarchicalArmJointRegressor
    """
    model = HierarchicalArmJointRegressor(
        in_channels=1,
        heatmap_size=heatmap_size,
        num_joints_per_arm=num_joints_per_arm,
        base_channels=base_channels,
        dropout=dropout,
        use_attention=use_attention,
        use_improved_attention=use_improved_attention
    )
    
    return model


def create_hierarchical_arm_joint_regressor_xy(
    heatmap_size: int = 50,
    num_joints_per_arm: int = 3,
    base_channels: int = 32,
    dropout: float = 0.5,
    use_attention: bool = True,
    use_improved_attention: bool = True
) -> HierarchicalArmJointRegressor:
    """
    xyヒートマップから腕の関節座標を階層的に回帰するモデルを作成（Attention機構付き）
    
    Args:
        heatmap_size: ヒートマップの解像度（デフォルト: 50×50）
        num_joints_per_arm: 1腕あたりの関節数（デフォルト: 3 = 肩、肘、手首）
        base_channels: ベースチャネル数
        dropout: Dropout率
        use_attention: Attention機構を使用するか
        use_improved_attention: 改善版Attentionを使用するか（3x3 Conv使用）
    
    Returns:
        model: HierarchicalArmJointRegressor
    """
    # 階層的モデルの構造は平面に依存しないため、同じクラスを使用
    model = HierarchicalArmJointRegressor(
        in_channels=1,
        heatmap_size=heatmap_size,
        num_joints_per_arm=num_joints_per_arm,
        base_channels=base_channels,
        dropout=dropout,
        use_attention=use_attention,
        use_improved_attention=use_improved_attention
    )
    
    return model

