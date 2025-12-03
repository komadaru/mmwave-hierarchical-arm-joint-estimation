#!/usr/bin/env python3
"""
Attentionマップから腕特化モデルと全身モデルのどちらを使うかを判定する分類器
エンドツーエンド学習に対応
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class ArmModelSelector(nn.Module):
    """
    Attentionマップからモデル選択を行う分類器（エンドツーエンド学習）
    
    入力: Attentionマップ（左腕用 + 右腕用）
    出力: 腕特化モデルを使用する確率（0-1）
    """
    
    def __init__(
        self,
        attention_size: Tuple[int, int] = (6, 6),  # (H/8, W/8) = (6, 6) for 50×50 heatmap
        num_joints_per_arm: int = 3,  # 肩、肘、手首
        hidden_dim: int = 128
    ):
        super().__init__()
        H, W = attention_size
        self.num_joints_per_arm = num_joints_per_arm
        self.attention_size = attention_size
        
        # 入力サイズ: 左腕(3関節) + 右腕(3関節) = 6関節 × H × W
        input_dim = num_joints_per_arm * 2 * H * W  # 6 * 6 * 6 = 216
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1),  # 2クラス分類（0 or 1）
            nn.Sigmoid()  # 腕特化モデルを使用する確率
        )
    
    def forward(
        self, 
        attention_left: torch.Tensor,  # (B, 3, H, W)
        attention_right: torch.Tensor   # (B, 3, H, W)
    ) -> torch.Tensor:
        """
        Args:
            attention_left: 左腕用Attentionマップ
            attention_right: 右腕用Attentionマップ
        
        Returns:
            probability: (B, 1) - 腕特化モデルを使用する確率（0-1）
        """
        # 左右のAttentionマップを結合
        attention_concat = torch.cat([attention_left, attention_right], dim=1)  # (B, 6, H, W)
        
        # 分類
        prob = self.classifier(attention_concat)  # (B, 1)
        
        return prob
    
    def get_num_parameters(self) -> int:
        """モデルのパラメータ数を取得"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HierarchicalArmJointRegressorWithSelector(nn.Module):
    """
    腕特化モデル + モデル選択器の統合クラス
    （既存のHierarchicalArmJointRegressorをラップ）
    """
    
    def __init__(
        self,
        regressor: nn.Module,  # HierarchicalArmJointRegressor
        selector: ArmModelSelector
    ):
        super().__init__()
        self.regressor = regressor
        self.selector = selector
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False,
        return_selection_prob: bool = False
    ):
        """
        Args:
            x: (B, in_channels, H, W) - ヒートマップ
            return_attention: Attentionマップも返すか
            return_selection_prob: 選択確率も返すか
        
        Returns:
            joint_coords: (B, num_joints * 2)
            または
            (joint_coords, ...): return_attention/return_selection_prob=Trueの場合
        """
        # 腕特化モデルで予測（Attentionマップも取得）
        if return_attention or return_selection_prob:
            result = self.regressor(x, return_attention=True)
            if isinstance(result, tuple):
                joint_coords, attention_dict = result
            else:
                # Attentionが無効な場合
                joint_coords = result
                attention_dict = None
            
            # 選択確率を計算
            if return_selection_prob and attention_dict is not None:
                attn_left = attention_dict['attention_left']
                attn_right = attention_dict['attention_right']
                selection_prob = self.selector(attn_left, attn_right)  # (B, 1)
            
            # 返り値を構築
            result_list = [joint_coords]
            if return_attention and attention_dict is not None:
                result_list.append(attention_dict)
            if return_selection_prob and attention_dict is not None:
                result_list.append(selection_prob)
            
            return tuple(result_list) if len(result_list) > 1 else result_list[0]
        else:
            # 既存の動作（Attentionマップを取得しない）
            return self.regressor(x, return_attention=False)
    
    def get_num_parameters(self) -> int:
        """モデルのパラメータ数を取得"""
        return (self.regressor.get_num_parameters() + 
                self.selector.get_num_parameters())


def create_arm_model_selector(
    attention_size: Tuple[int, int] = (6, 6),
    num_joints_per_arm: int = 3,
    hidden_dim: int = 128
) -> ArmModelSelector:
    """
    モデル選択器を作成
    
    Args:
        attention_size: Attentionマップのサイズ (H/8, W/8)
        num_joints_per_arm: 1腕あたりの関節数（デフォルト: 3 = 肩、肘、手首）
        hidden_dim: 隠れ層の次元数
    
    Returns:
        selector: ArmModelSelector
    """
    selector = ArmModelSelector(
        attention_size=attention_size,
        num_joints_per_arm=num_joints_per_arm,
        hidden_dim=hidden_dim
    )
    return selector

