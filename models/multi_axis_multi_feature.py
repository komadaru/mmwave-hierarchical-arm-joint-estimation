#!/usr/bin/env python3
"""
多軸・多特徴量ヒートマップモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .modules import DepthwiseSeparableConv, SEBlock, CrossAxisAttention


class StemBlock(nn.Module):
    """Stem: ペア内融合 & チャネル削減"""
    def __init__(self, in_channels=12, out_channels=32, groups=None):
        super().__init__()
        # Group Conv 1×1: 前後ペア融合
        # groupsが指定されていない場合、in_channels // 2の最大約数を使用
        if groups is None:
            mid_channels = in_channels // 2
            # mid_channelsの約数を見つける（最大6まで）
            for g in range(min(6, mid_channels), 0, -1):
                if mid_channels % g == 0:
                    groups = g
                    break
            else:
                groups = 1
        
        self.group_conv = nn.Conv2d(
            in_channels, in_channels // 2,
            kernel_size=1, groups=groups, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        
        # Depthwise Separable Conv
        self.dw_sep = DepthwiseSeparableConv(
            in_channels // 2, out_channels,
            stride=1, kernel_size=3
        )
        
        # SE注意
        self.se = SEBlock(out_channels, reduction=4)
    
    def forward(self, x):
        # x: (B, 12, 50, 50)
        x = self.group_conv(x)
        x = self.bn1(x)
        x = F.relu6(x)
        
        x = self.dw_sep(x)
        x = self.se(x)
        return x  # (B, 32, 50, 50)


class AxisFusionBlock(nn.Module):
    """Stage A: 軸内・特徴内融合"""
    def __init__(self, in_channels=32, out_channels=48):
        super().__init__()
        # Depthwise Separable Conv × 2
        # 1つ目: in_channels -> out_channels
        # 2つ目: out_channels -> out_channels
        self.dw_sep_blocks = nn.ModuleList([
            DepthwiseSeparableConv(in_channels, out_channels, stride=1),
            DepthwiseSeparableConv(out_channels, out_channels, stride=1)
        ])
    
    def forward(self, x):
        # x: (B, 32, 50, 50) - 既にStemで統合済み
        for block in self.dw_sep_blocks:
            x = block(x)
        return x  # (B, 48, 50, 50)


class CrossAxisFusionBlock(nn.Module):
    """Stage B: 異軸融合"""
    def __init__(self, in_channels=48, out_channels=64, num_axes=3):
        super().__init__()
        # チャネル圧縮
        self.compress = nn.Conv2d(
            in_channels * num_axes, out_channels,
            kernel_size=1, bias=False
        )
        self.bn_compress = nn.BatchNorm2d(out_channels)
        
        # Depthwise Separable Conv × 2
        self.dw_sep_blocks = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, stride=1)
            for _ in range(2)
        ])
        
        # クロスアテンション
        self.cross_attention = CrossAxisAttention(out_channels, num_axes)
    
    def forward(self, x_list):
        # x_list: [x_feat, y_feat, z_feat] 各 (B, 48, 50, 50)
        # 結合
        x = torch.cat(x_list, dim=1)  # (B, 144, 50, 50)
        
        # 圧縮
        x = self.compress(x)
        x = self.bn_compress(x)
        x = F.relu6(x)  # (B, 64, 50, 50)
        
        # Depthwise Separable
        for block in self.dw_sep_blocks:
            x = block(x)
        
        # クロスアテンション
        x = self.cross_attention(x)
        
        return x  # (B, 64, 50, 50)


class LightFPNDecoder(nn.Module):
    """Decoder: FPNライト"""
    def __init__(self, in_channels=64, skip_channels=48, num_classes=2):
        super().__init__()
        # スキップ接続の処理
        self.skip_conv = nn.Conv2d(skip_channels, in_channels, kernel_size=1, bias=False)
        self.bn_skip = nn.BatchNorm2d(in_channels)
        
        # 精緻化
        self.refine = DepthwiseSeparableConv(in_channels, in_channels, stride=1)
        
        # 最終出力
        self.final_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    def forward(self, x, skip_features=None):
        # x: (B, 64, 50, 50)
        # skip_features: (B, 48, 50, 50) - Stage Aのスキップ
        
        if skip_features is not None:
            skip = self.skip_conv(skip_features)
            skip = self.bn_skip(skip)
            skip = F.relu6(skip)
            x = x + skip  # スキップ接続
        
        # 精緻化
        x = self.refine(x)
        
        # 最終出力
        x = self.final_conv(x)  # (B, 2, 50, 50)
        return x


class MultiAxisMultiFeatureModel(nn.Module):
    """多軸・多特徴量ヒートマップモデル"""
    def __init__(
        self,
        in_channels: int = 12,  # 12チャネル（concatモード）
        spatial_bins: int = 50,
        feature_bins: int = 50,
        num_classes: int = 2,
        base_channels: int = 32
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.spatial_bins = spatial_bins
        self.feature_bins = feature_bins
        self.num_classes = num_classes
        
        # Stem
        # groupsは自動調整される（デフォルトでin_channels // 2の最大約数を使用）
        self.stem = StemBlock(in_channels, base_channels, groups=None)
        
        # Stage A: 軸内融合
        self.axis_fusion = AxisFusionBlock(base_channels, int(base_channels * 1.5))
        
        # Stage B: 異軸融合
        # 注意: 実際の実装では、3軸の特徴を分離してから融合する必要がある
        # 簡易実装として、ここでは1つの特徴マップを3つに分割して処理
        # 各軸のチャネル数は int(base_channels * 1.5) // 3
        axis_channels = int(base_channels * 1.5) // 3
        self.cross_axis_fusion = CrossAxisFusionBlock(
            axis_channels, base_channels * 2, num_axes=3
        )
        
        # Decoder
        self.decoder = LightFPNDecoder(
            base_channels * 2, int(base_channels * 1.5), num_classes
        )
    
    def forward(self, x):
        # x: (B, 12, 50, 50)
        
        # Stem
        x = self.stem(x)  # (B, 32, 50, 50)
        
        # Stage A
        x_stage_a = self.axis_fusion(x)  # (B, 48, 50, 50)
        
        # Stage B: 3軸の特徴を分離（簡易実装）
        # 実際には、StemやStage Aで軸ごとに処理する必要がある
        # ここでは、チャネルを3分割して処理
        # 各軸のチャネル数は int(base_channels * 1.5) // 3
        channels_per_axis = x_stage_a.size(1) // 3
        # 余りがある場合は最初の軸に追加
        remainder = x_stage_a.size(1) % 3
        
        x_list = []
        start_idx = 0
        for i in range(3):
            end_idx = start_idx + channels_per_axis + (1 if i < remainder else 0)
            x_list.append(x_stage_a[:, start_idx:end_idx, :, :])
            start_idx = end_idx
        
        x_stage_b = self.cross_axis_fusion(x_list)  # (B, 64, 50, 50)
        
        # Decoder
        output = self.decoder(x_stage_b, x_stage_a)  # (B, 2, 50, 50)
        
        return {
            'segmentation': output
        }


def create_multi_axis_model(
    in_channels: int = 12,
    spatial_bins: int = 50,
    feature_bins: int = 50,
    num_classes: int = 2,
    base_channels: int = 32
) -> MultiAxisMultiFeatureModel:
    """
    多軸・多特徴量モデルを作成
    
    Args:
        in_channels: 入力チャネル数（12: concat, 6: diff, 18: both）
        spatial_bins: 空間軸のビン数
        feature_bins: 特徴量軸のビン数
        num_classes: 出力クラス数（2: 末端/非末端）
        base_channels: ベースチャネル数
    
    Returns:
        model: MultiAxisMultiFeatureModel
    """
    return MultiAxisMultiFeatureModel(
        in_channels=in_channels,
        spatial_bins=spatial_bins,
        feature_bins=feature_bins,
        num_classes=num_classes,
        base_channels=base_channels
    )

