#!/usr/bin/env python3
"""
多軸・多特徴量ヒートマップモデル用の基本モジュール
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=kernel_size//2,
            groups=in_channels, bias=False
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = F.relu6(x)
        
        x = self.pw(x)
        x = self.bn2(x)
        x = F.relu6(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CrossAxisAttention(nn.Module):
    """Cross-Axis Attention (簡易版)"""
    def __init__(self, channels, num_axes=3):
        super().__init__()
        self.num_axes = num_axes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, num_axes, bias=False),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x: (B, channels, H, W) - 既に3軸統合済み
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        weights = self.fc(y)  # (B, num_axes)
        
        # 各軸の特徴に重みを適用（簡易実装）
        # 実際の実装では、軸ごとの特徴を分離してから重み付けする必要がある
        # チャネルを軸ごとに分割して重み付け
        channels_per_axis = c // self.num_axes
        
        if channels_per_axis * self.num_axes == c:
            # チャネルが均等に分割できる場合
            weights_expanded = weights.view(b, self.num_axes, 1, 1)
            x_split = x.view(b, self.num_axes, channels_per_axis, h, w)
            x_weighted = x_split * weights_expanded.unsqueeze(2)
            x = x_weighted.view(b, c, h, w)
        else:
            # チャネルが均等に分割できない場合、重みをチャネル全体に適用
            # 各軸の重みをチャネル数に応じて拡張
            weights_expanded = weights.view(b, self.num_axes, 1, 1)  # (B, num_axes, 1, 1)
            # チャネルを可能な限り均等に分割
            remainder = c % self.num_axes
            x_list = []
            start_idx = 0
            for i in range(self.num_axes):
                end_idx = start_idx + channels_per_axis + (1 if i < remainder else 0)
                x_axis = x[:, start_idx:end_idx, :, :]  # (B, channels_per_axis, H, W)
                # 重みを拡張: (B, 1, 1, 1) -> (B, channels_per_axis, H, W)にブロードキャスト
                x_axis_weighted = x_axis * weights_expanded[:, i:i+1, :, :]
                x_list.append(x_axis_weighted)
                start_idx = end_idx
            x = torch.cat(x_list, dim=1)  # (B, c, H, W)
        
        return x

