#!/usr/bin/env python3
"""
2D CNNベースの末端部位検出モデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal, Optional


class HeatmapDistalDetector2D(nn.Module):
    """
    2D CNNベースの末端部位検出モデル
    
    入力モード:
    - 'diff': 差分ヒートマップのみ (in_channels=1)
    - 'concat': 前フレーム+今フレーム (in_channels=2)
    - 'both': 前フレーム+今フレーム+差分 (in_channels=3)
    """
    
    def __init__(
        self,
        in_channels: int = 2,  # 1: 差分のみ, 2: 前+今, 3: 前+今+差分
        spatial_bins: int = 50,
        feature_bins: int = 50,
        num_classes: int = 2,  # 0: 非末端, 1: 末端
        base_channels: int = 32,
        use_global_classifier: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.spatial_bins = spatial_bins
        self.feature_bins = feature_bins
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.use_global_classifier = use_global_classifier
        
        # Encoder: 特徴抽出
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
        
        # Decoder: アップサンプリング（セグメンテーション用）
        self.decoder = nn.Sequential(
            # Upsample Block 1
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Upsample Block 2
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Upsample Block 3
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 最終出力層
            nn.Conv2d(base_channels, num_classes, kernel_size=1),
        )
        
        # オプション: グローバル特徴量を使用した分類
        if use_global_classifier:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Linear(base_channels * 8, base_channels * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(base_channels * 4, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, in_channels, spatial_bins, feature_bins)
        
        Returns:
            output: Dict
                - segmentation: (B, num_classes, spatial_bins, feature_bins) - セグメンテーションマップ
                - classification: (B, num_classes) - グローバル分類（use_global_classifier=Trueの場合のみ）
        """
        # Encoder
        features = self.encoder(x)  # (B, base_channels*8, H/8, W/8)
        
        # Decoder（セグメンテーション）
        segmentation = self.decoder(features)  # (B, num_classes, H, W)
        
        output = {
            'segmentation': segmentation
        }
        
        # グローバル分類（オプション）
        if self.use_global_classifier:
            global_features = self.global_pool(features).squeeze(-1).squeeze(-1)  # (B, base_channels*8)
            classification = self.classifier(global_features)  # (B, num_classes)
            output['classification'] = classification
        
        return output
    
    def get_num_parameters(self) -> int:
        """モデルのパラメータ数を取得"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn_2d_model(
    input_mode: Literal['diff', 'concat', 'both'] = 'diff',
    spatial_bins: int = 50,
    feature_bins: int = 50,
    num_classes: int = 2,
    base_channels: int = 32,
    use_global_classifier: bool = False
) -> HeatmapDistalDetector2D:
    """
    CNN 2Dモデルを作成
    
    Args:
        input_mode: 入力モード
            - 'diff': 差分ヒートマップのみ (in_channels=1)
            - 'concat': 前フレーム+今フレーム (in_channels=2)
            - 'both': 前フレーム+今フレーム+差分 (in_channels=3)
        spatial_bins: 空間軸のビン数
        feature_bins: 特徴量軸のビン数
        num_classes: クラス数
        base_channels: ベースチャネル数
        use_global_classifier: グローバル分類器を使用するか
    
    Returns:
        model: HeatmapDistalDetector2D
    """
    in_channels_map = {
        'diff': 1,
        'concat': 2,
        'both': 3
    }
    
    in_channels = in_channels_map[input_mode]
    
    model = HeatmapDistalDetector2D(
        in_channels=in_channels,
        spatial_bins=spatial_bins,
        feature_bins=feature_bins,
        num_classes=num_classes,
        base_channels=base_channels,
        use_global_classifier=use_global_classifier
    )
    
    return model

