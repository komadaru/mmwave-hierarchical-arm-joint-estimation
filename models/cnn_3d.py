#!/usr/bin/env python3
"""
3D CNNベースの末端部位検出モデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal, Optional


class HeatmapDistalDetector3D(nn.Module):
    """
    3D CNNベースの末端部位検出モデル
    複数の空間軸（x, y, z）または特徴量軸を統合
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # 複数の空間軸（x, y, z）または特徴量軸
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
        self.base_channels = base_channels
        
        # 3D Encoder
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels, base_channels, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # (B, base_channels, D/2, H/2, W/2)
            
            # Block 2
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # (B, base_channels*2, D/4, H/4, W/4)
            
            # Block 3
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # 3D→2D変換（空間軸を統合）
        self.spatial_pool = nn.AdaptiveAvgPool3d((1, spatial_bins, feature_bins))
        
        # 2D Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, num_classes, kernel_size=1),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, in_channels, spatial_bins, feature_bins)
               または (B, in_channels, depth, spatial_bins, feature_bins) for 3D
        
        Returns:
            output: Dict
                - segmentation: (B, num_classes, spatial_bins, feature_bins)
        """
        # 3D Encoder
        if x.dim() == 4:
            # 2D入力の場合は、depth次元を追加
            x = x.unsqueeze(2)  # (B, C, 1, H, W)
        
        features = self.encoder(x)  # (B, base_channels*4, D', H', W')
        
        # 3D→2D変換
        features_2d = self.spatial_pool(features).squeeze(2)  # (B, base_channels*4, H', W')
        
        # 2D Decoder
        segmentation = self.decoder(features_2d)  # (B, num_classes, H, W)
        
        return {
            'segmentation': segmentation
        }
    
    def get_num_parameters(self) -> int:
        """モデルのパラメータ数を取得"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiScaleHeatmapDistalDetector3D(nn.Module):
    """
    マルチスケール3D CNN
    複数の空間軸（x, y, z）を独立に処理して統合
    """
    
    def __init__(
        self,
        input_mode: Literal['diff', 'concat', 'both'] = 'diff',
        spatial_bins: int = 50,
        feature_bins: int = 50,
        num_classes: int = 2,
        base_channels: int = 32
    ):
        super().__init__()
        
        self.input_mode = input_mode
        in_channels_map = {
            'diff': 1,
            'concat': 2,
            'both': 3
        }
        in_channels = in_channels_map[input_mode]
        
        # 各空間軸用の2D CNN Encoder
        self.encoder_x = self._make_encoder(in_channels, base_channels)
        self.encoder_y = self._make_encoder(in_channels, base_channels)
        self.encoder_z = self._make_encoder(in_channels, base_channels)
        
        # 特徴量統合
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 8 * 3, base_channels * 8, kernel_size=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = self._make_decoder(base_channels * 8, num_classes)
    
    def _make_encoder(self, in_channels: int, base_channels: int) -> nn.Module:
        """2D CNN Encoderを作成"""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
    
    def _make_decoder(self, in_channels: int, num_classes: int) -> nn.Module:
        """2D CNN Decoderを作成"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels // 8, num_classes, kernel_size=1),
        )
    
    def forward(
        self,
        heatmap_x: torch.Tensor,
        heatmap_y: torch.Tensor,
        heatmap_z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            heatmap_x: (B, in_channels, spatial_bins, feature_bins) - x軸ヒートマップ
            heatmap_y: (B, in_channels, spatial_bins, feature_bins) - y軸ヒートマップ
            heatmap_z: (B, in_channels, spatial_bins, feature_bins) - z軸ヒートマップ
        
        Returns:
            output: Dict
                - segmentation: (B, num_classes, spatial_bins, feature_bins)
        """
        # 各軸を独立にエンコード
        feat_x = self.encoder_x(heatmap_x)
        feat_y = self.encoder_y(heatmap_y)
        feat_z = self.encoder_z(heatmap_z)
        
        # 特徴量を結合
        features = torch.cat([feat_x, feat_y, feat_z], dim=1)
        features = self.fusion(features)
        
        # デコード
        segmentation = self.decoder(features)
        
        return {
            'segmentation': segmentation
        }
    
    def get_num_parameters(self) -> int:
        """モデルのパラメータ数を取得"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn_3d_model(
    in_channels: int = 3,
    spatial_bins: int = 50,
    feature_bins: int = 50,
    num_classes: int = 2,
    base_channels: int = 32
) -> HeatmapDistalDetector3D:
    """
    3D CNNモデルを作成
    
    Args:
        in_channels: 入力チャネル数（複数の空間軸または特徴量軸）
        spatial_bins: 空間軸のビン数
        feature_bins: 特徴量軸のビン数
        num_classes: クラス数
        base_channels: ベースチャネル数
    
    Returns:
        model: HeatmapDistalDetector3D
    """
    model = HeatmapDistalDetector3D(
        in_channels=in_channels,
        spatial_bins=spatial_bins,
        feature_bins=feature_bins,
        num_classes=num_classes,
        base_channels=base_channels
    )
    
    return model


def create_multiscale_cnn_3d_model(
    input_mode: Literal['diff', 'concat', 'both'] = 'diff',
    spatial_bins: int = 50,
    feature_bins: int = 50,
    num_classes: int = 2,
    base_channels: int = 32
) -> MultiScaleHeatmapDistalDetector3D:
    """
    マルチスケール3D CNNモデルを作成
    
    Args:
        input_mode: 入力モード
            - 'diff': 差分ヒートマップのみ
            - 'concat': 前フレーム+今フレーム
            - 'both': 前フレーム+今フレーム+差分
        spatial_bins: 空間軸のビン数
        feature_bins: 特徴量軸のビン数
        num_classes: クラス数
        base_channels: ベースチャネル数
    
    Returns:
        model: MultiScaleHeatmapDistalDetector3D
    """
    model = MultiScaleHeatmapDistalDetector3D(
        input_mode=input_mode,
        spatial_bins=spatial_bins,
        feature_bins=feature_bins,
        num_classes=num_classes,
        base_channels=base_channels
    )
    
    return model

