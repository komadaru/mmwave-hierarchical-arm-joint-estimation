#!/usr/bin/env python3
"""
Vision Transformer (ViT) を使用した腕関節回帰モデル

timmライブラリを使用してViTを実装
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class VITArmJointRegressor(nn.Module):
    """
    Vision Transformerを使用した腕関節回帰モデル
    
    入力: ヒートマップ (B, 1, 50, 50)
    出力: 関節座標 (B, 12) - 6関節×2座標（正規化済み、0-1範囲）
    """
    
    def __init__(
        self,
        img_size: int = 50,
        patch_size: int = 5,
        in_chans: int = 1,
        num_joints: int = 6,
        model_size: str = 'small',  # 'small', 'medium', 'large'
        embed_dim: Optional[int] = None,
        depth: Optional[int] = None,
        num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0
    ):
        """
        Args:
            img_size: 入力画像サイズ（デフォルト: 50）
            patch_size: パッチサイズ（デフォルト: 5）
            in_chans: 入力チャネル数（デフォルト: 1 = グレースケール）
            num_joints: 関節数（デフォルト: 6）
            model_size: モデルサイズ（'small', 'medium', 'large'）
            embed_dim: エンベディング次元（Noneの場合はmodel_sizeに応じて自動設定）
            depth: Transformer層数（Noneの場合はmodel_sizeに応じて自動設定）
            num_heads: Attentionヘッド数（Noneの場合はmodel_sizeに応じて自動設定）
            mlp_ratio: MLPの拡大率
            qkv_bias: QKVにbiasを使用するか
            drop_rate: Dropout率
            attn_drop_rate: Attention Dropout率
            drop_path_rate: Drop Path率
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_joints = num_joints
        self.model_size = model_size
        
        # モデルサイズに応じたデフォルト設定
        if model_size == 'small':
            default_embed_dim = 384
            default_depth = 6
            default_num_heads = 6
        elif model_size == 'medium':
            default_embed_dim = 768
            default_depth = 12
            default_num_heads = 12
        elif model_size == 'large':
            default_embed_dim = 1024
            default_depth = 24
            default_num_heads = 16
        else:
            raise ValueError(f"Unknown model_size: {model_size}. Must be 'small', 'medium', or 'large'")
        
        embed_dim = embed_dim if embed_dim is not None else default_embed_dim
        depth = depth if depth is not None else default_depth
        num_heads = num_heads if num_heads is not None else default_num_heads
        
        # timmのViTを作成
        # 注意: timmのViTは通常224×224を想定しているが、カスタマイズ可能
        self.vit = timm.create_model(
            'vit_base_patch16_224',  # ベースモデル名（実際の設定は下で上書き）
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_classes=0  # 分類ヘッドは使用しない
        )
        
        # 出力ヘッド: 関節座標を回帰
        self.head = nn.Linear(embed_dim, num_joints * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) - ヒートマップ
        
        Returns:
            joint_coords: (B, num_joints * 2) - 関節座標（正規化済み、0-1範囲）
                [L_Shoulder_x, L_Shoulder_z/y, L_Elbow_x, L_Elbow_z/y, L_Wrist_x, L_Wrist_z/y,
                 R_Shoulder_x, R_Shoulder_z/y, R_Elbow_x, R_Elbow_z/y, R_Wrist_x, R_Wrist_z/y]
        """
        # ViTで特徴抽出
        features = self.vit.forward_features(x)  # (B, num_patches, embed_dim)
        
        # CLSトークンまたはGlobal Average Pooling
        # timmのViTは通常CLSトークンを使用
        if features.dim() == 3:
            # CLSトークンを使用（最初のトークン）
            cls_token = features[:, 0]  # (B, embed_dim)
        else:
            # Global Average Pooling
            cls_token = features.mean(dim=1)  # (B, embed_dim)
        
        # 関節座標を回帰
        joint_coords = self.head(cls_token)  # (B, num_joints * 2)
        
        return joint_coords
    
    def get_num_parameters(self) -> int:
        """モデルのパラメータ数を取得"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_vit_arm_joint_regressor(
    img_size: int = 50,
    patch_size: int = 5,
    num_joints: int = 6,
    model_size: str = 'small',
    embed_dim: Optional[int] = None,
    depth: Optional[int] = None,
    num_heads: Optional[int] = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.0
) -> VITArmJointRegressor:
    """
    Vision Transformerを使用した腕関節回帰モデルを作成
    
    Args:
        img_size: 入力画像サイズ（デフォルト: 50）
        patch_size: パッチサイズ（デフォルト: 5）
        num_joints: 関節数（デフォルト: 6）
        model_size: モデルサイズ（'small', 'medium', 'large'）
        embed_dim: エンベディング次元（Noneの場合はmodel_sizeに応じて自動設定）
        depth: Transformer層数（Noneの場合はmodel_sizeに応じて自動設定）
        num_heads: Attentionヘッド数（Noneの場合はmodel_sizeに応じて自動設定）
        drop_rate: Dropout率
        attn_drop_rate: Attention Dropout率
        drop_path_rate: Drop Path率
    
    Returns:
        model: VITArmJointRegressor
    """
    model = VITArmJointRegressor(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=1,
        num_joints=num_joints,
        model_size=model_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate
    )
    
    return model

