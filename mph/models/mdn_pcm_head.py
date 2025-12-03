"""
MDN (Mixture Density Network) PCM Head
マルチ仮説＋不確実性を考慮したPCM予測
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List


class MDNPCMHead(nn.Module):
    """
    MDN (Mixture Density Network) を使用したPCM予測
    
    入力: 特徴量 (B, C, D, H, W)
    出力: 複数仮説の平均・共分散・重み (B, J, K, 3), (B, J, K, 3), (B, J, K)
    """
    
    def __init__(self, in_channels: int, num_joints: int, num_components: int = 3):
        super().__init__()
        self.num_joints = num_joints
        self.num_components = num_components
        
        # 特徴量を関節数に投影
        self.joint_projection = nn.Conv3d(in_channels, num_joints, 1)
        
        # MDNパラメータの生成
        # 各関節・各成分に対して: μ(3) + σ(3) + π(1) = 7パラメータ
        self.mdn_head = nn.Conv3d(num_joints, num_joints * num_components * 7, 1)
        
        # 活性化関数
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        MDN PCM予測の実行
        
        Args:
            features: (B, C, D, H, W) 入力特徴量
            
        Returns:
            Dict containing:
                - mu: (B, J, K, 3) 各仮説の平均
                - sigma: (B, J, K, 3) 各仮説の標準偏差
                - pi: (B, J, K) 各仮説の重み
                - logits: (B, J, K) 重みのロジット
        """
        # 形状の柔軟な対応（学習最適化）
        if len(features.shape) == 4:
            # (B, C, H, W) -> (B, C, 1, H, W) に変換
            features = features.unsqueeze(2)
        
        B, C, D, H, W = features.shape
        
        # 関節特徴量の生成
        joint_features = self.joint_projection(features)  # (B, J, D, H, W)
        
        # MDNパラメータの生成
        mdn_params = self.mdn_head(joint_features)  # (B, J*K*7, D, H, W)
        
        # グローバル平均プーリング
        mdn_params = F.adaptive_avg_pool3d(mdn_params, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, J*K*7)
        
        # パラメータの分解
        mdn_params = mdn_params.view(B, self.num_joints, self.num_components, 7)  # (B, J, K, 7)
        
        # 平均 (μ)
        mu = mdn_params[..., :3]  # (B, J, K, 3)
        
        # 標準偏差 (σ) - 正の値にするためSoftplus
        sigma = self.softplus(mdn_params[..., 3:6]) + 1e-6  # (B, J, K, 3)
        
        # 重み (π) - Softmaxで正規化
        logits = mdn_params[..., 6]  # (B, J, K)
        pi = self.softmax(logits)  # (B, J, K)
        
        return {
            'mu': mu,
            'sigma': sigma,
            'pi': pi,
            'logits': logits
        }


class EvidentialPCMHead(nn.Module):
    """
    Evidential Deep Learning を使用したPCM予測
    
    入力: 特徴量 (B, C, D, H, W)
    出力: 不確実性を考慮した予測 (B, J, 3), (B, J), (B, J)
    """
    
    def __init__(self, in_channels: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        
        # 特徴量を関節数に投影
        self.joint_projection = nn.Conv3d(in_channels, num_joints, 1)
        
        # Evidentialパラメータの生成
        # 各関節に対して: μ(3) + ν(1) + α(1) + β(1) = 6パラメータ
        self.evidential_head = nn.Conv3d(num_joints, num_joints * 6, 1)
        
        # 活性化関数
        self.softplus = nn.Softplus()
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evidential PCM予測の実行
        
        Args:
            features: (B, C, D, H, W) 入力特徴量
            
        Returns:
            Dict containing:
                - mu: (B, J, 3) 予測平均
                - uncertainty: (B, J) 不確実性
                - confidence: (B, J) 信頼度
                - evidence: (B, J) エビデンス
        """
        # 形状の柔軟な対応（学習最適化）
        if len(features.shape) == 4:
            # (B, C, H, W) -> (B, C, 1, H, W) に変換
            features = features.unsqueeze(2)
        
        B, C, D, H, W = features.shape
        
        # 関節特徴量の生成
        joint_features = self.joint_projection(features)  # (B, J, D, H, W)
        
        # Evidentialパラメータの生成
        evidential_params = self.evidential_head(joint_features)  # (B, J*6, D, H, W)
        
        # グローバル平均プーリング
        evidential_params = F.adaptive_avg_pool3d(evidential_params, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, J*6)
        
        # パラメータの分解
        evidential_params = evidential_params.view(B, self.num_joints, 6)  # (B, J, 6)
        
        # 平均 (μ)
        mu = evidential_params[..., :3]  # (B, J, 3)
        
        # 不確実性パラメータ
        nu = self.softplus(evidential_params[..., 3]) + 1e-6  # (B, J)
        alpha = self.softplus(evidential_params[..., 4]) + 1e-6  # (B, J)
        beta = self.softplus(evidential_params[..., 5]) + 1e-6  # (B, J)
        
        # 不確実性と信頼度の計算
        # alpha > 1 を保証するため、alpha = alpha + 1
        alpha_safe = alpha + 1.0
        uncertainty = nu / alpha_safe  # (B, J)
        confidence = alpha_safe / (alpha_safe + beta)  # (B, J)
        evidence = alpha_safe + beta  # (B, J)
        
        return {
            'mu': mu,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'evidence': evidence,
            'nu': nu,
            'alpha': alpha,
            'beta': beta
        }


class KinematicHypothesisSelector(nn.Module):
    """
    物理制約を考慮した仮説選択器
    
    骨長制約、可視性制約、物理的妥当性を考慮して最良の仮説を選択
    """
    
    def __init__(self, bone_pairs: List[List[int]], num_joints: int):
        super().__init__()
        self.bone_pairs = bone_pairs
        self.num_joints = num_joints
        
        # 骨長制約のスコア計算
        self.bone_scorer = nn.Sequential(
            nn.Linear(len(bone_pairs), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 可視性制約のスコア計算
        self.visibility_scorer = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hypotheses: Dict[str, torch.Tensor], 
                visibility: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        物理制約を考慮した仮説選択
        
        Args:
            hypotheses: MDNの出力辞書
            visibility: (B, J) 可視性マスク
            
        Returns:
            Dict containing:
                - selected_mu: (B, J, 3) 選択された仮説の平均
                - selected_sigma: (B, J, 3) 選択された仮説の標準偏差
                - selection_scores: (B, J, K) 各仮説のスコア
        """
        B, J, K, _ = hypotheses['mu'].shape
        
        # 各仮説に対してスコアを計算
        scores = []
        for k in range(K):
            mu_k = hypotheses['mu'][:, :, k, :]  # (B, J, 3)
            
            # 骨長制約スコア
            bone_score = self._compute_bone_score(mu_k)  # (B, 1)
            
            # 可視性制約スコア
            visibility_score = self._compute_visibility_score(mu_k, visibility)  # (B, 1)
            
            # 物理的妥当性スコア
            physics_score = self._compute_physics_score(mu_k)  # (B, 1)
            
            # 総合スコア
            total_score = bone_score + visibility_score + physics_score
            scores.append(total_score)
        
        # スコアをスタック
        scores = torch.stack(scores, dim=-1)  # (B, 1, K)
        
        # 最良の仮説を選択
        best_indices = torch.argmax(scores, dim=-1)  # (B, 1)
        
        # 選択された仮説の取得
        # best_indices: (B, 1) -> (B, J, 1) に拡張
        best_indices_expanded = best_indices.unsqueeze(1).expand(-1, J, 1)  # (B, J, 1)
        # torch.gatherのインデックスは (B, J, 1, 3) の形状が必要
        gather_indices = best_indices_expanded.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B, J, 1, 3)
        selected_mu = torch.gather(hypotheses['mu'], 2, gather_indices).squeeze(2)
        selected_sigma = torch.gather(hypotheses['sigma'], 2, gather_indices).squeeze(2)
        
        return {
            'selected_mu': selected_mu,
            'selected_sigma': selected_sigma,
            'selection_scores': scores.squeeze(1)
        }
    
    def _compute_bone_score(self, mu: torch.Tensor) -> torch.Tensor:
        """骨長制約スコアの計算"""
        bone_lengths = []
        for i, j in self.bone_pairs:
            if i < self.num_joints and j < self.num_joints:
                bone_vec = mu[:, j] - mu[:, i]  # (B, 3)
                bone_length = torch.norm(bone_vec, dim=1)  # (B,)
                bone_lengths.append(bone_length)
        
        if bone_lengths:
            bone_lengths = torch.stack(bone_lengths, dim=1)  # (B, num_bones)
            return self.bone_scorer(bone_lengths)  # (B, 1)
        else:
            return torch.ones(mu.shape[0], 1, device=mu.device)
    
    def _compute_visibility_score(self, mu: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
        """可視性制約スコアの計算"""
        # 可視性が低い関節の予測をペナルティ
        # visibility: (B, J) -> 平均可視性を計算してスコア化
        visibility_mean = visibility.mean(dim=1, keepdim=True)  # (B, 1)
        return self.visibility_scorer(visibility_mean)  # (B, 1)
    
    def _compute_physics_score(self, mu: torch.Tensor) -> torch.Tensor:
        """物理的妥当性スコアの計算"""
        # 関節間の距離が合理的な範囲にあるかチェック
        B, J, _ = mu.shape
        physics_score = torch.ones(B, 1, device=mu.device)
        
        # 例: 関節間の最大距離が合理的な範囲内かチェック
        for i in range(J):
            for j in range(i+1, J):
                dist = torch.norm(mu[:, i] - mu[:, j], dim=1)  # (B,)
                # 合理的な範囲 (0.1m - 2.0m)
                reasonable = (dist > 0.1) & (dist < 2.0)
                physics_score *= reasonable.float().unsqueeze(1)
        
        return physics_score
