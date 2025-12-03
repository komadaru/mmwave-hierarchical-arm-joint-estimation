"""
マルチ仮説＋不確実性を考慮した統合PCMモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
from .mdn_pcm_head import MDNPCMHead, EvidentialPCMHead, KinematicHypothesisSelector


class MultiHypothesisPCMModel(nn.Module):
    """
    マルチ仮説＋不確実性を考慮した統合PCMモデル
    
    特徴:
    - MDN (Mixture Density Network) による複数仮説生成
    - Evidential Deep Learning による不確実性定量化
    - Kinematic Layer による物理制約での仮説選択
    - 可視性を考慮した重み付け
    """
    
    def __init__(self, in_channels: int, num_joints: int, bone_pairs: List[List[int]], 
                 num_components: int = 3, use_evidential: bool = True,
                 use_kinematic: bool = True):
        super().__init__()
        self.num_joints = num_joints
        self.num_components = num_components
        self.bone_pairs = bone_pairs
        self.use_evidential = use_evidential
        self.use_kinematic = use_kinematic
        
        # 特徴抽出器（学習最適化のため改良）
        self.feature_extractor = nn.Sequential(
            # 第1層: 初期特徴抽出
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 解像度を1/2に
            
            # 第2層: 中間特徴抽出
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 解像度を1/4に
            
            # 第3層: 高次特徴抽出
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 解像度を1/8に
            
            # 第4層: 最終特徴抽出
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)  # グローバル平均プーリング
        )
        
        # MDN Head
        self.mdn_head = MDNPCMHead(256, num_joints, num_components)
        
        # Evidential Head
        if use_evidential:
            self.evidential_head = EvidentialPCMHead(256, num_joints)
        
        # Kinematic Hypothesis Selector
        if use_kinematic:
            self.kinematic_selector = KinematicHypothesisSelector(bone_pairs, num_joints)
        
        # 可視性予測器
        self.visibility_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_joints),
            nn.Sigmoid()
        )
        
        # 骨長予測器
        self.bone_length_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(bone_pairs)),
            nn.Softplus()  # 正の値を保証
        )
    
    def forward(self, voxel_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        マルチ仮説＋不確実性を考慮したPCM予測
        
        Args:
            voxel_features: (B, C, D, H, W) ボクセル化された特徴量
            
        Returns:
            Dict containing:
                - mdn_output: MDNの出力
                - evidential_output: Evidentialの出力（オプション）
                - selected_hypothesis: 選択された仮説
                - visibility: 可視性予測
                - bone_lengths: 骨長予測
        """
        # 形状の柔軟な対応（学習最適化）
        if len(voxel_features.shape) == 4:
            # (B, C, H, W) -> (B, C, 1, H, W) に変換
            voxel_features = voxel_features.unsqueeze(2)
        
        B, C, D, H, W = voxel_features.shape
        
        # 特徴抽出
        features = self.feature_extractor(voxel_features)  # (B, 256, 1, 1, 1)
        features = features.squeeze(-1).squeeze(-1).squeeze(-1)  # (B, 256)
        
        # MDN予測（特徴抽出後の特徴量を使用）
        # 特徴量を3D形状に戻してMDN Headに渡す
        features_3d = features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, 256, 1, 1, 1)
        mdn_output = self.mdn_head(features_3d)
        
        # Evidential予測（特徴抽出後の特徴量を使用）
        evidential_output = None
        if self.use_evidential:
            evidential_output = self.evidential_head(features_3d)
        
        # 可視性予測
        visibility = self.visibility_predictor(features)  # (B, J)
        
        # 骨長予測
        bone_lengths = self.bone_length_predictor(features)  # (B, num_bones)
        
        # 仮説選択
        selected_hypothesis = None
        if self.use_kinematic:
            selected_hypothesis = self.kinematic_selector(mdn_output, visibility)
        
        return {
            'mdn_output': mdn_output,
            'evidential_output': evidential_output,
            'selected_hypothesis': selected_hypothesis,
            'visibility': visibility,
            'bone_lengths': bone_lengths
        }


class SparseMultiHypothesisPCMModel(nn.Module):
    """
    スパース点群向けマルチ仮説PCMモデル
    
    MinkowskiEngineを使用したスパース畳み込みと
    マルチ仮説＋不確実性を組み合わせたモデル
    """
    
    def __init__(self, in_channels: int, num_joints: int, bone_pairs: List[List[int]], 
                 num_components: int = 3, use_evidential: bool = True,
                 use_kinematic: bool = True):
        super().__init__()
        self.num_joints = num_joints
        self.num_components = num_components
        self.bone_pairs = bone_pairs
        self.use_evidential = use_evidential
        self.use_kinematic = use_kinematic
        
        # MinkowskiEngineのインポートを試行
        try:
            import MinkowskiEngine as ME
            self.ME = ME
            self.use_sparse = True
        except ImportError:
            print("Warning: MinkowskiEngine not available, falling back to dense model")
            self.use_sparse = False
        
        if self.use_sparse:
            # スパース畳み込み層
            self.sparse_conv1 = ME.MinkowskiConvolution(
                in_channels, 64, kernel_size=3, dimension=3
            )
            self.sparse_conv2 = ME.MinkowskiConvolution(
                64, 128, kernel_size=3, dimension=3
            )
            self.sparse_conv3 = ME.MinkowskiConvolution(
                128, 256, kernel_size=3, dimension=3
            )
            
            # スパース→密変換
            self.sparse_to_dense = ME.MinkowskiToDense()
        else:
            # 密畳み込み層（フォールバック）
            self.dense_conv1 = nn.Conv3d(in_channels, 64, 3, padding=1)
            self.dense_conv2 = nn.Conv3d(64, 128, 3, padding=1)
            self.dense_conv3 = nn.Conv3d(128, 256, 3, padding=1)
            self.dense_bn1 = nn.BatchNorm3d(64)
            self.dense_bn2 = nn.BatchNorm3d(128)
            self.dense_bn3 = nn.BatchNorm3d(256)
        
        # MDN Head
        self.mdn_head = MDNPCMHead(256, num_joints, num_components)
        
        # Evidential Head
        if use_evidential:
            self.evidential_head = EvidentialPCMHead(256, num_joints)
        
        # Kinematic Hypothesis Selector
        if use_kinematic:
            self.kinematic_selector = KinematicHypothesisSelector(bone_pairs, num_joints)
        
        # 可視性予測器
        self.visibility_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_joints),
            nn.Sigmoid()
        )
        
        # 骨長予測器
        self.bone_length_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(bone_pairs)),
            nn.Softplus()  # 正の値を保証
        )
    
    def forward(self, voxel_features: torch.Tensor, 
                sparse_coords: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        スパース点群向けマルチ仮説PCM予測
        
        Args:
            voxel_features: (B, C, D, H, W) ボクセル化された特徴量
            sparse_coords: (N, 4) スパース座標（バッチインデックス付き）
            
        Returns:
            Dict containing:
                - mdn_output: MDNの出力
                - evidential_output: Evidentialの出力（オプション）
                - selected_hypothesis: 選択された仮説
                - visibility: 可視性予測
                - bone_lengths: 骨長予測
        """
        B, C, D, H, W = voxel_features.shape
        
        if self.use_sparse and sparse_coords is not None:
            # スパース畳み込み
            sparse_features = self.ME.SparseTensor(
                features=voxel_features[sparse_coords[:, 0], :, sparse_coords[:, 1], 
                                      sparse_coords[:, 2], sparse_coords[:, 3]],
                coordinates=sparse_coords
            )
            
            # スパース畳み込み層
            x = self.sparse_conv1(sparse_features)
            x = self.sparse_conv2(x)
            x = self.sparse_conv3(x)
            
            # スパース→密変換
            dense_features = self.sparse_to_dense(x, (B, 256, D, H, W))
        else:
            # 密畳み込み（フォールバック）
            x = F.relu(self.dense_bn1(self.dense_conv1(voxel_features)))
            x = F.relu(self.dense_bn2(self.dense_conv2(x)))
            dense_features = F.relu(self.dense_bn3(self.dense_conv3(x)))
        
        # グローバル平均プーリング
        features = F.adaptive_avg_pool3d(dense_features, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, 256)
        
        # MDN予測
        mdn_output = self.mdn_head(dense_features)
        
        # Evidential予測
        evidential_output = None
        if self.use_evidential:
            evidential_output = self.evidential_head(dense_features)
        
        # 可視性予測
        visibility = self.visibility_predictor(features)  # (B, J)
        
        # 骨長予測
        bone_lengths = self.bone_length_predictor(features)  # (B, num_bones)
        
        # 仮説選択
        selected_hypothesis = None
        if self.use_kinematic:
            selected_hypothesis = self.kinematic_selector(mdn_output, visibility)
        
        return {
            'mdn_output': mdn_output,
            'evidential_output': evidential_output,
            'selected_hypothesis': selected_hypothesis,
            'visibility': visibility,
            'bone_lengths': bone_lengths
        }


class AdaptiveMultiHypothesisPCMModel(nn.Module):
    """
    適応的マルチ仮説PCMモデル
    
    点群密度に応じて適応的に仮説数を調整するモデル
    """
    
    def __init__(self, in_channels: int, num_joints: int, bone_pairs: List[List[int]], 
                 max_components: int = 5, use_evidential: bool = True,
                 use_kinematic: bool = True):
        super().__init__()
        self.num_joints = num_joints
        self.max_components = max_components
        self.bone_pairs = bone_pairs
        self.use_evidential = use_evidential
        self.use_kinematic = use_kinematic
        
        # 密度推定器
        self.density_estimator = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 適応的MDN Head
        self.adaptive_mdn_head = AdaptiveMDNPCMHead(256, num_joints, max_components)
        
        # Evidential Head
        if use_evidential:
            self.evidential_head = EvidentialPCMHead(256, num_joints)
        
        # Kinematic Hypothesis Selector
        if use_kinematic:
            self.kinematic_selector = KinematicHypothesisSelector(bone_pairs, num_joints)
        
        # 可視性予測器
        self.visibility_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_joints),
            nn.Sigmoid()
        )
        
        # 骨長予測器
        self.bone_length_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(bone_pairs)),
            nn.Softplus()  # 正の値を保証
        )
    
    def forward(self, voxel_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        適応的マルチ仮説PCM予測
        
        Args:
            voxel_features: (B, C, D, H, W) ボクセル化された特徴量
            
        Returns:
            Dict containing:
                - mdn_output: MDNの出力
                - evidential_output: Evidentialの出力（オプション）
                - selected_hypothesis: 選択された仮説
                - visibility: 可視性予測
                - bone_lengths: 骨長予測
                - density: 密度推定
        """
        B, C, D, H, W = voxel_features.shape
        
        # 密度推定
        density = self.density_estimator(voxel_features)  # (B, 1)
        
        # 適応的MDN予測
        mdn_output = self.adaptive_mdn_head(voxel_features, density)
        
        # Evidential予測（特徴抽出後の特徴量を使用）
        evidential_output = None
        if self.use_evidential:
            evidential_output = self.evidential_head(features_3d)
        
        # 可視性予測
        features = F.adaptive_avg_pool3d(voxel_features, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, C)
        visibility = self.visibility_predictor(features)  # (B, J)
        
        # 骨長予測
        bone_lengths = self.bone_length_predictor(features)  # (B, num_bones)
        
        # 仮説選択
        selected_hypothesis = None
        if self.use_kinematic:
            selected_hypothesis = self.kinematic_selector(mdn_output, visibility)
        
        return {
            'mdn_output': mdn_output,
            'evidential_output': evidential_output,
            'selected_hypothesis': selected_hypothesis,
            'visibility': visibility,
            'bone_lengths': bone_lengths,
            'density': density
        }


class AdaptiveMDNPCMHead(nn.Module):
    """
    適応的MDN PCM Head
    
    密度に応じて仮説数を動的に調整
    """
    
    def __init__(self, in_channels: int, num_joints: int, max_components: int = 5):
        super().__init__()
        self.num_joints = num_joints
        self.max_components = max_components
        
        # 特徴量を関節数に投影
        self.joint_projection = nn.Conv3d(in_channels, num_joints, 1)
        
        # 最大成分数でのMDNパラメータ生成
        self.mdn_head = nn.Conv3d(num_joints, num_joints * max_components * 7, 1)
        
        # 活性化関数
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, voxel_features: torch.Tensor, density: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        適応的MDN予測
        
        Args:
            voxel_features: (B, C, D, H, W) 入力特徴量
            density: (B, 1) 密度推定
            
        Returns:
            Dict containing:
                - mu: (B, J, K, 3) 各仮説の平均
                - sigma: (B, J, K, 3) 各仮説の標準偏差
                - pi: (B, J, K) 各仮説の重み
                - logits: (B, J, K) 重みのロジット
                - num_components: (B,) 各サンプルの成分数
        """
        B, C, D, H, W = voxel_features.shape
        
        # 関節特徴量の生成
        joint_features = self.joint_projection(voxel_features)  # (B, J, D, H, W)
        
        # MDNパラメータの生成
        mdn_params = self.mdn_head(joint_features)  # (B, J*K*7, D, H, W)
        
        # グローバル平均プーリング
        mdn_params = F.adaptive_avg_pool3d(mdn_params, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, J*K*7)
        
        # パラメータの分解
        mdn_params = mdn_params.view(B, self.num_joints, self.max_components, 7)  # (B, J, K, 7)
        
        # 密度に応じて成分数を調整
        num_components = torch.clamp(
            torch.round(density * self.max_components).long(), 
            min=1, max=self.max_components
        )  # (B,)
        
        # 平均 (μ)
        mu = mdn_params[..., :3]  # (B, J, K, 3)
        
        # 標準偏差 (σ) - 正の値にするためSoftplus
        sigma = self.softplus(mdn_params[..., 3:6]) + 1e-6  # (B, J, K, 3)
        
        # 重み (π) - Softmaxで正規化
        logits = mdn_params[..., 6]  # (B, J, K)
        pi = self.softmax(logits)  # (B, J, K)
        
        # 密度に応じて重みを調整
        for b in range(B):
            k = num_components[b].item()
            if k < self.max_components:
                # 使用しない成分の重みを0に設定
                pi[b, :, k:] = 0.0
                # 正規化
                pi[b] = pi[b] / pi[b].sum(dim=-1, keepdim=True)
        
        return {
            'mu': mu,
            'sigma': sigma,
            'pi': pi,
            'logits': logits,
            'num_components': num_components
        }
