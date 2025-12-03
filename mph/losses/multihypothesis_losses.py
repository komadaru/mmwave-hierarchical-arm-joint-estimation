"""
マルチ仮説＋不確実性を考慮した損失関数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from scipy.stats import chi2


def mdn_loss(mu: torch.Tensor, sigma: torch.Tensor, pi: torch.Tensor, 
              target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    MDN (Mixture Density Network) 損失関数
    
    Args:
        mu: (B, J, K, 3) 各仮説の平均
        sigma: (B, J, K, 3) 各仮説の標準偏差
        pi: (B, J, K) 各仮説の重み
        target: (B, J, 3) ターゲット座標
        reduction: 損失の縮約方法
        
    Returns:
        loss: スカラー損失値
    """
    B, J, K, _ = mu.shape
    
    # ターゲットを拡張
    target_expanded = target.unsqueeze(2).expand(-1, -1, K, -1)  # (B, J, K, 3)
    
    # 各仮説の負の対数尤度を計算
    nll = 0.5 * torch.log(2 * np.pi * sigma.pow(2).sum(dim=-1, keepdim=True)) + \
          0.5 * ((target_expanded - mu) / sigma).pow(2).sum(dim=-1, keepdim=True)  # (B, J, K, 1)
    
    # 重み付き負の対数尤度
    weighted_nll = pi.unsqueeze(-1) * nll  # (B, J, K, 1)
    
    # 最小の負の対数尤度を選択（最良の仮説）
    min_nll, _ = weighted_nll.min(dim=2)  # (B, J, 1)
    
    if reduction == 'mean':
        return min_nll.mean()
    elif reduction == 'sum':
        return min_nll.sum()
    else:
        return min_nll


def evidential_loss(mu: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                   target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Evidential Deep Learning 損失関数
    
    Args:
        mu: (B, J, 3) 予測平均
        nu: (B, J) 不確実性パラメータ
        alpha: (B, J) エビデンスパラメータ
        beta: (B, J) エビデンスパラメータ
        target: (B, J, 3) ターゲット座標
        reduction: 損失の縮約方法
        
    Returns:
        loss: スカラー損失値
    """
    # 予測誤差
    error = target - mu  # (B, J, 3)
    
    # 不確実性の計算
    # alpha > 1 を保証するため、alpha = alpha + 1
    alpha_safe = alpha + 1.0
    uncertainty = nu / alpha_safe  # (B, J)
    
    # Evidential損失
    # データ適合項
    data_fit = 0.5 * torch.log(2 * np.pi * uncertainty) + \
               0.5 * error.pow(2).sum(dim=-1) / uncertainty  # (B, J)
    
    # 正則化項
    regularization = torch.log(alpha_safe) - torch.log(beta)  # (B, J)
    
    # 総損失
    loss = data_fit + regularization  # (B, J)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def kinematic_constraint_loss(mu: torch.Tensor, bone_pairs: List[List[int]], 
                            target_lengths: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    物理制約損失関数（骨長制約）
    
    Args:
        mu: (B, J, 3) 予測座標
        bone_pairs: 骨の接続ペア
        target_lengths: (B, num_bones) ターゲット骨長
        reduction: 損失の縮約方法
        
    Returns:
        loss: スカラー損失値
    """
    B, J, _ = mu.shape
    num_bones = len(bone_pairs)
    
    # 予測骨長の計算
    pred_lengths = []
    for i, j in bone_pairs:
        if i < J and j < J:
            bone_vec = mu[:, j] - mu[:, i]  # (B, 3)
            bone_length = torch.norm(bone_vec, dim=1)  # (B,)
            pred_lengths.append(bone_length)
    
    if pred_lengths:
        pred_lengths = torch.stack(pred_lengths, dim=1)  # (B, num_bones)
        
        # 骨長制約損失
        length_loss = F.mse_loss(pred_lengths, target_lengths, reduction='none')  # (B, num_bones)
        
        if reduction == 'mean':
            return length_loss.mean()
        elif reduction == 'sum':
            return length_loss.sum()
        else:
            return length_loss
    else:
        return torch.tensor(0.0, device=mu.device)


def visibility_constraint_loss(mu: torch.Tensor, visibility: torch.Tensor, 
                             target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    可視性制約損失関数
    
    Args:
        mu: (B, J, 3) 予測座標
        visibility: (B, J) 可視性マスク
        target: (B, J, 3) ターゲット座標
        reduction: 損失の縮約方法
        
    Returns:
        loss: スカラー損失値
    """
    # 可視性が低い関節の予測誤差にペナルティ
    error = (mu - target).pow(2).sum(dim=-1)  # (B, J)
    
    # 可視性重み付き損失
    visibility_weight = 1.0 + (1.0 - visibility) * 2.0  # 可視性が低いほど重みを増加
    weighted_error = error * visibility_weight  # (B, J)
    
    if reduction == 'mean':
        return weighted_error.mean()
    elif reduction == 'sum':
        return weighted_error.sum()
    else:
        return weighted_error


def coverage_loss(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor,
                 confidence_level: float = 0.95, reduction: str = 'mean') -> torch.Tensor:
    """
    カバレッジ損失関数（信頼楕円内のGTカバレッジ）
    
    Args:
        mu: (B, J, 3) 予測平均
        sigma: (B, J, 3) 予測標準偏差
        target: (B, J, 3) ターゲット座標
        confidence_level: 信頼水準
        reduction: 損失の縮約方法
        
    Returns:
        loss: スカラー損失値
    """
    # マハラノビス距離の計算
    diff = target - mu  # (B, J, 3)
    
    # 対角共分散行列を仮定
    sigma_diag = sigma.pow(2)  # (B, J, 3)
    mahal_dist = (diff.pow(2) / sigma_diag).sum(dim=-1)  # (B, J)
    
    # 信頼楕円の閾値（3次元の場合）
    threshold = chi2.ppf(confidence_level, df=3)
    
    # カバレッジ率
    coverage = (mahal_dist < threshold).float()  # (B, J)
    
    # カバレッジ損失（1 - カバレッジ率）
    coverage_loss = 1.0 - coverage  # (B, J)
    
    if reduction == 'mean':
        return coverage_loss.mean()
    elif reduction == 'sum':
        return coverage_loss.sum()
    else:
        return coverage_loss


def calibration_loss(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor,
                    num_bins: int = 10, reduction: str = 'mean') -> torch.Tensor:
    """
    較正損失関数（不確実性の較正）
    
    Args:
        mu: (B, J, 3) 予測平均
        sigma: (B, J, 3) 予測標準偏差
        target: (B, J, 3) ターゲット座標
        num_bins: ビン数
        reduction: 損失の縮約方法
        
    Returns:
        loss: スカラー損失値
    """
    # 予測誤差
    error = (target - mu).pow(2).sum(dim=-1)  # (B, J)
    
    # 不確実性
    uncertainty = sigma.pow(2).sum(dim=-1)  # (B, J)
    
    # 較正誤差の計算
    # 不確実性をビンに分割
    uncertainty_flat = uncertainty.flatten()
    error_flat = error.flatten()
    
    # ビンの境界を計算
    bin_edges = torch.linspace(uncertainty_flat.min(), uncertainty_flat.max(), num_bins + 1)
    
    calibration_error = 0.0
    for i in range(num_bins):
        mask = (uncertainty_flat >= bin_edges[i]) & (uncertainty_flat < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_error = error_flat[mask].mean()
            bin_uncertainty = uncertainty_flat[mask].mean()
            calibration_error += (bin_error - bin_uncertainty).pow(2)
    
    if reduction == 'mean':
        return calibration_error / num_bins
    elif reduction == 'sum':
        return calibration_error
    else:
        return calibration_error


class MultiHypothesisLoss(nn.Module):
    """
    マルチ仮説＋不確実性を考慮した総合損失関数
    """
    
    def __init__(self, bone_pairs: List[List[int]], 
                 lambda_mdn: float = 1.0,
                 lambda_evidential: float = 0.5,
                 lambda_kinematic: float = 0.3,
                 lambda_visibility: float = 0.2,
                 lambda_coverage: float = 0.1,
                 lambda_calibration: float = 0.1):
        super().__init__()
        self.bone_pairs = bone_pairs
        self.lambda_mdn = lambda_mdn
        self.lambda_evidential = lambda_evidential
        self.lambda_kinematic = lambda_kinematic
        self.lambda_visibility = lambda_visibility
        self.lambda_coverage = lambda_coverage
        self.lambda_calibration = lambda_calibration
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        総合損失の計算
        
        Args:
            predictions: 予測結果の辞書
            targets: ターゲットの辞書
            
        Returns:
            Dict containing:
                - total_loss: 総合損失
                - mdn_loss: MDN損失
                - evidential_loss: Evidential損失
                - kinematic_loss: 物理制約損失
                - visibility_loss: 可視性制約損失
                - coverage_loss: カバレッジ損失
                - calibration_loss: 較正損失
        """
        losses = {}
        
        # MDN損失
        if 'mdn_mu' in predictions and 'mdn_sigma' in predictions and 'mdn_pi' in predictions:
            mdn_loss_val = mdn_loss(
                predictions['mdn_mu'], predictions['mdn_sigma'], predictions['mdn_pi'],
                targets['joints']
            )
            losses['mdn_loss'] = mdn_loss_val
        
        # Evidential損失
        if 'evidential_mu' in predictions and 'evidential_nu' in predictions:
            evidential_loss_val = evidential_loss(
                predictions['evidential_mu'], predictions['evidential_nu'],
                predictions['evidential_alpha'], predictions['evidential_beta'],
                targets['joints']
            )
            losses['evidential_loss'] = evidential_loss_val
        
        # 物理制約損失
        if 'selected_mu' in predictions and 'target_lengths' in targets:
            kinematic_loss_val = kinematic_constraint_loss(
                predictions['selected_mu'], self.bone_pairs, targets['target_lengths']
            )
            losses['kinematic_loss'] = kinematic_loss_val
        
        # 可視性制約損失
        if 'selected_mu' in predictions and 'visibility' in targets:
            visibility_loss_val = visibility_constraint_loss(
                predictions['selected_mu'], targets['visibility'], targets['joints']
            )
            losses['visibility_loss'] = visibility_loss_val
        
        # カバレッジ損失
        if 'selected_mu' in predictions and 'selected_sigma' in predictions:
            coverage_loss_val = coverage_loss(
                predictions['selected_mu'], predictions['selected_sigma'], targets['joints']
            )
            losses['coverage_loss'] = coverage_loss_val
        
        # 較正損失
        if 'selected_mu' in predictions and 'selected_sigma' in predictions:
            calibration_loss_val = calibration_loss(
                predictions['selected_mu'], predictions['selected_sigma'], targets['joints']
            )
            losses['calibration_loss'] = calibration_loss_val
        
        # 総合損失の計算
        total_loss = 0.0
        if 'mdn_loss' in losses:
            total_loss += self.lambda_mdn * losses['mdn_loss']
        if 'evidential_loss' in losses:
            total_loss += self.lambda_evidential * losses['evidential_loss']
        if 'kinematic_loss' in losses:
            total_loss += self.lambda_kinematic * losses['kinematic_loss']
        if 'visibility_loss' in losses:
            total_loss += self.lambda_visibility * losses['visibility_loss']
        if 'coverage_loss' in losses:
            total_loss += self.lambda_coverage * losses['coverage_loss']
        if 'calibration_loss' in losses:
            total_loss += self.lambda_calibration * losses['calibration_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
