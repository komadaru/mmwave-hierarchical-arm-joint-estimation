"""
マルチ仮説＋不確実性を考慮した評価指標
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
from scipy.stats import chi2


def coverage_metrics(pred_mu: torch.Tensor, pred_sigma: torch.Tensor, 
                   gt_joints: torch.Tensor, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
    """
    信頼楕円内のGTカバレッジを計算
    
    Args:
        pred_mu: (B, J, 3) 予測平均
        pred_sigma: (B, J, 3) 予測標準偏差
        gt_joints: (B, J, 3) GT座標
        confidence_levels: 信頼水準のリスト
        
    Returns:
        Dict containing coverage rates for each confidence level
    """
    coverage_rates = {}
    
    for conf_level in confidence_levels:
        # マハラノビス距離の計算
        diff = gt_joints - pred_mu  # (B, J, 3)
        
        # 対角共分散行列を仮定
        sigma_diag = pred_sigma.pow(2)  # (B, J, 3)
        mahal_dist = (diff.pow(2) / sigma_diag).sum(dim=-1)  # (B, J)
        
        # 信頼楕円の閾値（3次元の場合）
        threshold = chi2.ppf(conf_level, df=3)
        
        # カバレッジ率
        coverage_rate = (mahal_dist < threshold).float().mean().item()
        coverage_rates[f'coverage_{int(conf_level*100)}'] = coverage_rate
    
    return coverage_rates


def uncertainty_quality_metrics(pred_mu: torch.Tensor, pred_sigma: torch.Tensor, 
                               gt_joints: torch.Tensor) -> Dict[str, float]:
    """
    不確実性の品質を評価
    
    Args:
        pred_mu: (B, J, 3) 予測平均
        pred_sigma: (B, J, 3) 予測標準偏差
        gt_joints: (B, J, 3) GT座標
        
    Returns:
        Dict containing uncertainty quality metrics
    """
    # 予測誤差
    error = (gt_joints - pred_mu).pow(2).sum(dim=-1)  # (B, J)
    
    # 不確実性
    uncertainty = pred_sigma.pow(2).sum(dim=-1)  # (B, J)
    
    # 予測区間の幅
    interval_width = uncertainty.mean().item()
    
    # 不確実性の較正誤差
    calibration_error = (error - uncertainty).pow(2).mean().item()
    
    # 信頼度の較正
    confidence_calibration = compute_confidence_calibration(pred_mu, pred_sigma, gt_joints)
    
    return {
        'interval_width': interval_width,
        'calibration_error': calibration_error,
        'confidence_calibration': confidence_calibration
    }


def compute_confidence_calibration(pred_mu: torch.Tensor, pred_sigma: torch.Tensor, 
                                  gt_joints: torch.Tensor, num_bins: int = 10) -> float:
    """
    信頼度の較正を計算
    
    Args:
        pred_mu: (B, J, 3) 予測平均
        pred_sigma: (B, J, 3) 予測標準偏差
        gt_joints: (B, J, 3) GT座標
        num_bins: ビン数
        
    Returns:
        calibration error
    """
    # 予測誤差
    error = (gt_joints - pred_mu).pow(2).sum(dim=-1)  # (B, J)
    
    # 不確実性
    uncertainty = pred_sigma.pow(2).sum(dim=-1)  # (B, J)
    
    # 不確実性をビンに分割
    uncertainty_flat = uncertainty.flatten().cpu().numpy()
    error_flat = error.flatten().cpu().numpy()
    
    # ビンの境界を計算
    bin_edges = np.linspace(uncertainty_flat.min(), uncertainty_flat.max(), num_bins + 1)
    
    calibration_error = 0.0
    for i in range(num_bins):
        mask = (uncertainty_flat >= bin_edges[i]) & (uncertainty_flat < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_error = error_flat[mask].mean()
            bin_uncertainty = uncertainty_flat[mask].mean()
            calibration_error += (bin_error - bin_uncertainty) ** 2
    
    return calibration_error / num_bins


def hypothesis_selection_metrics(hypotheses: Dict[str, torch.Tensor], 
                                gt_joints: torch.Tensor) -> Dict[str, float]:
    """
    仮説選択の品質を評価
    
    Args:
        hypotheses: 仮説の辞書
        gt_joints: (B, J, 3) GT座標
        
    Returns:
        Dict containing hypothesis selection metrics
    """
    mu = hypotheses['mu']  # (B, J, K, 3)
    pi = hypotheses['pi']  # (B, J, K)
    
    B, J, K, _ = mu.shape
    
    # 各仮説の誤差を計算
    gt_expanded = gt_joints.unsqueeze(2).expand(-1, -1, K, -1)  # (B, J, K, 3)
    errors = (mu - gt_expanded).pow(2).sum(dim=-1)  # (B, J, K)
    
    # 最良の仮説の誤差
    best_errors, best_indices = errors.min(dim=-1)  # (B, J)
    
    # 選択された仮説の重み
    selected_weights = torch.gather(pi, 2, best_indices.unsqueeze(-1)).squeeze(-1)  # (B, J)
    
    # 重み付き誤差
    weighted_errors = best_errors * selected_weights  # (B, J)
    
    return {
        'best_error_mean': best_errors.mean().item(),
        'best_error_std': best_errors.std().item(),
        'weighted_error_mean': weighted_errors.mean().item(),
        'weighted_error_std': weighted_errors.std().item(),
        'selection_confidence': selected_weights.mean().item()
    }


def uncertainty_calibration_plot(pred_mu: torch.Tensor, pred_sigma: torch.Tensor, 
                                gt_joints: torch.Tensor, num_bins: int = 10) -> Dict[str, np.ndarray]:
    """
    不確実性較正プロット用のデータを生成
    
    Args:
        pred_mu: (B, J, 3) 予測平均
        pred_sigma: (B, J, 3) 予測標準偏差
        gt_joints: (B, J, 3) GT座標
        num_bins: ビン数
        
    Returns:
        Dict containing calibration plot data
    """
    # 予測誤差
    error = (gt_joints - pred_mu).pow(2).sum(dim=-1)  # (B, J)
    
    # 不確実性
    uncertainty = pred_sigma.pow(2).sum(dim=-1)  # (B, J)
    
    # 不確実性をビンに分割
    uncertainty_flat = uncertainty.flatten().cpu().numpy()
    error_flat = error.flatten().cpu().numpy()
    
    # ビンの境界を計算
    bin_edges = np.linspace(uncertainty_flat.min(), uncertainty_flat.max(), num_bins + 1)
    
    bin_centers = []
    bin_errors = []
    bin_uncertainties = []
    bin_counts = []
    
    for i in range(num_bins):
        mask = (uncertainty_flat >= bin_edges[i]) & (uncertainty_flat < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_errors.append(error_flat[mask].mean())
            bin_uncertainties.append(uncertainty_flat[mask].mean())
            bin_counts.append(mask.sum())
    
    return {
        'bin_centers': np.array(bin_centers),
        'bin_errors': np.array(bin_errors),
        'bin_uncertainties': np.array(bin_uncertainties),
        'bin_counts': np.array(bin_counts)
    }


def reliability_diagram(pred_mu: torch.Tensor, pred_sigma: torch.Tensor, 
                      gt_joints: torch.Tensor, num_bins: int = 10) -> Dict[str, np.ndarray]:
    """
    信頼性ダイアグラム用のデータを生成
    
    Args:
        pred_mu: (B, J, 3) 予測平均
        pred_sigma: (B, J, 3) 予測標準偏差
        gt_joints: (B, J, 3) GT座標
        num_bins: ビン数
        
    Returns:
        Dict containing reliability diagram data
    """
    # 予測誤差
    error = (gt_joints - pred_mu).pow(2).sum(dim=-1)  # (B, J)
    
    # 不確実性
    uncertainty = pred_sigma.pow(2).sum(dim=-1)  # (B, J)
    
    # 信頼度（不確実性の逆数）
    confidence = 1.0 / (uncertainty + 1e-6)  # (B, J)
    
    # 信頼度をビンに分割
    confidence_flat = confidence.flatten().cpu().numpy()
    error_flat = error.flatten().cpu().numpy()
    
    # ビンの境界を計算
    bin_edges = np.linspace(confidence_flat.min(), confidence_flat.max(), num_bins + 1)
    
    bin_centers = []
    bin_errors = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(num_bins):
        mask = (confidence_flat >= bin_edges[i]) & (confidence_flat < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_errors.append(error_flat[mask].mean())
            bin_confidences.append(confidence_flat[mask].mean())
            bin_counts.append(mask.sum())
    
    return {
        'bin_centers': np.array(bin_centers),
        'bin_errors': np.array(bin_errors),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts)
    }


class MultiHypothesisMetrics:
    """
    マルチ仮説＋不確実性を考慮した総合評価指標
    """
    
    def __init__(self, bone_pairs: List[List[int]]):
        self.bone_pairs = bone_pairs
    
    def compute_all_metrics(self, predictions: Dict[str, torch.Tensor], 
                           targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        全ての評価指標を計算
        
        Args:
            predictions: 予測結果の辞書
            targets: ターゲットの辞書
            
        Returns:
            Dict containing all metrics
        """
        metrics = {}
        
        # 基本指標
        if 'selected_mu' in predictions and 'joints' in targets:
            # MPJPE
            mpjpe = torch.norm(predictions['selected_mu'] - targets['joints'], dim=-1).mean().item()
            metrics['mpjpe'] = mpjpe
            
            # PCK
            pck = (torch.norm(predictions['selected_mu'] - targets['joints'], dim=-1) < 0.05).float().mean().item()
            metrics['pck'] = pck
        
        # カバレッジ指標
        if 'selected_mu' in predictions and 'selected_sigma' in predictions and 'joints' in targets:
            coverage_rates = coverage_metrics(
                predictions['selected_mu'], predictions['selected_sigma'], targets['joints']
            )
            metrics.update(coverage_rates)
        
        # 不確実性品質指標
        if 'selected_mu' in predictions and 'selected_sigma' in predictions and 'joints' in targets:
            uncertainty_metrics = uncertainty_quality_metrics(
                predictions['selected_mu'], predictions['selected_sigma'], targets['joints']
            )
            metrics.update(uncertainty_metrics)
        
        # 仮説選択指標
        if 'mu' in predictions and 'pi' in predictions and 'joints' in targets:
            hypothesis_metrics = hypothesis_selection_metrics(predictions, targets['joints'])
            metrics.update(hypothesis_metrics)
        
        return metrics
