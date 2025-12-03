#!/usr/bin/env python3
"""
グリッドヒートマップ生成とデータ前処理
"""

import numpy as np
from typing import Tuple, Optional, Dict, Literal, List
from scipy import ndimage


def normalize_value(
    value: float,
    original_range: Tuple[float, float],
    normalized_range: Tuple[float, float] = (-1.0, 1.0)
) -> float:
    """
    値を正規化
    
    Args:
        value: 元の値
        original_range: 元の範囲 (min, max)
        normalized_range: 正規化後の範囲 (min, max)
    
    Returns:
        normalized_value: 正規化後の値
    """
    orig_min, orig_max = original_range
    norm_min, norm_max = normalized_range
    
    if orig_max - orig_min < 1e-6:
        return norm_min
    
    # 線形正規化
    normalized = (value - orig_min) / (orig_max - orig_min) * (norm_max - norm_min) + norm_min
    return normalized


def denormalize_value(
    normalized_value: float,
    original_range: Tuple[float, float],
    normalized_range: Tuple[float, float] = (-1.0, 1.0)
) -> float:
    """
    正規化された値を元の範囲に戻す
    
    Args:
        normalized_value: 正規化後の値
        original_range: 元の範囲 (min, max)
        normalized_range: 正規化後の範囲 (min, max)
    
    Returns:
        original_value: 元の値
    """
    orig_min, orig_max = original_range
    norm_min, norm_max = normalized_range
    
    if norm_max - norm_min < 1e-6:
        return orig_min
    
    # 逆正規化
    original = (normalized_value - norm_min) / (norm_max - norm_min) * (orig_max - orig_min) + orig_min
    return original


def create_heatmap_grid(
    points: np.ndarray,  # (N, 6) [x, y, z, velocity, amplitude, energy_power]
    spatial_axis: Literal['x', 'y', 'z'] = 'x',
    feature_axis: Literal['energy_power', 'velocity'] = 'energy_power',
    spatial_bins: int = 50,
    feature_bins: int = 50,
    spatial_range: Optional[Tuple[float, float]] = None,
    feature_range: Optional[Tuple[float, float]] = None,
    normalize: bool = True,
    normalized_range: Tuple[float, float] = (-1.0, 1.0)
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    グリッドヒートマップを生成
    
    Args:
        points: (N, 6) - 点群データ [x, y, z, velocity, amplitude, energy_power]
        spatial_axis: 空間軸の選択 ('x', 'y', 'z')
        feature_axis: 特徴量軸の選択 ('energy_power', 'velocity')
        spatial_bins: 空間軸のビン数
        feature_bins: 特徴量軸のビン数
        spatial_range: 空間軸の範囲 (min, max)（Noneの場合は自動計算）
        feature_range: 特徴量軸の範囲 (min, max)（Noneの場合は自動計算）
        normalize: 正規化後にビン分割するか（Trueの場合、正規化後にビン分割）
        normalized_range: 正規化後の範囲 (min, max)
    
    Returns:
        heatmap: (spatial_bins, feature_bins) - ヒートマップ（点群数）
            【軸の統一】第0軸=空間軸、第1軸=特徴量軸
        spatial_range: (min, max) - 使用された空間軸の実際の範囲（メートル単位）
        feature_range: (min, max) - 使用された特徴量軸の実際の範囲
        spatial_range_norm: (min, max) - 正規化後の空間軸の範囲
        feature_range_norm: (min, max) - 正規化後の特徴量軸の範囲
    """
    if len(points) == 0:
        # 空の点群の場合はゼロヒートマップを返す
        heatmap = np.zeros((spatial_bins, feature_bins))
        if spatial_range is None:
            spatial_range = (0.0, 1.0)
        if feature_range is None:
            feature_range = (0.0, 1.0)
        spatial_range_norm = normalized_range
        feature_range_norm = normalized_range
        return heatmap, spatial_range, feature_range, spatial_range_norm, feature_range_norm
    
    # 空間軸と特徴量軸を抽出
    spatial_idx = {'x': 0, 'y': 1, 'z': 2}[spatial_axis]
    feature_idx = {'energy_power': 5, 'velocity': 3}[feature_axis]
    
    spatial_values = points[:, spatial_idx]
    feature_values = points[:, feature_idx]
    
    # 実際の範囲を決定（メートル単位）
    if spatial_range is None:
        spatial_range = (spatial_values.min(), spatial_values.max())
    if feature_range is None:
        # velocityの場合は、パーセンタイルベースの範囲を使用（外れ値に頑健）
        if feature_axis == 'velocity':
            # パーセンタイルベースの範囲（1パーセンタイル～99パーセンタイル）
            # これにより、外れ値を除外して適切な範囲を設定
            feature_min = np.percentile(feature_values, 1)
            feature_max = np.percentile(feature_values, 99)
            
            # 範囲が非常に小さい場合は、固定範囲を使用
            if feature_max - feature_min < 1e-3:  # 1mm/s未満の場合
                # velocityの典型的な範囲: 0.0 ～ 0.5 m/s（歩行速度程度）
                feature_range = (0.0, 0.5)
            else:
                # マージンを追加（10%）
                margin = (feature_max - feature_min) * 0.1
                feature_range = (max(0.0, feature_min - margin), feature_max + margin)
        else:
            # energy_powerの場合は通常のmin/max
            feature_range = (feature_values.min(), feature_values.max())
    
    # 範囲が0の場合はデフォルト範囲を使用
    if spatial_range[1] - spatial_range[0] < 1e-6:
        spatial_range = (spatial_range[0] - 0.5, spatial_range[0] + 0.5)
    if feature_range[1] - feature_range[0] < 1e-6:
        # velocityの場合は固定範囲を使用
        if feature_axis == 'velocity':
            feature_range = (0.0, 0.5)  # 0.0 ～ 0.5 m/s
        else:
            feature_range = (feature_range[0] - 0.5, feature_range[0] + 0.5)
    
    if normalize:
        # 正規化
        spatial_values_norm = np.array([
            normalize_value(v, spatial_range, normalized_range) for v in spatial_values
        ])
        feature_values_norm = np.array([
            normalize_value(v, feature_range, normalized_range) for v in feature_values
        ])
        
        # 正規化後の範囲でビン分割
        spatial_range_norm = normalized_range
        feature_range_norm = normalized_range
        
        heatmap, _, _ = np.histogram2d(
            feature_values_norm, spatial_values_norm,
            bins=[feature_bins, spatial_bins],
            range=[normalized_range, normalized_range]
        )
    else:
        # 正規化なし（既存の方法）
        spatial_range_norm = spatial_range
        feature_range_norm = feature_range
        
        heatmap, _, _ = np.histogram2d(
            feature_values, spatial_values,
            bins=[feature_bins, spatial_bins],
            range=[feature_range, spatial_range]
        )
    
    return heatmap.T, spatial_range, feature_range, spatial_range_norm, feature_range_norm  # (spatial_bins, feature_bins) に転置


def compute_heatmap_diff(
    heatmap_prev: np.ndarray,
    heatmap_curr: np.ndarray,
    method: Literal['absolute', 'relative', 'normalized'] = 'absolute'
) -> np.ndarray:
    """
    前フレームと今フレームのヒートマップ差分を計算
    
    Args:
        heatmap_prev: (H, W) - 前フレームのヒートマップ
        heatmap_curr: (H, W) - 今フレームのヒートマップ
        method: 差分計算方法
    
    Returns:
        diff: (H, W) - 差分ヒートマップ
    """
    if method == 'absolute':
        diff = np.abs(heatmap_curr - heatmap_prev)
    elif method == 'relative':
        # 前フレームが0の場合は絶対差分を使用
        mask = heatmap_prev > 0
        diff = np.zeros_like(heatmap_prev)
        diff[mask] = np.abs(heatmap_curr[mask] - heatmap_prev[mask]) / (heatmap_prev[mask] + 1e-6)
        diff[~mask] = heatmap_curr[~mask]
    elif method == 'normalized':
        # 正規化してから差分
        prev_min, prev_max = heatmap_prev.min(), heatmap_prev.max()
        curr_min, curr_max = heatmap_curr.min(), heatmap_curr.max()
        
        prev_range = prev_max - prev_min
        curr_range = curr_max - curr_min
        
        if prev_range > 1e-6:
            prev_norm = (heatmap_prev - prev_min) / prev_range
        else:
            prev_norm = np.zeros_like(heatmap_prev)
        
        if curr_range > 1e-6:
            curr_norm = (heatmap_curr - curr_min) / curr_range
        else:
            curr_norm = np.zeros_like(heatmap_curr)
        
        diff = np.abs(curr_norm - prev_norm)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return diff


def prepare_model_input(
    heatmap_prev: np.ndarray,
    heatmap_curr: np.ndarray,
    input_mode: Literal['diff', 'concat', 'both'] = 'diff',
    diff_method: Literal['absolute', 'relative', 'normalized'] = 'absolute'
) -> np.ndarray:
    """
    モデル入力用のテンソルを準備
    
    Args:
        heatmap_prev: (H, W) - 前フレームのヒートマップ
        heatmap_curr: (H, W) - 今フレームのヒートマップ
        input_mode: 入力モード
            - 'diff': 差分ヒートマップのみ (1チャネル)
            - 'concat': 前フレーム+今フレームを結合 (2チャネル)
            - 'both': 前フレーム+今フレーム+差分を結合 (3チャネル)
        diff_method: 差分計算方法（input_mode='diff'または'both'の場合に使用）
    
    Returns:
        input_tensor: (C, H, W) - モデル入力テンソル
    """
    if input_mode == 'diff':
        diff_heatmap = compute_heatmap_diff(heatmap_prev, heatmap_curr, method=diff_method)
        input_tensor = diff_heatmap[np.newaxis, :, :]  # (1, H, W)
    
    elif input_mode == 'concat':
        input_tensor = np.stack([heatmap_prev, heatmap_curr], axis=0)  # (2, H, W)
    
    elif input_mode == 'both':
        diff_heatmap = compute_heatmap_diff(heatmap_prev, heatmap_curr, method=diff_method)
        input_tensor = np.stack([heatmap_prev, heatmap_curr, diff_heatmap], axis=0)  # (3, H, W)
    
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    
    return input_tensor


def label_grid_cells(
    heatmap: np.ndarray,
    gt_joints: np.ndarray,  # (22, 3)
    spatial_axis: Literal['x', 'y', 'z'],
    spatial_range: Tuple[float, float],
    spatial_bins: int,
    dist_threshold: float = 0.10,  # デフォルトを10cmに変更
    normalize: bool = True,
    normalized_range: Tuple[float, float] = (-1.0, 1.0),
    moving_joint_indices: Optional[List[int]] = None  # 動いている関節のインデックスリスト（Noneの場合は全関節）
) -> np.ndarray:
    """
    各グリッドセルにラベルを付与（グリッドセル単位）
    
    Args:
        heatmap: (spatial_bins, feature_bins) - ヒートマップ（形状の取得用）
        gt_joints: (22, 3) - GT関節座標
        spatial_axis: 空間軸
        spatial_range: 空間軸の実際の範囲 (min, max)（メートル単位）
        spatial_bins: 空間軸のビン数
        dist_threshold: 距離閾値（メートル、デフォルト0.10m）
        normalize: 正規化後にビン分割したか
        normalized_range: 正規化後の範囲 (min, max)
        moving_joint_indices: 動いている関節のインデックスリスト（Noneの場合は全末端関節）
    
    Returns:
        labels: (spatial_bins, feature_bins) - 0: 非末端, 1: 末端
            【軸の統一】第0軸=空間軸、第1軸=特徴量軸
    """
    # 末端関節のインデックス
    DISTAL_JOINT_INDICES = [7, 8, 10, 11, 18, 19, 20, 21]  # L_Ankle, R_Ankle, L_Foot, R_Foot, L_Elbow, R_Elbow, L_Wrist, R_Wrist
    
    # 動いている関節のみを使用（指定されている場合）
    if moving_joint_indices is not None:
        joint_indices_to_label = [idx for idx in DISTAL_JOINT_INDICES if idx in moving_joint_indices]
    else:
        joint_indices_to_label = DISTAL_JOINT_INDICES
    
    labels = np.zeros((spatial_bins, heatmap.shape[1]), dtype=np.int64)
    
    if normalize:
        # 正規化後の範囲でビンサイズを計算
        norm_min, norm_max = normalized_range
        spatial_bin_size_norm = (norm_max - norm_min) / spatial_bins if spatial_bins > 0 else 1.0
        
        # dist_thresholdを正規化範囲に変換
        spatial_min, spatial_max = spatial_range
        spatial_range_size = spatial_max - spatial_min
        if spatial_range_size > 1e-6:
            dist_threshold_norm = dist_threshold / spatial_range_size * (norm_max - norm_min)
        else:
            dist_threshold_norm = (norm_max - norm_min) * 0.1  # デフォルト値
        
        threshold_bins = int(dist_threshold_norm / spatial_bin_size_norm) if spatial_bin_size_norm > 0 else 1
    else:
        # 正規化なし（既存の方法）
        spatial_min, spatial_max = spatial_range
        spatial_bin_size = (spatial_max - spatial_min) / spatial_bins if spatial_bins > 0 else 1.0
        threshold_bins = int(dist_threshold / spatial_bin_size) if spatial_bin_size > 0 else 1
    
    # 各末端関節について（動いている関節のみ）
    for joint_idx in joint_indices_to_label:
        if joint_idx >= len(gt_joints):
            continue
        
        joint_pos = gt_joints[joint_idx]
        spatial_value = joint_pos[{'x': 0, 'y': 1, 'z': 2}[spatial_axis]]
        
        if normalize:
            # GT関節を正規化してからビンにマッピング
            spatial_value_norm = normalize_value(spatial_value, spatial_range, normalized_range)
            norm_min, norm_max = normalized_range
            spatial_bin_size_norm = (norm_max - norm_min) / spatial_bins if spatial_bins > 0 else 1.0
            # 【Step4: +0.5の一貫性】ビン中心を扱うため、+0.5を追加
            spatial_center_bin = int((spatial_value_norm - norm_min) / spatial_bin_size_norm + 0.5)
        else:
            # 既存の方法
            spatial_min, spatial_max = spatial_range
            spatial_bin_size = (spatial_max - spatial_min) / spatial_bins if spatial_bins > 0 else 1.0
            # 【Step4: +0.5の一貫性】ビン中心を扱うため、+0.5を追加
            spatial_center_bin = int((spatial_value - spatial_min) / spatial_bin_size + 0.5)
        
        # 範囲内のビンをラベル付け
        bin_start = max(0, spatial_center_bin - threshold_bins)
        bin_end = min(spatial_bins, spatial_center_bin + threshold_bins + 1)
        
        labels[bin_start:bin_end, :] = 1
    
    return labels


def label_change_regions(
    diff_heatmap: np.ndarray,
    gt_joints: np.ndarray,  # (22, 3)
    spatial_axis: Literal['x', 'y', 'z'],
    spatial_range: Tuple[float, float],
    spatial_bins: int,
    change_threshold: float = 0.1,
    normalize: bool = True,
    normalized_range: Tuple[float, float] = (-1.0, 1.0)
) -> Dict:
    """
    変化領域とGT末端関節の重複度でラベル付け（変化領域単位）
    
    Args:
        diff_heatmap: (spatial_bins, feature_bins) - 差分ヒートマップ
        gt_joints: (22, 3) - GT関節座標
        spatial_axis: 空間軸
        spatial_range: 空間軸の範囲 (min, max)
        spatial_bins: 空間軸のビン数
        change_threshold: 変化の閾値
    
    Returns:
        region_labels: Dict - 各変化領域のラベルと重複度
    """
    # 末端関節のインデックス
    DISTAL_JOINT_INDICES = [7, 8, 10, 11, 18, 19, 20, 21]
    
    # 変化領域を検出
    change_mask = diff_heatmap > change_threshold
    
    if not np.any(change_mask):
        return {
            'regions': [],
            'num_regions': 0
        }
    
    labeled, num_regions = ndimage.label(change_mask)
    
    if normalize:
        # 正規化後の範囲でビンサイズを計算
        norm_min, norm_max = normalized_range
        spatial_bin_size_norm = (norm_max - norm_min) / spatial_bins if spatial_bins > 0 else 1.0
    else:
        spatial_min, spatial_max = spatial_range
        spatial_bin_size = (spatial_max - spatial_min) / spatial_bins if spatial_bins > 0 else 1.0
    
    region_labels = []
    for region_id in range(1, num_regions + 1):
        region_mask = (labeled == region_id)
        spatial_indices, feature_indices = np.where(region_mask)
        
        if len(spatial_indices) == 0:
            continue
        
        # 領域の空間範囲を計算（メートル単位）
        spatial_min_idx = spatial_indices.min()
        spatial_max_idx = spatial_indices.max()
        
        if normalize:
            # 正規化範囲からメートル単位に逆変換
            spatial_min, spatial_max = spatial_range
            spatial_center_norm_min = norm_min + spatial_min_idx * spatial_bin_size_norm
            spatial_center_norm_max = norm_min + (spatial_max_idx + 1) * spatial_bin_size_norm
            
            spatial_min_m = denormalize_value(spatial_center_norm_min, spatial_range, normalized_range)
            spatial_max_m = denormalize_value(spatial_center_norm_max, spatial_range, normalized_range)
        else:
            # 既存の方法
            spatial_min_m = spatial_min + spatial_min_idx * spatial_bin_size
            spatial_max_m = spatial_min + (spatial_max_idx + 1) * spatial_bin_size
        
        # GT末端関節との重複度を計算
        overlap_score = 0.0
        for joint_idx in DISTAL_JOINT_INDICES:
            if joint_idx >= len(gt_joints):
                continue
            
            joint_pos = gt_joints[joint_idx]
            joint_spatial = joint_pos[{'x': 0, 'y': 1, 'z': 2}[spatial_axis]]
            
            if spatial_min_m <= joint_spatial <= spatial_max_m:
                overlap_score += 1.0
        
        region_labels.append({
            'region_id': region_id,
            'label': 1 if overlap_score > 0 else 0,
            'overlap_score': overlap_score,
            'spatial_range': (spatial_min_m, spatial_max_m),
            'spatial_indices': (spatial_min_idx, spatial_max_idx),
            'feature_range': (feature_indices.min(), feature_indices.max()),
            'size': len(spatial_indices)
        })
    
    return {
        'regions': region_labels,
        'num_regions': num_regions
    }


def test_roundtrip_coordinate_conversion(
    spatial_value: float,
    spatial_range: Tuple[float, float],
    spatial_bins: int,
    normalize: bool = True,
    normalized_range: Tuple[float, float] = (-1.0, 1.0),
    tolerance: Optional[float] = None
) -> Dict:
    """
    【Step3: 往復テスト】座標変換の往復テスト
    
    Args:
        spatial_value: 元の空間座標値（メートル）
        spatial_range: 空間軸の範囲 (min, max)（メートル単位）
        spatial_bins: 空間軸のビン数
        normalize: 正規化を使用するか
        normalized_range: 正規化範囲
        tolerance: 許容誤差（Noneの場合はビンサイズの半分を自動計算）
    
    Returns:
        result: Dict - テスト結果
            - success: bool - テスト成功か
            - original_value: float - 元の値
            - converted_value: float - 変換後の値
            - error: float - 誤差
            - tolerance: float - 使用された許容誤差
    """
    # 許容誤差を計算（ビン量子化誤差を考慮）
    if tolerance is None:
        spatial_min, spatial_max = spatial_range
        spatial_bin_size = (spatial_max - spatial_min) / spatial_bins if spatial_bins > 0 else 1.0
        # ビンサイズ全体 + マージン（正規化/逆正規化の数値誤差も考慮）
        # 最悪の場合、元の値がビンの端にあり、最も遠いビン中心にマッピングされる可能性がある
        tolerance = spatial_bin_size * 1.0 + 2e-3  # ビンサイズ全体 + 2mmマージン
    
    if normalize:
        # 正規化
        spatial_value_norm = normalize_value(spatial_value, spatial_range, normalized_range)
        
        # ビンにマッピング（+0.5でビン中心を考慮）
        norm_min, norm_max = normalized_range
        spatial_bin_size_norm = (norm_max - norm_min) / spatial_bins if spatial_bins > 0 else 1.0
        spatial_center_bin = int((spatial_value_norm - norm_min) / spatial_bin_size_norm + 0.5)
        spatial_center_bin = max(0, min(spatial_bins - 1, spatial_center_bin))
        
        # ビン中心から正規化値に戻す
        spatial_center_norm = norm_min + (spatial_center_bin + 0.5) * spatial_bin_size_norm
        
        # 逆正規化
        converted_value = denormalize_value(spatial_center_norm, spatial_range, normalized_range)
    else:
        # 正規化なし
        spatial_min, spatial_max = spatial_range
        spatial_bin_size = (spatial_max - spatial_min) / spatial_bins if spatial_bins > 0 else 1.0
        spatial_center_bin = int((spatial_value - spatial_min) / spatial_bin_size + 0.5)
        spatial_center_bin = max(0, min(spatial_bins - 1, spatial_center_bin))
        
        # ビン中心から実座標に戻す
        converted_value = spatial_min + (spatial_center_bin + 0.5) * spatial_bin_size
    
    error = abs(spatial_value - converted_value)
    success = error < tolerance
    
    return {
        'success': success,
        'original_value': float(spatial_value),
        'converted_value': float(converted_value),
        'error': float(error),
        'bin_index': int(spatial_center_bin),
        'tolerance': float(tolerance)
    }


def create_multi_axis_heatmaps(
    points: np.ndarray,  # (N, 6) [x, y, z, velocity, amplitude, energy_power]
    spatial_bins: int = 50,
    feature_bins: int = 50,
    spatial_range: Optional[Dict[str, Tuple[float, float]]] = None,
    feature_range: Optional[Dict[str, Tuple[float, float]]] = None,
    normalize: bool = True,
    normalized_range: Tuple[float, float] = (-1.0, 1.0)
) -> Dict[str, Tuple[np.ndarray, Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """
    6つのヒートマップを生成（x/y/z × energy_power/velocity）
    
    Args:
        points: (N, 6) - 点群データ [x, y, z, velocity, amplitude, energy_power]
        spatial_bins: 空間軸のビン数
        feature_bins: 特徴量軸のビン数
        spatial_range: Dict[str, Tuple[float, float]] - 各空間軸の範囲 (min, max)（Noneの場合は自動計算）
        feature_range: Dict[str, Tuple[float, float]] - 各特徴量軸の範囲 (min, max)（Noneの場合は自動計算）
        normalize: 正規化後にビン分割するか
        normalized_range: 正規化後の範囲 (min, max)
    
    Returns:
        heatmaps: Dict[str, Tuple[heatmap, spatial_range, feature_range, spatial_range_norm, feature_range_norm]]
            - 'x_energy_power': (heatmap, spatial_range, feature_range, spatial_range_norm, feature_range_norm)
            - 'x_velocity': (heatmap, spatial_range, feature_range, spatial_range_norm, feature_range_norm)
            - 'y_energy_power': ...
            - 'y_velocity': ...
            - 'z_energy_power': ...
            - 'z_velocity': ...
    """
    heatmaps = {}
    
    for spatial_axis in ['x', 'y', 'z']:
        for feature_axis in ['energy_power', 'velocity']:
            key = f'{spatial_axis}_{feature_axis}'
            
            # 既存のcreate_heatmap_gridを使用
            spatial_range_axis = spatial_range.get(spatial_axis) if spatial_range else None
            feature_range_axis = feature_range.get(feature_axis) if feature_range else None
            
            heatmap, spatial_range_actual, feature_range_actual, spatial_range_norm, feature_range_norm = create_heatmap_grid(
                points,
                spatial_axis=spatial_axis,
                feature_axis=feature_axis,
                spatial_bins=spatial_bins,
                feature_bins=feature_bins,
                spatial_range=spatial_range_axis,
                feature_range=feature_range_axis,
                normalize=normalize,
                normalized_range=normalized_range
            )
            
            heatmaps[key] = (heatmap, spatial_range_actual, feature_range_actual, spatial_range_norm, feature_range_norm)
    
    return heatmaps


def prepare_multi_axis_input(
    heatmaps_prev: Dict[str, np.ndarray],
    heatmaps_curr: Dict[str, np.ndarray],
    input_mode: Literal['diff', 'concat', 'both'] = 'concat',
    diff_method: Literal['absolute', 'relative', 'normalized'] = 'absolute'
) -> np.ndarray:
    """
    12チャネル（または18チャネル）の入力テンソルを準備
    
    Args:
        heatmaps_prev: Dict[str, np.ndarray] - 前フレームの6ヒートマップ
            - キー: 'x_energy_power', 'x_velocity', 'y_energy_power', 'y_velocity', 'z_energy_power', 'z_velocity'
            - 値: (spatial_bins, feature_bins) のヒートマップ
        heatmaps_curr: Dict[str, np.ndarray] - 今フレームの6ヒートマップ（形式は同上）
        input_mode: 入力モード
            - 'diff': 差分のみ (6チャネル)
            - 'concat': 前フレーム+今フレーム (12チャネル)
            - 'both': 前フレーム+今フレーム+差分 (18チャネル)
        diff_method: 差分計算方法（input_mode='diff'または'both'の場合に使用）
    
    Returns:
        input_tensor: (C, spatial_bins, feature_bins) - Cは6, 12, 18のいずれか
    """
    channels = []
    
    # 順序: x-energy, x-velocity, y-energy, y-velocity, z-energy, z-velocity
    axis_order = ['x', 'y', 'z']
    feature_order = ['energy_power', 'velocity']
    
    if input_mode == 'diff':
        # 差分のみ
        for axis in axis_order:
            for feature in feature_order:
                key = f'{axis}_{feature}'
                diff = compute_heatmap_diff(
                    heatmaps_prev[key],
                    heatmaps_curr[key],
                    method=diff_method
                )
                channels.append(diff)
    
    elif input_mode == 'concat':
        # 前フレーム + 今フレーム
        for axis in axis_order:
            for feature in feature_order:
                key = f'{axis}_{feature}'
                channels.append(heatmaps_prev[key])
        
        for axis in axis_order:
            for feature in feature_order:
                key = f'{axis}_{feature}'
                channels.append(heatmaps_curr[key])
    
    elif input_mode == 'both':
        # 前フレーム + 今フレーム + 差分
        for axis in axis_order:
            for feature in feature_order:
                key = f'{axis}_{feature}'
                channels.append(heatmaps_prev[key])
        
        for axis in axis_order:
            for feature in feature_order:
                key = f'{axis}_{feature}'
                channels.append(heatmaps_curr[key])
        
        for axis in axis_order:
            for feature in feature_order:
                key = f'{axis}_{feature}'
                diff = compute_heatmap_diff(
                    heatmaps_prev[key],
                    heatmaps_curr[key],
                    method=diff_method
                )
                channels.append(diff)
    
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    
    input_tensor = np.stack(channels, axis=0)  # (C, spatial_bins, feature_bins)
    return input_tensor


def smoke_test_spatial_bins(
    prob_map: np.ndarray,  # (50, 50) - アップサンプル後の確率マップ
    gt_joints: np.ndarray,  # (22, 3) - GT関節座標
    spatial_axis: str,
    spatial_range: Tuple[float, float],
    spatial_bins: int = 50,
    normalize: bool = True,
    normalized_range: Tuple[float, float] = (-1.0, 1.0),
    threshold: float = 0.5
) -> Dict:
    """
    【Step5: スモークテスト】1フレームでGTの空間ビンと推論結果の空間ビンを比較
    
    Args:
        prob_map: (50, 50) - 確率マップ（アップサンプル後）
        gt_joints: (22, 3) - GT関節座標
        spatial_axis: 空間軸 ('x', 'y', 'z')
        spatial_range: 空間軸の範囲 (min, max)（メートル単位）
        spatial_bins: 空間軸のビン数（50）
        normalize: 正規化を使用するか
        normalized_range: 正規化範囲
        threshold: 確率閾値
    
    Returns:
        result: Dict - テスト結果
            - mean_bin_diff: float - 平均ビン差
            - max_bin_diff: float - 最大ビン差
            - matched_joints: int - マッチした関節数
            - total_joints: int - 総関節数
    """
    from heatmap_distal_detection.infer_heatmap_model import extract_distal_regions
    
    # 末端関節のインデックス
    DISTAL_JOINT_INDICES = [7, 8, 10, 11, 18, 19, 20, 21]
    
    # 推論結果の領域を抽出
    regions = extract_distal_regions(
        prob_map,
        threshold=threshold,
        min_region_size=5,
        use_weighted_center=True
    )
    
    # GT関節の空間ビンを計算
    gt_bins = []
    for joint_idx in DISTAL_JOINT_INDICES:
        if joint_idx >= len(gt_joints):
            continue
        
        joint_pos = gt_joints[joint_idx]
        spatial_value = joint_pos[{'x': 0, 'y': 1, 'z': 2}[spatial_axis]]
        
        if normalize:
            spatial_value_norm = normalize_value(spatial_value, spatial_range, normalized_range)
            norm_min, norm_max = normalized_range
            spatial_bin_size_norm = (norm_max - norm_min) / spatial_bins if spatial_bins > 0 else 1.0
            gt_bin = int((spatial_value_norm - norm_min) / spatial_bin_size_norm + 0.5)
        else:
            spatial_min, spatial_max = spatial_range
            spatial_bin_size = (spatial_max - spatial_min) / spatial_bins if spatial_bins > 0 else 1.0
            gt_bin = int((spatial_value - spatial_min) / spatial_bin_size + 0.5)
        
        gt_bin = max(0, min(spatial_bins - 1, gt_bin))
        gt_bins.append((joint_idx, gt_bin, spatial_value))
    
    # 推論結果の空間ビンを計算
    pred_bins = []
    for region in regions:
        if 'weighted_center' in region:
            spatial_center_idx, _ = region['weighted_center']
        else:
            spatial_center_idx, _ = region.get('geometric_center', (0.0, 0.0))
        
        pred_bin = int(spatial_center_idx + 0.5)  # 重み付き中心は連続値なので+0.5で丸める
        pred_bin = max(0, min(spatial_bins - 1, pred_bin))
        pred_bins.append(pred_bin)
    
    # マッチングとビン差を計算
    bin_diffs = []
    matched = set()
    
    for joint_idx, gt_bin, spatial_value in gt_bins:
        min_diff = float('inf')
        best_pred_idx = None
        
        for pred_idx, pred_bin in enumerate(pred_bins):
            if pred_idx in matched:
                continue
            
            diff = abs(gt_bin - pred_bin)
            if diff < min_diff:
                min_diff = diff
                best_pred_idx = pred_idx
        
        if best_pred_idx is not None:
            bin_diffs.append(min_diff)
            matched.add(best_pred_idx)
    
    if len(bin_diffs) == 0:
        return {
            'mean_bin_diff': float('inf'),
            'max_bin_diff': float('inf'),
            'matched_joints': 0,
            'total_joints': len(gt_bins),
            'bin_diffs': []
        }
    
    return {
        'mean_bin_diff': float(np.mean(bin_diffs)),
        'max_bin_diff': float(np.max(bin_diffs)),
        'matched_joints': len(bin_diffs),
        'total_joints': len(gt_bins),
        'bin_diffs': [float(d) for d in bin_diffs]
    }

