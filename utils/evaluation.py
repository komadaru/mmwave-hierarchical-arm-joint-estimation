#!/usr/bin/env python3
"""
評価指標の計算
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import ndimage


def compute_pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    ピクセル精度を計算
    
    Args:
        pred: (H, W) - 予測ラベル（0 or 1）
        target: (H, W) - 正解ラベル（0 or 1）
    
    Returns:
        accuracy: ピクセル精度
    """
    return np.mean(pred == target)


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """
    IoU (Intersection over Union) を計算
    
    Args:
        pred: (H, W) - 予測ラベル（0 or 1）
        target: (H, W) - 正解ラベル（0 or 1）
    
    Returns:
        iou: IoU値
    """
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_precision_recall_f1(
    pred: np.ndarray,
    target: np.ndarray
) -> Tuple[float, float, float]:
    """
    Precision, Recall, F1-Scoreを計算
    
    Args:
        pred: (H, W) - 予測ラベル（0 or 1）
        target: (H, W) - 正解ラベル（0 or 1）
    
    Returns:
        precision: Precision
        recall: Recall
        f1: F1-Score
    """
    tp = np.logical_and(pred == 1, target == 1).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def compute_spatial_accuracy(
    pred_regions: List[Dict],
    gt_joints: np.ndarray,  # (22, 3)
    spatial_axis: str,
    spatial_range: Tuple[float, float],
    spatial_bins: int,
    dist_threshold: float = 0.25,
    normalize: bool = True,
    normalized_range: Tuple[float, float] = (-1.0, 1.0),
    use_3d: bool = True,
    use_weighted_center: bool = True,
    metadata: Dict = None,
    output_spatial_bins: int = None
) -> Dict[str, float]:
    """
    空間精度を計算（検出領域とGT末端関節の位置誤差）
    
    Args:
        pred_regions: 検出された領域のリスト
        gt_joints: GT関節座標 (22, 3)
        spatial_axis: 空間軸 ('x', 'y', 'z') - use_3d=Falseの場合のみ使用
        spatial_range: 空間軸の範囲 (min, max)
        spatial_bins: データ収集時の空間軸のビン数（元のビン数、50など）
        dist_threshold: 距離閾値（メートル）
        normalize: 正規化を使用するか
        normalized_range: 正規化範囲
        use_3d: 3Dユークリッド距離を使用するか（Trueの場合、全軸で評価）
        use_weighted_center: 重み付き中心を使用するか
        metadata: メタデータ（他の軸の情報を含む）
        output_spatial_bins: モデルの実際の出力サイズ（48など）。Noneの場合はspatial_binsを使用
    
    Returns:
        metrics: Dict - 空間精度メトリクス
    """
    # 末端関節のインデックス
    DISTAL_JOINT_INDICES = [7, 8, 10, 11, 18, 19, 20, 21]
    
    if metadata is None:
        metadata = {}
    
    # 実際の出力サイズを決定（【Step2】アップサンプル後のサイズ、デフォルトは50）
    if output_spatial_bins is None:
        # メタデータから取得を試みる
        output_spatial_bins = metadata.get('output_spatial_bins', 50)  # デフォルトは50（アップサンプル後）
    
    # 座標変換には実際の出力サイズを使用
    actual_spatial_bins = output_spatial_bins
    
    # GT末端関節の3D位置を取得
    gt_joint_positions_3d = []
    for joint_idx in DISTAL_JOINT_INDICES:
        if joint_idx < len(gt_joints):
            joint_pos = gt_joints[joint_idx]  # (x, y, z)
            gt_joint_positions_3d.append(joint_pos)
    
    if len(gt_joint_positions_3d) == 0:
        return {
            'mean_error': float('inf'),
            'median_error': float('inf'),
            'min_error': float('inf'),
            'max_error': float('inf'),
            'within_threshold': 0.0,
            'mean_error_1d': float('inf'),
            'mean_error_3d': float('inf')
        }
    
    gt_joint_positions_3d = np.array(gt_joint_positions_3d)  # (N, 3)
    
    # 検出領域の3D位置を計算
    pred_positions_3d = []
    
    if use_3d:
        # 3D評価の場合、全軸の情報が必要
        # 多軸・多特徴量モードの場合、メタデータから全軸の情報を取得
        use_multi_axis = metadata.get('use_multi_axis', False) if metadata else False
        
        if use_multi_axis:
            # 多軸・多特徴量モード: メタデータから全軸の範囲を取得
            spatial_ranges_dict = metadata.get('spatial_ranges', {})
            if not spatial_ranges_dict:
                # フォールバック: 単一軸の範囲を使用
                spatial_ranges_dict = {
                    'x': spatial_range,
                    'y': spatial_range,
                    'z': spatial_range
                }
        else:
            # レガシーモード: 単一軸のみ
            spatial_ranges_dict = {
                spatial_axis: spatial_range
            }
        
        # 空間軸の範囲とビン情報
        spatial_min, spatial_max = spatial_range
        spatial_bin_size = (spatial_max - spatial_min) / spatial_bins if spatial_bins > 0 else 1.0
        
        for region in pred_regions:
            # 重み付き中心または幾何学的中心を使用
            if use_weighted_center and 'weighted_center' in region:
                spatial_center_idx, _ = region['weighted_center']
            else:
                spatial_center_idx, _ = region.get('geometric_center', 
                    ((region['spatial_range'][0] + region['spatial_range'][1]) / 2.0, 0.0))
            
            # 空間軸の位置をメートル単位に変換
            # 注意: 実際の出力サイズ（actual_spatial_bins）を使用
            # 【Step4: +0.5の一貫性】重み付き中心はビンインデックスの連続値なので、ビン中心を計算するために+0.5を追加
            if normalize:
                from heatmap_distal_detection.data_preprocessing import denormalize_value
                norm_min, norm_max = normalized_range
                spatial_bin_size_norm = (norm_max - norm_min) / actual_spatial_bins if actual_spatial_bins > 0 else 1.0
                # 重み付き中心はビンインデックスの連続値（0から始まる）なので、ビン中心を計算するために+0.5を追加
                spatial_center_norm = norm_min + (spatial_center_idx + 0.5) * spatial_bin_size_norm
                spatial_value = denormalize_value(spatial_center_norm, spatial_range, normalized_range)
            else:
                spatial_bin_size = (spatial_max - spatial_min) / actual_spatial_bins if actual_spatial_bins > 0 else 1.0
                # 重み付き中心はビンインデックスの連続値（0から始まる）なので、ビン中心を計算するために+0.5を追加
                spatial_value = spatial_min + (spatial_center_idx + 0.5) * spatial_bin_size
            
            # 3D位置を構築
            # 注意: 現在の実装では、検出領域は1つのヒートマップ（例：x-energy_power）から抽出されるため、
            # 他の軸の情報は直接取得できません。多軸・多特徴量モデルの場合、将来的には
            # 3つのヒートマップ（x, y, z）すべてから検出領域を抽出する必要があります。
            # 現時点では、検出された軸の位置のみを使用し、他の軸は0とします。
            if spatial_axis == 'x':
                pred_pos_3d = np.array([spatial_value, 0.0, 0.0])  # y, zは推定が必要
            elif spatial_axis == 'y':
                pred_pos_3d = np.array([0.0, spatial_value, 0.0])  # x, zは推定が必要
            else:  # z
                pred_pos_3d = np.array([0.0, 0.0, spatial_value])  # x, yは推定が必要
            
            pred_positions_3d.append(pred_pos_3d)
        
        pred_positions_3d = np.array(pred_positions_3d)  # (M, 3)
        
        # 3Dユークリッド距離でマッチング
        errors_3d = []
        errors_1d = []
        matched_pred = set()
        
        for gt_pos_3d in gt_joint_positions_3d:
            min_error_3d = float('inf')
            min_error_1d = float('inf')
            best_pred_idx = None
            
            for pred_idx, pred_pos_3d in enumerate(pred_positions_3d):
                if pred_idx in matched_pred:
                    continue
                
                # 3Dユークリッド距離
                error_3d = np.linalg.norm(gt_pos_3d - pred_pos_3d)
                
                # 1D距離（比較用）
                if spatial_axis == 'x':
                    error_1d = abs(gt_pos_3d[0] - pred_pos_3d[0])
                elif spatial_axis == 'y':
                    error_1d = abs(gt_pos_3d[1] - pred_pos_3d[1])
                else:  # z
                    error_1d = abs(gt_pos_3d[2] - pred_pos_3d[2])
                
                if error_3d < min_error_3d:
                    min_error_3d = error_3d
                    min_error_1d = error_1d
                    best_pred_idx = pred_idx
            
            if best_pred_idx is not None:
                errors_3d.append(min_error_3d)
                errors_1d.append(min_error_1d)
                matched_pred.add(best_pred_idx)
            else:
                # マッチしない場合は閾値以上の誤差として扱う
                errors_3d.append(dist_threshold * 2)
                errors_1d.append(dist_threshold * 2)
        
        errors_3d = np.array(errors_3d)
        errors_1d = np.array(errors_1d)
        
        return {
            'mean_error': float(np.mean(errors_3d)),
            'median_error': float(np.median(errors_3d)),
            'min_error': float(np.min(errors_3d)),
            'max_error': float(np.max(errors_3d)),
            'within_threshold': float(np.mean(errors_3d <= dist_threshold)),
            'mean_error_1d': float(np.mean(errors_1d)),
            'mean_error_3d': float(np.mean(errors_3d))
        }
    
    else:
        # 1D評価（既存の方法）
        spatial_min, spatial_max = spatial_range
        spatial_bin_size = (spatial_max - spatial_min) / spatial_bins if spatial_bins > 0 else 1.0
        
        # GT末端関節の空間位置を取得（1軸のみ）
        gt_spatial_positions = []
        for joint_pos in gt_joint_positions_3d:
            spatial_value = joint_pos[{'x': 0, 'y': 1, 'z': 2}[spatial_axis]]
            gt_spatial_positions.append(spatial_value)
        
        # 検出領域の空間中心を計算
        pred_spatial_positions = []
        for region in pred_regions:
            # 重み付き中心または幾何学的中心を使用
            if use_weighted_center and 'weighted_center' in region:
                spatial_center_idx, _ = region['weighted_center']
            else:
                spatial_center_idx, _ = region.get('geometric_center',
                    ((region['spatial_range'][0] + region['spatial_range'][1]) / 2.0, 0.0))
            
            # 注意: 実際の出力サイズ（actual_spatial_bins）を使用
            # 【Step4: +0.5の一貫性】重み付き中心はビンインデックスの連続値なので、ビン中心を計算するために+0.5を追加
            if normalize:
                from heatmap_distal_detection.data_preprocessing import denormalize_value
                norm_min, norm_max = normalized_range
                spatial_bin_size_norm = (norm_max - norm_min) / actual_spatial_bins if actual_spatial_bins > 0 else 1.0
                # 重み付き中心はビンインデックスの連続値（0から始まる）なので、ビン中心を計算するために+0.5を追加
                spatial_center_norm = norm_min + (spatial_center_idx + 0.5) * spatial_bin_size_norm
                spatial_center = denormalize_value(spatial_center_norm, spatial_range, normalized_range)
            else:
                spatial_bin_size = (spatial_max - spatial_min) / actual_spatial_bins if actual_spatial_bins > 0 else 1.0
                # 重み付き中心はビンインデックスの連続値（0から始まる）なので、ビン中心を計算するために+0.5を追加
                spatial_center = spatial_min + (spatial_center_idx + 0.5) * spatial_bin_size
            
            pred_spatial_positions.append(spatial_center)
        
        # 各GT関節に対して最も近い検出領域を見つける
        errors = []
        matched_pred = set()
        
        for gt_pos in gt_spatial_positions:
            min_error = float('inf')
            best_pred_idx = None
            
            for pred_idx, pred_pos in enumerate(pred_spatial_positions):
                if pred_idx in matched_pred:
                    continue
                
                error = abs(gt_pos - pred_pos)
                if error < min_error:
                    min_error = error
                    best_pred_idx = pred_idx
            
            if best_pred_idx is not None:
                errors.append(min_error)
                matched_pred.add(best_pred_idx)
            else:
                errors.append(dist_threshold * 2)
        
        errors = np.array(errors)
        
        return {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'within_threshold': float(np.mean(errors <= dist_threshold)),
            'mean_error_1d': float(np.mean(errors)),
            'mean_error_3d': None
        }


def evaluate_predictions(
    pred_prob_maps: List[np.ndarray],  # List of (H, W)
    target_labels: List[np.ndarray],  # List of (H, W)
    pred_regions_list: List[List[Dict]],
    gt_joints_list: List[np.ndarray],
    metadata_list: List[Dict],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    予測結果を評価
    
    Args:
        pred_prob_maps: 予測確率マップのリスト
        target_labels: 正解ラベルのリスト
        pred_regions_list: 検出領域のリスト
        gt_joints_list: GT関節座標のリスト
        metadata_list: メタデータのリスト
        threshold: 確率閾値
    
    Returns:
        metrics: Dict - 評価メトリクス
    """
    pixel_accuracies = []
    ious = []
    precisions = []
    recalls = []
    f1_scores = []
    spatial_errors = []
    spatial_within_thresholds = []
    
    for i in range(len(pred_prob_maps)):
        pred_prob = pred_prob_maps[i]
        target = target_labels[i]
        metadata = metadata_list[i]
        
        # 確率マップを二値化
        pred_binary = (pred_prob > threshold).astype(np.int64)
        target_binary = target.astype(np.int64)
        
        # ピクセル精度
        pixel_acc = compute_pixel_accuracy(pred_binary, target_binary)
        pixel_accuracies.append(pixel_acc)
        
        # IoU
        iou = compute_iou(pred_binary, target_binary)
        ious.append(iou)
        
        # Precision, Recall, F1
        precision, recall, f1 = compute_precision_recall_f1(pred_binary, target_binary)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        # 空間精度（GT関節がある場合のみ）
        if i < len(gt_joints_list) and i < len(pred_regions_list):
            gt_joints = gt_joints_list[i]
            pred_regions = pred_regions_list[i]
            
            # 多軸・多特徴量モードの場合は3D評価をデフォルトに
            use_multi_axis = metadata.get('use_multi_axis', False)
            use_3d_eval = use_multi_axis  # 多軸モードの場合は3D評価
            
            # 空間軸と範囲を取得（レガシーモードとの互換性）
            if use_multi_axis:
                # 多軸モード: x軸を代表として使用（将来的には全軸から検出領域を抽出）
                spatial_axis = 'x'
                spatial_ranges = metadata.get('spatial_ranges', {})
                spatial_range = tuple(spatial_ranges.get('x', (0.0, 1.0)))
            else:
                # レガシーモード
                spatial_axis = metadata.get('spatial_axis', 'x')
                spatial_range = tuple(metadata.get('spatial_range', (0.0, 1.0)))
            
            spatial_metrics = compute_spatial_accuracy(
                pred_regions=pred_regions,
                gt_joints=gt_joints,
                spatial_axis=spatial_axis,
                spatial_range=spatial_range,
                spatial_bins=metadata.get('spatial_bins', 50),
                dist_threshold=0.25,
                normalize=metadata.get('normalize', True),
                normalized_range=tuple(metadata.get('normalized_range', [-1.0, 1.0])),
                use_3d=use_3d_eval,  # 多軸モードの場合は3D評価
                use_weighted_center=True,
                metadata=metadata,
                output_spatial_bins=50  # 【Step2】アップサンプル後のサイズ（50x50）
            )
            
            spatial_errors.append(spatial_metrics['mean_error'])
            spatial_within_thresholds.append(spatial_metrics['within_threshold'])
    
    # 平均を計算
    metrics = {
        'pixel_accuracy': float(np.mean(pixel_accuracies)),
        'iou': float(np.mean(ious)),
        'precision': float(np.mean(precisions)),
        'recall': float(np.mean(recalls)),
        'f1_score': float(np.mean(f1_scores)),
    }
    
    if spatial_errors:
        metrics['spatial_mean_error'] = float(np.mean(spatial_errors))
        metrics['spatial_median_error'] = float(np.median(spatial_errors))
        metrics['spatial_within_threshold'] = float(np.mean(spatial_within_thresholds))
        
        # 1D/3D誤差の情報を追加（最初のサンプルから取得）
        if len(pred_regions_list) > 0 and len(gt_joints_list) > 0:
            # 最初のサンプルで1D/3D誤差を計算（メタデータから）
            sample_metadata = metadata_list[0] if metadata_list else {}
            use_multi_axis = sample_metadata.get('use_multi_axis', False)
            
            # 空間軸と範囲を取得
            if use_multi_axis:
                spatial_axis = 'x'
                spatial_ranges = sample_metadata.get('spatial_ranges', {})
                spatial_range = tuple(spatial_ranges.get('x', (0.0, 1.0)))
            else:
                spatial_axis = sample_metadata.get('spatial_axis', 'x')
                spatial_range = tuple(sample_metadata.get('spatial_range', (0.0, 1.0)))
            
            sample_spatial_metrics = compute_spatial_accuracy(
                pred_regions=pred_regions_list[0] if pred_regions_list else [],
                gt_joints=gt_joints_list[0] if gt_joints_list else None,
                spatial_axis=spatial_axis,
                spatial_range=spatial_range,
                spatial_bins=sample_metadata.get('spatial_bins', 50),
                dist_threshold=0.25,
                normalize=sample_metadata.get('normalize', True),
                normalized_range=tuple(sample_metadata.get('normalized_range', [-1.0, 1.0])),
                use_3d=use_multi_axis,  # 多軸モードの場合は3D評価
                use_weighted_center=True,
                metadata=sample_metadata,
                output_spatial_bins=50  # 【Step2】アップサンプル後のサイズ（50x50）
            )
            
            if 'mean_error_1d' in sample_spatial_metrics:
                metrics['mean_error_1d'] = sample_spatial_metrics['mean_error_1d']
            if 'mean_error_3d' in sample_spatial_metrics:
                metrics['mean_error_3d'] = sample_spatial_metrics['mean_error_3d']
    
    return metrics

