#!/usr/bin/env python3
"""
リアルタイム推論速度測定スクリプト

点群からヒートマップ生成、モデル推論、座標変換・統合までの
エンドツーエンドの処理時間を測定します。
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mph.postprocessing import load_points_from_npz
from heatmap_distal_detection.visualize_xz_heatmap import create_xz_heatmap, create_xy_heatmap
from heatmap_distal_detection.models.arm_joint_regressor_hierarchical import (
    create_hierarchical_arm_joint_regressor,
    create_hierarchical_arm_joint_regressor_xy
)
from heatmap_distal_detection.models.full_skeleton_regressor import (
    create_full_skeleton_regressor
)
from heatmap_distal_detection.models.full_skeleton_regressor_xy import (
    create_full_skeleton_regressor_xy
)
from heatmap_distal_detection.evaluate_hybrid_skeleton_3d import (
    extract_shoulder_coords_from_full_skeleton,
    combine_2d_predictions_to_3d,
    combine_hybrid_predictions,
    denormalize_coords_2d,
    extract_elbow_wrist_from_arm_pred,
    ELBOW_WRIST_NAMES
)


def load_joints_definition(joints_def_path: Optional[str] = None) -> Dict:
    """関節定義を読み込む"""
    possible_paths = [
        joints_def_path,
        "data_specs/joints_def_22.json",
        "../data_specs/joints_def_22.json",
        "../../data_specs/joints_def_22.json",
        "mmwave_pose_reconstraction_pj/data_specs/joints_def_22.json",
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    joints_def = json.load(f)
                    return joints_def
            except Exception as e:
                continue
    
    # デフォルト
    return {
        "joint_names": ["Pelvis","L_Hip","R_Hip","Spine1","L_Knee","R_Knee","Spine2","L_Ankle","R_Ankle","Spine3","L_Foot","R_Foot","Neck","L_Collar","R_Collar","Head","L_Shoulder","R_Shoulder","L_Elbow","R_Elbow","L_Wrist","R_Wrist"]
    }


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """ヒートマップを0-1範囲に正規化"""
    if heatmap.max() > heatmap.min():
        normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        normalized = np.zeros_like(heatmap)
    return normalized.astype(np.float32)


def measure_single_inference(
    point_cloud_path: str,
    full_model_xz: nn.Module,
    full_model_xy: nn.Module,
    arm_model_xz: nn.Module,
    arm_model_xy: nn.Module,
    device: str,
    x_range_xz: Tuple[float, float] = (-1.0, 1.0),
    z_range: Tuple[float, float] = (-1.0, 0.75),
    x_range_xy: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (2.0, 5.0),
    bins: int = 50,
    feature: str = 'energy_power',
    joint_names: List[str] = None
) -> Dict[str, float]:
    """
    単一サンプルの推論時間を測定
    
    Returns:
        timing_dict: 各処理ステップの時間（秒）とFPS
    """
    timing = {}
    
    # [1] 点群データ読み込み
    start = time.time()
    try:
        points, _ = load_points_from_npz(point_cloud_path, normalize=False)
    except Exception as e:
        print(f"Warning: Failed to load {point_cloud_path}: {e}")
        return None
    if device == 'cuda':
        torch.cuda.synchronize()
    timing['load_points'] = time.time() - start
    
    # [2] ヒートマップ生成（XZ平面）
    start = time.time()
    heatmap_xz, x_range_actual_xz, z_range_actual = create_xz_heatmap(
        points, feature=feature, bins=bins,
        x_range=x_range_xz, z_range=z_range
    )
    if device == 'cuda':
        torch.cuda.synchronize()
    timing['create_heatmap_xz'] = time.time() - start
    
    # [3] ヒートマップ生成（XY平面）
    start = time.time()
    heatmap_xy, x_range_actual_xy, y_range_actual = create_xy_heatmap(
        points, feature=feature, bins=bins,
        x_range=x_range_xy, y_range=y_range
    )
    if device == 'cuda':
        torch.cuda.synchronize()
    timing['create_heatmap_xy'] = time.time() - start
    
    # [4] ヒートマップ正規化
    start = time.time()
    heatmap_xz_norm = normalize_heatmap(heatmap_xz)
    heatmap_xy_norm = normalize_heatmap(heatmap_xy)
    # (1, H, W)に変換
    heatmap_xz_tensor = torch.from_numpy(heatmap_xz_norm).unsqueeze(0).unsqueeze(0).to(device)
    heatmap_xy_tensor = torch.from_numpy(heatmap_xy_norm).unsqueeze(0).unsqueeze(0).to(device)
    if device == 'cuda':
        torch.cuda.synchronize()
    timing['normalize_heatmap'] = time.time() - start
    
    # [5] Full Model推論（XZ平面）
    start = time.time()
    with torch.no_grad():
        full_pred_xz = full_model_xz(heatmap_xz_tensor)
    if device == 'cuda':
        torch.cuda.synchronize()
    timing['full_model_xz'] = time.time() - start
    
    # [6] Full Model推論（XY平面）
    start = time.time()
    with torch.no_grad():
        full_pred_xy = full_model_xy(heatmap_xy_tensor)
    if device == 'cuda':
        torch.cuda.synchronize()
    timing['full_model_xy'] = time.time() - start
    
    # [7] 肩座標抽出
    start = time.time()
    full_pred_xz_np = full_pred_xz.cpu().numpy()  # (1, 44)
    full_pred_xy_np = full_pred_xy.cpu().numpy()  # (1, 44)
    shoulder_coords_xz = extract_shoulder_coords_from_full_skeleton(full_pred_xz_np, joint_names)  # (1, 4)
    shoulder_coords_xy = extract_shoulder_coords_from_full_skeleton(full_pred_xy_np, joint_names)  # (1, 4)
    shoulder_coords_xz_tensor = torch.from_numpy(shoulder_coords_xz).to(device)
    shoulder_coords_xy_tensor = torch.from_numpy(shoulder_coords_xy).to(device)
    if device == 'cuda':
        torch.cuda.synchronize()
    timing['extract_shoulder'] = time.time() - start
    
    # [8] Arm Model推論（XZ平面）
    start = time.time()
    with torch.no_grad():
        arm_pred_xz = arm_model_xz(heatmap_xz_tensor, shoulder_coords=shoulder_coords_xz_tensor)
    if device == 'cuda':
        torch.cuda.synchronize()
    timing['arm_model_xz'] = time.time() - start
    
    # [9] Arm Model推論（XY平面）
    start = time.time()
    with torch.no_grad():
        arm_pred_xy = arm_model_xy(heatmap_xy_tensor, shoulder_coords=shoulder_coords_xy_tensor)
    if device == 'cuda':
        torch.cuda.synchronize()
    timing['arm_model_xy'] = time.time() - start
    
    # [10] 座標の非正規化と3D統合
    start = time.time()
    arm_pred_xz_np = arm_pred_xz.cpu().numpy()  # (1, 12)
    arm_pred_xy_np = arm_pred_xy.cpu().numpy()  # (1, 12)
    
    # Arm modelの出力を6関節（肩、肘、手首）に変換
    # Arm model出力: [L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist] = 12次元
    # これを3D座標に変換
    arm_pred_3d = combine_2d_predictions_to_3d(
        arm_pred_xz_np, arm_pred_xy_np,
        x_range_xz, z_range, x_range_xy, y_range
    )  # (1, 6, 3)
    
    # Full modelの予測も3Dに変換
    full_pred_3d = combine_2d_predictions_to_3d(
        full_pred_xz_np, full_pred_xy_np,
        x_range_xz, z_range, x_range_xy, y_range
    )  # (1, 22, 3)
    
    # ハイブリッド予測の統合
    # combine_hybrid_predictionsは(N, 22, 3)と(N, 6, 3)を期待
    hybrid_pred_3d = combine_hybrid_predictions(
        full_pred_3d, arm_pred_3d, joint_names
    )  # (1, 22, 3)
    if device == 'cuda':
        torch.cuda.synchronize()
    timing['postprocess'] = time.time() - start
    
    # エンドツーエンド時間
    timing['end_to_end'] = sum([
        timing['load_points'],
        timing['create_heatmap_xz'],
        timing['create_heatmap_xy'],
        timing['normalize_heatmap'],
        timing['full_model_xz'],
        timing['full_model_xy'],
        timing['extract_shoulder'],
        timing['arm_model_xz'],
        timing['arm_model_xy'],
        timing['postprocess']
    ])
    
    # FPS計算
    timing['fps'] = 1.0 / timing['end_to_end'] if timing['end_to_end'] > 0 else 0.0
    
    return timing


def measure_realtime_inference_speed(
    data_path: str,
    full_model_xz_path: str,
    full_model_xy_path: str,
    arm_model_xz_path: str,
    arm_model_xy_path: str,
    device: str = 'cuda',
    num_samples: int = 100,
    warmup: int = 10,
    x_range_xz: Tuple[float, float] = (-1.0, 1.0),
    z_range: Tuple[float, float] = (-1.0, 0.75),
    x_range_xy: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (2.0, 5.0),
    bins: int = 50,
    feature: str = 'energy_power',
    joints_def_path: Optional[str] = None
) -> Dict:
    """
    リアルタイム推論速度を測定
    
    Args:
        data_path: JSONL形式のデータファイルパス
        full_model_xz_path: Full Skeleton Model (XZ平面)のチェックポイントパス
        full_model_xy_path: Full Skeleton Model (XY平面)のチェックポイントパス
        arm_model_xz_path: Arm Model (XZ平面)のチェックポイントパス
        arm_model_xy_path: Arm Model (XY平面)のチェックポイントパス
        device: 'cuda' or 'cpu'
        num_samples: 測定するサンプル数
        warmup: ウォームアップ回数
        x_range_xz, z_range: XZ平面の範囲
        x_range_xy, y_range: XY平面の範囲
        bins: ヒートマップのビン数
        feature: ヒートマップの特徴量
        joints_def_path: 関節定義ファイルのパス
    
    Returns:
        results: 測定結果の辞書
    """
    print(f"Using device: {device}")
    
    # 関節定義を読み込み
    joints_def = load_joints_definition(joints_def_path)
    joint_names = joints_def['joint_names']
    
    # モデルをロード
    print("\nLoading models...")
    
    # Full Skeleton Model (XZ)
    print("  Loading Full Skeleton Model (XZ)...")
    checkpoint_xz = torch.load(full_model_xz_path, map_location=device)
    full_config_xz = checkpoint_xz.get('config', {})
    # binsはconfigに'bins'として保存されている可能性がある
    heatmap_size_xz = full_config_xz.get('heatmap_size', full_config_xz.get('bins', bins))
    full_model_xz = create_full_skeleton_regressor(
        heatmap_size=heatmap_size_xz,
        num_joints=full_config_xz.get('num_joints', 22),
        base_channels=full_config_xz.get('base_channels', 32),
        dropout=full_config_xz.get('dropout', 0.5)
    )
    if 'model_state_dict' in checkpoint_xz:
        full_model_xz.load_state_dict(checkpoint_xz['model_state_dict'])
    else:
        full_model_xz.load_state_dict(checkpoint_xz)
    full_model_xz.to(device)
    full_model_xz.eval()
    
    # Full Skeleton Model (XY)
    print("  Loading Full Skeleton Model (XY)...")
    checkpoint_xy = torch.load(full_model_xy_path, map_location=device)
    full_config_xy = checkpoint_xy.get('config', {})
    # binsはconfigに'bins'として保存されている可能性がある
    heatmap_size_xy = full_config_xy.get('heatmap_size', full_config_xy.get('bins', bins))
    full_model_xy = create_full_skeleton_regressor_xy(
        heatmap_size=heatmap_size_xy,
        num_joints=full_config_xy.get('num_joints', 22),
        base_channels=full_config_xy.get('base_channels', 32),
        dropout=full_config_xy.get('dropout', 0.5)
    )
    if 'model_state_dict' in checkpoint_xy:
        full_model_xy.load_state_dict(checkpoint_xy['model_state_dict'])
    else:
        full_model_xy.load_state_dict(checkpoint_xy)
    full_model_xy.to(device)
    full_model_xy.eval()
    
    # Arm Model (XZ)
    print("  Loading Arm Model (XZ)...")
    checkpoint_arm_xz = torch.load(arm_model_xz_path, map_location=device)
    arm_config_xz = checkpoint_arm_xz.get('config', {})
    # binsはconfigに'bins'として保存されている可能性がある
    heatmap_size_arm_xz = arm_config_xz.get('heatmap_size', arm_config_xz.get('bins', bins))
    arm_model_xz = create_hierarchical_arm_joint_regressor(
        heatmap_size=heatmap_size_arm_xz,
        num_joints_per_arm=arm_config_xz.get('num_joints_per_arm', 3),
        base_channels=arm_config_xz.get('base_channels', 32),
        dropout=arm_config_xz.get('dropout', 0.5),
        use_attention=arm_config_xz.get('use_attention', True),
        use_improved_attention=arm_config_xz.get('use_improved_attention', True)
    )
    if 'model_state_dict' in checkpoint_arm_xz:
        arm_model_xz.load_state_dict(checkpoint_arm_xz['model_state_dict'])
    else:
        arm_model_xz.load_state_dict(checkpoint_arm_xz)
    arm_model_xz.to(device)
    arm_model_xz.eval()
    
    # Arm Model (XY)
    print("  Loading Arm Model (XY)...")
    checkpoint_arm_xy = torch.load(arm_model_xy_path, map_location=device)
    arm_config_xy = checkpoint_arm_xy.get('config', {})
    # binsはconfigに'bins'として保存されている可能性がある
    heatmap_size_arm_xy = arm_config_xy.get('heatmap_size', arm_config_xy.get('bins', bins))
    arm_model_xy = create_hierarchical_arm_joint_regressor_xy(
        heatmap_size=heatmap_size_arm_xy,
        num_joints_per_arm=arm_config_xy.get('num_joints_per_arm', 3),
        base_channels=arm_config_xy.get('base_channels', 32),
        dropout=arm_config_xy.get('dropout', 0.5),
        use_attention=arm_config_xy.get('use_attention', True),
        use_improved_attention=arm_config_xy.get('use_improved_attention', True)
    )
    if 'model_state_dict' in checkpoint_arm_xy:
        arm_model_xy.load_state_dict(checkpoint_arm_xy['model_state_dict'])
    else:
        arm_model_xy.load_state_dict(checkpoint_arm_xy)
    arm_model_xy.to(device)
    arm_model_xy.eval()
    
    print("  All models loaded")
    
    # データを読み込み
    print(f"\nLoading data from {data_path}...")
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    if len(samples) == 0:
        raise ValueError(f"No samples found in {data_path}")
    
    # サンプル数を制限
    num_samples = min(num_samples, len(samples))
    samples = samples[:num_samples]
    
    print(f"  Loaded {len(samples)} samples")
    
    # ウォームアップ
    print(f"\nWarming up ({warmup} iterations)...")
    for i in range(warmup):
        sample = samples[i % len(samples)]
        file_path = sample.get('file_path', '')
        if file_path and os.path.exists(file_path):
            _ = measure_single_inference(
                file_path, full_model_xz, full_model_xy, arm_model_xz, arm_model_xy,
                device, x_range_xz, z_range, x_range_xy, y_range, bins, feature, joint_names
            )
    
    # 測定
    print(f"\nMeasuring inference speed ({num_samples} samples)...")
    all_timings = []
    
    for sample in tqdm(samples, desc="Measuring"):
        file_path = sample.get('file_path', '')
        if not file_path or not os.path.exists(file_path):
            continue
        
        timing = measure_single_inference(
            file_path, full_model_xz, full_model_xy, arm_model_xz, arm_model_xy,
            device, x_range_xz, z_range, x_range_xy, y_range, bins, feature, joint_names
        )
        
        if timing is not None:
            all_timings.append(timing)
    
    if len(all_timings) == 0:
        raise ValueError("No valid timings collected")
    
    # 統計を計算
    timing_keys = [
        'load_points', 'create_heatmap_xz', 'create_heatmap_xy', 'normalize_heatmap',
        'full_model_xz', 'full_model_xy', 'extract_shoulder',
        'arm_model_xz', 'arm_model_xy', 'postprocess', 'end_to_end', 'fps'
    ]
    
    results = {
        'num_samples': len(all_timings),
        'device': device,
        'timings': {}
    }
    
    for key in timing_keys:
        values = [t[key] for t in all_timings if key in t]
        if len(values) > 0:
            results['timings'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
    
    # 単位変換（ms）
    results['timings_ms'] = {}
    for key, stats in results['timings'].items():
        if key != 'fps':
            results['timings_ms'][key] = {
                k: v * 1000.0 for k, v in stats.items()
            }
        else:
            results['timings_ms'][key] = stats
    
    return results


def visualize_results(results: Dict, output_dir: Path):
    """結果を可視化"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 処理時間の内訳を棒グラフで表示
    timing_keys = [
        'load_points', 'create_heatmap_xz', 'create_heatmap_xy', 'normalize_heatmap',
        'full_model_xz', 'full_model_xy', 'extract_shoulder',
        'arm_model_xz', 'arm_model_xy', 'postprocess'
    ]
    
    labels = [
        'Load Points', 'Heatmap XZ', 'Heatmap XY', 'Normalize',
        'Full Model XZ', 'Full Model XY', 'Extract Shoulder',
        'Arm Model XZ', 'Arm Model XY', 'Postprocess'
    ]
    
    means = [results['timings_ms'][k]['mean'] for k in timing_keys]
    stds = [results['timings_ms'][k]['std'] for k in timing_keys]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(labels))
    width = 0.6
    
    bars = ax.bar(x, means, width, yerr=stds, capsize=5, alpha=0.8, color='steelblue')
    
    ax.set_xlabel('Processing Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Real-time Inference Speed Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # 値をバーの上に表示
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.5, f'{mean:.2f}ms', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_speed_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'inference_speed_breakdown.pdf', bbox_inches='tight')
    plt.close()
    
    # エンドツーエンド時間とFPSの分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # エンドツーエンド時間の分布
    end_to_end_mean = results['timings_ms']['end_to_end']['mean']
    end_to_end_std = results['timings_ms']['end_to_end']['std']
    fps_mean = results['timings']['fps']['mean']
    fps_std = results['timings']['fps']['std']
    
    ax1.barh(['End-to-End'], [end_to_end_mean], xerr=[end_to_end_std], 
             capsize=10, alpha=0.8, color='coral')
    ax1.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('End-to-End Inference Time', fontsize=12, fontweight='bold')
    ax1.text(end_to_end_mean + end_to_end_std + 1, 0, f'{end_to_end_mean:.2f}ms', 
             va='center', fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # FPS
    ax2.barh(['FPS'], [fps_mean], xerr=[fps_std], capsize=10, alpha=0.8, color='green')
    ax2.set_xlabel('FPS', fontsize=12, fontweight='bold')
    ax2.set_title('Inference Speed (FPS)', fontsize=12, fontweight='bold')
    ax2.text(fps_mean + fps_std + 1, 0, f'{fps_mean:.2f} FPS', 
             va='center', fontsize=11, fontweight='bold')
    ax2.axvline(x=30, color='orange', linestyle='--', alpha=0.5, label='Target (30 FPS)')
    ax2.axvline(x=15, color='red', linestyle='--', alpha=0.5, label='Minimum (15 FPS)')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_speed_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'inference_speed_summary.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 6のデータをJSON形式で保存（paper_figuresディレクトリにも保存）
    from datetime import datetime
    paper_figures_dir = Path("outputs/paper_figures/figures_data")
    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    
    figure_6_data = {
        'title': 'Real-time Inference Speed Breakdown',
        'type': 'bar_plot',
        'xlabel': 'Processing Step',
        'ylabel': 'Time (ms)',
        'processing_steps': labels,
        'data': {
            'mean_ms': means,
            'std_ms': stds
        },
        'end_to_end': {
            'mean_ms': end_to_end_mean,
            'std_ms': end_to_end_std
        },
        'fps': {
            'mean': fps_mean,
            'std': fps_std
        },
        'reference_fps': {
            'target': 30,
            'minimum': 15
        },
        'created_at': datetime.now().isoformat(),
        'figure_name': 'figure_6_inference_speed'
    }
    
    with open(paper_figures_dir / 'figure_6_inference_speed_data.json', 'w', encoding='utf-8') as f:
        json.dump(figure_6_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Visualizations saved to {output_dir}")
    print(f"✓ Figure 6 data saved to {paper_figures_dir / 'figure_6_inference_speed_data.json'}")


def main():
    parser = argparse.ArgumentParser(description='Measure real-time inference speed')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to JSONL data file')
    parser.add_argument('--full_model_xz', type=str, required=True,
                        help='Path to Full Skeleton Model (XZ) checkpoint')
    parser.add_argument('--full_model_xy', type=str, required=True,
                        help='Path to Full Skeleton Model (XY) checkpoint')
    parser.add_argument('--arm_model_xz', type=str, required=True,
                        help='Path to Arm Model (XZ) checkpoint')
    parser.add_argument('--arm_model_xy', type=str, required=True,
                        help='Path to Arm Model (XY) checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/inference_speed',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to measure')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--x_range_xz', type=float, nargs=2, default=[-1.0, 1.0],
                        help='X range for XZ plane')
    parser.add_argument('--z_range', type=float, nargs=2, default=[-1.0, 0.75],
                        help='Z range for XZ plane')
    parser.add_argument('--x_range_xy', type=float, nargs=2, default=[-1.0, 1.0],
                        help='X range for XY plane')
    parser.add_argument('--y_range', type=float, nargs=2, default=[2.0, 5.0],
                        help='Y range for XY plane')
    parser.add_argument('--bins', type=int, default=50,
                        help='Heatmap bins')
    parser.add_argument('--feature', type=str, default='energy_power',
                        choices=['velocity', 'energy_power'],
                        help='Heatmap feature')
    parser.add_argument('--joints_def', type=str, default=None,
                        help='Path to joints_def_22.json')
    
    args = parser.parse_args()
    
    # デバイス確認
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # 測定実行
    results = measure_realtime_inference_speed(
        data_path=args.data_path,
        full_model_xz_path=args.full_model_xz,
        full_model_xy_path=args.full_model_xy,
        arm_model_xz_path=args.arm_model_xz,
        arm_model_xy_path=args.arm_model_xy,
        device=args.device,
        num_samples=args.num_samples,
        warmup=args.warmup,
        x_range_xz=tuple(args.x_range_xz),
        z_range=tuple(args.z_range),
        x_range_xy=tuple(args.x_range_xy),
        y_range=tuple(args.y_range),
        bins=args.bins,
        feature=args.feature,
        joints_def_path=args.joints_def
    )
    
    # 結果を保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON形式で保存
    with open(output_dir / 'inference_speed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # CSV形式で保存（統計サマリー）
    summary_data = []
    for key, stats in results['timings_ms'].items():
        summary_data.append({
            'Step': key,
            'Mean (ms)': stats['mean'],
            'Std (ms)': stats['std'],
            'Min (ms)': stats['min'],
            'Max (ms)': stats['max'],
            'Median (ms)': stats['median']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / 'inference_speed_summary.csv', index=False)
    
    # 結果を表示
    print("\n" + "="*60)
    print("Real-time Inference Speed Results")
    print("="*60)
    print(f"Device: {results['device']}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"\nProcessing Time Breakdown (mean ± std):")
    print("-" * 60)
    for key, stats in results['timings_ms'].items():
        if key != 'fps':
            print(f"  {key:20s}: {stats['mean']:6.2f} ± {stats['std']:5.2f} ms")
        else:
            print(f"  {key:20s}: {stats['mean']:6.2f} ± {stats['std']:5.2f} FPS")
    
    print(f"\nEnd-to-End Time: {results['timings_ms']['end_to_end']['mean']:.2f} ms")
    print(f"Inference Speed: {results['timings']['fps']['mean']:.2f} FPS")
    print("="*60)
    
    # 可視化
    visualize_results(results, output_dir)
    
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()

