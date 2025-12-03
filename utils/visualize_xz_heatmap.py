#!/usr/bin/env python3
"""
xz平面の点群ヒートマップ可視化スクリプト
点群をxz平面に投影し、velocity/amplitudeで色付けしたヒートマップを作成
値が低いほど明るい色（白に近い）で表示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mph.postprocessing import load_points_from_npz
from heatmap_distal_detection.dataset import HeatmapDistalDataset


def load_joints_definition(joints_def_path: str) -> Dict:
    """
    joints_def_22.jsonから骨格構造を読み込む
    """
    possible_paths = [
        joints_def_path,
        "data_specs/joints_def_22.json",
        "../data_specs/joints_def_22.json",
        "../../data_specs/joints_def_22.json",
        "mmwave_pose_reconstraction_pj/data_specs/joints_def_22.json",
        "../mmwave_pose_reconstraction_pj/data_specs/joints_def_22.json",
        "/home/users/grad/2024/24t0010/joints_def_22.json",
        "/cluster/users/grad/2024/24t0010/mmwave_pose_reconstraction_pj/data_specs/joints_def_22.json"
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    joints_def = json.load(f)
                    print(f"Loaded joints definition from: {path}")
                    return joints_def
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue
    
    print("Warning: joints_def_22.json not found, using default bone pairs")
    return {
        "bones": [[0,3],[3,6],[6,9],[9,12],[12,15],[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[9,13],[13,16],[16,18],[18,20],[9,14],[14,17],[17,19],[19,21]],
        "joint_names": ["Pelvis","L_Hip","R_Hip","Spine1","L_Knee","R_Knee","Spine2","L_Ankle","R_Ankle","Spine3","L_Foot","R_Foot","Neck","L_Collar","R_Collar","Head","L_Shoulder","R_Shoulder","L_Elbow","R_Elbow","L_Wrist","R_Wrist"]
    }


def get_bone_pairs(joints_def: Dict) -> List[Tuple[int, int]]:
    """
    骨格接続のリストを取得
    """
    bones = joints_def.get("bones", [])
    return [tuple(bone) for bone in bones]


def create_2d_heatmap(
    points: np.ndarray,  # (N, 6) [x, y, z, velocity, amplitude, energy_power]
    axis1: str = 'x',  # 'x', 'y', or 'z'
    axis2: str = 'z',  # 'x', 'y', or 'z'
    feature: str = 'velocity',  # 'velocity' or 'energy_power'
    bins: int = 50,
    axis1_range: Optional[Tuple[float, float]] = None,
    axis2_range: Optional[Tuple[float, float]] = None,
    invert_colormap: bool = True  # True: 値が低いほど明るい色
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    2D平面のヒートマップを作成（任意の2軸の組み合わせ）
    
    Args:
        points: (N, 6) 点群データ
        axis1: 第1軸 ('x', 'y', or 'z')
        axis2: 第2軸 ('x', 'y', or 'z')
        feature: 色付けに使用する特徴量 ('velocity' or 'energy_power')
        bins: ビン数
        axis1_range: 第1軸の範囲 (min, max)（Noneの場合は自動計算）
        axis2_range: 第2軸の範囲 (min, max)（Noneの場合は自動計算）
        invert_colormap: Trueの場合、値が低いほど明るい色
    
    Returns:
        heatmap: (bins, bins) ヒートマップ（特徴量の平均値）
        axis1_range: (min, max) 使用された第1軸の範囲
        axis2_range: (min, max) 使用された第2軸の範囲
    """
    if len(points) == 0:
        return np.zeros((bins, bins)), (0.0, 1.0), (0.0, 1.0)
    
    # 座標を抽出
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis1_idx = axis_map[axis1]
    axis2_idx = axis_map[axis2]
    axis1_coords = points[:, axis1_idx]  # (N,)
    axis2_coords = points[:, axis2_idx]  # (N,)
    
    # 特徴量を抽出
    feature_idx = {'velocity': 3, 'energy_power': 5}[feature]
    feature_values = points[:, feature_idx]  # (N,)
    
    # 範囲を決定
    if axis1_range is None:
        axis1_range = (axis1_coords.min(), axis1_coords.max())
    if axis2_range is None:
        axis2_range = (axis2_coords.min(), axis2_coords.max())
    
    # 範囲が0の場合はデフォルト範囲を使用
    if axis1_range[1] - axis1_range[0] < 1e-6:
        axis1_range = (axis1_range[0] - 0.5, axis1_range[0] + 0.5)
    if axis2_range[1] - axis2_range[0] < 1e-6:
        axis2_range = (axis2_range[0] - 0.5, axis2_range[0] + 0.5)
    
    # 値が非常に小さい場合（10^-30未満）、正規化してからヒストグラムを作成
    # これにより、np.histogram2dが正しく処理できるようになる
    feature_min = feature_values.min()
    feature_max = feature_values.max()
    needs_normalization = feature_max < 1e-30
    
    if needs_normalization:
        # 特徴量を0-1の範囲に正規化
        range_size = feature_max - feature_min
        if range_size > 0:
            feature_values_normalized = (feature_values - feature_min) / range_size
        else:
            # すべて同じ値の場合
            feature_values_normalized = np.ones_like(feature_values) * 0.5
        weights_to_use = feature_values_normalized
    else:
        weights_to_use = feature_values
    
    # 2Dヒストグラムを作成（特徴量を重みとして使用）
    # 注意: np.histogram2d(x, y, ...)は(y_bins, x_bins)の形状を返す
    # つまり、axis1_coordsがx軸、axis2_coordsがy軸の場合、
    # heatmap.shape = (axis2_bins, axis1_bins) = (y_bins, x_bins)
    # imshowは最初の次元をy軸、2番目の次元をx軸として解釈するため、転置は不要
    heatmap, axis1_edges, axis2_edges = np.histogram2d(
        axis1_coords, axis2_coords,
        bins=bins,
        range=[axis1_range, axis2_range],
        weights=weights_to_use  # 特徴量を重みとして使用（正規化済みの場合もある）
    )
    
    # デバッグ: ビンの境界を確認（無効化：学習時のパフォーマンス向上のため）
    # print(f"  [DEBUG] axis1 ({axis1}) edges: [{axis1_edges[0]:.3f}, {axis1_edges[-1]:.3f}], {len(axis1_edges)-1} bins")
    # print(f"  [DEBUG] axis2 ({axis2}) edges: [{axis2_edges[0]:.3f}, {axis2_edges[-1]:.3f}], {len(axis2_edges)-1} bins")
    # print(f"  [DEBUG] heatmap shape (before transpose): {heatmap.shape} = (axis2_bins, axis1_bins)")
    # print(f"  [DEBUG] axis1_range requested: {axis1_range}, axis2_range requested: {axis2_range}")
    
    # 点群数をカウント（0除算を避けるため）
    count_map, _, _ = np.histogram2d(
        axis1_coords, axis2_coords,
        bins=bins,
        range=[axis1_range, axis2_range]
    )
    
    # 平均値を計算（重み付き合計を点群数で割る）
    mask = count_map > 0
    heatmap[mask] = heatmap[mask] / count_map[mask]
    # データがない領域はNaNに設定（明るく表示されないようにする）
    heatmap[~mask] = np.nan
    
    # 正規化した場合、ヒートマップは既に0-1の範囲になっている
    # 正規化していない場合でも、後で正規化される可能性がある
    
    # デバッグ情報（無効化：学習時のパフォーマンス向上のため）
    # valid_values = heatmap[~np.isnan(heatmap)]
    # if len(valid_values) > 0:
    #     print(f"  [DEBUG] Heatmap stats: min={valid_values.min():.6e}, max={valid_values.max():.6e}, mean={valid_values.mean():.6e}, num_valid={len(valid_values)}/{heatmap.size}")
    #     ...
    # else:
    #     print(f"  [DEBUG] Heatmap: All values are NaN!")
    
    # 転置が必要: np.histogram2dは(axis2_bins, axis1_bins)の形状を返す
    # 例えば、axis1='x', axis2='z'の場合、heatmap.shape = (z_bins, x_bins)
    # imshowは最初の次元をy軸（縦軸）、2番目の次元をx軸（横軸）として解釈する
    # extent=[x_min, x_max, y_min, y_max]で、配列の[0,0]が(x_min, y_min)、[-1,-1]が(x_max, y_max)に対応
    # したがって、xz平面の場合、z軸がy軸として表示されるのを防ぐために転置が必要
    # 転置後: (axis1_bins, axis2_bins) = (x_bins, z_bins)
    # extent=[axis1_range, axis2_range] = [x_min, x_max, z_min, z_max]で正しく表示できる
    heatmap_transposed = heatmap.T
    # デバッグ出力（無効化：学習時のパフォーマンス向上のため）
    # print(f"  [DEBUG] heatmap shape (before transpose): {heatmap.shape} = (axis2_bins, axis1_bins)")
    # print(f"  [DEBUG] heatmap shape (after transpose): {heatmap_transposed.shape} = (axis1_bins, axis2_bins)")
    
    return heatmap_transposed, axis1_range, axis2_range


def create_xz_heatmap(
    points: np.ndarray,  # (N, 6) [x, y, z, velocity, amplitude, energy_power]
    feature: str = 'velocity',  # 'velocity' or 'energy_power'
    bins: int = 50,
    x_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None,
    invert_colormap: bool = True  # True: 値が低いほど明るい色
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    xz平面のヒートマップを作成（後方互換性のためのラッパー）
    """
    return create_2d_heatmap(points, 'x', 'z', feature, bins, x_range, z_range, invert_colormap)


def create_xy_heatmap(
    points: np.ndarray,  # (N, 6) [x, y, z, velocity, amplitude, energy_power]
    feature: str = 'velocity',  # 'velocity' or 'energy_power'
    bins: int = 50,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    invert_colormap: bool = True  # True: 値が低いほど明るい色
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    xy平面のヒートマップを作成（後方互換性のためのラッパー）
    """
    return create_2d_heatmap(points, 'x', 'y', feature, bins, x_range, y_range, invert_colormap)


def create_xy_plane_animation(
    dataset_path: str,
    output_path: str,
    max_frames: Optional[int] = None,
    sequence_id: Optional[str] = None,
    fps: int = 15,
    feature: str = 'velocity',  # 'velocity' or 'energy_power'
    joints_def_path: Optional[str] = None,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    point_size: float = 20.0,
    joint_size: float = 100.0
):
    """
    xy平面から見た点群+GT骨格のアニメーションを作成（上から見た図）
    
    Args:
        dataset_path: データセットパス（JSONL）
        output_path: 出力アニメーションファイルパス
        max_frames: 最大フレーム数（Noneの場合は全フレーム）
        sequence_id: 特定のシーケンスIDのみ可視化（Noneの場合は全シーケンス）
        fps: フレームレート
        feature: 色付けに使用する特徴量 ('velocity' or 'energy_power')
        joints_def_path: joints_def_22.jsonのパス（Noneの場合は自動検索）
        x_range: x軸の固定範囲（Noneの場合は自動計算）
        y_range: y軸の固定範囲（Noneの場合は自動計算）
        point_size: 点群のサイズ
        joint_size: 関節のサイズ
    """
    # 骨格定義を読み込み
    joints_def = load_joints_definition(joints_def_path or "data_specs/joints_def_22.json")
    bone_pairs = get_bone_pairs(joints_def)
    joint_names = joints_def.get("joint_names", [])
    
    print(f"Loaded {len(bone_pairs)} bone pairs from joints definition")
    
    # データセットを読み込み
    dataset = HeatmapDistalDataset(dataset_path)
    
    # シーケンスIDでフィルタリング
    if sequence_id is not None:
        filtered_data = []
        for i, sample in enumerate(dataset.data):
            if sample.get('sequence_id') == sequence_id:
                filtered_data.append((i, sample))
    else:
        filtered_data = [(i, dataset.data[i]) for i in range(len(dataset))]
    
    if max_frames is not None:
        filtered_data = filtered_data[:max_frames]
    
    print(f"Visualizing {len(filtered_data)} frames...")
    
    # フレームデータを準備
    frames_data = []
    for idx, sample in filtered_data:
        # GT関節
        gt_joints = None
        if 'gt_joints' in sample and sample['gt_joints'] is not None:
            gt_joints = np.array(sample['gt_joints'])  # (22, 3)
        
        # 点群パス
        file_path = sample.get('file_path', '')
        
        # 点群を読み込み
        points_curr = None
        if file_path and os.path.exists(file_path):
            try:
                points_curr, _ = load_points_from_npz(file_path, normalize=False)
            except:
                pass
        
        frames_data.append({
            'frame_idx': idx,
            'frame_num': sample.get('frame_num', idx),
            'sequence_id': sample.get('sequence_id', ''),
            'gt_joints': gt_joints,
            'points_curr': points_curr
        })
    
    if len(frames_data) == 0:
        print("No frames to visualize")
        return
    
    # 特徴量の範囲を計算（全フレームを通して）
    feature_idx = {'velocity': 3, 'energy_power': 5}[feature]
    all_features = []
    for frame in frames_data:
        if frame['points_curr'] is not None and frame['points_curr'].shape[1] > feature_idx:
            all_features.append(frame['points_curr'][:, feature_idx])
    
    if len(all_features) > 0:
        all_features = np.concatenate(all_features)
        feature_min, feature_max = np.min(all_features), np.max(all_features)
        print(f"Feature ({feature}) range: [{feature_min:.6e}, {feature_max:.6e}]")
    else:
        feature_min, feature_max = 0.0, 1.0
        print(f"[WARNING] No feature values found, using default range [0.0, 1.0]")
    
    # 座標範囲を計算
    if x_range is not None and y_range is not None:
        # 固定範囲を使用
        x_min, x_max = x_range
        y_min, y_max = y_range
        print(f"Using fixed ranges: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    else:
        # 座標範囲を計算（全フレームを通して）
        all_points = []
        all_joints = []
        for frame in frames_data:
            if frame['points_curr'] is not None:
                all_points.append(frame['points_curr'][:, :3])
            if frame['gt_joints'] is not None:
                all_joints.append(frame['gt_joints'])
        
        if len(all_points) > 0:
            all_points = np.concatenate(all_points, axis=0)
            x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        else:
            x_min, x_max = -1.0, 1.0
            y_min, y_max = -1.0, 1.0
        
        if len(all_joints) > 0:
            all_joints = np.concatenate(all_joints, axis=0)
            x_min = min(x_min, np.min(all_joints[:, 0]))
            x_max = max(x_max, np.max(all_joints[:, 0]))
            y_min = min(y_min, np.min(all_joints[:, 1]))
            y_max = max(y_max, np.max(all_joints[:, 1]))
        
        # マージンを追加
        margin = 0.2
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        print(f"Calculated ranges from data: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    
    # カラーマップ（値が低いほど明るい色）
    if feature == 'velocity':
        cmap = 'viridis_r'  # 反転（値が低いほど明るい）
    else:  # energy_power
        cmap = 'plasma_r'  # 反転（値が低いほど明るい）
    
    # 図を作成
    fig, ax = plt.subplots(figsize=(10, 10))
    cbar = None
    
    # アニメーション関数
    def animate(frame_idx):
        nonlocal cbar
        
        frame = frames_data[frame_idx]
        ax.clear()
        
        # 点群を描画（xy平面に投影）
        if frame['points_curr'] is not None:
            points = frame['points_curr']
            coords = points[:, :3]  # (N, 3)
            features = points[:, feature_idx]  # (N,)
            
            # 値が非常に小さい場合、正規化して表示
            if feature_max < 1e-30:
                range_size = feature_max - feature_min
                if range_size > 0:
                    features_normalized = (features - feature_min) / range_size
                else:
                    features_normalized = np.ones_like(features) * 0.5
                features_to_show = features_normalized
                vmin = 0.0
                vmax = 1.0
            else:
                features_to_show = features
                vmin = feature_min
                vmax = feature_max
            
            # xy平面に投影（x, y座標のみ使用）
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=features_to_show,
                cmap=cmap,
                s=point_size,
                alpha=0.7,
                vmin=vmin,
                vmax=vmax,
                edgecolors='none'
            )
            
            # カラーバーを追加（最初のフレームのみ）
            if frame_idx == 0 and cbar is None:
                cbar = plt.colorbar(scatter, ax=ax, label=feature)
        
        # GT骨格を描画（xy平面に投影）
        if frame['gt_joints'] is not None:
            joints = frame['gt_joints']
            
            # xy平面に投影（x, y座標のみ使用）
            ax.scatter(
                joints[:, 0], joints[:, 1],
                c='red', marker='o', s=joint_size, alpha=0.9, label='GT Joints',
                edgecolors='darkred', linewidths=2, zorder=10
            )
            
            # 骨格線を描画
            for parent, child in bone_pairs:
                if parent < len(joints) and child < len(joints):
                    ax.plot(
                        [joints[parent, 0], joints[child, 0]],
                        [joints[parent, 1], joints[child, 1]],
                        'r-', linewidth=3, alpha=0.9, zorder=5
                    )
        
        # プロットの設定
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'XY Plane View (Top View) - Frame {frame["frame_num"]} - {frame["sequence_id"]}', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    # アニメーションを作成
    print(f"Creating animation with {len(frames_data)} frames at {fps} fps...")
    anim = FuncAnimation(
        fig, animate, frames=len(frames_data),
        interval=1000/fps, blit=False, repeat=True
    )
    
    # 保存
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving animation to {output_path}...")
    try:
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"✓ Animation saved: {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Trying with ffmpeg writer...")
        try:
            anim.save(output_path.replace('.gif', '.mp4'), writer='ffmpeg', fps=fps)
            print(f"✓ Animation saved as MP4: {output_path.replace('.gif', '.mp4')}")
        except Exception as e2:
            print(f"Error saving with ffmpeg: {e2}")
            print("Please install pillow or ffmpeg to save animations")
    
    plt.close()


def create_xz_heatmap_animation(
    dataset_path: str,
    output_path: str,
    max_frames: Optional[int] = None,
    sequence_id: Optional[str] = None,
    fps: int = 15,
    feature: str = 'velocity',  # 'velocity' or 'energy_power'
    bins: int = 50,
    joints_def_path: Optional[str] = None,
    x_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None
):
    """
    xz平面のヒートマップ + 点群+GT骨格のアニメーションを作成
    
    Args:
        dataset_path: データセットパス（JSONL）
        output_path: 出力アニメーションファイルパス
        max_frames: 最大フレーム数（Noneの場合は全フレーム）
        sequence_id: 特定のシーケンスIDのみ可視化（Noneの場合は全シーケンス）
        fps: フレームレート
        feature: 色付けに使用する特徴量 ('velocity' or 'energy_power')
        bins: ヒートマップのビン数
        joints_def_path: joints_def_22.jsonのパス（Noneの場合は自動検索）
    """
    # 骨格定義を読み込み
    joints_def = load_joints_definition(joints_def_path or "data_specs/joints_def_22.json")
    bone_pairs = get_bone_pairs(joints_def)
    joint_names = joints_def.get("joint_names", [])
    
    print(f"Loaded {len(bone_pairs)} bone pairs from joints definition")
    
    # データセットを読み込み
    dataset = HeatmapDistalDataset(dataset_path)
    
    # シーケンスIDでフィルタリング
    if sequence_id is not None:
        filtered_data = []
        for i, sample in enumerate(dataset.data):
            if sample.get('sequence_id') == sequence_id:
                filtered_data.append((i, sample))
    else:
        filtered_data = [(i, dataset.data[i]) for i in range(len(dataset))]
    
    if max_frames is not None:
        filtered_data = filtered_data[:max_frames]
    
    print(f"Visualizing {len(filtered_data)} frames...")
    
    # フレームデータを準備
    frames_data = []
    for idx, sample in filtered_data:
        # GT関節
        gt_joints = None
        if 'gt_joints' in sample and sample['gt_joints'] is not None:
            gt_joints = np.array(sample['gt_joints'])  # (22, 3)
        
        # 点群パス
        file_path = sample.get('file_path', '')
        
        # 点群を読み込み
        points_curr = None
        if file_path and os.path.exists(file_path):
            try:
                points_curr, _ = load_points_from_npz(file_path, normalize=False)
            except:
                pass
        
        frames_data.append({
            'frame_idx': idx,
            'frame_num': sample.get('frame_num', idx),
            'sequence_id': sample.get('sequence_id', ''),
            'gt_joints': gt_joints,
            'points_curr': points_curr
        })
    
    if len(frames_data) == 0:
        print("No frames to visualize")
        return
    
    # 特徴量の範囲を計算（全フレームを通して）
    feature_idx = {'velocity': 3, 'energy_power': 5}[feature]
    all_features = []
    for frame in frames_data:
        if frame['points_curr'] is not None and frame['points_curr'].shape[1] > feature_idx:
            all_features.append(frame['points_curr'][:, feature_idx])
    
    if len(all_features) > 0:
        all_features = np.concatenate(all_features)
        feature_min, feature_max = np.min(all_features), np.max(all_features)
        
        # デバッグ: 特徴量の統計情報
        print(f"\n[DEBUG] Global feature ({feature}) stats:")
        print(f"  Min: {feature_min:.6e}, Max: {feature_max:.6e}")
        print(f"  Mean: {np.mean(all_features):.6e}, Std: {np.std(all_features):.6e}")
        print(f"  Non-zero values: {(all_features != 0).sum()}/{len(all_features)} ({(all_features != 0).sum()/len(all_features)*100:.1f}%)")
        
        # 値がすべて0または非常に小さい場合の警告と対処
        if feature_max - feature_min < 1e-10:
            print(f"  [WARNING] Feature range is too small! All values might be 0 or very small.")
            print(f"  [WARNING] Consider using a different feature or checking data loading.")
            # デフォルト範囲を使用（後で実際の値に基づいて更新）
            if feature_max == feature_min:
                # すべて同じ値の場合、少し範囲を広げる
                feature_min = feature_max * 0.9 if feature_max > 0 else -1.0
                feature_max = feature_max * 1.1 if feature_max > 0 else 1.0
        elif feature_max - feature_min < 1e-30:
            # 非常に小さい範囲の場合、ログスケールを検討するが、まずは線形スケールで試す
            print(f"  [INFO] Feature range is very small ({feature_max - feature_min:.2e}), using linear scale")
    else:
        feature_min, feature_max = 0.0, 1.0
        print(f"[WARNING] No feature values found, using default range [0.0, 1.0]")
    
    print(f"[DEBUG] Final feature range for colormap: [{feature_min:.6e}, {feature_max:.6e}]")
    
    # 図を作成（4列: xyヒートマップ、xzヒートマップ、yzヒートマップ、点群+GT骨格）
    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    ax_heatmap_xy = fig.add_subplot(gs[0, 0])
    ax_heatmap_xz = fig.add_subplot(gs[0, 1])
    ax_heatmap_yz = fig.add_subplot(gs[0, 2])
    ax_3d = fig.add_subplot(gs[0, 3], projection='3d')
    
    # カラーマップ（値が低いほど明るい色）
    if feature == 'velocity':
        cmap = 'viridis_r'  # 反転（値が低いほど明るい）
    else:  # energy_power
        cmap = 'plasma_r'  # 反転（値が低いほど明るい）
    
    # カラーバー用の変数
    cbar_heatmap_xy = None
    cbar_heatmap_xz = None
    cbar_heatmap_yz = None
    cbar_3d = None
    
    # 座標範囲を計算（全フレームを通して、全軸に対応）
    if x_range is not None and z_range is not None:
        # 固定範囲を使用
        x_min, x_max = x_range
        z_min, z_max = z_range
        # y範囲も計算
        all_points_y = []
        for frame in frames_data:
            if frame['points_curr'] is not None:
                all_points_y.append(frame['points_curr'][:, 1])
        if len(all_points_y) > 0:
            all_points_y = np.concatenate(all_points_y)
            y_min, y_max = np.min(all_points_y), np.max(all_points_y)
            margin = 0.2
            y_min -= margin
            y_max += margin
        else:
            y_min, y_max = -1.0, 1.0
        print(f"Using fixed ranges: x=[{x_min:.2f}, {x_max:.2f}], z=[{z_min:.2f}, {z_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    else:
        # 座標範囲を計算（全フレームを通して、全軸に対応）
        all_points = []
        all_joints = []
        for frame in frames_data:
            if frame['points_curr'] is not None:
                all_points.append(frame['points_curr'][:, :3])
            if frame['gt_joints'] is not None:
                all_joints.append(frame['gt_joints'])
        
        if len(all_points) > 0:
            all_points = np.concatenate(all_points, axis=0)
            x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
            z_min, z_max = np.min(all_points[:, 2]), np.max(all_points[:, 2])
        else:
            x_min, x_max = -1.0, 1.0
            y_min, y_max = -1.0, 1.0
            z_min, z_max = -1.0, 1.0
        
        if len(all_joints) > 0:
            all_joints = np.concatenate(all_joints, axis=0)
            x_min = min(x_min, np.min(all_joints[:, 0]))
            x_max = max(x_max, np.max(all_joints[:, 0]))
            y_min = min(y_min, np.min(all_joints[:, 1]))
            y_max = max(y_max, np.max(all_joints[:, 1]))
            z_min = min(z_min, np.min(all_joints[:, 2]))
            z_max = max(z_max, np.max(all_joints[:, 2]))
        
        # マージンを追加
        margin = 0.2
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        z_min -= margin
        z_max += margin
        print(f"Calculated ranges from data: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}], z=[{z_min:.2f}, {z_max:.2f}]")
    
    # アニメーション関数
    def animate(frame_idx):
        nonlocal cbar_heatmap_xy, cbar_heatmap_xz, cbar_heatmap_yz, cbar_3d, feature_min, feature_max
        
        frame = frames_data[frame_idx]
        
        # すべてのヒートマップをクリア
        ax_heatmap_xy.clear()
        ax_heatmap_xz.clear()
        ax_heatmap_yz.clear()
        ax_3d.clear()
        
        # ヒートマップを作成（xy, xz, yz）
        if frame['points_curr'] is not None:
            points = frame['points_curr']
            
            # デバッグ: 点群の統計情報（最初のフレームのみ）
            if frame_idx == 0:
                print(f"\n[DEBUG] Frame {frame_idx} point cloud stats:")
                print(f"  Total points: {len(points)}")
                print(f"  Points shape: {points.shape}")
                print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
                print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
                print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
            
            # 各フレームの点群の実際の範囲を計算
            # これにより、点群がヒートマップの範囲全体に広がるようになる
            points_x_min, points_x_max = points[:, 0].min(), points[:, 0].max()
            points_y_min, points_y_max = points[:, 1].min(), points[:, 1].max()
            points_z_min, points_z_max = points[:, 2].min(), points[:, 2].max()
            
            # マージンを追加（点群が境界に接しないように）
            margin = 0.05
            points_x_min -= margin
            points_x_max += margin
            points_y_min -= margin
            points_y_max += margin
            points_z_min -= margin
            points_z_max += margin
            
            # xyヒートマップ（各フレームの点群の実際の範囲を使用）
            heatmap_xy, xy_range1, xy_range2 = create_2d_heatmap(
                points,
                axis1='x', axis2='y',
                feature=feature,
                bins=bins,
                axis1_range=(points_x_min, points_x_max),
                axis2_range=(points_y_min, points_y_max)
            )
            
            # xzヒートマップ（各フレームの点群の実際の範囲を使用）
            heatmap_xz, xz_range1, xz_range2 = create_2d_heatmap(
                points,
                axis1='x', axis2='z',
                feature=feature,
                bins=bins,
                axis1_range=(points_x_min, points_x_max),
                axis2_range=(points_z_min, points_z_max)
            )
            
            # yzヒートマップ（各フレームの点群の実際の範囲を使用）
            heatmap_yz, yz_range1, yz_range2 = create_2d_heatmap(
                points,
                axis1='y', axis2='z',
                feature=feature,
                bins=bins,
                axis1_range=(points_y_min, points_y_max),
                axis2_range=(points_z_min, points_z_max)
            )
            
            # 各ヒートマップの統計情報を取得（最初のフレームのみ）
            if frame_idx == 0:
                for name, heatmap in [('XY', heatmap_xy), ('XZ', heatmap_xz), ('YZ', heatmap_yz)]:
                    valid_heatmap = heatmap[~np.isnan(heatmap)]
                    if len(valid_heatmap) > 0:
                        print(f"  [DEBUG] {name} Heatmap stats: min={valid_heatmap.min():.6e}, max={valid_heatmap.max():.6e}, mean={valid_heatmap.mean():.6e}, num_valid={len(valid_heatmap)}/{heatmap.size}")
            
            # 各ヒートマップの表示範囲を決定（各フレームの点群の実際の範囲を使用）
            heatmaps_to_show = [
                ('xy', heatmap_xy, xy_range1, xy_range2, ax_heatmap_xy, cbar_heatmap_xy),
                ('xz', heatmap_xz, xz_range1, xz_range2, ax_heatmap_xz, cbar_heatmap_xz),
                ('yz', heatmap_yz, yz_range1, yz_range2, ax_heatmap_yz, cbar_heatmap_yz)
            ]
            
            for name, heatmap, axis1_range, axis2_range, ax, cbar_ref in heatmaps_to_show:
                valid_heatmap = heatmap[~np.isnan(heatmap)]
                if len(valid_heatmap) > 0:
                    heatmap_min = valid_heatmap.min()
                    heatmap_max = valid_heatmap.max()
                    
                    # デバッグ出力（最初のフレームのみ）
                    if frame_idx == 0:
                        print(f"  [DEBUG] {name} Heatmap value range: [{heatmap_min:.6f}, {heatmap_max:.6f}]")
                        print(f"  [DEBUG] {name} Heatmap percentile (1%, 50%, 99%): [{np.percentile(valid_heatmap, 1):.6f}, {np.percentile(valid_heatmap, 50):.6f}, {np.percentile(valid_heatmap, 99):.6f}]")
                    
                    # **重要**: ヒートマップの値を0-1の範囲に正規化してから表示
                    # これにより、値の範囲に関わらず色の変化が見えるようになる
                    range_size = heatmap_max - heatmap_min
                    if range_size > 1e-10:
                        # 正規化: (value - min) / (max - min)
                        heatmap_normalized = heatmap.copy()
                        heatmap_normalized[~np.isnan(heatmap)] = (heatmap[~np.isnan(heatmap)] - heatmap_min) / range_size
                        heatmap_to_show = heatmap_normalized
                        vmin_actual = 0.0
                        vmax_actual = 1.0
                        
                        if frame_idx == 0:
                            print(f"  [DEBUG] {name} Normalized heatmap range: [0.0, 1.0] (original: [{heatmap_min:.6f}, {heatmap_max:.6f}])")
                            # 追加デバッグ: 正規化後の統計情報
                            valid_normalized = heatmap_normalized[~np.isnan(heatmap_normalized)]
                            if len(valid_normalized) > 0:
                                print(f"  [DEBUG] {name} Normalized stats: min={valid_normalized.min():.6f}, max={valid_normalized.max():.6f}, mean={valid_normalized.mean():.6f}, num_valid={len(valid_normalized)}/{heatmap_normalized.size}")
                    else:
                        # 範囲が非常に小さい場合、すべて0.5に設定
                        heatmap_to_show = heatmap.copy()
                        heatmap_to_show[~np.isnan(heatmap)] = 0.5
                        vmin_actual = 0.0
                        vmax_actual = 1.0
                        
                        if frame_idx == 0:
                            print(f"  [WARNING] {name} Heatmap range is too small ({range_size:.6e}), using constant value 0.5")
                else:
                    vmin_actual = 0.0
                    vmax_actual = 1.0
                    heatmap_to_show = heatmap
                    if frame_idx == 0:
                        print(f"  [WARNING] {name} Heatmap: All values are NaN!")
                
                # 追加デバッグ: 表示前のheatmap_to_showの統計情報
                if frame_idx == 0:
                    valid_to_show = heatmap_to_show[~np.isnan(heatmap_to_show)]
                    if len(valid_to_show) > 0:
                        print(f"  [DEBUG] {name} heatmap_to_show stats: min={valid_to_show.min():.6f}, max={valid_to_show.max():.6f}, mean={valid_to_show.mean():.6f}, num_valid={len(valid_to_show)}/{heatmap_to_show.size}")
                    else:
                        print(f"  [ERROR] {name} heatmap_to_show: All values are NaN!")
                
                # ヒートマップを描画
                # 注意: imshowのextentは[x_min, x_max, y_min, y_max]の順序
                # heatmapは転置後(axis1_bins, axis2_bins) = (x_bins, z_bins)の形状
                # imshowは最初の次元をy軸（縦軸）、2番目の次元をx軸（横軸）として解釈する
                # 転置により、extent=[axis1_range, axis2_range] = [x_min, x_max, z_min, z_max]で正しく表示できる
                if frame_idx == 0:
                    print(f"  [DEBUG] {name} imshow extent: [{axis1_range[0]:.3f}, {axis1_range[1]:.3f}, {axis2_range[0]:.3f}, {axis2_range[1]:.3f}]")
                    print(f"  [DEBUG] {name} heatmap_to_show shape: {heatmap_to_show.shape}")
                    print(f"  [DEBUG] {name} vmin={vmin_actual:.6f}, vmax={vmax_actual:.6f}")
                
                # NaNをマスクして黒色で表示する
                # 補間を'nearest'にすることで、NaNが周囲に広がるのを防ぐ
                H = np.ma.masked_invalid(heatmap_to_show)  # NaNをマスク
                
                # カラーマップを設定（NaN=黒）
                cmap_obj = plt.cm.get_cmap(cmap)
                try:
                    # Matplotlib >= 3.5
                    cmap_obj = cmap_obj.with_extremes(bad='black')
                except AttributeError:
                    # Matplotlib < 3.5
                    cmap_obj = cmap_obj.copy()
                    cmap_obj.set_bad('black')
                
                im = ax.imshow(
                    H,
                    extent=[axis1_range[0], axis1_range[1], axis2_range[0], axis2_range[1]],
                    origin='lower',
                    aspect='auto',
                    cmap=cmap_obj,
                    interpolation='nearest',  # 補間を切ることでNaNが広がらない
                    vmin=vmin_actual,
                    vmax=vmax_actual,
                )
                
                ax.set_xlabel(f'{name[0].upper()} (m)')
                ax.set_ylabel(f'{name[1].upper()} (m)')
                ax.set_title(f'{name.upper()} Heatmap ({feature})')
                
                # カラーバーを追加（最初のフレームのみ）
                if frame_idx == 0 and cbar_ref is None:
                    if name == 'xy':
                        cbar_heatmap_xy = plt.colorbar(im, ax=ax, label=feature)
                    elif name == 'xz':
                        cbar_heatmap_xz = plt.colorbar(im, ax=ax, label=feature)
                    elif name == 'yz':
                        cbar_heatmap_yz = plt.colorbar(im, ax=ax, label=feature)
        
        # 点群+GT骨格を描画
        if frame['points_curr'] is not None:
            points = frame['points_curr']
            coords = points[:, :3]  # (N, 3)
            features = points[:, feature_idx]  # (N,)
            
            # 値が非常に小さい場合、正規化して表示
            if feature_max < 1e-30:
                # 特徴量を0-1の範囲に正規化
                range_size = feature_max - feature_min
                if range_size > 0:
                    features_normalized = (features - feature_min) / range_size
                else:
                    features_normalized = np.ones_like(features) * 0.5
                features_to_show = features_normalized
                vmin_3d = 0.0
                vmax_3d = 1.0
            else:
                features_to_show = features
                vmin_3d = feature_min
                vmax_3d = feature_max
            
            # xz平面に投影（y座標を0に設定）
            coords_xz = coords.copy()
            coords_xz[:, 1] = 0.0  # y座標を0に
            
            scatter = ax_3d.scatter(
                coords_xz[:, 0], coords_xz[:, 1], coords_xz[:, 2],
                c=features_to_show,
                cmap=cmap,
                s=50,
                alpha=0.7,
                vmin=vmin_3d,
                vmax=vmax_3d
            )
            
            # カラーバーを追加（最初のフレームのみ）
            if frame_idx == 0 and cbar_3d is None:
                cbar_3d = plt.colorbar(scatter, ax=ax_3d, label=feature)
        
        # GT骨格を描画
        if frame['gt_joints'] is not None:
            joints = frame['gt_joints']
            # xz平面に投影（y座標を0に設定）
            joints_xz = joints.copy()
            joints_xz[:, 1] = 0.0  # y座標を0に
            
            ax_3d.scatter(
                joints_xz[:, 0], joints_xz[:, 1], joints_xz[:, 2],
                c='red', marker='o', s=150, alpha=0.9, label='GT Joints',
                edgecolors='darkred', linewidths=1.5
            )
            
            # 骨格線を描画
            for parent, child in bone_pairs:
                if parent < len(joints) and child < len(joints):
                    ax_3d.plot(
                        [joints_xz[parent, 0], joints_xz[child, 0]],
                        [joints_xz[parent, 1], joints_xz[child, 1]],
                        [joints_xz[parent, 2], joints_xz[child, 2]],
                        'r-', linewidth=3, alpha=0.9
                    )
        
        # 3Dプロットの設定
        ax_3d.set_xlim(x_min, x_max)
        ax_3d.set_ylim(-0.5, 0.5)  # y座標は固定
        ax_3d.set_zlim(z_min, z_max)
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        seq_id = frame['sequence_id'] if frame['sequence_id'] else 'N/A'
        ax_3d.set_title(f'Point Cloud ({feature}) + GT Skeleton (XZ plane)\nSeq: {seq_id}, Frame {frame["frame_num"]}/{len(frames_data)-1}')
        ax_3d.view_init(elev=0, azim=270)  # xz平面から見る（z軸周りに180度回転して正面向き）
        
        plt.tight_layout()
    
    # アニメーションを作成
    anim = FuncAnimation(
        fig, animate, frames=len(frames_data),
        interval=1000/fps, repeat=True, blit=False
    )
    
    # 保存
    print(f"Saving animation to {output_path}...")
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"✓ Animation saved: {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Trying with ffmpeg writer...")
        try:
            anim.save(output_path.replace('.gif', '.mp4'), writer='ffmpeg', fps=fps)
            print(f"✓ Animation saved as MP4: {output_path.replace('.gif', '.mp4')}")
        except Exception as e2:
            print(f"Error saving with ffmpeg: {e2}")
            print("Please install pillow or ffmpeg to save animations")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize heatmap + point cloud + GT skeleton')
    parser.add_argument('--mode', type=str, default='xz',
                       choices=['xz', 'xy'],
                       help='Visualization mode: xz (side view) or xy (top view)')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset (JSONL)')
    parser.add_argument('--output_path', type=str, default='outputs/animation.gif',
                       help='Output animation file path')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to visualize')
    parser.add_argument('--sequence_id', type=str, default=None,
                       help='Specific sequence ID to visualize (None for all)')
    parser.add_argument('--fps', type=int, default=15,
                       help='Frame rate for animation')
    parser.add_argument('--feature', type=str, default='velocity',
                       choices=['velocity', 'energy_power'],
                       help='Feature to use for coloring (velocity or energy_power)')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of bins for heatmap (only for xz mode)')
    parser.add_argument('--joints_def', type=str, default=None,
                       help='Path to joints_def_22.json (None for auto-search)')
    parser.add_argument('--x_range', type=float, nargs=2, default=None,
                       metavar=('X_MIN', 'X_MAX'),
                       help='Fixed x-axis range (e.g., --x_range -0.5 0.5). If not specified, range is calculated from data.')
    parser.add_argument('--y_range', type=float, nargs=2, default=None,
                       metavar=('Y_MIN', 'Y_MAX'),
                       help='Fixed y-axis range (for xy mode, e.g., --y_range -1.0 1.0). If not specified, range is calculated from data.')
    parser.add_argument('--z_range', type=float, nargs=2, default=None,
                       metavar=('Z_MIN', 'Z_MAX'),
                       help='Fixed z-axis range (for xz mode, e.g., --z_range -1.0 0.75). If not specified, range is calculated from data.')
    parser.add_argument('--point_size', type=float, default=20.0,
                       help='Point cloud size (for xy mode)')
    parser.add_argument('--joint_size', type=float, default=100.0,
                       help='Joint size (for xy mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'xy':
        # xy平面アニメーション
        x_range = tuple(args.x_range) if args.x_range is not None else None
        y_range = tuple(args.y_range) if args.y_range is not None else None
        
        create_xy_plane_animation(
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            max_frames=args.max_frames,
            sequence_id=args.sequence_id,
            fps=args.fps,
            feature=args.feature,
            joints_def_path=args.joints_def,
            x_range=x_range,
            y_range=y_range,
            point_size=args.point_size,
            joint_size=args.joint_size
        )
    else:
        # xz平面アニメーション（既存）
        x_range = tuple(args.x_range) if args.x_range is not None else None
        z_range = tuple(args.z_range) if args.z_range is not None else None
        
        create_xz_heatmap_animation(
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            max_frames=args.max_frames,
            sequence_id=args.sequence_id,
            fps=args.fps,
            feature=args.feature,
            bins=args.bins,
            joints_def_path=args.joints_def,
            x_range=x_range,
            z_range=z_range
        )


if __name__ == '__main__':
    main()

