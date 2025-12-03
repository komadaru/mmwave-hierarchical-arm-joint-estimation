#!/usr/bin/env python3
"""
点群（特徴量で色付け）+ GT骨格 vs ヒートマップのアニメーション可視化
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
    
    Args:
        joints_def_path: joints_def_22.jsonのパス
    
    Returns:
        joints_def: 関節定義の辞書
    """
    # 複数のパスを試す
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
    
    # デフォルトの骨格接続（joints_def_22.jsonのbonesフィールドから）
    print("Warning: joints_def_22.json not found, using default bone pairs")
    return {
        "bones": [[0,3],[3,6],[6,9],[9,12],[12,15],[0,1],[0,2],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[9,13],[13,16],[16,18],[18,20],[9,14],[14,17],[17,19],[19,21]],
        "joint_names": ["Pelvis","L_Hip","R_Hip","Spine1","L_Knee","R_Knee","Spine2","L_Ankle","R_Ankle","Spine3","L_Foot","R_Foot","Neck","L_Collar","R_Collar","Head","L_Shoulder","R_Shoulder","L_Elbow","R_Elbow","L_Wrist","R_Wrist"]
    }


def get_bone_pairs(joints_def: Dict) -> List[Tuple[int, int]]:
    """
    骨格接続のリストを取得
    
    Args:
        joints_def: 関節定義の辞書
    
    Returns:
        bone_pairs: 骨格接続のリスト [(parent, child), ...]
    """
    bones = joints_def.get("bones", [])
    return [tuple(bone) for bone in bones]


def create_comparison_animation(
    dataset_path: str,
    output_path: str,
    max_frames: Optional[int] = None,
    sequence_id: Optional[str] = None,
    fps: int = 15,
    feature_to_show: str = 'energy_power',  # 'velocity', 'amplitude', 'energy_power'
    joints_def_path: Optional[str] = None
):
    """
    点群（特徴量で色付け）+ GT骨格 vs ヒートマップのアニメーションを作成
    
    Args:
        dataset_path: データセットパス（JSONL）
        output_path: 出力アニメーションファイルパス
        max_frames: 最大フレーム数（Noneの場合は全フレーム）
        sequence_id: 特定のシーケンスIDのみ可視化（Noneの場合は全シーケンス）
        fps: フレームレート
        feature_to_show: 点群の色付けに使用する特徴量
        joints_def_path: joints_def_22.jsonのパス（Noneの場合は自動検索）
    """
    # 骨格定義を読み込み
    joints_def = load_joints_definition(joints_def_path or "data_specs/joints_def_22.json")
    bone_pairs = get_bone_pairs(joints_def)
    joint_names = joints_def.get("joint_names", [])
    
    print(f"Loaded {len(bone_pairs)} bone pairs from joints definition")
    print(f"Joint names: {joint_names}")
    
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
        # メタデータから情報を取得
        metadata = sample.get('metadata', {}) if isinstance(sample.get('metadata'), dict) else {}
        
        # GT関節
        gt_joints = None
        if 'gt_joints' in sample and sample['gt_joints'] is not None:
            gt_joints = np.array(sample['gt_joints'])  # (22, 3)
        
        # 点群パス
        file_path = sample.get('file_path', '')
        prev_file_path = sample.get('prev_file_path', '')
        
        # 点群を読み込み
        points_curr = None
        points_prev = None
        
        if file_path and os.path.exists(file_path):
            try:
                points_curr, _ = load_points_from_npz(file_path, normalize=False)
            except:
                pass
        
        if prev_file_path and os.path.exists(prev_file_path):
            try:
                points_prev, _ = load_points_from_npz(prev_file_path, normalize=False)
            except:
                pass
        
        # ヒートマップを取得（多軸モードの場合）
        heatmaps = {}
        use_multi_axis = sample.get('use_multi_axis', False)
        
        if use_multi_axis:
            # 6つのヒートマップを取得（現在フレーム）
            if 'heatmaps_curr' in sample:
                heatmaps_curr_dict = sample['heatmaps_curr']
                for axis in ['x', 'y', 'z']:
                    for feature in ['energy_power', 'velocity']:
                        key = f'{axis}_{feature}'
                        if key in heatmaps_curr_dict:
                            heatmaps[key] = np.array(heatmaps_curr_dict[key])  # (50, 50)
        else:
            # レガシーモード（単一ヒートマップ）
            if 'heatmap_curr' in sample:
                heatmaps['single'] = np.array(sample['heatmap_curr'])
            elif 'heatmap' in sample:
                heatmaps['single'] = np.array(sample['heatmap'])
        
        frames_data.append({
            'frame_idx': idx,
            'frame_num': sample.get('frame_num', idx),
            'sequence_id': sample.get('sequence_id', ''),
            'gt_joints': gt_joints,
            'points_curr': points_curr,
            'points_prev': points_prev,
            'heatmaps': heatmaps,
            'metadata': metadata
        })
    
    if len(frames_data) == 0:
        print("No frames to visualize")
        return
    
    # 座標範囲を計算
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
    
    # 特徴量の範囲を計算
    feature_idx = {'velocity': 3, 'amplitude': 4, 'energy_power': 5}[feature_to_show]
    all_features = []
    for frame in frames_data:
        if frame['points_curr'] is not None and frame['points_curr'].shape[1] > feature_idx:
            all_features.append(frame['points_curr'][:, feature_idx])
    
    if len(all_features) > 0:
        all_features = np.concatenate(all_features)
        feature_min, feature_max = np.min(all_features), np.max(all_features)
    else:
        feature_min, feature_max = 0.0, 1.0
    
    # 図を作成
    use_multi_axis = frames_data[0]['heatmaps'] and 'x_energy_power' in frames_data[0]['heatmaps']
    
    if use_multi_axis:
        # 多軸モード: 点群+GT骨格（3D） + 6つのヒートマップ
        # 3Dプロットを大きくするため、figsizeを大きくし、GridSpecでレイアウトを調整
        fig = plt.figure(figsize=(24, 14))  # (20, 12)から(24, 14)に拡大
        
        # GridSpecを使用してレイアウトを定義（3行×4列、左側2列を3Dプロットに使用）
        # width_ratios: 左側2列を大きく、右側2列を小さく
        gs = GridSpec(3, 4, figure=fig, width_ratios=[2, 2, 1, 1], height_ratios=[1, 1, 1], 
                      hspace=0.3, wspace=0.3)
        
        # 点群+GT骨格（3D）- 左側の2列×3行分を使用（より大きな領域）
        ax_3d = fig.add_subplot(gs[:, 0:2], projection='3d')
        
        # 6つのヒートマップ - 右側の2列×3行分を使用
        axes_heatmaps = []
        for i, axis in enumerate(['x', 'y', 'z']):
            for j, feature in enumerate(['energy_power', 'velocity']):
                row = i
                col = 2 + j
                ax = fig.add_subplot(gs[row, col])
                axes_heatmaps.append((ax, f'{axis}_{feature}'))
    else:
        # レガシーモード: 点群+GT骨格（3D） + 1つのヒートマップ
        fig = plt.figure(figsize=(20, 10))  # (16, 8)から(20, 10)に拡大
        
        # 3Dプロットをより大きく（左側70%、右側30%）
        ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax_heatmap = fig.add_subplot(1, 2, 2)
        axes_heatmaps = [(ax_heatmap, 'single')]
        
        # 3Dプロットの位置とサイズを調整
        pos_3d = ax_3d.get_position()
        ax_3d.set_position([pos_3d.x0, pos_3d.y0, pos_3d.width * 1.2, pos_3d.height * 1.2])
    
    # カラーバー用の変数
    cbar_3d = None
    cbars_heatmap = [None] * len(axes_heatmaps)
    
    # アニメーション関数
    def animate(frame_idx):
        nonlocal cbar_3d, cbars_heatmap
        
        frame = frames_data[frame_idx]
        
        # 3Dプロットをクリア
        ax_3d.clear()
        
        # 点群を描画
        if frame['points_curr'] is not None:
            points = frame['points_curr']
            coords = points[:, :3]  # (N, 3)
            features = points[:, feature_idx]  # (N,)
            
            scatter = ax_3d.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                c=features,
                cmap='viridis',
                s=50,  # 10から50に変更（5倍大きく）
                alpha=0.7,  # 0.6から0.7に変更（少し濃く）
                vmin=feature_min,
                vmax=feature_max
            )
            
            # カラーバーを追加（最初のフレームのみ）
            if frame_idx == 0 and cbar_3d is None:
                cbar_3d = plt.colorbar(scatter, ax=ax_3d, label=feature_to_show)
        
        # GT骨格を描画
        if frame['gt_joints'] is not None:
            joints = frame['gt_joints']
            ax_3d.scatter(
                joints[:, 0], joints[:, 1], joints[:, 2],
                c='red', marker='o', s=150, alpha=0.9, label='GT Joints',  # s=50から150に変更（3倍大きく）
                edgecolors='darkred', linewidths=1.5  # エッジを追加して見やすく
            )
            
            # 骨格線を描画
            for parent, child in bone_pairs:
                if parent < len(joints) and child < len(joints):
                    ax_3d.plot(
                        [joints[parent, 0], joints[child, 0]],
                        [joints[parent, 1], joints[child, 1]],
                        [joints[parent, 2], joints[child, 2]],
                        'r-', linewidth=3, alpha=0.9  # linewidth=2から3に変更（太く）
                    )
        
        # 3Dプロットの設定
        ax_3d.set_xlim(x_min, x_max)
        ax_3d.set_ylim(y_min, y_max)
        ax_3d.set_zlim(z_min, z_max)
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        seq_id = frame['sequence_id'] if frame['sequence_id'] else 'N/A'
        ax_3d.set_title(f'Point Cloud ({feature_to_show}) + GT Skeleton (XZ plane view)\nSeq: {seq_id}, Frame {frame["frame_num"]}/{len(frames_data)-1}')
        ax_3d.view_init(elev=0, azim=90)  # xz平面から見る（elev=0: 水平、azim=90: y軸方向から見る）
        
        # ヒートマップを描画
        for ax_idx, (ax, key) in enumerate(axes_heatmaps):
            ax.clear()
            
            if key in frame['heatmaps']:
                heatmap = frame['heatmaps'][key]
                im = ax.imshow(
                    heatmap.T,  # 転置して正しい向きに
                    aspect='auto',
                    origin='lower',
                    cmap='hot',
                    interpolation='nearest'
                )
                ax.set_title(f'Heatmap: {key}')
                ax.set_xlabel('Spatial Axis (bins)')
                ax.set_ylabel('Feature Axis (bins)')
                
                # カラーバーを追加（最初のフレームのみ）
                if frame_idx == 0 and cbars_heatmap[ax_idx] is None:
                    cbars_heatmap[ax_idx] = plt.colorbar(im, ax=ax, label='Point Count')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Heatmap: {key} (N/A)')
        
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
    parser = argparse.ArgumentParser(description='Visualize point cloud + GT skeleton vs heatmaps')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset (JSONL)')
    parser.add_argument('--output_path', type=str, default='outputs/heatmap_comparison_animation.gif',
                       help='Output animation file path')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to visualize')
    parser.add_argument('--sequence_id', type=str, default=None,
                       help='Specific sequence ID to visualize (None for all)')
    parser.add_argument('--fps', type=int, default=15,
                       help='Frame rate for animation')
    parser.add_argument('--feature', type=str, default='energy_power',
                       choices=['velocity', 'amplitude', 'energy_power'],
                       help='Feature to use for point cloud coloring')
    parser.add_argument('--joints_def', type=str, default=None,
                       help='Path to joints_def_22.json (None for auto-search)')
    
    args = parser.parse_args()
    
    create_comparison_animation(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        max_frames=args.max_frames,
        sequence_id=args.sequence_id,
        fps=args.fps,
        feature_to_show=args.feature,
        joints_def_path=args.joints_def
    )


if __name__ == '__main__':
    main()

