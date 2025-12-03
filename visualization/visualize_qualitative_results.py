#!/usr/bin/env python3
"""
Figure 8: 定性結果（Qualitative Results）の可視化

GT骨格、提案手法、Direct Regression、Motion-Filteredモデルの予測を比較
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import sys
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_joints_def(json_path: str) -> dict:
    """joints_def_22.jsonを読み込む"""
    # パスが存在しない場合、相対パスを試す
    if not os.path.exists(json_path):
        # スクリプトの場所から相対パスを計算
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir / json_path,
            script_dir.parent.parent / 'data_specs' / 'joints_def_22.json',
            script_dir / '..' / '..' / 'data_specs' / 'joints_def_22.json',
            Path(json_path) if json_path != 'data_specs/joints_def_22.json' else None
        ]
        
        for path in possible_paths:
            if path and path.exists():
                json_path = str(path)
                break
        else:
            raise FileNotFoundError(f"Could not find joints_def_22.json. Tried: {json_path} and relative paths")
    
    with open(json_path, 'r') as f:
        return json.load(f)


def load_evaluation_results(json_path: str) -> dict:
    """評価結果を読み込む"""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_skeleton_connections(joints_def: dict) -> List[Tuple[int, int]]:
    """骨格の接続関係を取得"""
    connections = []
    if 'skeleton' in joints_def:
        for connection in joints_def['skeleton']:
            connections.append((connection[0], connection[1]))
    return connections


def visualize_qualitative_comparison(
    pred_ours: np.ndarray,  # (N, 6, 3) - 提案手法の予測
    pred_direct: Optional[np.ndarray] = None,  # (N, 6, 3) - Direct Regressionの予測
    pred_motion: Optional[np.ndarray] = None,  # (N, 6, 3) - Motion-Filteredの予測
    gt_full: Optional[np.ndarray] = None,  # (N, 22, 3) - 全骨格のGT
    gt_arm: Optional[np.ndarray] = None,  # (N, 6, 3) - 腕関節のGT
    joints_def_path: Optional[str] = None,
    num_frames: int = 10,
    output_dir: Path = Path("outputs/paper_figures"),
    frame_indices: Optional[List[int]] = None
):
    """
    定性結果を可視化
    
    Args:
        pred_ours: 提案手法の予測
        pred_direct: Direct Regressionの予測（オプション）
        pred_motion: Motion-Filteredモデルの予測（オプション）
        gt_full: 全骨格のGT（オプション）
        gt_arm: 腕関節のGT
        joints_def_path: joints_def_22.jsonのパス
        num_frames: 可視化するフレーム数
        output_dir: 出力ディレクトリ
        frame_indices: 可視化するフレームのインデックス（指定された場合）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 関節定義を読み込む
    # デフォルトパスを自動検出
    if joints_def_path is None:
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir.parent.parent / 'data_specs' / 'joints_def_22.json',
            script_dir / '..' / '..' / 'data_specs' / 'joints_def_22.json',
            Path('data_specs/joints_def_22.json'),
            Path('mmwave_pose_reconstraction_pj/data_specs/joints_def_22.json'),
        ]
        for path in possible_paths:
            if path.exists():
                joints_def_path = str(path.resolve())
                break
        else:
            joints_def_path = 'data_specs/joints_def_22.json'  # フォールバック
    
    joints_def = load_joints_def(joints_def_path)
    joint_names = joints_def.get('joint_names', [])
    
    # 腕関節のインデックス
    arm_joint_indices = [16, 18, 20, 17, 19, 21]  # L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist
    arm_joint_names = ['L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']
    
    # フレームインデックスを決定
    if frame_indices is None:
        total_frames = len(pred_ours)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    # 各フレームを可視化
    for frame_idx in tqdm(frame_indices, desc="Creating qualitative visualizations"):
        if frame_idx >= len(pred_ours):
            continue
        
        fig = plt.figure(figsize=(16, 12))
        
        # 3つのビュー（3D、正面、側面）
        from mpl_toolkits.mplot3d import Axes3D
        
        # 3D表示
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
                
        # GT骨格（全骨格）
        if gt_full is not None and frame_idx < len(gt_full):
            gt_full_frame = gt_full[frame_idx]
            valid_gt = ~np.isnan(gt_full_frame).any(axis=1)
            if valid_gt.sum() > 0:
                ax1.scatter(gt_full_frame[valid_gt, 0], 
                          gt_full_frame[valid_gt, 1], 
                          gt_full_frame[valid_gt, 2],
                          c='black', s=50, marker='o', label='GT (All)', alpha=0.6)
        
        # GT骨格（腕関節のみ）
        if gt_arm is not None and frame_idx < len(gt_arm):
            gt_arm_frame = gt_arm[frame_idx]
            valid_gt_arm = ~np.isnan(gt_arm_frame).any(axis=1)
            if valid_gt_arm.sum() > 0:
                ax1.scatter(gt_arm_frame[valid_gt_arm, 0], 
                          gt_arm_frame[valid_gt_arm, 1], 
                          gt_arm_frame[valid_gt_arm, 2],
                          c='black', s=100, marker='o', label='GT (Arm)', edgecolors='white', linewidths=1)
        
        # 提案手法の予測
        pred_ours_frame = pred_ours[frame_idx]
        valid_pred = ~np.isnan(pred_ours_frame).any(axis=1)
        if valid_pred.sum() > 0:
            ax1.scatter(pred_ours_frame[valid_pred, 0], 
                      pred_ours_frame[valid_pred, 1], 
                      pred_ours_frame[valid_pred, 2],
                      c='red', s=100, marker='^', label='Ours', edgecolors='white', linewidths=1)
        
        # Direct Regressionの予測
        if pred_direct is not None and frame_idx < len(pred_direct):
            pred_direct_frame = pred_direct[frame_idx]
            valid_direct = ~np.isnan(pred_direct_frame).any(axis=1)
            if valid_direct.sum() > 0:
                ax1.scatter(pred_direct_frame[valid_direct, 0], 
                          pred_direct_frame[valid_direct, 1], 
                          pred_direct_frame[valid_direct, 2],
                          c='blue', s=80, marker='s', label='Direct Regression', alpha=0.7)
        
        # Motion-Filteredの予測
        if pred_motion is not None and frame_idx < len(pred_motion):
            pred_motion_frame = pred_motion[frame_idx]
            valid_motion = ~np.isnan(pred_motion_frame).any(axis=1)
            if valid_motion.sum() > 0:
                ax1.scatter(pred_motion_frame[valid_motion, 0], 
                          pred_motion_frame[valid_motion, 1], 
                          pred_motion_frame[valid_motion, 2],
                          c='purple', s=80, marker='d', label='Motion-Filtered', alpha=0.7)
        
        ax1.set_xlabel('X (m)', fontsize=10)
        ax1.set_ylabel('Y (m)', fontsize=10)
        ax1.set_zlabel('Z (m)', fontsize=10)
        ax1.set_title(f'Frame {frame_idx}: 3D View', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        
        # 2D表示（正面、側面）
        views = [
            {'title': 'Front View (XY)', 'axes': [0, 1], 'labels': ['X (m)', 'Y (m)']},
            {'title': 'Side View (XZ)', 'axes': [0, 2], 'labels': ['X (m)', 'Z (m)']}
        ]
        
        for view_idx, view in enumerate(views):
            ax = fig.add_subplot(2, 2, view_idx + 2)
            ax_idx, ax_idy = view['axes']
            
            # GT骨格
            if gt_arm is not None and frame_idx < len(gt_arm):
                gt_arm_frame = gt_arm[frame_idx]
                valid_gt_arm = ~np.isnan(gt_arm_frame).any(axis=1)
                if valid_gt_arm.sum() > 0:
                    ax.scatter(gt_arm_frame[valid_gt_arm, ax_idx], 
                              gt_arm_frame[valid_gt_arm, ax_idy],
                              c='black', s=100, marker='o', label='GT', edgecolors='white', linewidths=1)
            
            # 提案手法の予測
            pred_ours_frame = pred_ours[frame_idx]
            valid_pred = ~np.isnan(pred_ours_frame).any(axis=1)
            if valid_pred.sum() > 0:
                ax.scatter(pred_ours_frame[valid_pred, ax_idx], 
                          pred_ours_frame[valid_pred, ax_idy],
                          c='red', s=100, marker='^', label='Ours', edgecolors='white', linewidths=1)
            
            # Direct Regressionの予測
            if pred_direct is not None and frame_idx < len(pred_direct):
                pred_direct_frame = pred_direct[frame_idx]
                valid_direct = ~np.isnan(pred_direct_frame).any(axis=1)
                if valid_direct.sum() > 0:
                    ax.scatter(pred_direct_frame[valid_direct, ax_idx], 
                              pred_direct_frame[valid_direct, ax_idy],
                              c='blue', s=80, marker='s', label='Direct Regression', alpha=0.7)
            
            ax.set_xlabel(view['labels'][0], fontsize=10)
            ax.set_ylabel(view['labels'][1], fontsize=10)
            ax.set_title(f'Frame {frame_idx}: {view["title"]}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_aspect('equal')
        
        plt.suptitle(f'Qualitative Results: Frame {frame_idx}', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_path = output_dir / f'figure_8_qualitative_frame_{frame_idx}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created qualitative visualization for frame {frame_idx}: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize qualitative comparison results')
    parser.add_argument('--ours_result', type=str, required=True,
                       help='Path to our method evaluation result JSON')
    parser.add_argument('--direct_result', type=str, default=None,
                       help='Path to Direct Regression evaluation result JSON (optional)')
    parser.add_argument('--motion_result', type=str, default=None,
                       help='Path to Motion-Filtered evaluation result JSON (optional)')
    # デフォルトパスを自動検出
    script_dir = Path(__file__).parent
    default_joints_def = None
    possible_paths = [
        script_dir.parent.parent / 'data_specs' / 'joints_def_22.json',
        script_dir / '..' / '..' / 'data_specs' / 'joints_def_22.json',
        Path('data_specs/joints_def_22.json'),
        Path('mmwave_pose_reconstraction_pj/data_specs/joints_def_22.json'),
    ]
    for path in possible_paths:
        if path.exists():
            default_joints_def = str(path.resolve())
            break
    
    if default_joints_def is None:
        default_joints_def = 'data_specs/joints_def_22.json'  # フォールバック
    
    parser.add_argument('--joints_def', type=str, default=default_joints_def,
                       help='Path to joints_def_22.json')
    parser.add_argument('--num_frames', type=int, default=10,
                       help='Number of frames to visualize')
    parser.add_argument('--frame_indices', type=int, nargs='+', default=None,
                       help='Specific frame indices to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/paper_figures',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 評価結果を読み込む
    print(f"Loading our method results from {args.ours_result}...")
    ours_data = load_evaluation_results(args.ours_result)
    pred_ours = np.array(ours_data.get('pred_3d', []))
    gt_arm = np.array(ours_data.get('gt_3d', []))
    gt_full = np.array(ours_data.get('gt_full_3d', [])) if 'gt_full_3d' in ours_data else None
    
    pred_direct = None
    if args.direct_result:
        print(f"Loading Direct Regression results from {args.direct_result}...")
        direct_data = load_evaluation_results(args.direct_result)
        pred_direct = np.array(direct_data.get('pred_3d', []))
    
    pred_motion = None
    if args.motion_result:
        print(f"Loading Motion-Filtered results from {args.motion_result}...")
        motion_data = load_evaluation_results(args.motion_result)
        pred_motion = np.array(motion_data.get('pred_3d', []))
    
    print(f"Loaded {len(pred_ours)} samples")
    
    # 可視化
    output_dir = Path(args.output_dir)
    visualize_qualitative_comparison(
        pred_ours=pred_ours,
        pred_direct=pred_direct,
        pred_motion=pred_motion,
        gt_full=gt_full,
        gt_arm=gt_arm,
        joints_def_path=args.joints_def,
        num_frames=args.num_frames,
        output_dir=output_dir,
        frame_indices=args.frame_indices
    )
    
    print("\n✓ Qualitative visualization completed!")


if __name__ == '__main__':
    main()

