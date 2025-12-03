#!/usr/bin/env python3
"""
wristとelbowの予測vsGT骨格アニメーションを作成するスクリプト

予測については、wristとelbowのみを使用し、それを赤色でプロット
それ以外のキーポイントはGTのものを使用し、黒色でプロット
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 関節インデックス（joints_def_22.jsonから）
# L_Shoulder (16), L_Elbow (18), L_Wrist (20)
# R_Shoulder (17), R_Elbow (19), R_Wrist (21)
ARM_JOINT_INDICES = [16, 18, 20, 17, 19, 21]
ARM_JOINT_NAMES = ['L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']

# wristとelbowのインデックス（ARM_JOINT_INDICES内でのインデックス）
WRIST_ELBOW_INDICES = [1, 2, 4, 5]  # L_Elbow, L_Wrist, R_Elbow, R_Wrist


def load_joints_def(json_path: str) -> dict:
    """joints_def_22.jsonを読み込む"""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_evaluation_results(json_path: str) -> dict:
    """evaluate_arm_joint_3d.pyの結果を読み込む"""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_full_skeleton_gt(gt_arm_joints: np.ndarray, gt_full_joints: np.ndarray = None) -> np.ndarray:
    """
    腕の関節のみのGTから、全骨格のGTを構築
    
    Args:
        gt_arm_joints: (6, 3) - 腕の関節のみのGT座標
        gt_full_joints: (22, 3) - 全関節のGT座標（オプション、利用可能な場合）
    
    Returns:
        full_skeleton: (22, 3) - 全骨格のGT座標
    """
    full_skeleton = np.full((22, 3), np.nan, dtype=np.float32)
    
    if gt_full_joints is not None and not np.isnan(gt_full_joints).all():
        # 全関節のGTが利用可能な場合はそれを使用
        full_skeleton = gt_full_joints.copy()
    else:
        # 腕の関節のみのGTを使用
        for i, joint_idx in enumerate(ARM_JOINT_INDICES):
            if not np.isnan(gt_arm_joints[i]).any():
                full_skeleton[joint_idx] = gt_arm_joints[i]
    
    return full_skeleton


def create_hybrid_skeleton(pred_arm_joints: np.ndarray, gt_arm_joints: np.ndarray, 
                          gt_full_joints: np.ndarray = None) -> np.ndarray:
    """
    予測とGTを組み合わせた骨格を作成
    
    Args:
        pred_arm_joints: (6, 3) - 予測された腕の関節座標
        gt_arm_joints: (6, 3) - GTの腕の関節座標
        gt_full_joints: (22, 3) - 全関節のGT座標（オプション）
    
    Returns:
        hybrid_skeleton: (22, 3) - 予測とGTを組み合わせた骨格
    """
    # まず全骨格のGTを取得
    hybrid_skeleton = get_full_skeleton_gt(gt_arm_joints, gt_full_joints)
    
    # wristとelbowのみ予測値に置き換え
    for i in WRIST_ELBOW_INDICES:
        joint_idx = ARM_JOINT_INDICES[i]
        if not np.isnan(pred_arm_joints[i]).any():
            hybrid_skeleton[joint_idx] = pred_arm_joints[i]
    
    return hybrid_skeleton


def plot_skeleton(ax, skeleton: np.ndarray, joints_def: dict, 
                  color: str = 'black', linewidth: float = 2.0, 
                  markersize: float = 8.0, alpha: float = 1.0,
                  highlight_joints: list = None):
    """
    骨格をプロット
    
    Args:
        ax: matplotlibのaxes
        skeleton: (22, 3) - 骨格座標
        joints_def: joints_def_22.jsonの内容
        color: プロットの色
        linewidth: 線の太さ
        markersize: マーカーのサイズ
        alpha: 透明度
        highlight_joints: 強調表示する関節のインデックスリスト（Noneの場合は全て同じ色）
    """
    bones = joints_def['bones']
    
    if highlight_joints is None:
        # 通常の描画（全て同じ色）
        # 骨を描画
        for bone in bones:
            joint1_idx, joint2_idx = bone
            if (not np.isnan(skeleton[joint1_idx]).any() and 
                not np.isnan(skeleton[joint2_idx]).any()):
                ax.plot([skeleton[joint1_idx, 0], skeleton[joint2_idx, 0]],
                       [skeleton[joint1_idx, 1], skeleton[joint2_idx, 1]],
                       [skeleton[joint1_idx, 2], skeleton[joint2_idx, 2]],
                       color=color, linewidth=linewidth, alpha=alpha)
        
        # 関節を描画
        for i in range(len(skeleton)):
            if not np.isnan(skeleton[i]).any():
                ax.scatter(skeleton[i, 0], skeleton[i, 1], skeleton[i, 2],
                          color=color, s=markersize**2, alpha=alpha)
    else:
        # 強調表示する関節がある場合
        highlight_set = set(highlight_joints)
        
        # 骨を描画（強調表示する関節を含む骨は強調色、それ以外は通常色）
        for bone in bones:
            joint1_idx, joint2_idx = bone
            if (not np.isnan(skeleton[joint1_idx]).any() and 
                not np.isnan(skeleton[joint2_idx]).any()):
                # どちらかの関節が強調対象の場合、強調色で描画
                if joint1_idx in highlight_set or joint2_idx in highlight_set:
                    bone_color = 'red'
                    bone_linewidth = linewidth * 1.2
                else:
                    bone_color = color
                    bone_linewidth = linewidth
                
                ax.plot([skeleton[joint1_idx, 0], skeleton[joint2_idx, 0]],
                       [skeleton[joint1_idx, 1], skeleton[joint2_idx, 1]],
                       [skeleton[joint1_idx, 2], skeleton[joint2_idx, 2]],
                       color=bone_color, linewidth=bone_linewidth, alpha=alpha)
        
        # 関節を描画
        for i in range(len(skeleton)):
            if not np.isnan(skeleton[i]).any():
                if i in highlight_set:
                    joint_color = 'red'
                    joint_size = markersize * 1.2
                else:
                    joint_color = color
                    joint_size = markersize
                ax.scatter(skeleton[i, 0], skeleton[i, 1], skeleton[i, 2],
                          color=joint_color, s=joint_size**2, alpha=alpha)


def create_animation(eval_json_path: str, joints_def_path: str, 
                    output_path: str, fps: int = 30, 
                    start_frame: int = 0, end_frame: int = None,
                    view_angle: tuple = (30, 45)):
    """
    アニメーションを作成
    
    Args:
        eval_json_path: evaluate_arm_joint_3d.pyの結果JSONファイル
        joints_def_path: joints_def_22.jsonのパス
        output_path: 出力動画ファイルのパス
        fps: フレームレート
        start_frame: 開始フレーム
        end_frame: 終了フレーム（Noneの場合は全フレーム）
        view_angle: 3Dプロットの視点角度 (elev, azim)
    """
    # データを読み込む
    print(f"Loading evaluation results from {eval_json_path}...")
    eval_results = load_evaluation_results(eval_json_path)
    
    print(f"Loading joints definition from {joints_def_path}...")
    joints_def = load_joints_def(joints_def_path)
    
    # 予測とGTの3D座標を取得
    pred_3d = np.array(eval_results['pred_3d'])  # (N, 6, 3)
    gt_3d = np.array(eval_results['gt_3d'])  # (N, 6, 3)
    frame_ids = eval_results.get('frame_ids', [])  # フレームID情報
    gt_full_3d_list = eval_results.get('gt_full_3d', [])
    gt_full_3d = None
    if len(gt_full_3d_list) > 0:
        try:
            # NoneをNaNに変換してからNumPy配列に変換
            gt_full_3d_processed = []
            for frame in gt_full_3d_list:
                frame_processed = []
                for joint in frame:
                    if joint is None or (isinstance(joint, list) and all(x is None for x in joint)):
                        frame_processed.append([np.nan, np.nan, np.nan])
                    else:
                        joint_processed = [np.nan if x is None else float(x) for x in joint]
                        frame_processed.append(joint_processed)
                gt_full_3d_processed.append(frame_processed)
            
            gt_full_3d = np.array(gt_full_3d_processed, dtype=np.float32)  # (N, 22, 3) - 全骨格のGT
            # 形状を確認
            if len(gt_full_3d.shape) < 2 or gt_full_3d.shape[0] == 0:
                gt_full_3d = None
            else:
                # デバッグ: gt_full_3dの有効性を確認
                valid_frames = ~np.isnan(gt_full_3d).all(axis=(1, 2))
                print(f"\n[Debug] GT Full Skeleton Statistics:")
                print(f"  Total frames: {len(gt_full_3d)}")
                print(f"  Valid frames (not all NaN): {valid_frames.sum()}/{len(valid_frames)} ({valid_frames.sum()/len(valid_frames)*100:.1f}%)")
                if valid_frames.sum() > 0:
                    # 有効なフレームの最初の数フレームで、座標が変化しているか確認
                    valid_indices = np.where(valid_frames)[0]
                    if len(valid_indices) >= 2:
                        frame1_idx = valid_indices[0]
                        frame2_idx = valid_indices[min(1, len(valid_indices)-1)]
                        frame1 = gt_full_3d[frame1_idx]
                        frame2 = gt_full_3d[frame2_idx]
                        # 有効な関節のみで差分を計算
                        valid_joints1 = ~np.isnan(frame1).any(axis=1)
                        valid_joints2 = ~np.isnan(frame2).any(axis=1)
                        common_valid = valid_joints1 & valid_joints2
                        if common_valid.sum() > 0:
                            diff = np.abs(frame1[common_valid] - frame2[common_valid])
                            max_diff = diff.max()
                            mean_diff = diff.mean()
                            print(f"  Coordinate change between frame {frame1_idx} and {frame2_idx}:")
                            print(f"    Max difference: {max_diff:.4f} m")
                            print(f"    Mean difference: {mean_diff:.4f} m")
                            if max_diff < 0.001:  # 1mm未満の変化
                                print(f"    ⚠️  Warning: GT coordinates are not changing between frames!")
                                print(f"    This suggests that gt_full_3d may not be correctly matched to frames.")
                        else:
                            print(f"  ⚠️  Warning: No common valid joints between frames {frame1_idx} and {frame2_idx}")
                else:
                    print(f"  ⚠️  Warning: No valid GT full skeleton frames found!")
        except (ValueError, TypeError) as e:
            print(f"Warning: Failed to process gt_full_3d: {e}")
            gt_full_3d = None
    valid_mask = np.array(eval_results['valid_mask'])  # (N, 6)
    
    num_frames = len(pred_3d)
    if end_frame is None:
        end_frame = num_frames
    
    # フレーム範囲を調整
    start_frame = max(0, start_frame)
    end_frame = min(num_frames, end_frame)
    num_frames_to_plot = end_frame - start_frame
    
    print(f"Creating animation for frames {start_frame} to {end_frame} ({num_frames_to_plot} frames)...")
    
    # 座標の範囲を計算（全フレームから）
    all_gt_coords = gt_3d[valid_mask].reshape(-1, 3)
    all_pred_coords = pred_3d[valid_mask].reshape(-1, 3)
    all_coords = [all_gt_coords, all_pred_coords]
    
    # 全骨格のGTも含める（利用可能な場合）
    if gt_full_3d is not None and len(gt_full_3d) > 0 and gt_full_3d.shape[0] > 0:
        try:
            # 有効なフレームを取得（全てNaNでないフレーム）
            # 数値型であることを確認
            if np.issubdtype(gt_full_3d.dtype, np.floating) or np.issubdtype(gt_full_3d.dtype, np.integer):
                valid_frames = ~np.isnan(gt_full_3d).all(axis=(1, 2))
                if valid_frames.sum() > 0:
                    gt_full_valid = gt_full_3d[valid_frames]
                    all_coords.append(gt_full_valid.reshape(-1, 3))
        except (TypeError, ValueError):
            # 型エラーの場合はスキップ
            pass
    
    all_coords = np.vstack(all_coords)
    
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()
    
    # マージンを追加
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    x_min -= x_range * margin
    x_max += x_range * margin
    y_min -= y_range * margin
    y_max += y_range * margin
    z_min -= z_range * margin
    z_max += z_range * margin
    
    # 図を作成（横並びに2つのサブプロット）
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(121, projection='3d')  # 左: 予測+GT骨格
    ax2 = fig.add_subplot(122, projection='3d')  # 右: GT骨格
    
    # 初期化関数
    def init():
        ax1.clear()
        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])
        ax1.set_zlim([z_min, z_max])
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Predicted+GT Skeleton\n(Red=Predicted Wrist/Elbow)\nFrame 0')
        ax1.view_init(elev=view_angle[0], azim=view_angle[1])
        
        ax2.clear()
        ax2.set_xlim([x_min, x_max])
        ax2.set_ylim([y_min, y_max])
        ax2.set_zlim([z_min, z_max])
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('GT Skeleton\n(Red=GT Wrist/Elbow)\nFrame 0')
        ax2.view_init(elev=view_angle[0], azim=view_angle[1])
        return []
    
    # アニメーション関数
    def animate(frame_idx):
        ax1.clear()
        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])
        ax1.set_zlim([z_min, z_max])
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        ax2.clear()
        ax2.set_xlim([x_min, x_max])
        ax2.set_ylim([y_min, y_max])
        ax2.set_zlim([z_min, z_max])
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        
        actual_frame = start_frame + frame_idx
        ax1.set_title(f'Predicted+GT Skeleton\n(Red=Predicted Wrist/Elbow)\nFrame {actual_frame}')
        ax2.set_title(f'GT Skeleton\n(Red=GT Wrist/Elbow)\nFrame {actual_frame}')
        ax1.view_init(elev=view_angle[0], azim=view_angle[1])
        ax2.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # 予測とGTを組み合わせた骨格を作成
        gt_arm = gt_3d[actual_frame]  # (6, 3)
        pred_arm = pred_3d[actual_frame]  # (6, 3)
        gt_full_frame = None
        if gt_full_3d is not None and len(gt_full_3d) > 0 and gt_full_3d.shape[0] > actual_frame:
            gt_full_frame = gt_full_3d[actual_frame]  # (22, 3)
            
            # デバッグ: 最初の数フレームでフレームIDとGTの有効性を確認
            if frame_idx < 3 and len(frame_ids) > actual_frame:
                frame_info = frame_ids[actual_frame]
                frame_id_str = frame_info.get('frame_id', 'N/A')
                sequence_id = frame_info.get('sequence_id', 'N/A')
                frame_num = frame_info.get('frame_num', 'N/A')
                valid_joints_count = (~np.isnan(gt_full_frame).any(axis=1)).sum() if gt_full_frame is not None else 0
                print(f"  Frame {actual_frame}: frame_id={frame_id_str}, seq={sequence_id}, frame_num={frame_num}, valid_joints={valid_joints_count}/22")
        
        if valid_mask[actual_frame].any():
            # GT骨格を取得（全22関節）
            gt_skeleton = get_full_skeleton_gt(gt_arm, gt_full_frame)  # (22, 3)
            
            # 予測とGTを組み合わせた骨格を作成（wristとelbowのみ予測値を使用）
            hybrid_skeleton = create_hybrid_skeleton(pred_arm, gt_arm, gt_full_frame)  # (22, 3)
            
            # wristとelbowの関節インデックス（joints_def_22.jsonのインデックス）
            # L_Elbow: 18, L_Wrist: 20, R_Elbow: 19, R_Wrist: 21
            highlight_joints = [18, 20, 19, 21]  # wristとelbowのインデックス
            highlight_set = set(highlight_joints)
            bones = joints_def['bones']
            
            # ===== 左側: 予測+GT骨格 =====
            # 全骨格を描画（通常の黒色）
            plot_skeleton(ax1, hybrid_skeleton, joints_def, color='black', 
                         linewidth=2.0, markersize=6.0, alpha=1.0,
                         highlight_joints=None)
            
            # 予測wrist/elbowの関節を赤色で描画
            for joint_idx in highlight_joints:
                if not np.isnan(hybrid_skeleton[joint_idx]).any():
                    ax1.scatter(hybrid_skeleton[joint_idx, 0], 
                              hybrid_skeleton[joint_idx, 1], 
                              hybrid_skeleton[joint_idx, 2],
                              color='red', s=10.0**2, alpha=1.0)
            
            # wrist/elbowに接続する骨を赤色で描画（予測骨格）
            for bone in bones:
                joint1_idx, joint2_idx = bone
                # どちらかの関節がwrist/elbowの場合
                if joint1_idx in highlight_set or joint2_idx in highlight_set:
                    if (not np.isnan(hybrid_skeleton[joint1_idx]).any() and 
                        not np.isnan(hybrid_skeleton[joint2_idx]).any()):
                        ax1.plot([hybrid_skeleton[joint1_idx, 0], hybrid_skeleton[joint2_idx, 0]],
                               [hybrid_skeleton[joint1_idx, 1], hybrid_skeleton[joint2_idx, 1]],
                               [hybrid_skeleton[joint1_idx, 2], hybrid_skeleton[joint2_idx, 2]],
                               color='red', linewidth=3.0, alpha=1.0, linestyle='-')
            
            # ===== 右側: GT骨格 =====
            # 全骨格を描画（通常の黒色）
            plot_skeleton(ax2, gt_skeleton, joints_def, color='black', 
                         linewidth=2.0, markersize=6.0, alpha=1.0,
                         highlight_joints=None)
            
            # GT wrist/elbowの関節を赤色で描画（×マーカー）
            for joint_idx in highlight_joints:
                if not np.isnan(gt_skeleton[joint_idx]).any():
                    ax2.scatter(gt_skeleton[joint_idx, 0], 
                              gt_skeleton[joint_idx, 1], 
                              gt_skeleton[joint_idx, 2],
                              color='red', s=10.0**2, alpha=1.0, marker='x', linewidths=3)
            
            # wrist/elbowに接続する骨を赤色で描画（GT骨格）
            for bone in bones:
                joint1_idx, joint2_idx = bone
                # どちらかの関節がwrist/elbowの場合
                if joint1_idx in highlight_set or joint2_idx in highlight_set:
                    if (not np.isnan(gt_skeleton[joint1_idx]).any() and 
                        not np.isnan(gt_skeleton[joint2_idx]).any()):
                        ax2.plot([gt_skeleton[joint1_idx, 0], gt_skeleton[joint2_idx, 0]],
                               [gt_skeleton[joint1_idx, 1], gt_skeleton[joint2_idx, 1]],
                               [gt_skeleton[joint1_idx, 2], gt_skeleton[joint2_idx, 2]],
                               color='red', linewidth=3.0, alpha=1.0, linestyle='--')
        
        return []
    
    # アニメーションを作成
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=num_frames_to_plot, 
                                  interval=1000/fps, blit=False, repeat=True)
    
    # 保存
    print(f"Saving animation to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 出力形式に応じてwriterを選択
    output_ext = output_path.suffix.lower()
    
    if output_ext == '.mp4':
        # MP4形式の場合はffmpegを試す
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(str(output_path), writer=writer)
            print(f"Animation saved to {output_path} (MP4 format)")
        except RuntimeError:
            print("Warning: ffmpeg is not available. Saving as GIF instead...")
            # GIF形式で保存
            gif_path = output_path.with_suffix('.gif')
            try:
                Writer = animation.writers['pillow']
                writer = Writer(fps=fps)
                anim.save(str(gif_path), writer=writer)
                print(f"Animation saved to {gif_path} (GIF format)")
            except RuntimeError:
                print("Error: Neither ffmpeg nor pillow is available.")
                print("Please install ffmpeg or pillow to save animations.")
                raise
    elif output_ext == '.gif':
        # GIF形式の場合はpillowを使用
        try:
            Writer = animation.writers['pillow']
            writer = Writer(fps=fps)
            anim.save(str(output_path), writer=writer)
            print(f"Animation saved to {output_path} (GIF format)")
        except RuntimeError:
            print("Error: pillow is not available.")
            print("Please install pillow to save GIF animations: pip install pillow")
            raise
    else:
        # その他の形式はffmpegを試す
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(str(output_path), writer=writer)
            print(f"Animation saved to {output_path}")
        except RuntimeError:
            print("Warning: ffmpeg is not available. Trying GIF format...")
            # GIF形式で保存
            gif_path = output_path.with_suffix('.gif')
            try:
                Writer = animation.writers['pillow']
                writer = Writer(fps=fps)
                anim.save(str(gif_path), writer=writer)
                print(f"Animation saved to {gif_path} (GIF format)")
            except RuntimeError:
                print("Error: Neither ffmpeg nor pillow is available.")
                print("Please install ffmpeg or pillow to save animations.")
                raise


def main():
    parser = argparse.ArgumentParser(description='Create animation of predicted vs GT arm joints')
    parser.add_argument('--eval_json', type=str, required=True,
                       help='Path to evaluation results JSON (from evaluate_arm_joint_3d.py)')
    parser.add_argument('--joints_def', type=str, 
                       default='../../data_specs/joints_def_22.json',
                       help='Path to joints_def_22.json')
    parser.add_argument('--output', type=str, required=True,
                       help='Output animation file path (e.g., animation.mp4)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frame rate for animation (default: 30)')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Start frame index (default: 0)')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='End frame index (default: None, all frames)')
    parser.add_argument('--elev', type=float, default=30.0,
                       help='Elevation angle for 3D plot (default: 30.0)')
    parser.add_argument('--azim', type=float, default=45.0,
                       help='Azimuth angle for 3D plot (default: 45.0)')
    
    args = parser.parse_args()
    
    # パスを解決
    eval_json_path = Path(args.eval_json)
    if not eval_json_path.is_absolute():
        eval_json_path = Path(__file__).parent / eval_json_path
    
    joints_def_path = Path(args.joints_def)
    if not joints_def_path.is_absolute():
        joints_def_path = Path(__file__).parent / joints_def_path
    
    create_animation(
        str(eval_json_path),
        str(joints_def_path),
        args.output,
        fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        view_angle=(args.elev, args.azim)
    )


if __name__ == '__main__':
    main()

