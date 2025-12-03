#!/usr/bin/env python3
"""
論文用のすべての図表を作成するメインスクリプト

このスクリプトを実行すると、すべての図表が作成されます。
"""

import subprocess
import sys
from pathlib import Path
import argparse
from tqdm import tqdm


def run_script(script_path: str, args: list = None):
    """スクリプトを実行"""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"⚠️  Error running {script_path}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Create all paper figures and tables')
    parser.add_argument('--model_path_xz', type=str, 
                       default='outputs/improved_attention_xz/best_model.pth',
                       help='Path to XZ plane model for attention visualization (must be trained with --use_attention)')
    parser.add_argument('--model_path_xy', type=str,
                       default='outputs/improved_attention_xy/best_model.pth',
                       help='Path to XY plane model for attention visualization (must be trained with --use_attention)')
    parser.add_argument('--test_data_xz', type=str,
                       default='data/full_skeleton_data/val.jsonl',
                       help='Test data path for XZ plane')
    parser.add_argument('--test_data_xy', type=str,
                       default='data/full_skeleton_data_xy/val.jsonl',
                       help='Test data path for XY plane')
    parser.add_argument('--ours_result', type=str,
                       default='outputs/evaluation_arm_joint_3d_full_skeleton.json',
                       help='Path to our method evaluation result')
    parser.add_argument('--direct_result', type=str, default=None,
                       help='Path to Direct Regression evaluation result (optional)')
    parser.add_argument('--motion_result', type=str, default=None,
                       help='Path to Motion-Filtered evaluation result (optional)')
    parser.add_argument('--skip_attention', action='store_true',
                       help='Skip attention map visualization (requires model inference)')
    parser.add_argument('--skip_inference', action='store_true',
                       help='Skip inference speed measurement')
    parser.add_argument('--skip_qualitative', action='store_true',
                       help='Skip qualitative results visualization')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Creating All Paper Figures and Tables")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # タスクリストを作成
    tasks = []
    
    # 1. Figure 1: アーキテクチャ図
    tasks.append(("Figure 1: Architecture Diagram", 'visualize_architecture.py', None, True))
    
    # 2. Figure 2-3, 5, 7, 9: 各種比較図とTable 1-5
    tasks.append(("Figures 2, 3, 5, 7, 9 and Tables 1-5", 'create_paper_figures.py', None, True))
    
    # 3. Figure 4: 関節ごとのAttentionスケーリング値の可視化
    if not args.skip_attention:
        tasks.append(("Figure 4: Attention Scaling Values", 'visualize_attention_scaling.py', [
            '--model_path', args.model_path_xz,
            '--test_data', args.test_data_xz,
            '--plane', 'xz',
            '--x_range', '-1.0', '1.0',
            '--z_range', '-1.0', '0.75',
            '--bins', '50',
            '--feature', 'energy_power',
            '--num_samples', '100',
            '--output_dir', 'outputs/paper_figures'
        ], True))
    
    # 4. Figure 6: 推論速度の比較
    if not args.skip_inference:
        tasks.append(("Figure 6: Inference Speed", 'measure_inference_speed.py', None, True))
    
    # 5. Figure 8: 定性結果
    if not args.skip_qualitative:
        cmd_args = [
            '--ours_result', args.ours_result,
            '--num_frames', '10'
        ]
        if args.direct_result:
            cmd_args.extend(['--direct_result', args.direct_result])
        if args.motion_result:
            cmd_args.extend(['--motion_result', args.motion_result])
        tasks.append(("Figure 8: Qualitative Results", 'visualize_qualitative_results.py', cmd_args, True))
    
    # タスクを実行
    for desc, script, script_args, should_run in tqdm(tasks, desc="Creating all figures"):
        if should_run:
            total_count += 1
            if run_script(script, script_args):
                success_count += 1
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Successfully created: {success_count}/{total_count} figures/tables")
    print(f"Output directory: outputs/paper_figures/")
    print("=" * 60)
    
    if success_count == total_count:
        print("✓ All figures and tables created successfully!")
    else:
        print(f"⚠️  {total_count - success_count} figure(s)/table(s) failed to create")


if __name__ == '__main__':
    main()

