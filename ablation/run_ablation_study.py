#!/usr/bin/env python3
"""
アブレーションスタディを実行するスクリプト

以下のアブレーションスタディを実行:
1. Attention機構の効果
   - Attentionあり vs Attentionなし
2. 階層的回帰の効果
   - 階層的回帰 vs 独立回帰
3. 全身モデル統合の効果
   - 全身モデルの肩 vs 腕特化モデルの肩
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_training(
    plane: str,
    train_data: str,
    val_data: str,
    output_dir: str,
    model_type: str = "hierarchical",
    use_attention: bool = True,
    x_range: tuple = (-1.0, 1.0),
    z_range: tuple = (-1.0, 0.75),
    y_range: tuple = (2.0, 5.0),
    bins: int = 50,
    feature: str = "energy_power",
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    normalize_heatmap: bool = True,
    early_stopping_patience: int = 10
) -> str:
    """
    モデルの学習を実行
    
    Returns:
        model_path: 学習済みモデルのパス
    """
    cmd = [
        "python", "train_arm_joint_regressor.py",
        "--plane", plane,
        "--train_data", train_data,
        "--val_data", val_data,
        "--output_dir", output_dir,
        "--model_type", model_type,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--learning_rate", str(learning_rate),
        "--x_range", str(x_range[0]), str(x_range[1]),
        "--bins", str(bins),
        "--feature", feature,
        "--early_stopping_patience", str(early_stopping_patience),
    ]
    
    if plane == "xz":
        cmd.extend(["--z_range", str(z_range[0]), str(z_range[1])])
    elif plane == "xy":
        cmd.extend(["--y_range", str(y_range[0]), str(y_range[1])])
    
    if use_attention:
        cmd.append("--use_attention")
    
    if normalize_heatmap:
        cmd.append("--normalize_heatmap")
    
    print(f"\n{'='*60}")
    print(f"Training: {model_type}, Attention: {use_attention}, Plane: {plane}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    # 学習の進捗を表示（subprocessの出力をリアルタイムで表示）
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, bufsize=1, universal_newlines=True)
    
    # 出力をリアルタイムで表示
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
    
    process.wait()
    result = subprocess.CompletedProcess(cmd, process.returncode, 
                                        stdout=''.join(output_lines), stderr='')
    
    if result.returncode != 0:
        print(f"Error during training:")
        print(result.stderr)
        raise RuntimeError(f"Training failed for {model_type}, attention={use_attention}, plane={plane}")
    
    model_path = os.path.join(output_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return model_path


def run_evaluation_3d(
    model_path_xz: str,
    model_path_xy: str,
    test_data_xz: str,
    test_data_xy: str,
    output_path: str,
    full_model_xz: str = None,
    full_model_xy: str = None,
    full_skeleton_data: str = None,
    x_range_xz: tuple = (-1.0, 1.0),
    z_range: tuple = (-1.0, 0.75),
    x_range_xy: tuple = (-1.0, 1.0),
    y_range: tuple = (2.0, 5.0),
    bins: int = 50,
    feature: str = "energy_power",
    batch_size: int = 32,
    normalize_heatmap: bool = True
) -> Dict:
    """
    3D評価を実行
    
    Returns:
        metrics: 評価結果の辞書
    """
    cmd = [
        "python", "evaluate_arm_joint_3d.py",
        "--model_path_xz", model_path_xz,
        "--model_path_xy", model_path_xy,
        "--test_data_xz", test_data_xz,
        "--test_data_xy", test_data_xy,
        "--output_path", output_path,
        "--x_range_xz", str(x_range_xz[0]), str(x_range_xz[1]),
        "--z_range", str(z_range[0]), str(z_range[1]),
        "--x_range_xy", str(x_range_xy[0]), str(x_range_xy[1]),
        "--y_range", str(y_range[0]), str(y_range[1]),
        "--bins", str(bins),
        "--feature", feature,
        "--batch_size", str(batch_size),
    ]
    
    if normalize_heatmap:
        cmd.append("--normalize_heatmap")
    
    if full_model_xz and full_model_xy:
        cmd.extend(["--full_model_xz", full_model_xz])
        cmd.extend(["--full_model_xy", full_model_xy])
    
    if full_skeleton_data:
        cmd.extend(["--full_skeleton_data", full_skeleton_data])
    
    print(f"\n{'='*60}")
    print(f"Evaluating 3D: {os.path.basename(model_path_xz)}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    # 評価の進捗を表示（subprocessの出力をリアルタイムで表示）
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, bufsize=1, universal_newlines=True)
    
    # 出力をリアルタイムで表示
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
    
    process.wait()
    result = subprocess.CompletedProcess(cmd, process.returncode, 
                                        stdout=''.join(output_lines), stderr='')
    
    if result.returncode != 0:
        print(f"Error during evaluation:")
        print(result.stderr)
        raise RuntimeError(f"Evaluation failed for {model_path_xz}")
    
    # 結果を読み込み
    with open(output_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def ablation_study_1_attention(
    train_data_xz: str,
    val_data_xz: str,
    train_data_xy: str,
    val_data_xy: str,
    test_data_xz: str,
    test_data_xy: str,
    output_base_dir: str,
    full_model_xz: str = None,
    full_model_xy: str = None,
    full_skeleton_data: str = None,
    x_range_xz: tuple = (-1.0, 1.0),
    z_range: tuple = (-1.0, 0.75),
    x_range_xy: tuple = (-1.0, 1.0),
    y_range: tuple = (2.0, 5.0),
    bins: int = 50,
    feature: str = "energy_power",
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    normalize_heatmap: bool = True,
    early_stopping_patience: int = 10
) -> Dict:
    """
    アブレーションスタディ1: Attention機構の効果
    - Attentionあり vs Attentionなし
    """
    print("\n" + "="*80)
    print("Ablation Study 1: Attention Mechanism")
    print("="*80)
    
    results = {}
    
    # 進捗バー
    pbar = tqdm(total=4, desc="Study 1 Progress", unit="step")
    
    # Attentionあり
    print("\n[1/2] Training with Attention...")
    pbar.set_description("Study 1: Training with Attention (XZ)")
    output_dir_attn_xz = os.path.join(output_base_dir, "ablation_attention_xz")
    output_dir_attn_xy = os.path.join(output_base_dir, "ablation_attention_xy")
    
    model_path_xz_attn = run_training(
        plane="xz",
        train_data=train_data_xz,
        val_data=val_data_xz,
        output_dir=output_dir_attn_xz,
        model_type="hierarchical",
        use_attention=True,
        x_range=x_range_xz,
        z_range=z_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        normalize_heatmap=normalize_heatmap,
        early_stopping_patience=early_stopping_patience
    )
    pbar.update(1)
    pbar.set_description("Study 1: Training with Attention (XY)")
    
    model_path_xy_attn = run_training(
        plane="xy",
        train_data=train_data_xy,
        val_data=val_data_xy,
        output_dir=output_dir_attn_xy,
        model_type="hierarchical",
        use_attention=True,
        x_range=x_range_xy,
        y_range=y_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        normalize_heatmap=normalize_heatmap,
        early_stopping_patience=early_stopping_patience
    )
    pbar.update(1)
    pbar.set_description("Study 1: Evaluating with Attention")
    
    eval_path_attn = os.path.join(output_base_dir, "ablation_attention_3d.json")
    metrics_attn = run_evaluation_3d(
        model_path_xz=model_path_xz_attn,
        model_path_xy=model_path_xy_attn,
        test_data_xz=test_data_xz,
        test_data_xy=test_data_xy,
        output_path=eval_path_attn,
        full_model_xz=full_model_xz,
        full_model_xy=full_model_xy,
        full_skeleton_data=full_skeleton_data,
        x_range_xz=x_range_xz,
        z_range=z_range,
        x_range_xy=x_range_xy,
        y_range=y_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        normalize_heatmap=normalize_heatmap
    )
    results['with_attention'] = metrics_attn
    pbar.update(1)
    
    # Attentionなし
    print("\n[2/2] Training without Attention...")
    pbar.set_description("Study 1: Training without Attention (XZ)")
    output_dir_noattn_xz = os.path.join(output_base_dir, "ablation_no_attention_xz")
    output_dir_noattn_xy = os.path.join(output_base_dir, "ablation_no_attention_xy")
    
    model_path_xz_noattn = run_training(
        plane="xz",
        train_data=train_data_xz,
        val_data=val_data_xz,
        output_dir=output_dir_noattn_xz,
        model_type="hierarchical",
        use_attention=False,
        x_range=x_range_xz,
        z_range=z_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        normalize_heatmap=normalize_heatmap,
        early_stopping_patience=early_stopping_patience
    )
    pbar.update(1)
    pbar.set_description("Study 1: Training without Attention (XY)")
    
    model_path_xy_noattn = run_training(
        plane="xy",
        train_data=train_data_xy,
        val_data=val_data_xy,
        output_dir=output_dir_noattn_xy,
        model_type="hierarchical",
        use_attention=False,
        x_range=x_range_xy,
        y_range=y_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        normalize_heatmap=normalize_heatmap,
        early_stopping_patience=early_stopping_patience
    )
    pbar.update(1)
    pbar.set_description("Study 1: Evaluating without Attention")
    
    eval_path_noattn = os.path.join(output_base_dir, "ablation_no_attention_3d.json")
    metrics_noattn = run_evaluation_3d(
        model_path_xz=model_path_xz_noattn,
        model_path_xy=model_path_xy_noattn,
        test_data_xz=test_data_xz,
        test_data_xy=test_data_xy,
        output_path=eval_path_noattn,
        full_model_xz=full_model_xz,
        full_model_xy=full_model_xy,
        full_skeleton_data=full_skeleton_data,
        x_range_xz=x_range_xz,
        z_range=z_range,
        x_range_xy=x_range_xy,
        y_range=y_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        normalize_heatmap=normalize_heatmap
    )
    results['without_attention'] = metrics_noattn
    pbar.update(1)
    pbar.close()
    
    return results


def ablation_study_2_hierarchical(
    train_data_xz: str,
    val_data_xz: str,
    train_data_xy: str,
    val_data_xy: str,
    test_data_xz: str,
    test_data_xy: str,
    output_base_dir: str,
    full_model_xz: str = None,
    full_model_xy: str = None,
    full_skeleton_data: str = None,
    x_range_xz: tuple = (-1.0, 1.0),
    z_range: tuple = (-1.0, 0.75),
    x_range_xy: tuple = (-1.0, 1.0),
    y_range: tuple = (2.0, 5.0),
    bins: int = 50,
    feature: str = "energy_power",
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    normalize_heatmap: bool = True,
    early_stopping_patience: int = 10
) -> Dict:
    """
    アブレーションスタディ2: 階層的回帰の効果
    - 階層的回帰 vs 独立回帰
    """
    print("\n" + "="*80)
    print("Ablation Study 2: Hierarchical Regression")
    print("="*80)
    
    results = {}
    
    # 進捗バー
    pbar = tqdm(total=4, desc="Study 2 Progress", unit="step")
    
    # 階層的回帰（Attentionあり）
    print("\n[1/2] Training Hierarchical Model...")
    pbar.set_description("Study 2: Training Hierarchical (XZ)")
    output_dir_hier_xz = os.path.join(output_base_dir, "ablation_hierarchical_xz")
    output_dir_hier_xy = os.path.join(output_base_dir, "ablation_hierarchical_xy")
    
    model_path_xz_hier = run_training(
        plane="xz",
        train_data=train_data_xz,
        val_data=val_data_xz,
        output_dir=output_dir_hier_xz,
        model_type="hierarchical",
        use_attention=True,
        x_range=x_range_xz,
        z_range=z_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        normalize_heatmap=normalize_heatmap,
        early_stopping_patience=early_stopping_patience
    )
    pbar.update(1)
    pbar.set_description("Study 2: Training Hierarchical (XY)")
    
    model_path_xy_hier = run_training(
        plane="xy",
        train_data=train_data_xy,
        val_data=val_data_xy,
        output_dir=output_dir_hier_xy,
        model_type="hierarchical",
        use_attention=True,
        x_range=x_range_xy,
        y_range=y_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        normalize_heatmap=normalize_heatmap,
        early_stopping_patience=early_stopping_patience
    )
    pbar.update(1)
    pbar.set_description("Study 2: Evaluating Hierarchical")
    
    eval_path_hier = os.path.join(output_base_dir, "ablation_hierarchical_3d.json")
    metrics_hier = run_evaluation_3d(
        model_path_xz=model_path_xz_hier,
        model_path_xy=model_path_xy_hier,
        test_data_xz=test_data_xz,
        test_data_xy=test_data_xy,
        output_path=eval_path_hier,
        full_model_xz=full_model_xz,
        full_model_xy=full_model_xy,
        full_skeleton_data=full_skeleton_data,
        x_range_xz=x_range_xz,
        z_range=z_range,
        x_range_xy=x_range_xy,
        y_range=y_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        normalize_heatmap=normalize_heatmap
    )
    results['hierarchical'] = metrics_hier
    pbar.update(1)
    
    # 独立回帰（standard）
    print("\n[2/2] Training Standard Model (Independent Regression)...")
    pbar.set_description("Study 2: Training Standard (XZ)")
    output_dir_std_xz = os.path.join(output_base_dir, "ablation_standard_xz")
    output_dir_std_xy = os.path.join(output_base_dir, "ablation_standard_xy")
    
    model_path_xz_std = run_training(
        plane="xz",
        train_data=train_data_xz,
        val_data=val_data_xz,
        output_dir=output_dir_std_xz,
        model_type="standard",
        use_attention=False,  # standardモデルはAttentionをサポートしていない
        x_range=x_range_xz,
        z_range=z_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        normalize_heatmap=normalize_heatmap,
        early_stopping_patience=early_stopping_patience
    )
    pbar.update(1)
    pbar.set_description("Study 2: Training Standard (XY)")
    
    model_path_xy_std = run_training(
        plane="xy",
        train_data=train_data_xy,
        val_data=val_data_xy,
        output_dir=output_dir_std_xy,
        model_type="standard",
        use_attention=False,
        x_range=x_range_xy,
        y_range=y_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        normalize_heatmap=normalize_heatmap,
        early_stopping_patience=early_stopping_patience
    )
    pbar.update(1)
    pbar.set_description("Study 2: Evaluating Standard")
    
    eval_path_std = os.path.join(output_base_dir, "ablation_standard_3d.json")
    metrics_std = run_evaluation_3d(
        model_path_xz=model_path_xz_std,
        model_path_xy=model_path_xy_std,
        test_data_xz=test_data_xz,
        test_data_xy=test_data_xy,
        output_path=eval_path_std,
        full_model_xz=full_model_xz,
        full_model_xy=full_model_xy,
        full_skeleton_data=full_skeleton_data,
        x_range_xz=x_range_xz,
        z_range=z_range,
        x_range_xy=x_range_xy,
        y_range=y_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        normalize_heatmap=normalize_heatmap
    )
    results['independent'] = metrics_std
    pbar.update(1)
    pbar.close()
    
    return results


def ablation_study_3_full_body_integration(
    train_data_xz: str,
    val_data_xz: str,
    train_data_xy: str,
    val_data_xy: str,
    test_data_xz: str,
    test_data_xy: str,
    output_base_dir: str,
    full_model_xz: str,
    full_model_xy: str,
    full_skeleton_data: str,
    x_range_xz: tuple = (-1.0, 1.0),
    z_range: tuple = (-1.0, 0.75),
    x_range_xy: tuple = (-1.0, 1.0),
    y_range: tuple = (2.0, 5.0),
    bins: int = 50,
    feature: str = "energy_power",
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    normalize_heatmap: bool = True,
    early_stopping_patience: int = 10
) -> Dict:
    """
    アブレーションスタディ3: 全身モデル統合の効果
    - 全身モデルの肩 vs 腕特化モデルの肩
    """
    print("\n" + "="*80)
    print("Ablation Study 3: Full-Body Integration")
    print("="*80)
    
    results = {}
    
    # 進捗バー
    pbar = tqdm(total=3, desc="Study 3 Progress", unit="step")
    
    # 階層的モデルを学習（腕特化モデルの肩を使用）
    print("\n[1/2] Training Hierarchical Model (using arm model's shoulders)...")
    pbar.set_description("Study 3: Training Hierarchical (XZ)")
    output_dir_arm_xz = os.path.join(output_base_dir, "ablation_arm_shoulder_xz")
    output_dir_arm_xy = os.path.join(output_base_dir, "ablation_arm_shoulder_xy")
    
    model_path_xz_arm = run_training(
        plane="xz",
        train_data=train_data_xz,
        val_data=val_data_xz,
        output_dir=output_dir_arm_xz,
        model_type="hierarchical",
        use_attention=True,
        x_range=x_range_xz,
        z_range=z_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        normalize_heatmap=normalize_heatmap,
        early_stopping_patience=early_stopping_patience
    )
    pbar.update(1)
    pbar.set_description("Study 3: Training Hierarchical (XY)")
    
    model_path_xy_arm = run_training(
        plane="xy",
        train_data=train_data_xy,
        val_data=val_data_xy,
        output_dir=output_dir_arm_xy,
        model_type="hierarchical",
        use_attention=True,
        x_range=x_range_xy,
        y_range=y_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        normalize_heatmap=normalize_heatmap,
        early_stopping_patience=early_stopping_patience
    )
    pbar.update(1)
    pbar.set_description("Study 3: Evaluating with Arm Shoulders")
    
    eval_path_arm = os.path.join(output_base_dir, "ablation_arm_shoulder_3d.json")
    metrics_arm = run_evaluation_3d(
        model_path_xz=model_path_xz_arm,
        model_path_xy=model_path_xy_arm,
        test_data_xz=test_data_xz,
        test_data_xy=test_data_xy,
        output_path=eval_path_arm,
        full_model_xz=None,  # 全身モデルを使用しない
        full_model_xy=None,
        full_skeleton_data=full_skeleton_data,
        x_range_xz=x_range_xz,
        z_range=z_range,
        x_range_xy=x_range_xy,
        y_range=y_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        normalize_heatmap=normalize_heatmap
    )
    results['arm_shoulder'] = metrics_arm
    pbar.update(1)
    
    # 全身モデルの肩を使用
    print("\n[2/2] Evaluating with Full-Body Shoulders...")
    pbar.set_description("Study 3: Evaluating with Full Shoulders")
    eval_path_full = os.path.join(output_base_dir, "ablation_full_shoulder_3d.json")
    metrics_full = run_evaluation_3d(
        model_path_xz=model_path_xz_arm,  # 同じモデルを使用
        model_path_xy=model_path_xy_arm,
        test_data_xz=test_data_xz,
        test_data_xy=test_data_xy,
        output_path=eval_path_full,
        full_model_xz=full_model_xz,  # 全身モデルを使用
        full_model_xy=full_model_xy,
        full_skeleton_data=full_skeleton_data,
        x_range_xz=x_range_xz,
        z_range=z_range,
        x_range_xy=x_range_xy,
        y_range=y_range,
        bins=bins,
        feature=feature,
        batch_size=batch_size,
        normalize_heatmap=normalize_heatmap
    )
    results['full_shoulder'] = metrics_full
    pbar.update(1)
    pbar.close()
    
    return results


def summarize_results(all_results: Dict, output_path: str):
    """
    アブレーションスタディの結果をまとめて保存
    """
    summary = {
        'ablation_study_1_attention': {
            'with_attention': {
                'overall_mpjpe': all_results['study1']['with_attention'].get('mpjpe_3d_cm'),
                'per_joint': {
                    'L_Shoulder': all_results['study1']['with_attention'].get('L_Shoulder_mpjpe_3d_cm'),
                    'L_Elbow': all_results['study1']['with_attention'].get('L_Elbow_mpjpe_3d_cm'),
                    'L_Wrist': all_results['study1']['with_attention'].get('L_Wrist_mpjpe_3d_cm'),
                    'R_Shoulder': all_results['study1']['with_attention'].get('R_Shoulder_mpjpe_3d_cm'),
                    'R_Elbow': all_results['study1']['with_attention'].get('R_Elbow_mpjpe_3d_cm'),
                    'R_Wrist': all_results['study1']['with_attention'].get('R_Wrist_mpjpe_3d_cm'),
                }
            },
            'without_attention': {
                'overall_mpjpe': all_results['study1']['without_attention'].get('mpjpe_3d_cm'),
                'per_joint': {
                    'L_Shoulder': all_results['study1']['without_attention'].get('L_Shoulder_mpjpe_3d_cm'),
                    'L_Elbow': all_results['study1']['without_attention'].get('L_Elbow_mpjpe_3d_cm'),
                    'L_Wrist': all_results['study1']['without_attention'].get('L_Wrist_mpjpe_3d_cm'),
                    'R_Shoulder': all_results['study1']['without_attention'].get('R_Shoulder_mpjpe_3d_cm'),
                    'R_Elbow': all_results['study1']['without_attention'].get('R_Elbow_mpjpe_3d_cm'),
                    'R_Wrist': all_results['study1']['without_attention'].get('R_Wrist_mpjpe_3d_cm'),
                }
            }
        },
        'ablation_study_2_hierarchical': {
            'hierarchical': {
                'overall_mpjpe': all_results['study2']['hierarchical'].get('mpjpe_3d_cm'),
                'per_joint': {
                    'L_Shoulder': all_results['study2']['hierarchical'].get('L_Shoulder_mpjpe_3d_cm'),
                    'L_Elbow': all_results['study2']['hierarchical'].get('L_Elbow_mpjpe_3d_cm'),
                    'L_Wrist': all_results['study2']['hierarchical'].get('L_Wrist_mpjpe_3d_cm'),
                    'R_Shoulder': all_results['study2']['hierarchical'].get('R_Shoulder_mpjpe_3d_cm'),
                    'R_Elbow': all_results['study2']['hierarchical'].get('R_Elbow_mpjpe_3d_cm'),
                    'R_Wrist': all_results['study2']['hierarchical'].get('R_Wrist_mpjpe_3d_cm'),
                }
            },
            'independent': {
                'overall_mpjpe': all_results['study2']['independent'].get('mpjpe_3d_cm'),
                'per_joint': {
                    'L_Shoulder': all_results['study2']['independent'].get('L_Shoulder_mpjpe_3d_cm'),
                    'L_Elbow': all_results['study2']['independent'].get('L_Elbow_mpjpe_3d_cm'),
                    'L_Wrist': all_results['study2']['independent'].get('L_Wrist_mpjpe_3d_cm'),
                    'R_Shoulder': all_results['study2']['independent'].get('R_Shoulder_mpjpe_3d_cm'),
                    'R_Elbow': all_results['study2']['independent'].get('R_Elbow_mpjpe_3d_cm'),
                    'R_Wrist': all_results['study2']['independent'].get('R_Wrist_mpjpe_3d_cm'),
                }
            }
        },
        'ablation_study_3_full_body': {
            'arm_shoulder': {
                'overall_mpjpe': all_results['study3']['arm_shoulder'].get('mpjpe_3d_cm'),
                'per_joint': {
                    'L_Shoulder': all_results['study3']['arm_shoulder'].get('L_Shoulder_mpjpe_3d_cm'),
                    'L_Elbow': all_results['study3']['arm_shoulder'].get('L_Elbow_mpjpe_3d_cm'),
                    'L_Wrist': all_results['study3']['arm_shoulder'].get('L_Wrist_mpjpe_3d_cm'),
                    'R_Shoulder': all_results['study3']['arm_shoulder'].get('R_Shoulder_mpjpe_3d_cm'),
                    'R_Elbow': all_results['study3']['arm_shoulder'].get('R_Elbow_mpjpe_3d_cm'),
                    'R_Wrist': all_results['study3']['arm_shoulder'].get('R_Wrist_mpjpe_3d_cm'),
                }
            },
            'full_shoulder': {
                'overall_mpjpe': all_results['study3']['full_shoulder'].get('mpjpe_3d_cm'),
                'per_joint': {
                    'L_Shoulder': all_results['study3']['full_shoulder'].get('L_Shoulder_mpjpe_3d_cm'),
                    'L_Elbow': all_results['study3']['full_shoulder'].get('L_Elbow_mpjpe_3d_cm'),
                    'L_Wrist': all_results['study3']['full_shoulder'].get('L_Wrist_mpjpe_3d_cm'),
                    'R_Shoulder': all_results['study3']['full_shoulder'].get('R_Shoulder_mpjpe_3d_cm'),
                    'R_Elbow': all_results['study3']['full_shoulder'].get('R_Elbow_mpjpe_3d_cm'),
                    'R_Wrist': all_results['study3']['full_shoulder'].get('R_Wrist_mpjpe_3d_cm'),
                }
            }
        }
    }
    
    # JSONで保存
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 表形式でも表示
    print("\n" + "="*80)
    print("Ablation Study Summary")
    print("="*80)
    
    print("\n[Study 1] Attention Mechanism:")
    print(f"  With Attention:    {summary['ablation_study_1_attention']['with_attention']['overall_mpjpe']:.2f} cm")
    print(f"  Without Attention: {summary['ablation_study_1_attention']['without_attention']['overall_mpjpe']:.2f} cm")
    improvement = summary['ablation_study_1_attention']['without_attention']['overall_mpjpe'] - summary['ablation_study_1_attention']['with_attention']['overall_mpjpe']
    print(f"  Improvement: {improvement:.2f} cm ({improvement/summary['ablation_study_1_attention']['without_attention']['overall_mpjpe']*100:.1f}%)")
    
    print("\n[Study 2] Hierarchical Regression:")
    print(f"  Hierarchical: {summary['ablation_study_2_hierarchical']['hierarchical']['overall_mpjpe']:.2f} cm")
    print(f"  Independent:  {summary['ablation_study_2_hierarchical']['independent']['overall_mpjpe']:.2f} cm")
    improvement = summary['ablation_study_2_hierarchical']['independent']['overall_mpjpe'] - summary['ablation_study_2_hierarchical']['hierarchical']['overall_mpjpe']
    print(f"  Improvement: {improvement:.2f} cm ({improvement/summary['ablation_study_2_hierarchical']['independent']['overall_mpjpe']*100:.1f}%)")
    
    print("\n[Study 3] Full-Body Integration:")
    print(f"  Arm Shoulder: {summary['ablation_study_3_full_body']['arm_shoulder']['overall_mpjpe']:.2f} cm")
    print(f"  Full Shoulder: {summary['ablation_study_3_full_body']['full_shoulder']['overall_mpjpe']:.2f} cm")
    improvement = summary['ablation_study_3_full_body']['arm_shoulder']['overall_mpjpe'] - summary['ablation_study_3_full_body']['full_shoulder']['overall_mpjpe']
    print(f"  Improvement: {improvement:.2f} cm ({improvement/summary['ablation_study_3_full_body']['arm_shoulder']['overall_mpjpe']*100:.1f}%)")
    
    print(f"\nSummary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies for hierarchical arm joint regressor")
    
    # データパス
    parser.add_argument("--train_data_xz", type=str, required=True, help="Training data path (xz plane)")
    parser.add_argument("--val_data_xz", type=str, required=True, help="Validation data path (xz plane)")
    parser.add_argument("--train_data_xy", type=str, required=True, help="Training data path (xy plane)")
    parser.add_argument("--val_data_xy", type=str, required=True, help="Validation data path (xy plane)")
    parser.add_argument("--test_data_xz", type=str, required=True, help="Test data path (xz plane)")
    parser.add_argument("--test_data_xy", type=str, required=True, help="Test data path (xy plane)")
    
    # 出力パス
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory for ablation studies")
    
    # 全身モデル（オプション）
    parser.add_argument("--full_model_xz", type=str, default=None, help="Full skeleton model path (xz plane)")
    parser.add_argument("--full_model_xy", type=str, default=None, help="Full skeleton model path (xy plane)")
    parser.add_argument("--full_skeleton_data", type=str, default=None, help="Full skeleton data path for GT")
    
    # 学習パラメータ
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    
    # データパラメータ
    parser.add_argument("--x_range_xz", type=float, nargs=2, default=[-1.0, 1.0])
    parser.add_argument("--z_range", type=float, nargs=2, default=[-1.0, 0.75])
    parser.add_argument("--x_range_xy", type=float, nargs=2, default=[-1.0, 1.0])
    parser.add_argument("--y_range", type=float, nargs=2, default=[2.0, 5.0])
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--feature", type=str, default="energy_power", choices=["velocity", "energy_power"])
    parser.add_argument("--normalize_heatmap", action="store_true", help="Normalize heatmap")
    
    # 実行するアブレーションスタディ
    parser.add_argument("--study", type=str, nargs="+", 
                       choices=["1", "2", "3", "all"],
                       default=["all"],
                       help="Which ablation studies to run (1: Attention, 2: Hierarchical, 3: Full-Body)")
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    # 全体の進捗バー
    studies_to_run = []
    if "1" in args.study or "all" in args.study:
        studies_to_run.append("Study 1: Attention")
    if "2" in args.study or "all" in args.study:
        studies_to_run.append("Study 2: Hierarchical")
    if "3" in args.study or "all" in args.study:
        if args.full_model_xz is not None and args.full_model_xy is not None:
            studies_to_run.append("Study 3: Full-Body")
    
    overall_pbar = tqdm(total=len(studies_to_run), desc="Overall Progress", unit="study")
    
    # アブレーションスタディ1: Attention機構の効果
    if "1" in args.study or "all" in args.study:
        overall_pbar.set_description("Running Study 1: Attention")
        results1 = ablation_study_1_attention(
            train_data_xz=args.train_data_xz,
            val_data_xz=args.val_data_xz,
            train_data_xy=args.train_data_xy,
            val_data_xy=args.val_data_xy,
            test_data_xz=args.test_data_xz,
            test_data_xy=args.test_data_xy,
            output_base_dir=args.output_dir,
            full_model_xz=args.full_model_xz,
            full_model_xy=args.full_model_xy,
            full_skeleton_data=args.full_skeleton_data,
            x_range_xz=tuple(args.x_range_xz),
            z_range=tuple(args.z_range),
            x_range_xy=tuple(args.x_range_xy),
            y_range=tuple(args.y_range),
            bins=args.bins,
            feature=args.feature,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            normalize_heatmap=args.normalize_heatmap,
            early_stopping_patience=args.early_stopping_patience
        )
        all_results['study1'] = results1
        overall_pbar.update(1)
    
    # アブレーションスタディ2: 階層的回帰の効果
    if "2" in args.study or "all" in args.study:
        overall_pbar.set_description("Running Study 2: Hierarchical")
        results2 = ablation_study_2_hierarchical(
            train_data_xz=args.train_data_xz,
            val_data_xz=args.val_data_xz,
            train_data_xy=args.train_data_xy,
            val_data_xy=args.val_data_xy,
            test_data_xz=args.test_data_xz,
            test_data_xy=args.test_data_xy,
            output_base_dir=args.output_dir,
            full_model_xz=args.full_model_xz,
            full_model_xy=args.full_model_xy,
            full_skeleton_data=args.full_skeleton_data,
            x_range_xz=tuple(args.x_range_xz),
            z_range=tuple(args.z_range),
            x_range_xy=tuple(args.x_range_xy),
            y_range=tuple(args.y_range),
            bins=args.bins,
            feature=args.feature,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            normalize_heatmap=args.normalize_heatmap,
            early_stopping_patience=args.early_stopping_patience
        )
        all_results['study2'] = results2
        overall_pbar.update(1)
    
    # アブレーションスタディ3: 全身モデル統合の効果
    if "3" in args.study or "all" in args.study:
        if args.full_model_xz is None or args.full_model_xy is None:
            print("Warning: Full model paths not provided. Skipping Study 3.")
        else:
            overall_pbar.set_description("Running Study 3: Full-Body")
            results3 = ablation_study_3_full_body_integration(
                train_data_xz=args.train_data_xz,
                val_data_xz=args.val_data_xz,
                train_data_xy=args.train_data_xy,
                val_data_xy=args.val_data_xy,
                test_data_xz=args.test_data_xz,
                test_data_xy=args.test_data_xy,
                output_base_dir=args.output_dir,
                full_model_xz=args.full_model_xz,
                full_model_xy=args.full_model_xy,
                full_skeleton_data=args.full_skeleton_data,
                x_range_xz=tuple(args.x_range_xz),
                z_range=tuple(args.z_range),
                x_range_xy=tuple(args.x_range_xy),
                y_range=tuple(args.y_range),
                bins=args.bins,
                feature=args.feature,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                normalize_heatmap=args.normalize_heatmap,
                early_stopping_patience=args.early_stopping_patience
            )
            all_results['study3'] = results3
            overall_pbar.update(1)
    
    overall_pbar.close()
    
    # 結果をまとめる
    print("\n" + "="*80)
    print("Summarizing results...")
    print("="*80)
    summary_path = os.path.join(args.output_dir, "ablation_study_summary.json")
    summarize_results(all_results, summary_path)
    
    print("\n" + "="*80)
    print("All ablation studies completed!")
    print("="*80)


if __name__ == "__main__":
    main()

