#!/usr/bin/env python3
"""
モデル選択器の評価スクリプト

学習済みモデル選択器の性能を評価
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heatmap_distal_detection.models.arm_model_selector import create_arm_model_selector
from heatmap_distal_detection.train_arm_model_selector import ModelSelectionDataset


def evaluate_model_selector(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    split_name: str = 'test'
) -> Dict:
    """
    モデル選択器を評価
    
    Args:
        model: 学習済みモデル選択器
        test_loader: テストデータローダー
        device: デバイス
        output_dir: 出力ディレクトリ
        split_name: データセット名（'test' or 'val'）
    
    Returns:
        metrics: 評価指標の辞書
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_arm_mpjpe = []
    all_full_mpjpe = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {split_name}"):
            attention_left = batch['attention_left'].to(device)  # (B, 3, 6, 6)
            attention_right = batch['attention_right'].to(device)  # (B, 3, 6, 6)
            labels = batch['label'].numpy()  # (B,)
            arm_mpjpe = batch['arm_mpjpe'].numpy()  # (B,)
            full_mpjpe = batch['full_mpjpe'].numpy()  # (B,)
            
            # 予測
            probs = model(attention_left, attention_right).squeeze().cpu().numpy()  # (B,)
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_arm_mpjpe.extend(arm_mpjpe)
            all_full_mpjpe.extend(full_mpjpe)
    
    # NumPy配列に変換
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_arm_mpjpe = np.array(all_arm_mpjpe)
    all_full_mpjpe = np.array(all_full_mpjpe)
    
    # メトリクス計算
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # ROC曲線
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # 分類レポート
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # 選択結果に基づくMPJPE分析
    arm_selected_mask = all_preds == 1
    full_selected_mask = all_preds == 0
    
    arm_selected_mpjpe = all_arm_mpjpe[arm_selected_mask]
    full_selected_mpjpe = all_full_mpjpe[full_selected_mask]
    
    # 実際に選択されたモデルのMPJPE
    selected_mpjpe = np.zeros(len(all_labels))
    selected_mpjpe[arm_selected_mask] = arm_selected_mpjpe
    selected_mpjpe[full_selected_mask] = full_mpjpe[full_selected_mask]
    
    # 理想的な選択（常に良い方を選ぶ場合）のMPJPE
    ideal_selection = (all_arm_mpjpe < all_full_mpjpe).astype(int)
    ideal_mpjpe = np.zeros(len(all_labels))
    ideal_mpjpe[ideal_selection == 1] = all_arm_mpjpe[ideal_selection == 1]
    ideal_mpjpe[ideal_selection == 0] = all_full_mpjpe[ideal_selection == 0]
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'mpjpe_analysis': {
            'arm_selected_count': int(arm_selected_mask.sum()),
            'full_selected_count': int(full_selected_mask.sum()),
            'arm_selected_mpjpe_mean': float(np.nanmean(arm_selected_mpjpe)) if len(arm_selected_mpjpe) > 0 else np.nan,
            'full_selected_mpjpe_mean': float(np.nanmean(full_selected_mpjpe)) if len(full_selected_mpjpe) > 0 else np.nan,
            'selected_mpjpe_mean': float(np.nanmean(selected_mpjpe)),
            'selected_mpjpe_std': float(np.nanstd(selected_mpjpe)),
            'ideal_mpjpe_mean': float(np.nanmean(ideal_mpjpe)),
            'improvement_ratio': float((np.nanmean(ideal_mpjpe) - np.nanmean(selected_mpjpe)) / np.nanmean(ideal_mpjpe) * 100) if np.nanmean(ideal_mpjpe) > 0 else 0.0
        }
    }
    
    # 結果を表示
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({split_name})")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print(f"\nMPJPE Analysis:")
    print(f"  Arm model selected: {arm_selected_mask.sum()} samples")
    print(f"  Full model selected: {full_selected_mask.sum()} samples")
    print(f"  Selected MPJPE (mean): {np.nanmean(selected_mpjpe):.6f}")
    print(f"  Ideal MPJPE (mean): {np.nanmean(ideal_mpjpe):.6f}")
    print(f"  Improvement ratio: {metrics['mpjpe_analysis']['improvement_ratio']:.2f}%")
    print(f"{'='*60}\n")
    
    # ROC曲線をプロット
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({split_name})')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    roc_path = os.path.join(output_dir, f'roc_curve_{split_name}.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {roc_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate model selector')
    
    # モデルパス
    parser.add_argument('--model_xz', type=str, required=True,
                        help='Path to model checkpoint (xz plane)')
    parser.add_argument('--model_xy', type=str, required=True,
                        help='Path to model checkpoint (xy plane)')
    
    # データパス
    parser.add_argument('--test_data_xz', type=str, required=True,
                        help='Path to test data (xz plane)')
    parser.add_argument('--test_data_xy', type=str, required=True,
                        help='Path to test data (xy plane)')
    parser.add_argument('--val_data_xz', type=str, default=None,
                        help='Path to val data (xz plane, optional)')
    parser.add_argument('--val_data_xy', type=str, default=None,
                        help='Path to val data (xy plane, optional)')
    
    # 出力
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for evaluation results')
    
    # 評価設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # モデルを読み込み
    print("Loading models...")
    
    # xz平面のモデル
    checkpoint_xz = torch.load(args.model_xz, map_location=device)
    config_xz = checkpoint_xz.get('config', {})
    model_xz = create_arm_model_selector(
        attention_size=tuple(config_xz.get('attention_size', [6, 6])),
        num_joints_per_arm=config_xz.get('num_joints_per_arm', 3),
        hidden_dim=config_xz.get('hidden_dim', 128)
    )
    model_xz.load_state_dict(checkpoint_xz['model_state_dict'])
    model_xz.to(device)
    model_xz.eval()
    print(f"  XZ model loaded from: {args.model_xz}")
    
    # xy平面のモデル
    checkpoint_xy = torch.load(args.model_xy, map_location=device)
    config_xy = checkpoint_xy.get('config', {})
    model_xy = create_arm_model_selector(
        attention_size=tuple(config_xy.get('attention_size', [6, 6])),
        num_joints_per_arm=config_xy.get('num_joints_per_arm', 3),
        hidden_dim=config_xy.get('hidden_dim', 128)
    )
    model_xy.load_state_dict(checkpoint_xy['model_state_dict'])
    model_xy.to(device)
    model_xy.eval()
    print(f"  XY model loaded from: {args.model_xy}")
    
    # データセットを読み込み
    print("\nLoading datasets...")
    test_dataset_xz = ModelSelectionDataset(args.test_data_xz, plane='xz')
    test_dataset_xy = ModelSelectionDataset(args.test_data_xy, plane='xy')
    
    test_loader_xz = DataLoader(
        test_dataset_xz, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader_xy = DataLoader(
        test_dataset_xy, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # xz平面の評価
    print(f"\n{'='*60}")
    print("Evaluating XZ model")
    print(f"{'='*60}")
    output_dir_xz = os.path.join(args.output_dir, 'xz')
    os.makedirs(output_dir_xz, exist_ok=True)
    metrics_xz = evaluate_model_selector(
        model_xz, test_loader_xz, device, output_dir_xz, split_name='test'
    )
    
    # 結果を保存
    metrics_path_xz = os.path.join(output_dir_xz, 'test_metrics.json')
    with open(metrics_path_xz, 'w') as f:
        json.dump(metrics_xz, f, indent=2)
    print(f"Metrics saved to: {metrics_path_xz}")
    
    # xy平面の評価
    print(f"\n{'='*60}")
    print("Evaluating XY model")
    print(f"{'='*60}")
    output_dir_xy = os.path.join(args.output_dir, 'xy')
    os.makedirs(output_dir_xy, exist_ok=True)
    metrics_xy = evaluate_model_selector(
        model_xy, test_loader_xy, device, output_dir_xy, split_name='test'
    )
    
    # 結果を保存
    metrics_path_xy = os.path.join(output_dir_xy, 'test_metrics.json')
    with open(metrics_path_xy, 'w') as f:
        json.dump(metrics_xy, f, indent=2)
    print(f"Metrics saved to: {metrics_path_xy}")
    
    # 検証データがある場合も評価
    if args.val_data_xz is not None and args.val_data_xy is not None:
        val_dataset_xz = ModelSelectionDataset(args.val_data_xz, plane='xz')
        val_dataset_xy = ModelSelectionDataset(args.val_data_xy, plane='xy')
        
        val_loader_xz = DataLoader(
            val_dataset_xz, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        val_loader_xy = DataLoader(
            val_dataset_xy, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        
        # xz平面の検証データ評価
        metrics_xz_val = evaluate_model_selector(
            model_xz, val_loader_xz, device, output_dir_xz, split_name='val'
        )
        metrics_path_xz_val = os.path.join(output_dir_xz, 'val_metrics.json')
        with open(metrics_path_xz_val, 'w') as f:
            json.dump(metrics_xz_val, f, indent=2)
        
        # xy平面の検証データ評価
        metrics_xy_val = evaluate_model_selector(
            model_xy, val_loader_xy, device, output_dir_xy, split_name='val'
        )
        metrics_path_xy_val = os.path.join(output_dir_xy, 'val_metrics.json')
        with open(metrics_path_xy_val, 'w') as f:
            json.dump(metrics_xy_val, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Evaluation completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

