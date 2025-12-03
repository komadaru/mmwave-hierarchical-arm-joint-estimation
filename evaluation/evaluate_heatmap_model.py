#!/usr/bin/env python3
"""
ヒートマップベース末端部位検出モデルの評価スクリプト
valデータでモデルを評価し、詳細なメトリクスを計算・保存
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heatmap_distal_detection.dataset import HeatmapDistalDataset, collate_fn
from heatmap_distal_detection.models.cnn_2d import create_cnn_2d_model
from heatmap_distal_detection.models.cnn_3d import create_cnn_3d_model, create_multiscale_cnn_3d_model
from heatmap_distal_detection.evaluation import (
    compute_pixel_accuracy,
    compute_iou,
    compute_precision_recall_f1,
    compute_spatial_accuracy,
    evaluate_predictions
)
from heatmap_distal_detection.infer_heatmap_model import extract_distal_regions, classify_distal_regions
from heatmap_distal_detection.train_heatmap_model import create_model
from heatmap_distal_detection.data_preprocessing import (
    test_roundtrip_coordinate_conversion,
    smoke_test_spatial_bins
)


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    gt_joints_list: List[np.ndarray],
    threshold: float = 0.5,
    min_region_size: int = 5,
    min_separation: float = 0.3,
    max_regions: int = 4
) -> Dict:
    """
    モデルを評価
    
    Args:
        model: 評価するモデル
        data_loader: データローダー
        device: デバイス
        gt_joints_list: GT関節のリスト（データセットから読み込んだもの）
        threshold: 確率閾値
        min_region_size: 最小領域サイズ
        min_separation: 最小分離距離
        max_regions: 最大領域数
    
    Returns:
        results: 評価結果の辞書
    """
    model.eval()
    
    # 予測結果を保存
    pred_prob_maps = []
    target_labels = []
    pred_regions_list = []
    metadata_list = []
    
    sample_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = batch['input'].to(device)
            labels = batch['label'].cpu().numpy()  # (B, H, W)
            metadata = batch['metadata']
            
            # フォワードパス
            outputs = model(inputs)
            segmentation = outputs['segmentation']  # (B, num_classes, H, W)
            
            # 確率に変換
            probs = torch.softmax(segmentation, dim=1)  # (B, num_classes, H, W)
            distal_probs = probs[:, 1, :, :]  # (B, H, W) - 末端部位の確率
            
            # 【Step2: 48→50へのアップサンプル】モデル出力(48x48)を50x50にアップサンプル
            # データ収集時のビン数(50)に合わせる
            if distal_probs.shape[-1] != 50 or distal_probs.shape[-2] != 50:
                # 48x48 -> 50x50 にアップサンプル
                distal_probs_upsampled = torch.nn.functional.interpolate(
                    distal_probs.unsqueeze(1),  # (B, 1, H, W)
                    size=(50, 50),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)  # (B, 50, 50)
            else:
                distal_probs_upsampled = distal_probs
            
            distal_probs_upsampled = distal_probs_upsampled.cpu().numpy()
            
            # ラベルも50x50にアップサンプル（確率マップと形状を合わせる）
            labels_upsampled = []
            for b in range(labels.shape[0]):
                label = labels[b]  # (H, W)
                if label.shape[0] != 50 or label.shape[1] != 50:
                    # 48x48 -> 50x50 にアップサンプル（nearest neighborでラベルを保持）
                    label_tensor = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
                    label_upsampled = torch.nn.functional.interpolate(
                        label_tensor,
                        size=(50, 50),
                        mode='nearest'  # ラベルなのでnearest neighbor
                    ).squeeze(0).squeeze(0).numpy().astype(label.dtype)  # (50, 50)
                else:
                    label_upsampled = label
                labels_upsampled.append(label_upsampled)
            
            # 各サンプルについて処理
            for b in range(distal_probs_upsampled.shape[0]):
                prob_map = distal_probs_upsampled[b]  # (50, 50)
                target = labels_upsampled[b]  # (50, 50)
                
                # 領域を抽出（重み付き中心を使用）
                regions = extract_distal_regions(
                    prob_map,
                    threshold=threshold,
                    min_region_size=min_region_size,
                    use_weighted_center=True
                )
                
                # 空間的分離
                distal_regions = classify_distal_regions(
                    regions,
                    min_separation=min_separation,
                    max_regions=max_regions
                )
                
                # メタデータをコピー（GT関節情報を含む）
                meta = metadata[b].copy()
                
                # 結果を保存
                pred_prob_maps.append(prob_map)
                target_labels.append(target)
                pred_regions_list.append(distal_regions)
                metadata_list.append(meta)
                
                sample_idx += 1
    
    # GT関節リストを構築（メタデータから取得、またはデータセットから読み込んだもの）
    eval_gt_joints_list = []
    for i, meta in enumerate(metadata_list):
        # まずメタデータから取得を試みる
        if 'gt_joints' in meta and meta['gt_joints'] is not None:
            eval_gt_joints_list.append(np.array(meta['gt_joints']))
        # 次にデータセットから読み込んだものを使用
        elif i < len(gt_joints_list) and gt_joints_list[i] is not None:
            eval_gt_joints_list.append(gt_joints_list[i])
        else:
            eval_gt_joints_list.append(None)
    
    # 評価メトリクスを計算
    metrics = evaluate_predictions(
        pred_prob_maps=pred_prob_maps,
        target_labels=target_labels,
        pred_regions_list=pred_regions_list,
        gt_joints_list=eval_gt_joints_list,
        metadata_list=metadata_list,
        threshold=threshold
    )
    
    return {
        'metrics': metrics,
        'predictions': {
            'prob_maps': [pm.tolist() for pm in pred_prob_maps],
            'regions': pred_regions_list,
            'num_samples': len(pred_prob_maps)
        }
    }


def load_gt_joints_from_dataset(data_path: str) -> List[np.ndarray]:
    """
    データセットからGT関節を読み込む
    
    Args:
        data_path: JSONLファイルのパス
    
    Returns:
        gt_joints_list: GT関節のリスト
    """
    gt_joints_list = []
    
    with open(data_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            if 'gt_joints' in sample:
                gt_joints = np.array(sample['gt_joints'])
                gt_joints_list.append(gt_joints)
            else:
                gt_joints_list.append(None)
    
    return gt_joints_list


def main():
    parser = argparse.ArgumentParser(description='Evaluate heatmap-based distal detection model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation data (JSONL)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for region extraction')
    parser.add_argument('--min_region_size', type=int, default=5,
                       help='Minimum region size (pixels)')
    parser.add_argument('--min_separation', type=float, default=0.3,
                       help='Minimum separation between regions (normalized)')
    parser.add_argument('--max_regions', type=int, default=4,
                       help='Maximum number of regions to detect')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction results (prob maps and regions)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots after evaluation')
    
    args = parser.parse_args()
    
    # デバイス
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # チェックポイントを読み込み
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']
    
    # モデルを作成
    in_channels = config.get('in_channels', None)
    model = create_model(
        model_type=config['model_type'],
        input_mode=config['input_mode'],
        spatial_bins=config['spatial_bins'],
        feature_bins=config['feature_bins'],
        num_classes=2,
        base_channels=config['base_channels'],
        in_channels=in_channels if config['model_type'] == 'multi_axis' else None
    )
    
    # モデルの重みを読み込み
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Model type: {config['model_type']}")
    print(f"  Input mode: {config['input_mode']}")
    print(f"  Spatial bins: {config['spatial_bins']}, Feature bins: {config['feature_bins']}")
    print(f"  Base channels: {config['base_channels']}")
    
    # データセットを読み込み
    print(f"\nLoading validation data from {args.val_data}...")
    val_dataset = HeatmapDistalDataset(args.val_data)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"  Validation samples: {len(val_dataset)}")
    
    # GT関節を読み込み（データセットから）
    print("Loading GT joints from dataset...")
    gt_joints_list = load_gt_joints_from_dataset(args.val_data)
    num_gt_joints = sum(1 for gj in gt_joints_list if gj is not None)
    print(f"  Loaded GT joints for {num_gt_joints} / {len(gt_joints_list)} samples")
    
    # 【Step3: 往復テスト】座標変換の往復テストを実行
    print("\nRunning roundtrip coordinate conversion test...")
    roundtrip_results = []
    for i in range(min(10, len(val_dataset))):  # 最初の10サンプルでテスト
        sample = val_dataset[i]
        meta = sample['metadata']
        
        if 'gt_joints' not in meta or meta['gt_joints'] is None:
            continue
        
        gt_joints = np.array(meta['gt_joints'])
        
        # 多軸・多特徴量モードの対応
        use_multi_axis = meta.get('use_multi_axis', False)
        if use_multi_axis:
            spatial_axis = 'x'  # 代表軸
            spatial_ranges = meta.get('spatial_ranges', {})
            spatial_range = tuple(spatial_ranges.get('x', (0.0, 1.0)))
        else:
            spatial_axis = meta.get('spatial_axis', 'x')
            spatial_range = tuple(meta.get('spatial_range', (0.0, 1.0)))
        
        if spatial_range[0] is None or spatial_range[1] is None:
            continue  # spatial_rangeが無効な場合はスキップ
        
        spatial_bins = meta.get('spatial_bins', 50)
        normalize = meta.get('normalize', True)
        normalized_range = tuple(meta.get('normalized_range', [-1.0, 1.0]))
        
        # 末端関節の1つでテスト
        DISTAL_JOINT_INDICES = [7, 8, 10, 11, 18, 19, 20, 21]
        for joint_idx in DISTAL_JOINT_INDICES[:1]:  # 最初の1つだけ
            if joint_idx >= len(gt_joints):
                continue
            spatial_value = gt_joints[joint_idx][{'x': 0, 'y': 1, 'z': 2}[spatial_axis]]
            
            roundtrip_result = test_roundtrip_coordinate_conversion(
                spatial_value=spatial_value,
                spatial_range=spatial_range,
                spatial_bins=spatial_bins,
                normalize=normalize,
                normalized_range=normalized_range
            )
            roundtrip_results.append(roundtrip_result)
            break
    
    if roundtrip_results:
        success_count = sum(1 for r in roundtrip_results if r['success'])
        avg_error = np.mean([r['error'] for r in roundtrip_results])
        max_error = np.max([r['error'] for r in roundtrip_results])
        tolerance_used = roundtrip_results[0].get('tolerance', 0.0)
        
        print(f"  Roundtrip test: {success_count}/{len(roundtrip_results)} passed")
        print(f"    Average error: {avg_error:.6f} m, Max error: {max_error:.6f} m")
        print(f"    Tolerance: {tolerance_used:.6f} m (bin_size/2 + margin)")
        
        if success_count < len(roundtrip_results):
            print(f"  Warning: Some roundtrip tests failed!")
            for r in roundtrip_results:
                if not r['success']:
                    print(f"    Error: {r['error']:.6f} m > tolerance: {r.get('tolerance', 0.0):.6f} m (original: {r['original_value']:.4f} m, converted: {r['converted_value']:.4f} m, bin: {r['bin_index']})")
    else:
        print("  No valid samples found for roundtrip test")
    
    # 評価を実行
    print("\nRunning evaluation...")
    results = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        gt_joints_list=gt_joints_list,
        threshold=args.threshold,
        min_region_size=args.min_region_size,
        min_separation=args.min_separation,
        max_regions=args.max_regions
    )
    
    # 【Step5: スモークテスト】1フレームでGTと推論結果のビンを比較
    print("\nRunning smoke test (spatial bin comparison)...")
    smoke_test_results = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if sample_idx >= 1:  # 最初の1フレームのみ
                break
            
            inputs = batch['input'].to(device)
            metadata = batch['metadata']
            
            # フォワードパス
            outputs = model(inputs)
            segmentation = outputs['segmentation']
            probs = torch.softmax(segmentation, dim=1)
            distal_probs = probs[:, 1, :, :]
            
            # アップサンプル
            if distal_probs.shape[-1] != 50 or distal_probs.shape[-2] != 50:
                distal_probs_upsampled = torch.nn.functional.interpolate(
                    distal_probs.unsqueeze(1),
                    size=(50, 50),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            else:
                distal_probs_upsampled = distal_probs
            
            prob_map = distal_probs_upsampled[0].cpu().numpy()  # (50, 50)
            meta = metadata[0]
            
            if 'gt_joints' in meta and meta['gt_joints'] is not None:
                gt_joints = np.array(meta['gt_joints'])
                
                # 多軸・多特徴量モードの対応
                use_multi_axis = meta.get('use_multi_axis', False)
                if use_multi_axis:
                    spatial_axis = 'x'  # 代表軸
                    spatial_ranges = meta.get('spatial_ranges', {})
                    spatial_range = tuple(spatial_ranges.get('x', (0.0, 1.0)))
                else:
                    spatial_axis = meta.get('spatial_axis', 'x')
                    spatial_range = tuple(meta.get('spatial_range', (0.0, 1.0)))
                
                spatial_bins = meta.get('spatial_bins', 50)
                normalize = meta.get('normalize', True)
                normalized_range = tuple(meta.get('normalized_range', [-1.0, 1.0]))
                
                smoke_result = smoke_test_spatial_bins(
                    prob_map=prob_map,
                    gt_joints=gt_joints,
                    spatial_axis=spatial_axis,
                    spatial_range=spatial_range,
                    spatial_bins=spatial_bins,
                    normalize=normalize,
                    normalized_range=normalized_range,
                    threshold=args.threshold
                )
                smoke_test_results.append(smoke_result)
                
                print(f"  Smoke test result:")
                print(f"    Mean bin diff: {smoke_result['mean_bin_diff']:.2f} bins")
                print(f"    Max bin diff: {smoke_result['max_bin_diff']:.2f} bins")
                print(f"    Matched joints: {smoke_result['matched_joints']}/{smoke_result['total_joints']}")
                if smoke_result['mean_bin_diff'] < 0.5:
                    print(f"    ✓ PASSED (mean bin diff < 0.5)")
                else:
                    print(f"    ✗ FAILED (mean bin diff >= 0.5)")
            
            sample_idx += 1
    
    # メトリクスを表示
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    metrics = results['metrics']
    
    print(f"\nPixel-level Metrics:")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"  IoU: {metrics['iou']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    if 'spatial_mean_error' in metrics:
        print(f"\nSpatial Metrics:")
        print(f"  Mean Error (1D): {metrics['spatial_mean_error']:.4f} m")
        print(f"  Median Error (1D): {metrics['spatial_median_error']:.4f} m")
        if 'mean_error_1d' in metrics and metrics.get('mean_error_1d') is not None:
            print(f"  Mean Error (1D, detailed): {metrics['mean_error_1d']:.4f} m")
        if 'mean_error_3d' in metrics and metrics.get('mean_error_3d') is not None:
            print(f"  Mean Error (3D Euclidean): {metrics['mean_error_3d']:.4f} m")
        print(f"  Within Threshold (0.25m): {metrics['spatial_within_threshold']:.4f}")
    
    print("="*80)
    
    # 結果を保存
    output_path = output_dir / 'evaluation_results.json'
    
    # JSONシリアライズ可能な形式に変換
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    save_data = {
        'metrics': convert_to_serializable(metrics),
        'config': {
            'model_path': args.model_path,
            'val_data': args.val_data,
            'threshold': args.threshold,
            'min_region_size': args.min_region_size,
            'min_separation': args.min_separation,
            'max_regions': args.max_regions,
            **config
        },
        'num_samples': len(results['predictions']['prob_maps'])
    }
    
    if args.save_predictions:
        save_data['predictions'] = {
            'regions': results['predictions']['regions'],
            'num_samples': results['predictions']['num_samples']
        }
        # 確率マップは大きいので、オプションで保存
        print("\nNote: Probability maps are not saved to reduce file size.")
        print("      Use --save_predictions to save regions only.")
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nEvaluation results saved to: {output_path}")
    
    # メトリクスのサマリーをテキストファイルに保存
    summary_path = output_dir / 'evaluation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Evaluation Results Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Validation Data: {args.val_data}\n")
        f.write(f"Number of Samples: {save_data['num_samples']}\n\n")
        
        f.write("Pixel-level Metrics:\n")
        f.write(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n")
        f.write(f"  IoU: {metrics['iou']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n\n")
        
        if 'spatial_mean_error' in metrics:
            f.write("Spatial Metrics:\n")
            f.write(f"  Mean Error (1D): {metrics['spatial_mean_error']:.4f} m\n")
            f.write(f"  Median Error (1D): {metrics['spatial_median_error']:.4f} m\n")
            if 'mean_error_1d' in metrics and metrics.get('mean_error_1d') is not None:
                f.write(f"  Mean Error (1D, detailed): {metrics['mean_error_1d']:.4f} m\n")
            if 'mean_error_3d' in metrics and metrics.get('mean_error_3d') is not None:
                f.write(f"  Mean Error (3D Euclidean): {metrics['mean_error_3d']:.4f} m\n")
            f.write(f"  Within Threshold (0.25m): {metrics['spatial_within_threshold']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Evaluation summary saved to: {summary_path}")
    
    # 可視化を生成（オプション）
    if args.visualize:
        print("\nGenerating visualizations...")
        try:
            from heatmap_distal_detection.visualize_evaluation_results import (
                load_evaluation_results,
                visualize_error_distribution,
                visualize_metrics_comparison
            )
            
            visualize_error_distribution(
                metrics,
                str(output_dir / 'error_distribution.png')
            )
            
            visualize_metrics_comparison(
                metrics,
                str(output_dir / 'metrics_comparison.png')
            )
            
            print(f"Visualizations saved to: {output_dir}")
        except Exception as e:
            print(f"Warning: Failed to generate visualizations: {e}")


if __name__ == '__main__':
    main()

