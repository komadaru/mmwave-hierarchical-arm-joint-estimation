#!/usr/bin/env python3
"""
ヒートマップデータセット
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from pathlib import Path


class HeatmapDistalDataset(Dataset):
    """
    グリッドヒートマップベース末端部位検出用データセット
    """
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None
    ):
        """
        Args:
            data_path: JSONL形式のデータファイルパス
            transform: データ拡張（オプション）
        """
        self.data_path = data_path
        self.transform = transform
        
        # データを読み込み
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        データサンプルを取得
        
        Returns:
            sample: Dict
                - input: (C, H, W) - モデル入力テンソル
                - label: (H, W) - ラベル（0 or 1）
                - metadata: Dict - メタデータ
        """
        sample = self.data[idx]
        
        # 入力テンソルを取得
        input_tensor = np.array(sample['input_tensor'], dtype=np.float32)  # (C, H, W)
        
        # ラベルを取得
        labels = np.array(sample['labels'], dtype=np.int64)  # (H, W)
        
        # NumPy配列をTorchテンソルに変換
        input_tensor = torch.from_numpy(input_tensor)
        labels = torch.from_numpy(labels)
        
        # ラベルを50x50にリサイズ（モデルの出力サイズに合わせる）
        # 注意: 多軸・多特徴量モデルの出力が50x50になるため、ラベルも50x50にリサイズ
        # 既存のデータが48x48の場合は50x50にアップサンプル
        if labels.shape != (50, 50):
            labels = labels.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            labels = torch.nn.functional.interpolate(
                labels.float(),
                size=(50, 50),
                mode='nearest'
            ).squeeze(0).squeeze(0).long()  # (50, 50)
        
        # データ拡張（オプション）
        if self.transform is not None:
            input_tensor, labels = self.transform(input_tensor, labels)
        
        # 多軸・多特徴量モードかどうかを判定
        use_multi_axis = sample.get('use_multi_axis', False)
        
        metadata = {
            'frame_id': sample.get('frame_id', ''),
            'sequence_id': sample.get('sequence_id', ''),
            'frame_num': sample.get('frame_num', -1),
            'input_mode': sample.get('input_mode', 'diff'),
            'spatial_bins': sample.get('spatial_bins', 50),
            'feature_bins': sample.get('feature_bins', 50),
            'normalize': sample.get('normalize', True),
            'normalized_range': sample.get('normalized_range', [-1.0, 1.0]),
            'use_multi_axis': use_multi_axis
        }
        
        # 多軸・多特徴量モードの場合
        if use_multi_axis:
            metadata['spatial_ranges'] = sample.get('spatial_ranges', None)
            metadata['feature_ranges'] = sample.get('feature_ranges', None)
            metadata['spatial_ranges_norm'] = sample.get('spatial_ranges_norm', None)
            metadata['feature_ranges_norm'] = sample.get('feature_ranges_norm', None)
        else:
            # レガシーモード
            metadata['spatial_axis'] = sample.get('spatial_axis', 'x')
            metadata['feature_axis'] = sample.get('feature_axis', 'energy_power')
            metadata['spatial_range'] = sample.get('spatial_range', None)
            metadata['feature_range'] = sample.get('feature_range', None)
            metadata['spatial_range_norm'] = sample.get('spatial_range_norm', None)
            metadata['feature_range_norm'] = sample.get('feature_range_norm', None)
        
        # GT関節がある場合は追加
        if 'gt_joints' in sample:
            metadata['gt_joints'] = sample['gt_joints']
        
        return {
            'input': input_tensor,
            'label': labels,
            'metadata': metadata
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    バッチを結合
    
    Args:
        batch: データサンプルのリスト
    
    Returns:
        batched: Dict
            - input: (B, C, H, W)
            - label: (B, H, W)
            - metadata: List[Dict]
    """
    inputs = torch.stack([item['input'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    metadata = [item['metadata'] for item in batch]
    
    return {
        'input': inputs,
        'label': labels,
        'metadata': metadata
    }

