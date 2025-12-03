#!/usr/bin/env python3
"""
Tracking Layer用のユーティリティ関数
"""

import re
from typing import Tuple, Optional
from pathlib import Path


def extract_sequence_and_frame(file_path: str) -> Tuple[Optional[str], Optional[int]]:
    """
    ファイルパスからシーケンスIDとフレーム番号を抽出
    
    Args:
        file_path: ファイルパス（例: "sequence_0_frame_1000_pcm_data.npz"）
    
    Returns:
        (sequence_id, frame_number): シーケンスIDとフレーム番号のタプル
    """
    # ファイル名を取得
    filename = Path(file_path).name
    
    # パターン1: sequence_X_frame_Y_pcm_data.npz
    pattern1 = r'sequence_(\d+)_frame_(\d+)_pcm_data'
    match1 = re.search(pattern1, filename)
    if match1:
        sequence_id = f"sequence_{match1.group(1)}"
        frame_number = int(match1.group(2))
        return sequence_id, frame_number
    
    # パターン2: sequence_X_frame_Y.npz
    pattern2 = r'sequence_(\d+)_frame_(\d+)'
    match2 = re.search(pattern2, filename)
    if match2:
        sequence_id = f"sequence_{match2.group(1)}"
        frame_number = int(match2.group(2))
        return sequence_id, frame_number
    
    # パターン3: パスにsequence_Xが含まれる場合
    if 'sequence_' in file_path:
        seq_match = re.search(r'sequence_(\d+)', file_path)
        if seq_match:
            sequence_id = f"sequence_{seq_match.group(1)}"
            frame_match = re.search(r'frame_(\d+)', filename)
            if frame_match:
                frame_number = int(frame_match.group(1))
                return sequence_id, frame_number
    
    return None, None

