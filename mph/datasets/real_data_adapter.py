"""
実データ用のアダプター
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class RealDataAdapter:
    """
    実データ用のアダプター
    """
    
    def __init__(self, joints_def_path: str = None):
        self.joints_def_path = joints_def_path
        self.joint_names = None
        self.bones = None
        self.standard_bone_lengths = None
        self._load_joint_definitions()
    
    def _load_joint_definitions(self):
        """関節定義を読み込み"""
        # 複数のパスを試す（クラスタ環境も含む）
        possible_paths = [
            self.joints_def_path,
            "/home/users/grad/2024/24t0010/joints_def_22.json",  # クラスタ環境のパス（最優先）
            "data_specs/joints_def_22.json",
            "../data_specs/joints_def_22.json",
            "../../data_specs/joints_def_22.json",
            "/cluster/users/grad/2024/24t0010/mmwave_pose_reconstraction_pj/data_specs/joints_def_22.json",
            "/home/users/grad/2024/24t0010/mmwave_pose_reconstraction_pj/data_specs/joints_def_22.json",
            "/mnt/c/Users/kodam/PycharmProjects/M1Lab/mmwave_pose_reconstraction_pj/data_specs/joints_def_22.json"
        ]
        
        joints_def = None
        for path in possible_paths:
            if path and os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        joints_def = json.load(f)
                    print(f"Loaded joint definitions from: {path}")
                    break
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
        
        if joints_def:
            self.joint_names = joints_def['joint_names']
            self.bones = joints_def['bones']
            self.standard_bone_lengths = joints_def['standard_bone_length_m']
            
            print(f"  Joint names: {len(self.joint_names)} joints")
            print(f"  Bones: {len(self.bones)} bone pairs")
            print(f"  Standard bone lengths: {len(self.standard_bone_lengths)} pairs")
        else:
            print("Warning: joints_def_22.json not found, using default definitions")
            self._set_default_definitions()
    
    def _set_default_definitions(self):
        """デフォルトの関節定義を設定"""
        self.joint_names = [
            "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
            "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
            "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
            "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
        ]
        self.bones = [
            [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [0, 1], [0, 2],
            [1, 4], [4, 7], [7, 10], [2, 5], [5, 8], [8, 11], [9, 13],
            [13, 16], [16, 18], [18, 20], [9, 14], [14, 17], [17, 19], [19, 21]
        ]
        self.standard_bone_lengths = {
            "0-3": 0.15, "3-6": 0.15, "6-9": 0.20, "9-12": 0.15, "12-15": 0.10,
            "0-1": 0.20, "0-2": 0.20, "1-4": 0.45, "4-7": 0.40, "7-10": 0.25,
            "2-5": 0.45, "5-8": 0.40, "8-11": 0.25, "9-13": 0.15, "13-16": 0.20,
            "16-18": 0.30, "18-20": 0.25, "9-14": 0.15, "14-17": 0.20,
            "17-19": 0.30, "19-21": 0.25
        }
    
    def map_24_to_22_joints(self, joints_24: np.ndarray) -> np.ndarray:
        """
        24関節から22関節にマッピング
        """
        # 24関節から22関節へのマッピング（手首を手として使用）
        mapping_indices = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,  # 既存の22関節
            20,  # L_Wrist -> left_hand (手首を手として使用)
            21,  # R_Wrist -> right_hand (手首を手として使用)
        ]
        
        joints_22 = joints_24[mapping_indices]
        return joints_22
    
    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        点群データの正規化
        """
        # 中心化
        center = points[:, :3].mean(axis=0)
        points_centered = points.copy()
        points_centered[:, :3] = points[:, :3] - center
        
        # スケーリング（最大範囲で正規化）
        max_range = np.linalg.norm(points_centered[:, :3], axis=1).max()
        if max_range > 0:
            points_centered[:, :3] = points_centered[:, :3] / (max_range / 2)
        
        return points_centered
    
    def compute_bone_lengths(self, joints: np.ndarray) -> np.ndarray:
        """
        骨長を計算
        """
        bone_lengths = []
        for i, j in self.bones:
            if i < len(joints) and j < len(joints):
                bone_vec = joints[j] - joints[i]
                bone_length = np.linalg.norm(bone_vec)
                bone_lengths.append(bone_length)
        
        return np.array(bone_lengths)
    
    def compute_visibility(self, joints: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        可視性を計算（関節の位置に基づく簡易版）
        """
        # 関節が原点に近すぎる場合は非可視とする
        distances = np.linalg.norm(joints, axis=1)
        visibility = (distances > threshold).astype(np.float32)
        return visibility


class PCMDatasetLoader(Dataset):
    """
    PCMデータセット用のDataLoader
    """
    
    def __init__(self, data_root: str, split: str = 'train', 
                 adapter: RealDataAdapter = None, 
                 voxel_size: Tuple[int, int, int] = (32, 32, 32),
                 normalize: bool = True):
        self.data_root = data_root
        self.split = split
        self.voxel_size = voxel_size
        self.normalize = normalize
        self.adapter = adapter or RealDataAdapter()
        
        # データファイルのパスを取得
        self.data_files = self._get_data_files()
        
        print(f"Loaded {len(self.data_files)} files for {split} split")
    
    def _get_data_files(self) -> List[str]:
        """データファイルのパスを取得"""
        split_dir = os.path.join(self.data_root, self.split)
        
        # データディレクトリが見つからない場合、代替パスを試行
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory not found: {split_dir}")
            print("Trying alternative data paths...")
            
            # データセットディレクトリの存在確認とディレクトリ構造の確認
            print(f"Checking data_root: {self.data_root}")
            print(f"Path exists: {os.path.exists(self.data_root)}")
            print(f"Path is directory: {os.path.isdir(self.data_root) if os.path.exists(self.data_root) else 'N/A'}")
            
            if os.path.exists(self.data_root):
                print(f"Data root exists. Contents:")
                try:
                    contents = os.listdir(self.data_root)
                    print(f"  Total items: {len(contents)}")
                    print(f"  Items (first 20): {contents[:20]}")
                    
                    # train/val/testディレクトリを探す（大文字小文字を区別しない）
                    dirs = [d for d in contents if os.path.isdir(os.path.join(self.data_root, d))]
                    print(f"  Directories found: {dirs}")
                    
                    # split名にマッチするディレクトリを探す（大文字小文字を区別しない）
                    matching_dirs = [d for d in dirs if self.split.lower() in d.lower()]
                    if matching_dirs:
                        print(f"  Directories matching '{self.split}': {matching_dirs}")
                        split_dir = os.path.join(self.data_root, matching_dirs[0])
                        print(f"  Using directory: {split_dir}")
                        if os.path.exists(split_dir):
                            files = [f for f in os.listdir(split_dir) if f.endswith('.npz')]
                            if files:
                                print(f"  Found {len(files)} .npz files")
                                # 時系列順にソート（シーケンスID → フレーム番号の順）
                                try:
                                    from mph.postprocessing.tracking_utils import extract_sequence_and_frame
                                except ImportError:
                                    # フォールバック: ファイル名でソート
                                    files.sort()
                                    return [os.path.join(split_dir, f) for f in files]
                                files_with_info = []
                                for f in files:
                                    full_path = os.path.join(split_dir, f)
                                    seq_id, frame_num = extract_sequence_and_frame(full_path)
                                    # ソート用のキーを作成
                                    if seq_id and frame_num is not None:
                                        sort_key = (seq_id, frame_num)
                                    else:
                                        # 抽出失敗時はファイル名でソート
                                        sort_key = (f, 0)
                                    files_with_info.append((sort_key, full_path))
                                files_with_info.sort(key=lambda x: x[0])
                                files = [f[1] for f in files_with_info]
                                print(f"  Sorted {len(files)} files by sequence_id and frame_number")
                                return files
                    
                    # 直接data_rootにnpzファイルがある場合（train/val/testディレクトリがない構造）
                    npz_files = [f for f in contents if f.endswith('.npz')]
                    if npz_files:
                        print(f"Found {len(npz_files)} .npz files directly in {self.data_root}")
                        # split名でフィルタリング（例: train, val, test）
                        files = [f for f in npz_files if self.split.lower() in f.lower()]
                        if files:
                            print(f"Found {len(files)} files matching split '{self.split}'")
                            # 時系列順にソート
                            try:
                                from mph.postprocessing.tracking_utils import extract_sequence_and_frame
                            except ImportError:
                                # フォールバック: ファイル名でソート
                                files.sort()
                                return [os.path.join(self.data_root, f) for f in files]
                            files_with_info = []
                            for f in files:
                                full_path = os.path.join(self.data_root, f)
                                seq_id, frame_num = extract_sequence_and_frame(full_path)
                                if seq_id and frame_num is not None:
                                    sort_key = (seq_id, frame_num)
                                else:
                                    sort_key = (f, 0)
                                files_with_info.append((sort_key, full_path))
                            files_with_info.sort(key=lambda x: x[0])
                            files = [f[1] for f in files_with_info]
                            print(f"  Sorted {len(files)} files by sequence_id and frame_number")
                            return files
                        else:
                            print(f"Warning: No files match split '{self.split}'")
                            print(f"  Available files (first 10): {npz_files[:10]}")
                except PermissionError as e:
                    print(f"Permission error accessing directory: {e}")
                except Exception as e:
                    print(f"Error listing directory: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"ERROR: Data root does not exist: {self.data_root}")
            
            # 代替パスの候補をより広く検索
            base_dirs = [
                "/cluster/users/grad/2024/24t0010",
                "/home/users/grad/2024/24t0010",  # ログに表示されているパス
                "/home/24t0010",
                "/mnt/d",
            ]
            
            alternative_paths = []
            print("Searching for dataset in alternative locations...")
            for base in base_dirs:
                print(f"  Checking base: {base}")
                if not os.path.exists(base):
                    print(f"    Base directory does not exist")
                    continue
                
                # pcm_dataset_test_v2 を試行
                dataset_dir = os.path.join(base, "pcm_dataset_test_v2")
                print(f"    Checking: {dataset_dir}")
                if os.path.exists(dataset_dir):
                    print(f"      Dataset directory exists!")
                    # splitサブディレクトリを確認
                    alt_path = os.path.join(dataset_dir, self.split)
                    if os.path.exists(alt_path):
                        print(f"      Split directory '{self.split}' found!")
                        alternative_paths.append(alt_path)
                    else:
                        print(f"      Split directory '{self.split}' not found")
                        # ディレクトリ内の内容を確認
                        try:
                            contents = os.listdir(dataset_dir)
                            dirs = [d for d in contents if os.path.isdir(os.path.join(dataset_dir, d))]
                            print(f"      Available subdirectories: {dirs}")
                            # 類似名を探す（大文字小文字を区別しない）
                            matching_dirs = [d for d in dirs if self.split.lower() in d.lower()]
                            if matching_dirs:
                                print(f"      Found similar directory: {matching_dirs[0]}")
                                alternative_paths.append(os.path.join(dataset_dir, matching_dirs[0]))
                        except Exception as e:
                            print(f"      Error listing directory: {e}")
                
                # pcm_dataset_test を試行
                dataset_dir = os.path.join(base, "pcm_dataset_test")
                print(f"    Checking: {dataset_dir}")
                if os.path.exists(dataset_dir):
                    print(f"      Dataset directory exists!")
                    alt_path = os.path.join(dataset_dir, self.split)
                    if os.path.exists(alt_path):
                        print(f"      Split directory '{self.split}' found!")
                        alternative_paths.append(alt_path)
            
            if alternative_paths:
                split_dir = alternative_paths[0]
                print(f"✅ Found alternative path: {split_dir}")
                # data_rootも更新
                self.data_root = os.path.dirname(split_dir)
                print(f"Updated data_root to: {self.data_root}")
            else:
                # 最後の試み: ユーザーのホームディレクトリを確認
                home_dir = os.path.expanduser("~")
                print(f"Trying user home directory: {home_dir}")
                for dataset_name in ["pcm_dataset_test_v2", "pcm_dataset_test"]:
                    dataset_dir = os.path.join(home_dir, dataset_name)
                    if os.path.exists(dataset_dir):
                        alt_path = os.path.join(dataset_dir, self.split)
                        if os.path.exists(alt_path):
                            print(f"✅ Found dataset in home directory: {alt_path}")
                            alternative_paths.append(alt_path)
                            split_dir = alt_path
                            self.data_root = dataset_dir
                            break
                
                if not alternative_paths:
                    raise FileNotFoundError(
                        f"Split directory not found: {split_dir}\n"
                        f"Original data root: {self.data_root}\n"
                        f"Tried alternative paths but none found.\n"
                        f"Please check if the dataset is properly located.\n"
                        f"Expected structure: {{data_root}}/{{train,val,test}}/\n"
                        f"\n"
                        f"To check available directories, run:\n"
                        f"  ls -la /cluster/users/grad/2024/24t0010/ | grep pcm\n"
                        f"  find /cluster/users/grad/2024/24t0010 -type d -name '*pcm*' 2>/dev/null"
                    )
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        files = [f for f in os.listdir(split_dir) if f.endswith('.npz')]
        if len(files) == 0:
            raise FileNotFoundError(
                f"No .npz files found in {split_dir}\n"
                f"Directory contents: {os.listdir(split_dir)[:10]}"
            )
        
        # 時系列順にソート（シーケンスID → フレーム番号の順）
        try:
            from mph.postprocessing.tracking_utils import extract_sequence_and_frame
        except ImportError:
            # フォールバック: ファイル名でソート
            files.sort()
            return [os.path.join(split_dir, f) for f in files]
        files_with_info = []
        for f in files:
            full_path = os.path.join(split_dir, f)
            seq_id, frame_num = extract_sequence_and_frame(full_path)
            if seq_id and frame_num is not None:
                sort_key = (seq_id, frame_num)
            else:
                # 抽出失敗時はファイル名でソート
                sort_key = (f, 0)
            files_with_info.append((sort_key, full_path))
        files_with_info.sort(key=lambda x: x[0])
        files = [f[1] for f in files_with_info]
        print(f"  Sorted {len(files)} files by sequence_id and frame_number")
        
        return files
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """データサンプルを取得"""
        file_path = self.data_files[idx]
        
        try:
            # データを読み込み
            data = np.load(file_path)
            
            # 点群データ
            points = data['points']  # (N, 6)
            if self.normalize:
                points = self.adapter.normalize_points(points)
            
            # 関節データ（24関節から22関節にマッピング）
            joints_24 = data['joints_3d']  # (55, 3) -> 最初の24関節を使用
            joints_22 = joints_24[:22]  # 最初の22関節を使用
            
            # 可視性
            visibility_24 = data['visibility']  # (24,)
            visibility_22 = visibility_24[:22]  # 最初の22関節の可視性
            
            # 骨長を計算
            bone_lengths = self.adapter.compute_bone_lengths(joints_22)
            
            # 点群特徴量を抽出
            # 順序: [x, y, z, velocity, amplitude, energy_power]
            point_coords = points[:, :3]  # (N, 3) - x, y, z
            point_features = points[:, 3:]  # (N, 3) - velocity, amplitude, energy_power
            
            # ボクセル化用の特徴量を準備
            voxel_features = self._prepare_voxel_features(point_coords, point_features)
            
            # 点群データを固定長にパディング（バッチ化のため）
            max_points = 1000  # 最大点群数
            if len(points) > max_points:
                # ランダムサンプリング
                indices = np.random.choice(len(points), max_points, replace=False)
                points = points[indices]
            elif len(points) < max_points:
                # パディング
                padding = np.zeros((max_points - len(points), 6))
                points = np.vstack([points, padding])
            
            return {
                'voxel_features': voxel_features,
                'gt_joints': torch.from_numpy(joints_22).float(),
                'visibility': torch.from_numpy(visibility_22).float(),
                'bone_lengths': torch.from_numpy(bone_lengths).float(),
                'points': torch.from_numpy(points).float(),
                'file_path': file_path
            }
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # エラー時はダミーデータを返す
            return self._get_dummy_sample()
    
    def _prepare_voxel_features(self, coords: np.ndarray, features: np.ndarray) -> torch.Tensor:
        """
        ボクセル化用の特徴量を準備
        """
        # 簡易的なボクセル化（実際の実装では距離適応スプラットを使用）
        D, H, W = self.voxel_size
        
        # ボクセル特徴量を初期化（6次元特徴量）
        voxel_features = torch.zeros(6, D, H, W)
        
        # 座標をボクセルインデックスに変換
        coords_norm = (coords + 1) * np.array([W-1, H-1, D-1]) / 2
        coords_norm = np.clip(coords_norm, 0, [W-1, H-1, D-1])
        
        # 各点をボクセルに配置
        for i in range(len(coords)):
            x, y, z = coords_norm[i].astype(int)
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                # 特徴量を配置
                # 順序: [velocity, amplitude, energy_power]
                if len(features[i]) >= 3:
                    voxel_features[0, z, y, x] = features[i, 1]  # amplitude (次元4)
                    voxel_features[1, z, y, x] = features[i, 0]  # velocity (次元3)
                    voxel_features[2, z, y, x] = features[i, 2]  # energy_power (次元5)
                else:
                    voxel_features[0, z, y, x] = 1.0  # default amplitude
                    voxel_features[1, z, y, x] = 0.0  # default velocity
                    voxel_features[2, z, y, x] = 100.0  # default energy_power
                
                voxel_features[3, z, y, x] = 1.0  # occupancy
                voxel_features[4, z, y, x] = coords[i, 0]  # x coordinate
                voxel_features[5, z, y, x] = coords[i, 1]  # y coordinate
        
        return voxel_features
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """ダミーサンプルを返す"""
        D, H, W = self.voxel_size
        return {
            'voxel_features': torch.randn(6, D, H, W),
            'gt_joints': torch.randn(22, 3),
            'visibility': torch.ones(22),
            'bone_lengths': torch.rand(21) * 0.5,
            'points': torch.randn(1000, 6),  # 固定長に変更
            'file_path': 'dummy'
        }


def create_real_data_dataloader(data_root: str, split: str = 'train',
                               batch_size: int = 4, num_workers: int = 4,
                               shuffle: bool = True) -> DataLoader:
    """
    実データ用のDataLoaderを作成
    """
    dataset = PCMDatasetLoader(data_root, split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
