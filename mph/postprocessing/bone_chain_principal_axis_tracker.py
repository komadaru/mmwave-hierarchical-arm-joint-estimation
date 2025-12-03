#!/usr/bin/env python3
"""
骨チェーン主軸ベース追跡モジュール

個々の点を前フレームとマッチングし、新規点を検出。
新規点を固定された部位（右腕・左腕・右脚・左脚）ごとにクラスタリングし、
各クラスタの主軸（PCAの第1主成分）を計算して、キーポイントを配置する。
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional
import json
import os

# 末端部位検出モデル用のインポート
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available, distal detection model cannot be loaded")

# 信頼性計算関数をインポート（または定義）
try:
    from .motion_gated_tracking import compute_point_reliability
except ImportError:
    # インポートできない場合は定義
    def compute_point_reliability(
        energy_power: np.ndarray,
        velocity: np.ndarray,
        energy_threshold: float = 100.0,
        velocity_threshold: float = 0.1
    ) -> np.ndarray:
        """点群の信頼性を計算"""
        low_energy = energy_power < energy_threshold
        low_velocity = np.abs(velocity) < velocity_threshold
        reliability = np.ones(len(energy_power), dtype=np.float32)
        both_low = low_energy & low_velocity
        reliability[both_low] = 0.4
        return reliability

# Phase 1-3モジュールをインポート（オプション）
try:
    from .robust_centerline_extraction import (
        RobustPreprocessor,
        TorsoModeler,
        AttributeBasedClusterer,
        IRLSPCA,  # Phase 2
        CenterlineExtractor,  # Phase 3
        TemporalIntegration,  # Phase 3
        TemporalStateManager  # Phase 3
    )
    ROBUST_CENTERLINE_AVAILABLE = True
except ImportError:
    ROBUST_CENTERLINE_AVAILABLE = False
    IRLSPCA = None
    CenterlineExtractor = None
    TemporalIntegration = None
    TemporalStateManager = None


class BoneChainPrincipalAxisTracker:
    """
    骨チェーンの主軸に基づいてキーポイントを配置
    joints_def_22.jsonのbones定義から、右腕・左腕・右脚・左脚を固定で定義
    """
    
    def __init__(
        self,
        joints_def_path: Optional[str] = None,
        max_match_distance: float = 0.15,
        feature_similarity_weight: float = 0.3,
        position_weight: float = 0.7,
        min_new_neighbors: int = 2,
        new_point_radius: float = 0.15,
        bone_chain_radius: float = 0.30,
        use_motion_reliability: bool = True,  # 動きの信頼性計算を使用するか
        energy_threshold: float = 100.0,  # energy_powerが低いと判定する閾値
        velocity_threshold: float = 0.1,  # velocityが低いと判定する閾値
        min_reliability_for_principal_axis: float = 0.3,  # 主軸計算に使用する最小信頼性（0.3 = 動いている部位の点も含む）
        use_robust_centerline: bool = False,  # Phase 1のロバスト中心線抽出を使用するか（デフォルトFalse）
        use_irls_pca: bool = False,  # Phase 2: IRLSロバストPCAを使用するか（デフォルトFalse）
        use_centerline_extraction: bool = False,  # Phase 3: 方向場→中心線抽出を使用するか（デフォルトFalse）
        use_temporal_integration: bool = False,  # Phase 3: 時間統合を使用するか（デフォルトFalse）
        use_hmm_state: bool = False,  # Phase 3: HMM/FSM状態管理を使用するか（デフォルトFalse）
        robust_centerline_params: Optional[Dict] = None,  # Phase 1-3モジュールのパラメータ
        use_distal_detection_model: bool = False,  # 末端部位検出モデルを使用するか（デフォルトFalse）
        distal_model_path: Optional[str] = None,  # 末端部位検出モデルのパス（.pkl for RF）
        distal_grid_size: float = 0.3,  # グリッドセルのサイズ（m）
        distal_confidence_threshold: float = 0.5  # 信頼度閾値（0.0-1.0）
    ):
        """
        Args:
            joints_def_path: 関節定義ファイルのパス（Noneの場合はデフォルトパスを試行）
            max_match_distance: 最大マッチング距離（m）
            feature_similarity_weight: 特徴量類似度の重み
            position_weight: 位置の重み
            min_new_neighbors: 新規点の最小近傍点数
            new_point_radius: 新規点の近傍検索半径（m）
            bone_chain_radius: 骨チェーン付近の検索半径（m）
            use_motion_reliability: 動きの信頼性計算を使用するか（デフォルトTrue）
            energy_threshold: energy_powerが低いと判定する閾値（デフォルト100.0）
            velocity_threshold: velocityが低いと判定する閾値（デフォルト0.1）
            min_reliability_for_principal_axis: 主軸計算に使用する最小信頼性（デフォルト0.5）
        """
        self.max_match_distance = max_match_distance
        self.feature_similarity_weight = feature_similarity_weight
        self.position_weight = position_weight
        self.min_new_neighbors = min_new_neighbors
        self.new_point_radius = new_point_radius
        self.bone_chain_radius = bone_chain_radius
        
        # 関節定義を読み込み
        if joints_def_path is None:
            # デフォルトパスを試行
            possible_paths = [
                '../data_specs/joints_def_22.json',
                'data_specs/joints_def_22.json',
                os.path.join(os.path.dirname(__file__), '../../data_specs/joints_def_22.json'),
                os.path.join(os.path.dirname(__file__), '../../../data_specs/joints_def_22.json'),
            ]
            
            joints_def_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    joints_def_path = path
                    break
            
            if joints_def_path is None:
                raise FileNotFoundError(
                    "joints_def_22.json not found. Please specify joints_def_path explicitly."
                )
        
        with open(joints_def_path, 'r') as f:
            self.joints_def = json.load(f)
        
        # 固定された部位定義（joints_def_22.jsonのbones定義から抽出）
        # 関節名: ["Pelvis","L_Hip","R_Hip","Spine1","L_Knee","R_Knee","Spine2",
        #          "L_Ankle","R_Ankle","Spine3","L_Foot","R_Foot","Neck",
        #          "L_Collar","R_Collar","Head","L_Shoulder","R_Shoulder",
        #          "L_Elbow","R_Elbow","L_Wrist","R_Wrist"]
        self.body_parts = {
            'right_leg': [0, 2, 5, 8, 11],      # Pelvis -> R_Hip -> R_Knee -> R_Ankle -> R_Foot
            'left_leg': [0, 1, 4, 7, 10],       # Pelvis -> L_Hip -> L_Knee -> L_Ankle -> L_Foot
            'right_arm': [9, 14, 17, 19, 21],   # Spine3 -> R_Collar -> R_Shoulder -> R_Elbow -> R_Wrist
            'left_arm': [9, 13, 16, 18, 20]     # Spine3 -> L_Collar -> L_Shoulder -> L_Elbow -> L_Wrist
        }
        
        # 各部位の起点（胴体に近い関節）
        self.body_part_anchors = {
            'right_leg': 0,   # Pelvis
            'left_leg': 0,    # Pelvis
            'right_arm': 9,   # Spine3
            'left_arm': 9     # Spine3
        }
        
        # 前フレームの点群を保存（次フレーム用）
        self.prev_points = None
        self.prev_features = None
        
        # 動きの信頼性計算パラメータ
        self.use_motion_reliability = use_motion_reliability
        self.energy_threshold = energy_threshold
        self.velocity_threshold = velocity_threshold
        self.min_reliability_for_principal_axis = min_reliability_for_principal_axis
        
        # 末端部位検出モデルのパラメータ
        self.use_distal_detection_model = use_distal_detection_model
        self.distal_model = None
        self.distal_grid_size = distal_grid_size
        self.distal_confidence_threshold = distal_confidence_threshold
        
        # 末端部位検出モデルを読み込み
        if self.use_distal_detection_model:
            if distal_model_path is None:
                raise ValueError("distal_model_path must be specified when use_distal_detection_model=True")
            
            if not JOBLIB_AVAILABLE:
                raise ImportError("joblib is required for distal detection model. Install with: pip install joblib")
            
            if not os.path.exists(distal_model_path):
                raise FileNotFoundError(f"Distal detection model not found: {distal_model_path}")
            
            try:
                self.distal_model = joblib.load(distal_model_path)
                print(f"Loaded distal detection model from: {distal_model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load distal detection model: {e}")
        
        # Phase 1: ロバスト中心線抽出（オプション）
        self.use_robust_centerline = use_robust_centerline and ROBUST_CENTERLINE_AVAILABLE
        
        # Phase 2: IRLSロバストPCA（Phase 1が有効な場合のみ）
        self.use_irls_pca = use_irls_pca and self.use_robust_centerline
        
        # Phase 3: 高度な機能（Phase 1が有効な場合のみ）
        self.use_centerline_extraction = use_centerline_extraction and self.use_robust_centerline
        self.use_temporal_integration = use_temporal_integration and self.use_robust_centerline
        self.use_hmm_state = use_hmm_state and self.use_robust_centerline
        
        if self.use_robust_centerline:
            # パラメータのデフォルト値
            robust_params = robust_centerline_params or {}
            preprocessor_params = robust_params.get('preprocessor', {})
            torso_params = robust_params.get('torso', {})
            clusterer_params = robust_params.get('clusterer', {})
            
            # Phase 1モジュールを初期化
            self.robust_preprocessor = RobustPreprocessor(
                use_torso_relative=preprocessor_params.get('use_torso_relative', True),
                k_nn=preprocessor_params.get('k_nn', 16),
                time_window_sec=preprocessor_params.get('time_window_sec', None) if self.use_temporal_integration else None
            )
            self.torso_modeler = TorsoModeler(
                ransac_threshold=torso_params.get('ransac_threshold', 0.015),
                min_inliers=torso_params.get('min_inliers', 7),
                halo_delta=torso_params.get('halo_delta', 0.06)
            )
            self.attribute_clusterer = AttributeBasedClusterer(
                alpha_amp=clusterer_params.get('alpha_amp', 1.0),
                beta_power=clusterer_params.get('beta_power', 1.2),
                gamma_vel=clusterer_params.get('gamma_vel', 0.8),
                eta_density=clusterer_params.get('eta_density', 0.5),
                alpha_apr=clusterer_params.get('alpha_apr', 0.8),
                k_lambda=clusterer_params.get('k_lambda', 0.7),
                cluster_method=clusterer_params.get('cluster_method', 'dbscan'),
                min_cluster_size=clusterer_params.get('min_cluster_size', 5),
                min_samples=clusterer_params.get('min_samples', 3),
                eps=clusterer_params.get('eps', 0.15),
                sigma_x=clusterer_params.get('sigma_x', 0.05)
            )
            
            # Phase 2: IRLSロバストPCAモジュール
            if self.use_irls_pca and IRLSPCA is not None:
                irls_params = robust_params.get('irls_pca', {})
                self.irls_pca_module = IRLSPCA(
                    max_iterations=irls_params.get('max_iterations', 20),
                    convergence_threshold=irls_params.get('convergence_threshold', 1e-4),
                    robust_weight_threshold=irls_params.get('robust_weight_threshold', 0.1),
                    weight_function=irls_params.get('weight_function', 'huber')
                )
            else:
                self.irls_pca_module = None
            
            # Phase 3: 方向場→中心線抽出モジュール
            if self.use_centerline_extraction and CenterlineExtractor is not None:
                centerline_params = robust_params.get('centerline_extractor', {})
                self.centerline_extractor = CenterlineExtractor(
                    k_pca=centerline_params.get('k_pca', 20),
                    irls_iterations=centerline_params.get('irls_iterations', 4),
                    sigma_x=centerline_params.get('sigma_x', 0.05),
                    sigma_theta=centerline_params.get('sigma_theta', 0.12),
                    bspline_smooth=centerline_params.get('bspline_smooth', 0.1),
                    k_nn=centerline_params.get('k_nn', 15)
                )
            else:
                self.centerline_extractor = None
            
            # Phase 3: 時間統合機能
            if self.use_temporal_integration and TemporalIntegration is not None:
                temporal_params = robust_params.get('temporal_integration', {})
                self.temporal_integrator = TemporalIntegration(
                    time_window_sec=preprocessor_params.get('time_window_sec', 0.4),
                    fps=temporal_params.get('fps', 15.0),
                    decay_factor=temporal_params.get('decay_factor', 0.9)
                )
            else:
                self.temporal_integrator = None
            
            # Phase 3: HMM/FSM状態管理モジュール
            if self.use_hmm_state and TemporalStateManager is not None:
                hmm_params = robust_params.get('hmm_state', {})
                self.state_manager = TemporalStateManager(
                    active_threshold=hmm_params.get('active_threshold', 0.6),
                    deactive_threshold=hmm_params.get('deactive_threshold', 0.2),
                    hold_seconds=hmm_params.get('hold_seconds', 1.2),
                    fps=hmm_params.get('fps', 15.0)
                )
            else:
                self.state_manager = None
        else:
            self.robust_preprocessor = None
            self.torso_modeler = None
            self.attribute_clusterer = None
            self.irls_pca_module = None
            self.centerline_extractor = None
            self.temporal_integrator = None
            self.state_manager = None
    
    def match_points_and_detect_new(
        self,
        curr_points: np.ndarray,  # (N, 6)
        curr_features: np.ndarray,  # (N, 3)
        prev_points: Optional[np.ndarray] = None,  # (M, 6)
        prev_features: Optional[np.ndarray] = None  # (M, 3)
    ) -> np.ndarray:
        """
        点マッチングを行い、新規点を検出
        
        Args:
            curr_points: (N, 6) - 現在フレームの点群 [x, y, z, velocity, amplitude, energy_power]
            curr_features: (N, 3) - 現在フレームの特徴量 [velocity, amplitude, energy_power]
            prev_points: (M, 6) - 前フレームの点群（Noneの場合はself.prev_pointsを使用）
            prev_features: (M, 3) - 前フレームの特徴量（Noneの場合はself.prev_featuresを使用）
        
        Returns:
            new_points: (L, 6) - 新規点群
        """
        if prev_points is None:
            prev_points = self.prev_points
        if prev_features is None:
            prev_features = self.prev_features
        
        if prev_points is None or len(prev_points) == 0:
            # 前フレームがない場合は全て新規点
            self.prev_points = curr_points.copy()
            self.prev_features = curr_features.copy()
            return curr_points
        
        new_indices = []
        prev_tree = cKDTree(prev_points[:, :3])
        
        for curr_idx, curr_point in enumerate(curr_points):
            curr_pos = curr_point[:3]
            curr_feat = curr_features[curr_idx]
            
            # 位置近接点を検索
            neighbor_indices = prev_tree.query_ball_point(curr_pos, self.max_match_distance)
            
            if len(neighbor_indices) == 0:
                new_indices.append(curr_idx)
                continue
            
            # 特徴量類似度で最も近い点を選択
            best_match = None
            best_score = -float('inf')
            
            for prev_idx in neighbor_indices:
                prev_pos = prev_points[prev_idx, :3]
                prev_feat = prev_features[prev_idx]
                
                pos_dist = np.linalg.norm(curr_pos - prev_pos)
                position_score = 1.0 / (1.0 + pos_dist)
                
                # 特徴量の差分を計算（velocity, amplitude, energy_power）
                feat_diff = np.abs(curr_feat - prev_feat)
                feat_similarity = 1.0 / (1.0 + np.mean(feat_diff))
                
                combined_score = (
                    self.position_weight * position_score +
                    self.feature_similarity_weight * feat_similarity
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = prev_idx
            
            if best_match is None:
                new_indices.append(curr_idx)
        
        # ノイズフィルタリング（周辺に一定数以上の点があるかチェック）
        filtered_new_indices = []
        if len(new_indices) > 0:
            new_points_coords = curr_points[new_indices, :3]
            if len(new_points_coords) > 0:
                new_tree = cKDTree(new_points_coords)
                
                for i, new_idx in enumerate(new_indices):
                    new_pos = curr_points[new_idx, :3]
                    neighbor_count = len(new_tree.query_ball_point(new_pos, self.new_point_radius))
                    if neighbor_count >= self.min_new_neighbors:
                        filtered_new_indices.append(new_idx)
        
        # 動きの信頼性計算でフィルタリング（オプション）
        if self.use_motion_reliability and len(filtered_new_indices) > 0:
            new_points_subset = curr_points[filtered_new_indices]  # (L, 6)
            energy_power = new_points_subset[:, 5]  # energy_power
            velocity = new_points_subset[:, 3]  # velocity
            
            # 信頼性を計算
            reliability = compute_point_reliability(
                energy_power, velocity,
                self.energy_threshold, self.velocity_threshold
            )
            
            # 信頼性が低すぎる点（ノイズ）を除外（最小信頼性0.2）
            min_reliability = 0.2
            reliable_mask = reliability >= min_reliability
            filtered_new_indices = [filtered_new_indices[i] for i in range(len(filtered_new_indices)) 
                                   if reliable_mask[i]]
        
        # 前フレームを更新
        self.prev_points = curr_points.copy()
        self.prev_features = curr_features.copy()
        
        return curr_points[filtered_new_indices] if filtered_new_indices else np.zeros((0, 6))
    
    def classify_body_part_clusters(
        self,
        new_points: np.ndarray,  # (L, 6)
        regression_output: np.ndarray  # (22, 3)
    ) -> Dict[str, np.ndarray]:
        """
        新規点を固定された部位（右腕・左腕・右脚・左脚）ごとに分類
        
        Args:
            new_points: (L, 6) - 新規点群
            regression_output: (22, 3) - 回帰出力の関節座標
        
        Returns:
            body_part_clusters: Dict[str, np.ndarray] - {部位名: (N_i, 6)}
        """
        if len(new_points) == 0:
            return {}
        
        body_part_clusters = {part: [] for part in self.body_parts.keys()}
        new_coords = new_points[:, :3]  # (L, 3)
        
        # 動きの信頼性を計算（オプション）
        reliability = None
        if self.use_motion_reliability and len(new_points) > 0:
            energy_power = new_points[:, 5]  # energy_power
            velocity = new_points[:, 3]  # velocity
            reliability = compute_point_reliability(
                energy_power, velocity,
                self.energy_threshold, self.velocity_threshold
            )
        
        for part_name, joint_indices in self.body_parts.items():
            # この部位の骨チェーン上の関節座標を取得
            chain_positions = regression_output[joint_indices]  # (len(joint_indices), 3)
            
            # 骨チェーン上の点からの最小距離を計算
            min_distances = []
            for new_point in new_coords:
                # 各骨セグメント（連続する2関節）への距離を計算
                segment_distances = []
                for i in range(len(joint_indices) - 1):
                    j0, j1 = joint_indices[i], joint_indices[i + 1]
                    p0, p1 = chain_positions[i], chain_positions[i + 1]
                    
                    # 点から線分への距離を計算
                    segment_vec = p1 - p0
                    point_vec = new_point - p0
                    
                    # 線分上の最近接点を計算
                    segment_norm_sq = np.dot(segment_vec, segment_vec)
                    if segment_norm_sq < 1e-6:
                        # 骨の長さが0の場合は、点から関節への距離を使用
                        dist = np.linalg.norm(new_point - p0)
                    else:
                        t = np.clip(np.dot(point_vec, segment_vec) / segment_norm_sq, 0, 1)
                        closest_point = p0 + t * segment_vec
                        dist = np.linalg.norm(new_point - closest_point)
                    
                    segment_distances.append(dist)
                
                min_dist = min(segment_distances) if segment_distances else float('inf')
                min_distances.append(min_dist)
            
            # 半径内の新規点を抽出
            in_range = np.array(min_distances) <= self.bone_chain_radius
            
            if np.any(in_range):
                body_part_clusters[part_name] = new_points[in_range]
        
        # 空のクラスタを削除
        body_part_clusters = {k: v for k, v in body_part_clusters.items() if len(v) > 0}
        
        return body_part_clusters
    
    def detect_distal_cells_with_model(
        self,
        points: np.ndarray,  # (N, 6) 全点群
        regression_output: np.ndarray  # (22, 3) 回帰出力（胴体統計量計算用）
    ) -> List[Dict]:
        """
        末端部位検出モデルを使用してグリッドセルを判定
        
        Args:
            points: (N, 6) - 全点群 [x, y, z, velocity, amplitude, energy_power]
            regression_output: (22, 3) - 回帰出力の関節座標（胴体統計量計算用）
        
        Returns:
            distal_cells: List[Dict] - 末端部位として判定されたセル
                - center: (3,) セル中心座標
                - confidence: float 信頼度
                - point_indices: np.ndarray セル内の点群インデックス
        """
        if not self.use_distal_detection_model or self.distal_model is None:
            return []
        
        # collect_distal_grid_dataから必要な関数をインポート
        import sys
        import importlib.util
        
        # ファイルパスを構築
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(current_dir))
        distal_grid_module_path = os.path.join(project_root, "collect_distal_grid_data.py")
        
        if not os.path.exists(distal_grid_module_path):
            # 別のパスを試行
            alt_paths = [
                os.path.join(os.path.dirname(current_dir), "collect_distal_grid_data.py"),
                os.path.join(os.path.dirname(os.path.dirname(current_dir)), "collect_distal_grid_data.py"),
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    distal_grid_module_path = alt_path
                    break
            else:
                print(f"Warning: collect_distal_grid_data.py not found")
                return []
        
        try:
            spec = importlib.util.spec_from_file_location(
                "collect_distal_grid_data",
                distal_grid_module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            create_spatial_grid = module.create_spatial_grid
            extract_grid_cell_features = module.extract_grid_cell_features
            compute_torso_statistics = module.compute_torso_statistics
        except Exception as e:
            print(f"Warning: Failed to import distal detection functions: {e}")
            return []
        
        # 点群座標と特徴量を分離
        point_coords = points[:, :3]  # (N, 3)
        point_features = points[:, 3:6]  # (N, 3)
        
        # 胴体統計量を計算
        torso_stats = compute_torso_statistics(
            point_coords, point_features, regression_output, torso_radius=0.4
        )
        
        # 空間グリッド分割
        grid_info = create_spatial_grid(point_coords, grid_size=self.distal_grid_size)
        grid_cells = grid_info['grid_cells']
        
        # 各セルに対して推論
        distal_cells = []
        for cell in grid_cells:
            cell_indices = cell['indices']
            cell_center = cell['cell_center']
            
            # セル内の点群が少ない場合はスキップ
            if len(cell_indices) < 3:
                continue
            
            # セル内の点群と特徴量を取得
            points_in_cell = point_coords[cell_indices]
            features_in_cell = point_features[cell_indices]
            
            # 特徴量を抽出
            features = extract_grid_cell_features(
                points_in_cell, features_in_cell, cell_center,
                max_points=100, torso_stats=torso_stats, gt_joints=regression_output
            )
            
            # 推論
            try:
                proba = self.distal_model.predict_proba(features.reshape(1, -1))[0]
                confidence = proba[1]  # クラス1（末端部位）の確率
                
                if confidence >= self.distal_confidence_threshold:
                    distal_cells.append({
                        'center': cell_center,
                        'confidence': float(confidence),
                        'point_indices': cell_indices
                    })
            except Exception as e:
                # 推論エラーは無視して続行
                continue
        
        return distal_cells
    
    def classify_body_part_clusters_hybrid(
        self,
        new_points: np.ndarray,  # (L, 6) 新規点群
        all_points: np.ndarray,  # (N, 6) 全点群
        regression_output: np.ndarray,  # (22, 3)
        distal_cells: List[Dict]  # 末端部位検出モデルの結果
    ) -> Dict[str, np.ndarray]:
        """
        末端部位検出モデル + 回帰出力の骨チェーン近接度で部位分類（ハイブリッド）
        
        Args:
            new_points: (L, 6) - 新規点群
            all_points: (N, 6) - 全点群
            regression_output: (22, 3) - 回帰出力の関節座標
            distal_cells: List[Dict] - 末端部位検出モデルの結果
        
        Returns:
            body_part_clusters: Dict[str, np.ndarray] - {部位名: (N_i, 6)}
        """
        body_part_clusters = {part: [] for part in self.body_parts.keys()}
        
        if len(distal_cells) == 0:
            # 末端部位セルがない場合は既存のロジックを使用
            return self.classify_body_part_clusters(new_points, regression_output)
        
        # 1. 末端部位セル内の新規点を抽出
        distal_point_indices = set()
        new_coords = new_points[:, :3]
        
        for cell in distal_cells:
            cell_center = cell['center']
            cell_radius = self.distal_grid_size / 2.0  # セル半径
            
            # 新規点がセル内にあるかチェック
            distances_to_cell = np.linalg.norm(new_coords - cell_center, axis=1)
            in_cell_mask = distances_to_cell < cell_radius
            
            # セル内の新規点のインデックスを追加
            for i in np.where(in_cell_mask)[0]:
                distal_point_indices.add(i)
        
        distal_point_indices = np.array(list(distal_point_indices))
        
        # 2. 末端部位点 + 回帰出力の骨チェーン近接度で部位分類
        if len(distal_point_indices) > 0:
            distal_new_points = new_points[distal_point_indices]
            
            # 既存の骨チェーン近接度判定を使用
            for part_name, joint_indices in self.body_parts.items():
                chain_positions = regression_output[joint_indices]
                new_coords_distal = distal_new_points[:, :3]
                
                # 各新規点から骨チェーンへの最小距離を計算
                min_distances = []
                for new_point in new_coords_distal:
                    segment_distances = []
                    for i in range(len(joint_indices) - 1):
                        p0, p1 = chain_positions[i], chain_positions[i + 1]
                        
                        segment_vec = p1 - p0
                        point_vec = new_point - p0
                        segment_norm_sq = np.dot(segment_vec, segment_vec)
                        
                        if segment_norm_sq < 1e-6:
                            dist = np.linalg.norm(new_point - p0)
                        else:
                            t = np.clip(np.dot(point_vec, segment_vec) / segment_norm_sq, 0, 1)
                            closest_point = p0 + t * segment_vec
                            dist = np.linalg.norm(new_point - closest_point)
                        
                        segment_distances.append(dist)
                    
                    min_dist = min(segment_distances) if segment_distances else float('inf')
                    min_distances.append(min_dist)
                
                # 半径内の新規点を抽出
                in_range = np.array(min_distances) <= self.bone_chain_radius
                
                if np.any(in_range):
                    matched_points = distal_new_points[in_range]
                    if len(body_part_clusters[part_name]) == 0:
                        body_part_clusters[part_name] = matched_points
                    else:
                        body_part_clusters[part_name] = np.vstack([
                            body_part_clusters[part_name], matched_points
                        ])
        
        # 3. フォールバック: 末端部位検出モデルで検出されなかった点は既存ロジックを使用
        non_distal_indices = np.setdiff1d(np.arange(len(new_points)), distal_point_indices)
        if len(non_distal_indices) > 0:
            # 既存のclassify_body_part_clustersを使用
            fallback_clusters = self.classify_body_part_clusters(
                new_points[non_distal_indices], regression_output
            )
            # マージ
            for part_name, points in fallback_clusters.items():
                if len(points) > 0:
                    if len(body_part_clusters[part_name]) > 0:
                        body_part_clusters[part_name] = np.vstack([
                            body_part_clusters[part_name], points
                        ])
                    else:
                        body_part_clusters[part_name] = points
        
        # 空のクラスタを削除
        body_part_clusters = {k: v for k, v in body_part_clusters.items() if len(v) > 0}
        
        return body_part_clusters
    
    def classify_body_part_clusters_robust(
        self,
        new_points: np.ndarray,  # (L, 6)
        all_points: np.ndarray,  # (N, 6) 全点群（胴体モデル化用）
        regression_output: np.ndarray  # (22, 3)
    ) -> Dict[str, np.ndarray]:
        """
        ロバスト中心線抽出（Phase 1）を使用して新規点を部位ごとに分類
        
        Args:
            new_points: (L, 6) - 新規点群
            all_points: (N, 6) - 全点群（胴体モデル化用）
            regression_output: (22, 3) - 回帰出力の関節座標
        
        Returns:
            body_part_clusters: Dict[str, np.ndarray] - {部位名: (N_i, 6)}
        """
        if not self.use_robust_centerline or len(all_points) == 0:
            # フォールバック: 既存の実装を使用
            return self.classify_body_part_clusters(new_points, regression_output)
        
        # 1. 前処理
        processed = self.robust_preprocessor.preprocess(all_points)
        
        # 2. 胴体モデル化
        # 胴体領域を推定（回帰出力からPelvis, Spine1, Spine2, Spine3を中心とした領域）
        torso_joint_indices = [0, 3, 6, 9]  # Pelvis, Spine1, Spine2, Spine3
        torso_positions = regression_output[torso_joint_indices]
        torso_center = np.mean(torso_positions, axis=0)
        torso_distances = np.linalg.norm(processed['coords'] - torso_center, axis=1)
        # 胴体半径の2倍以内の点を胴体領域とする
        torso_radius = 0.20  # 20cm
        torso_region = np.where(torso_distances < torso_radius * 2)[0]
        
        # 胴体モデルを再計算（胴体領域を指定）
        if len(torso_region) > 0:
            processed_with_torso = self.robust_preprocessor.preprocess(
                all_points, torso_region=torso_region
            )
        else:
            processed_with_torso = processed
        
        torso_result = self.torso_modeler.fit_torso(
            processed_with_torso['coords'], all_points
        )
        mask_w = self.torso_modeler.compute_torso_halo_weight(
            processed_with_torso['coords'],
            torso_result['z_axis'],
            torso_result['center'],
            torso_result['radius']
        )
        
        # 3. 属性込み距離クラスタリング
        features = {
            'amp_torso': processed_with_torso['torso_relative_amp'],
            'pwr_torso': processed_with_torso['torso_relative_pwr'],
            'vel': processed_with_torso['velocity_normalized'],
            'density': processed_with_torso['density']
        }
        
        cluster_result = self.attribute_clusterer.cluster(
            processed_with_torso['coords'],
            features,
            processed_with_torso['torso_stats'],
            torso_result['a_torso'],
            mask_w
        )
        
        # 4. クラスタを部位ごとに分類
        body_part_clusters = {part: [] for part in self.body_parts.keys()}
        
        # labelsからclusters辞書を構築
        labels = cluster_result['labels']
        clusters_dict = {}
        for idx, label in enumerate(labels):
            if label == -1:  # ノイズはスキップ
                continue
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(idx)
        
        for label, cluster_indices in clusters_dict.items():
            if label == -1:  # ノイズクラスタ
                continue
            
            # クラスタの点群を取得
            cluster_points = all_points[cluster_indices]
            cluster_coords = cluster_points[:, :3]
            
            # 各部位への距離を計算
            best_part = None
            best_distance = float('inf')
            
            for part_name, joint_indices in self.body_parts.items():
                # 骨チェーン上の点からの最小距離を計算
                chain_positions = regression_output[joint_indices]
                min_dist = float('inf')
                
                for point_coord in cluster_coords:
                    for i in range(len(joint_indices) - 1):
                        j0, j1 = joint_indices[i], joint_indices[i + 1]
                        p0, p1 = chain_positions[i], chain_positions[i + 1]
                        
                        segment_vec = p1 - p0
                        point_vec = point_coord - p0
                        segment_norm_sq = np.dot(segment_vec, segment_vec)
                        
                        if segment_norm_sq < 1e-6:
                            dist = np.linalg.norm(point_coord - p0)
                        else:
                            t = np.clip(np.dot(point_vec, segment_vec) / segment_norm_sq, 0, 1)
                            closest_point = p0 + t * segment_vec
                            dist = np.linalg.norm(point_coord - closest_point)
                        
                        min_dist = min(min_dist, dist)
                
                if min_dist < best_distance:
                    best_distance = min_dist
                    best_part = part_name
            
            # 最適な部位に割り当て
            if best_part is not None and best_distance < self.bone_chain_radius:
                if len(body_part_clusters[best_part]) == 0:
                    body_part_clusters[best_part] = cluster_points
                else:
                    body_part_clusters[best_part] = np.vstack([
                        body_part_clusters[best_part],
                        cluster_points
                    ])
        
        # 空のクラスタを削除
        body_part_clusters = {k: v for k, v in body_part_clusters.items() if len(v) > 0}
        
        return body_part_clusters
    
    def compute_principal_axis_robust(
        self,
        cluster_points: np.ndarray  # (N, 6)
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        ロバスト中心線抽出（Phase 1+2）を使用して主軸を計算
        
        Args:
            cluster_points: (N, 6) - クラスタの点群
        
        Returns:
            principal_axis: (3,) or None - 主軸ベクトル（正規化済み、点が少ない場合はNone）
            center: (3,) - クラスタの重心
        """
        if not self.use_robust_centerline or len(cluster_points) < 3:
            # フォールバック: 既存の実装を使用
            return self.compute_principal_axis(cluster_points)
        
        # Phase 2: IRLSロバストPCAを使用
        if self.use_irls_pca and self.irls_pca_module is not None:
            # 点群座標を取得
            coords = cluster_points[:, :3]  # (N, 3)
            
            # 初期重心を計算（中央値を使用）
            initial_center = np.median(coords, axis=0)
            
            # IRLSロバストPCAで主軸を計算
            principal_axis, center, weights = self.irls_pca_module.compute_principal_axis(
                coords, initial_center=initial_center
            )
            
            return principal_axis, center
        
        # Phase 1: 既存のPCAを使用（フォールバック）
        return self.compute_principal_axis(cluster_points)
    
    def compute_principal_axis(
        self,
        cluster_points: np.ndarray  # (N, 6)
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        クラスタの主軸（PCAの第1主成分）を計算（信頼性閾値を満たす点のみ使用）
        
        Args:
            cluster_points: (N, 6) - クラスタの点群
        
        Returns:
            principal_axis: (3,) or None - 主軸ベクトル（正規化済み、点が少ない場合はNone）
            center: (3,) - クラスタの重心
        """
        if len(cluster_points) < 3:
            center = np.mean(cluster_points[:, :3], axis=0) if len(cluster_points) > 0 else np.zeros(3)
            return None, center
        
        coords = cluster_points[:, :3]  # (N, 3)
        
        # 動きの信頼性を計算（オプション）
        if self.use_motion_reliability and len(cluster_points) > 0:
            energy_power = cluster_points[:, 5]  # energy_power
            velocity = cluster_points[:, 3]      # velocity
            
            # 🔧 変更: energy_power < 100.0 AND velocity < 0.1 を満たす点（動いている部位）のみを使用
            low_energy = energy_power < self.energy_threshold
            low_velocity = np.abs(velocity) < self.velocity_threshold
            moving_mask = low_energy & low_velocity  # 動いている部位の点のみ
            
            moving_coords = coords[moving_mask]
            
            # フィルタリング後の点数をチェック
            if len(moving_coords) < 3:
                # 動いている部位の点が3点未満の場合は、全ての点を使用（フォールバック）
                coords = cluster_points[:, :3]  # 元の点群を使用
                center = np.mean(coords, axis=0)
            else:
                # 動いている部位の点のみを使用
                coords = moving_coords
                center = np.mean(coords, axis=0)
        else:
            # 信頼性計算を使用しない場合は全ての点を使用
            center = np.mean(coords, axis=0)
        
        coords_centered = coords - center  # (N, 3)
        
        # 再度点数チェック（フィルタリング後）
        if len(coords_centered) < 3:
            return None, center
        
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            # sklearnがインストールされていない場合は、簡易版のPCAを使用
            cov = np.cov(coords_centered.T)  # (3, 3)
            
            # 固有値分解
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            # 最大固有値に対応する固有ベクトルを取得
            max_idx = np.argmax(eigenvalues)
            principal_axis = eigenvectors[:, max_idx].real  # (3,)
            principal_axis = principal_axis / (np.linalg.norm(principal_axis) + 1e-6)  # 正規化
            
            return principal_axis, center
        
        # PCAで主軸を計算（信頼性の高い点のみ使用）
        pca = PCA(n_components=1)
        pca.fit(coords_centered)
        principal_axis = pca.components_[0]  # (3,) - 主軸ベクトル（第1主成分）
        principal_axis = principal_axis / (np.linalg.norm(principal_axis) + 1e-6)  # 正規化
        
        return principal_axis, center
    
    def align_keypoints_to_principal_axis(
        self,
        part_name: str,
        principal_axis: np.ndarray,  # (3,)
        cluster_center: np.ndarray,  # (3,)
        regression_output: np.ndarray  # (22, 3)
    ) -> Dict[int, np.ndarray]:
        """
        キーポイントを主軸に沿うように配置
        
        Args:
            part_name: 部位名（'right_arm', 'left_arm', 'right_leg', 'left_leg'）
            principal_axis: (3,) - クラスタの主軸
            cluster_center: (3,) - クラスタの重心
            regression_output: (22, 3) - 回帰出力の関節座標
        
        Returns:
            aligned_keypoints: Dict[int, np.ndarray] - {関節インデックス: (3,)} 更新された関節座標
        """
        joint_indices = self.body_parts[part_name]
        anchor_joint_idx = self.body_part_anchors[part_name]
        
        if len(joint_indices) < 2:
            return {}
        
        aligned_keypoints = {}
        
        # 起点（最初の関節）を取得
        start_joint = joint_indices[0]
        start_pos = regression_output[start_joint].copy()
        aligned_keypoints[start_joint] = start_pos
        
        # 主軸の方向を決定（回帰出力の方向と内積を取って、符号を決定）
        if len(joint_indices) >= 2:
            reg_direction = regression_output[joint_indices[1]] - start_pos
            reg_direction_norm = np.linalg.norm(reg_direction)
            if reg_direction_norm > 1e-6:
                reg_direction = reg_direction / reg_direction_norm
                
                # 主軸と回帰出力の方向の内積
                dot_product = np.dot(principal_axis, reg_direction)
                
                # 内積が負の場合は主軸を反転
                if dot_product < 0:
                    principal_axis = -principal_axis
        
        # 各関節を主軸方向に沿って配置
        current_pos = start_pos
        
        for i in range(1, len(joint_indices)):
            joint_idx = joint_indices[i]
            prev_joint_idx = joint_indices[i - 1]
            
            # 回帰出力から骨の長さを取得
            reg_bone_vec = regression_output[joint_idx] - regression_output[prev_joint_idx]
            bone_length = np.linalg.norm(reg_bone_vec)
            
            if bone_length < 1e-6:
                # 骨の長さが0の場合は、標準的な骨の長さを使用
                bone_key = f"{prev_joint_idx}-{joint_idx}"
                standard_lengths = self.joints_def.get('standard_bone_length_m', {})
                if bone_key in standard_lengths:
                    bone_length = standard_lengths[bone_key]
                else:
                    # デフォルト値（部位に応じて調整）
                    if 'leg' in part_name:
                        bone_length = 0.40  # 脚のデフォルト骨長
                    else:
                        bone_length = 0.25  # 腕のデフォルト骨長
            
            # 主軸方向に沿って配置
            new_joint_pos = current_pos + principal_axis * bone_length
            aligned_keypoints[joint_idx] = new_joint_pos
            
            current_pos = new_joint_pos
        
        return aligned_keypoints
    
    def process_frame(
        self,
        curr_points: np.ndarray,  # (N, 6)
        curr_features: np.ndarray,  # (N, 3)
        regression_output: np.ndarray,  # (22, 3)
        prev_points: Optional[np.ndarray] = None,  # (M, 6)
        prev_features: Optional[np.ndarray] = None  # (M, 3)
    ) -> np.ndarray:
        """
        フレーム処理のメイン関数
        
        Args:
            curr_points: (N, 6) - 現在フレームの点群 [x, y, z, velocity, amplitude, energy_power]
            curr_features: (N, 3) - 現在フレームの特徴量 [velocity, amplitude, energy_power]
            regression_output: (22, 3) - 回帰出力の関節座標
            prev_points: (M, 6) - 前フレームの点群（Noneの場合はself.prev_pointsを使用）
            prev_features: (M, 3) - 前フレームの特徴量（Noneの場合はself.prev_featuresを使用）
        
        Returns:
            updated_joints: (22, 3) - 更新された関節座標
        """
        updated_joints = regression_output.copy()
        
        # Phase 3: 時間統合（オプション）
        if self.use_temporal_integration and self.temporal_integrator is not None:
            # 時間統合を実行
            integrated_points = self.temporal_integrator.integrate(curr_points)
            # 統合後の点群から特徴量を抽出（時間統合により点数が増える可能性があるため）
            integrated_features = integrated_points[:, 3:6] if len(integrated_points) > 0 else curr_features
        else:
            integrated_points = curr_points
            integrated_features = curr_features
        
        # 1. 点マッチング → 新規点検出
        new_points = self.match_points_and_detect_new(
            integrated_points, integrated_features, prev_points, prev_features
        )
        
        if len(new_points) == 0:
            return updated_joints
        
        # 2. 固定された部位（右腕・左腕・右脚・左脚）ごとに新規点を分類
        if self.use_distal_detection_model and self.distal_model is not None:
            # 末端部位検出モデルを使用（ハイブリッド）
            distal_cells = self.detect_distal_cells_with_model(
                integrated_points, regression_output
            )
            body_part_clusters = self.classify_body_part_clusters_hybrid(
                new_points, integrated_points, regression_output, distal_cells
            )
        elif self.use_robust_centerline:
            body_part_clusters = self.classify_body_part_clusters_robust(
                new_points, integrated_points, regression_output
            )
        else:
            body_part_clusters = self.classify_body_part_clusters(new_points, regression_output)
        
        # Phase 3: HMM/FSM状態管理（オプション）
        if self.use_hmm_state and self.state_manager is not None:
            # TODO: 状態を更新
            pass
        
        # 3. 各部位の主軸を計算し、キーポイントを配置
        for part_name, cluster_points in body_part_clusters.items():
            # Phase 3: 方向場→中心線抽出（オプション）
            if self.use_centerline_extraction and self.centerline_extractor is not None:
                # TODO: 方向場→中心線抽出を使用
                # centerline_result = self.centerline_extractor.extract_centerline(...)
                # principal_axis = centerline_result['spline'].direction
                # cluster_center = centerline_result['centerline'][...]
                # 現時点ではフォールバック
                principal_axis, cluster_center = self.compute_principal_axis_robust(cluster_points)
            elif self.use_robust_centerline:
                # Phase 1+2: ロバスト中心線抽出
                principal_axis, cluster_center = self.compute_principal_axis_robust(cluster_points)
            else:
                # 既存実装
                principal_axis, cluster_center = self.compute_principal_axis(cluster_points)
            
            if principal_axis is None:
                continue
            
            # キーポイントを主軸に沿って配置
            aligned_keypoints = self.align_keypoints_to_principal_axis(
                part_name, principal_axis, cluster_center, regression_output
            )
            
            # 更新された関節座標を適用
            for joint_idx, new_pos in aligned_keypoints.items():
                updated_joints[joint_idx] = new_pos
        
        return updated_joints
    
    def process_frame_with_axes(
        self,
        curr_points: np.ndarray,  # (N, 6)
        curr_features: np.ndarray,  # (N, 3)
        regression_output: np.ndarray,  # (22, 3)
        prev_points: Optional[np.ndarray] = None,  # (M, 6)
        prev_features: Optional[np.ndarray] = None  # (M, 3)
    ) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        フレーム処理のメイン関数（主軸情報も返す）
        
        Args:
            curr_points: (N, 6) - 現在フレームの点群 [x, y, z, velocity, amplitude, energy_power]
            curr_features: (N, 3) - 現在フレームの特徴量 [velocity, amplitude, energy_power]
            regression_output: (22, 3) - 回帰出力の関節座標
            prev_points: (M, 6) - 前フレームの点群（Noneの場合はself.prev_pointsを使用）
            prev_features: (M, 3) - 前フレームの特徴量（Noneの場合はself.prev_featuresを使用）
        
        Returns:
            updated_joints: (22, 3) - 更新された関節座標
            principal_axes: Dict[str, Tuple[np.ndarray, np.ndarray]] - {部位名: (主軸, 重心)}
        """
        updated_joints = regression_output.copy()
        principal_axes = {}
        
        # Phase 3: 時間統合（オプション）
        if self.use_temporal_integration and self.temporal_integrator is not None:
            # 時間統合を実行
            integrated_points = self.temporal_integrator.integrate(curr_points)
            # 統合後の点群から特徴量を抽出（時間統合により点数が増える可能性があるため）
            integrated_features = integrated_points[:, 3:6] if len(integrated_points) > 0 else curr_features
        else:
            integrated_points = curr_points
            integrated_features = curr_features
        
        # 1. 点マッチング → 新規点検出
        new_points = self.match_points_and_detect_new(
            integrated_points, integrated_features, prev_points, prev_features
        )
        
        if len(new_points) == 0:
            return updated_joints, principal_axes
        
        # 2. 固定された部位（右腕・左腕・右脚・左脚）ごとに新規点を分類
        if self.use_distal_detection_model and self.distal_model is not None:
            # 末端部位検出モデルを使用（ハイブリッド）
            distal_cells = self.detect_distal_cells_with_model(
                integrated_points, regression_output
            )
            body_part_clusters = self.classify_body_part_clusters_hybrid(
                new_points, integrated_points, regression_output, distal_cells
            )
        elif self.use_robust_centerline:
            body_part_clusters = self.classify_body_part_clusters_robust(
                new_points, integrated_points, regression_output
            )
        else:
            body_part_clusters = self.classify_body_part_clusters(new_points, regression_output)
        
        # Phase 3: HMM/FSM状態管理（オプション）
        if self.use_hmm_state and self.state_manager is not None:
            # TODO: 状態を更新
            pass
        
        # 3. 各部位の主軸を計算し、キーポイントを配置
        for part_name, cluster_points in body_part_clusters.items():
            # Phase 3: 方向場→中心線抽出（オプション）
            if self.use_centerline_extraction and self.centerline_extractor is not None:
                # 中心線抽出を実行
                centerline_result = self.centerline_extractor.extract_centerline(
                    cluster_points[:, :3],  # 座標のみ
                    mask_w=None,
                    z_axis=None
                )
                
                # 中心線から主軸を取得
                if len(centerline_result['centerline']) >= 2:
                    # 中心線の最初と最後の点から主軸を計算
                    centerline = centerline_result['centerline']
                    direction = centerline[-1] - centerline[0]
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-6:
                        principal_axis = direction / direction_norm
                        cluster_center = np.mean(centerline, axis=0)
                    else:
                        # フォールバック
                        principal_axis, cluster_center = self.compute_principal_axis_robust(cluster_points)
                else:
                    # フォールバック
                    principal_axis, cluster_center = self.compute_principal_axis_robust(cluster_points)
            elif self.use_robust_centerline:
                # Phase 1+2: ロバスト中心線抽出
                principal_axis, cluster_center = self.compute_principal_axis_robust(cluster_points)
            else:
                # 既存実装
                principal_axis, cluster_center = self.compute_principal_axis(cluster_points)
            
            if principal_axis is None:
                continue
            
            # 主軸の方向を決定（回帰出力の方向と内積を取って、符号を決定）
            joint_indices = self.body_parts[part_name]
            if len(joint_indices) >= 2:
                start_joint = joint_indices[0]
                start_pos = regression_output[start_joint]
                reg_direction = regression_output[joint_indices[1]] - start_pos
                reg_direction_norm = np.linalg.norm(reg_direction)
                if reg_direction_norm > 1e-6:
                    reg_direction = reg_direction / reg_direction_norm
                    dot_product = np.dot(principal_axis, reg_direction)
                    if dot_product < 0:
                        principal_axis = -principal_axis
            
            # 主軸情報を保存
            principal_axes[part_name] = (principal_axis.copy(), cluster_center.copy())
            
            # キーポイントを主軸に沿って配置
            aligned_keypoints = self.align_keypoints_to_principal_axis(
                part_name, principal_axis, cluster_center, regression_output
            )
            
            # 更新された関節座標を適用
            for joint_idx, new_pos in aligned_keypoints.items():
                updated_joints[joint_idx] = new_pos
        
        return updated_joints, principal_axes
    
    def reset(self):
        """
        前フレームの状態をリセット（新しいシーケンス開始時など）
        Phase 3のコンポーネントもリセット
        """
        self.prev_points = None
        self.prev_features = None
        
        # Phase 3: 時間統合と状態管理をリセット
        if self.temporal_integrator is not None:
            self.temporal_integrator.reset()
        
        if self.state_manager is not None:
            self.state_manager.reset()

