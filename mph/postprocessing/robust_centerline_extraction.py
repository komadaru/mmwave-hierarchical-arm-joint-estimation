#!/usr/bin/env python3
"""
Robust Centerline Extraction for mmWave Radar Pose Tracking

Phase 1実装:
- RobustPreprocessor: ロバスト前処理
- TorsoModeler: 胴体モデル化（RANSAC円柱）
- AttributeBasedClusterer: 属性込み距離クラスタリング

Phase 2実装:
- IRLSPCA: IRLSロバストPCA（外れ値に強い主軸計算）

Phase 3実装:
- CenterlineExtractor: 方向場→中心線抽出
- TemporalIntegration: 時間統合機能
- TemporalStateManager: HMM/FSM状態管理
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
import warnings

# Phase 3依存関係
try:
    from scipy.interpolate import splprep, splev, BSpline
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    BSpline = None
    pdist = None
    squareform = None

warnings.filterwarnings('ignore')


class RobustPreprocessor:
    """
    ロバスト前処理モジュール
    
    特徴量のロバスト正規化（median/IQR）、胴体相対正規化、
    密度指標計算、時間統合を実行
    """
    
    def __init__(
        self,
        use_torso_relative: bool = True,
        k_nn: int = 16,
        time_window_sec: Optional[float] = None,  # 0.3-0.5秒
        fps: float = 15.0
    ):
        """
        Args:
            use_torso_relative: 胴体相対正規化を使用するか
            k_nn: k近傍数（密度計算用）
            time_window_sec: 時間統合窓（秒）。Noneの場合は時間統合なし
            fps: フレームレート
        """
        self.use_torso_relative = use_torso_relative
        self.k_nn = k_nn
        self.time_window_sec = time_window_sec
        self.fps = fps
        self.time_window_frames = int(time_window_sec * fps) if time_window_sec else None
    
    def preprocess(
        self,
        points: np.ndarray,  # (N, 6) [x, y, z, velocity, amplitude, energy_power]
        torso_region: Optional[np.ndarray] = None  # 胴体領域の点群インデックス
    ) -> Dict[str, np.ndarray]:
        """
        前処理を実行
        
        Returns:
            {
                'coords': (N, 3),
                'amplitude_normalized': (N,),
                'energy_power_normalized': (N,),
                'velocity_normalized': (N,),
                'density': (N,),
                'torso_relative_amp': (N,),
                'torso_relative_pwr': (N,),
                'torso_stats': {'median_amp': float, 'iqr_amp': float, ...}
            }
        """
        if len(points) == 0:
            return self._empty_result()
        
        coords = points[:, :3]  # (N, 3)
        velocity = points[:, 3]
        amplitude = points[:, 4]
        energy_power = points[:, 5]
        
        # 1. ロバスト正規化（median/IQR）
        amp_normalized = self._robust_normalize(amplitude)
        pwr_normalized = self._robust_normalize(energy_power)
        vel_normalized = self._robust_normalize(velocity)
        
        # 2. 密度指標計算
        density = self._compute_density(coords)
        
        # 3. 胴体相対正規化
        torso_relative_amp = None
        torso_relative_pwr = None
        torso_stats = {}
        
        if self.use_torso_relative and torso_region is not None and len(torso_region) > 0:
            torso_amp = amplitude[torso_region]
            torso_pwr = energy_power[torso_region]
            
            m_amp_torso = np.median(torso_amp)
            iqr_amp_torso = np.percentile(torso_amp, 75) - np.percentile(torso_amp, 25)
            m_pwr_torso = np.median(torso_pwr)
            iqr_pwr_torso = np.percentile(torso_pwr, 75) - np.percentile(torso_pwr, 25)
            
            torso_relative_amp = (amplitude - m_amp_torso) / (iqr_amp_torso + 1e-6)
            torso_relative_pwr = (energy_power - m_pwr_torso) / (iqr_pwr_torso + 1e-6)
            
            torso_stats = {
                'median_amp': m_amp_torso,
                'iqr_amp': iqr_amp_torso,
                'median_pwr': m_pwr_torso,
                'iqr_pwr': iqr_pwr_torso
            }
        
        return {
            'coords': coords,
            'amplitude_normalized': amp_normalized,
            'energy_power_normalized': pwr_normalized,
            'velocity_normalized': vel_normalized,
            'density': density,
            'torso_relative_amp': torso_relative_amp,
            'torso_relative_pwr': torso_relative_pwr,
            'torso_stats': torso_stats
        }
    
    def _robust_normalize(self, x: np.ndarray) -> np.ndarray:
        """median/IQRによるロバスト正規化"""
        median = np.median(x)
        q75 = np.percentile(x, 75)
        q25 = np.percentile(x, 25)
        iqr = q75 - q25
        
        if iqr < 1e-6:
            return np.zeros_like(x)
        
        return (x - median) / (iqr + 1e-6)
    
    def _compute_density(self, coords: np.ndarray) -> np.ndarray:
        """kNN距離による密度指標"""
        if len(coords) < 2:
            return np.zeros(len(coords))
        
        try:
            nbrs = NearestNeighbors(n_neighbors=min(self.k_nn + 1, len(coords)), algorithm='auto')
            nbrs.fit(coords)
            distances, _ = nbrs.kneighbors(coords)
            
            # kNN距離の平均（最初の要素は自分自身なので除外）
            density = np.mean(distances[:, 1:], axis=1)
            density = 1.0 / (density + 1e-6)  # 距離の逆数で密度を表現
            
            return density
        except Exception:
            return np.ones(len(coords))
    
    def _empty_result(self) -> Dict:
        """空の結果を返す"""
        return {
            'coords': np.array([]).reshape(0, 3),
            'amplitude_normalized': np.array([]),
            'energy_power_normalized': np.array([]),
            'velocity_normalized': np.array([]),
            'density': np.array([]),
            'torso_relative_amp': None,
            'torso_relative_pwr': None,
            'torso_stats': {}
        }


class TorsoModeler:
    """
    胴体モデル化モジュール
    
    RANSACベースの円柱/楕円柱フィッティングで胴体軸を推定
    """
    
    def __init__(
        self,
        ransac_threshold: float = 0.015,  # RANSACの閾値（m）
        min_inliers: int = 7,  # 最小インライア数
        halo_delta: float = 0.06  # 胴体ハローの半径増分（m）
    ):
        """
        Args:
            ransac_threshold: RANSACのインライア閾値
            min_inliers: 最小インライア数
            halo_delta: 胴体ハローの半径増分
        """
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.halo_delta = halo_delta
    
    def fit_torso(
        self,
        points: np.ndarray,  # (N, 6) または (N, 3)
        points_full: Optional[np.ndarray] = None  # 全点群（オプション）
    ) -> Dict:
        """
        胴体軸を推定
        
        Args:
            points: 胴体領域の点群 (N, 6) または (N, 3)
            points_full: 全点群（オプション）
        
        Returns:
            {
                'z_axis': (3,) - 胴体軸方向ベクトル（正規化済み）
                'center': (3,) - 胴体中心
                'radius': float - 胴体半径
                'a_torso': float - 胴体アクティビティ指標
            }
        """
        if len(points) < self.min_inliers:
            # フォールバック: 簡易PCA
            coords = points[:, :3] if points.shape[1] >= 3 else points
            center = np.mean(coords, axis=0)
            coords_centered = coords - center
            
            if len(coords_centered) < 3:
                # デフォルト値
                return {
                    'z_axis': np.array([0, 0, 1]),
                    'center': center,
                    'radius': 0.15,
                    'a_torso': 0.0
                }
            
            # PCAで主軸を計算
            cov = np.cov(coords_centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            max_idx = np.argmax(eigenvalues)
            z_axis = eigenvectors[:, max_idx].real
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)
            
            # 半径を推定（距離の中央値）
            distances = np.linalg.norm(coords_centered, axis=1)
            radius = np.median(distances) if len(distances) > 0 else 0.15
            
            # アクティビティ指標（簡易版）
            a_torso = 0.5  # デフォルト値
            
            return {
                'z_axis': z_axis,
                'center': center,
                'radius': radius,
                'a_torso': a_torso
            }
        
        coords = points[:, :3] if points.shape[1] >= 3 else points
        
        # 簡易PCAベースの胴体推定（完全なRANSACは将来実装）
        center = np.median(coords, axis=0)  # ロバスト中心
        coords_centered = coords - center
        
        # PCAで主軸を計算
        cov = np.cov(coords_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        max_idx = np.argmax(eigenvalues)
        z_axis = eigenvectors[:, max_idx].real
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)
        
        # 半径を推定（主軸からの距離の中央値）
        # 点から主軸への距離を計算
        distances_to_axis = []
        for pt in coords_centered:
            # 主軸への投影
            proj = np.dot(pt, z_axis) * z_axis
            dist = np.linalg.norm(pt - proj)
            distances_to_axis.append(dist)
        
        radius = np.median(distances_to_axis) if len(distances_to_axis) > 0 else 0.15
        
        # アクティビティ指標（速度の分散/平均）
        if points.shape[1] >= 6:
            velocity = points[:, 3]
            vel_mean = np.mean(np.abs(velocity))
            vel_std = np.std(velocity)
            a_torso = vel_std / (vel_mean + 1e-6)
        else:
            a_torso = 0.5
        
        return {
            'z_axis': z_axis,
            'center': center,
            'radius': radius,
            'a_torso': a_torso
        }
    
    def compute_torso_halo_weight(
        self,
        points: np.ndarray,  # (N, 3)
        z_axis: np.ndarray,  # (3,)
        center: np.ndarray,  # (3,)
        radius: float
    ) -> np.ndarray:
        """
        胴体ハロー重みを計算
        
        胴体ハロー内の点は重みを下げる（胴体点を抑制）
        
        Args:
            points: (N, 3) 点群座標
            z_axis: (3,) 胴体軸方向
            center: (3,) 胴体中心
            radius: 胴体半径
        
        Returns:
            weights: (N,) 重み（0-1）
        """
        if len(points) == 0:
            return np.array([])
        
        weights = np.ones(len(points))
        halo_radius = radius + self.halo_delta
        
        for i, pt in enumerate(points):
            pt_centered = pt - center
            # 主軸への投影
            proj = np.dot(pt_centered, z_axis) * z_axis
            dist_to_axis = np.linalg.norm(pt_centered - proj)
            
            if dist_to_axis < halo_radius:
                # ハロー内の点は重みを下げる
                weight = max(0.1, 1.0 - (dist_to_axis / halo_radius) ** 2)
                weights[i] = weight
        
        return weights


class AttributeBasedClusterer:
    """
    属性込み距離クラスタリングモジュール
    
    位置と属性（amplitude, power, velocity, density）を組み合わせた
    複合距離でクラスタリング
    """
    
    def __init__(
        self,
        alpha_amp: float = 1.0,  # amplitudeの重み
        beta_power: float = 1.2,  # energy_powerの重み
        gamma_vel: float = 0.8,  # velocityの重み
        eta_density: float = 0.5,  # densityの重み
        alpha_apr: float = 0.8,  # amplitude-power ratioの重み
        k_lambda: float = 0.7,  # 動的重み調整パラメータ
        cluster_method: str = 'dbscan',  # 'dbscan' or 'optics'
        min_cluster_size: int = 5,
        min_samples: int = 3,
        eps: float = 0.15,  # DBSCANのeps
        sigma_x: float = 0.05  # 位置距離のスケール
    ):
        """
        Args:
            alpha_amp: amplitude正規化値の重み
            beta_power: energy_power正規化値の重み
            gamma_vel: velocity正規化値の重み
            eta_density: densityの重み
            alpha_apr: amplitude-power ratioの重み
            k_lambda: 胴体アクティビティによる動的重み調整の係数
            cluster_method: クラスタリング手法
            min_cluster_size: 最小クラスタサイズ
            min_samples: 最小サンプル数
            eps: DBSCANのeps
            sigma_x: 位置距離のスケール（m）
        """
        self.alpha_amp = alpha_amp
        self.beta_power = beta_power
        self.gamma_vel = gamma_vel
        self.eta_density = eta_density
        self.alpha_apr = alpha_apr
        self.k_lambda = k_lambda
        self.cluster_method = cluster_method
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.eps = eps
        self.sigma_x = sigma_x
    
    def cluster(
        self,
        points: np.ndarray,  # (N, 3)
        features: Dict[str, np.ndarray],  # 正規化された特徴量
        torso_stats: Dict,
        a_torso: float,  # 胴体アクティビティ指標
        mask_w: np.ndarray  # 胴体ハロー重み
    ) -> Dict:
        """
        クラスタリングを実行
        
        Args:
            points: (N, 3) 点群座標
            features: {
                'amplitude_normalized': (N,),
                'energy_power_normalized': (N,),
                'velocity_normalized': (N,),
                'density': (N,),
                ...
            }
            torso_stats: 胴体統計情報
            a_torso: 胴体アクティビティ指標
            mask_w: (N,) 胴体ハロー重み
        
        Returns:
            {
                'labels': (N,) クラスタラベル
                'n_clusters': int クラスタ数
            }
        """
        if len(points) < self.min_cluster_size:
            return {
                'labels': np.zeros(len(points), dtype=int),
                'n_clusters': 0
            }
        
        try:
            from sklearn.cluster import DBSCAN
            
            # 複合距離行列を計算（ベクトル化版）
            n = len(points)
            
            # 1. 位置距離行列をベクトル化して計算
            if SCIPY_AVAILABLE and pdist is not None and squareform is not None:
                # scipyを使って高速に計算
                pos_distances = squareform(pdist(points, metric='euclidean')) / self.sigma_x
            else:
                # フォールバック: NumPyのベクトル演算で計算
                # points[i] - points[j] を一括計算
                # points[i] は (n, 3)、points[j] は (n, 3)
                # ブロードキャストを使って (n, n, 3) の差分を計算
                diff = points[:, None, :] - points[None, :, :]  # (n, n, 3)
                pos_distances = np.linalg.norm(diff, axis=2) / self.sigma_x  # (n, n)
            
            # 2. 属性距離行列をベクトル化して計算
            attr_distances = np.zeros((n, n))
            
            if 'amplitude_normalized' in features:
                amp_diff = np.abs(features['amplitude_normalized'][:, None] - 
                                features['amplitude_normalized'][None, :])
                attr_distances += self.alpha_amp * amp_diff
            
            if 'energy_power_normalized' in features:
                pwr_diff = np.abs(features['energy_power_normalized'][:, None] - 
                                features['energy_power_normalized'][None, :])
                attr_distances += self.beta_power * pwr_diff
            
            if 'velocity_normalized' in features:
                vel_diff = np.abs(features['velocity_normalized'][:, None] - 
                                features['velocity_normalized'][None, :])
                attr_distances += self.gamma_vel * vel_diff
            
            if 'density' in features:
                den_diff = np.abs(features['density'][:, None] - 
                                features['density'][None, :])
                attr_distances += self.eta_density * den_diff
            
            # 3. 複合距離行列 = 位置距離 + 属性距離
            distance_matrix = pos_distances + attr_distances
            
            # 対称行列にする（既に対称になっているはずだが、念のため）
            distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
            
            # DBSCANでクラスタリング
            # 距離行列を利用するため、precomputedを使用
            clusterer = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric='precomputed'
            )
            labels = clusterer.fit_predict(distance_matrix)
            
            # ノイズ（-1）を除外してクラスタ数をカウント
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels >= 0])
            
            return {
                'labels': labels,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            # フォールバック: 簡易クラスタリング
            return {
                'labels': np.zeros(len(points), dtype=int),
                'n_clusters': 0
            }


class IRLSPCA:
    """
    Phase 2: IRLSロバストPCAモジュール
    
    Iteratively Reweighted Least Squares (IRLS) を使用した
    外れ値に強い主軸計算
    """
    
    def __init__(
        self,
        max_iterations: int = 20,
        convergence_threshold: float = 1e-4,
        robust_weight_threshold: float = 0.1,  # 重みがこの値未満の点は外れ値として扱う
        weight_function: str = 'huber'  # 'huber' or 'bisquare'
    ):
        """
        Args:
            max_iterations: 最大反復回数
            convergence_threshold: 収束判定閾値
            robust_weight_threshold: ロバスト重みの閾値
            weight_function: 重み関数の種類
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.robust_weight_threshold = robust_weight_threshold
        self.weight_function = weight_function
    
    def compute_principal_axis(
        self,
        points: np.ndarray,  # (N, 3) 点群座標
        initial_center: Optional[np.ndarray] = None  # 初期重心（オプション）
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        IRLSロバストPCAで主軸を計算
        
        Args:
            points: (N, 3) 点群座標
            initial_center: (3,) 初期重心（Noneの場合は中央値を使用）
        
        Returns:
            principal_axis: (3,) 主軸ベクトル（正規化済み）
            center: (3,) ロバスト重心
            weights: (N,) 各点の重み（0-1）
        """
        if len(points) < 3:
            center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
            principal_axis = np.array([1.0, 0.0, 0.0])
            weights = np.ones(len(points))
            return principal_axis, center, weights
        
        # 初期重心（ロバスト中央値を使用）
        if initial_center is None:
            center = np.median(points, axis=0)
        else:
            center = initial_center.copy()
        
        # 初期化: 全ての点に等しい重み
        weights = np.ones(len(points))
        
        # IRLS反復
        for iteration in range(self.max_iterations):
            prev_center = center.copy()
            
            # 1. 重み付き重心を計算
            if np.sum(weights) > 1e-6:
                center = np.average(points, axis=0, weights=weights)
            else:
                center = np.median(points, axis=0)
            
            # 2. 中心化
            points_centered = points - center
            
            # 3. 重み付き共分散行列を計算
            if np.sum(weights) > 1e-6:
                # 重みを正規化
                weights_normalized = weights / (np.sum(weights) + 1e-6)
                
                # 重み付き共分散行列
                cov = np.zeros((3, 3))
                for i in range(len(points_centered)):
                    pt = points_centered[i].reshape(3, 1)
                    cov += weights_normalized[i] * np.dot(pt, pt.T)
            else:
                # フォールバック: 通常の共分散
                cov = np.cov(points_centered.T)
            
            # 4. 固有値分解で主軸を計算
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
            
            # 最大固有値に対応する固有ベクトル
            max_idx = np.argmax(eigenvalues)
            principal_axis = eigenvectors[:, max_idx]
            principal_axis = principal_axis / (np.linalg.norm(principal_axis) + 1e-6)
            
            # 5. 残差を計算（各点から主軸への距離）
            residuals = np.zeros(len(points_centered))
            for i, pt in enumerate(points_centered):
                # 主軸への投影
                proj = np.dot(pt, principal_axis) * principal_axis
                residual = np.linalg.norm(pt - proj)
                residuals[i] = residual
            
            # 6. ロバスト重みを更新
            if np.std(residuals) > 1e-6:
                # 標準化残差
                median_residual = np.median(residuals)
                mad_residual = np.median(np.abs(residuals - median_residual))  # Median Absolute Deviation
                if mad_residual > 1e-6:
                    standardized_residuals = (residuals - median_residual) / (1.4826 * mad_residual + 1e-6)
                else:
                    standardized_residuals = residuals / (np.std(residuals) + 1e-6)
                
                # 重み関数を適用
                if self.weight_function == 'huber':
                    weights = self._huber_weight(standardized_residuals)
                elif self.weight_function == 'bisquare':
                    weights = self._bisquare_weight(standardized_residuals)
                else:
                    weights = self._huber_weight(standardized_residuals)
            else:
                # 残差が均一な場合は全ての点に等しい重み
                weights = np.ones(len(points))
            
            # 7. 収束判定
            center_change = np.linalg.norm(center - prev_center)
            if center_change < self.convergence_threshold:
                break
        
        return principal_axis, center, weights
    
    def _huber_weight(self, r: np.ndarray, k: float = 1.345) -> np.ndarray:
        """
        Huber重み関数
        
        Args:
            r: 標準化残差
            k: Huber閾値（デフォルト1.345）
        
        Returns:
            weights: (N,) 重み
        """
        weights = np.ones_like(r)
        abs_r = np.abs(r)
        
        # |r| <= k の場合は重み1
        # |r| > k の場合は重み k/|r|
        mask = abs_r > k
        weights[mask] = k / (abs_r[mask] + 1e-6)
        
        return weights
    
    def _bisquare_weight(self, r: np.ndarray, k: float = 4.685) -> np.ndarray:
        """
        Bisquare重み関数
        
        Args:
            r: 標準化残差
            k: Bisquare閾値（デフォルト4.685）
        
        Returns:
            weights: (N,) 重み
        """
        weights = np.zeros_like(r)
        abs_r = np.abs(r)
        
        # |r| <= k の場合は重み (1 - (r/k)^2)^2
        # |r| > k の場合は重み0
        mask = abs_r <= k
        weights[mask] = (1 - (abs_r[mask] / k) ** 2) ** 2
        
        return weights


class CenterlineExtractor:
    """
    Phase 3: 方向場→中心線抽出モジュール
    
    IRLSロバストPCAによる方向場計算、方向重み付きkNNグラフ構築、
    最長経路抽出、Bスプライン平滑化を実行
    """
    
    def __init__(
        self,
        k_pca: int = 20,
        irls_iterations: int = 4,
        tukey_c_scale: float = 1.5,
        sigma_x: float = 0.05,
        sigma_theta: float = 0.12,  # ≈20°
        nms_ang_deg: float = 12.0,
        nms_dist_m: float = 0.06,
        bspline_smooth: float = 0.1,
        k_nn: int = 15
    ):
        """
        Args:
            k_pca: PCA計算に使用するk近傍数
            irls_iterations: IRLS反復回数
            tukey_c_scale: Tukey核のスケール
            sigma_x: 位置距離のスケール
            sigma_theta: 方向角度のスケール
            nms_ang_deg: NMSの角度閾値（度）
            nms_dist_m: NMSの距離閾値（m）
            bspline_smooth: Bスプライン平滑化パラメータ
            k_nn: kNNグラフのk
        """
        self.k_pca = k_pca
        self.irls_iterations = irls_iterations
        self.tukey_c_scale = tukey_c_scale
        self.sigma_x = sigma_x
        self.sigma_theta = sigma_theta
        self.nms_ang_deg = nms_ang_deg
        self.nms_dist_m = nms_dist_m
        self.bspline_smooth = bspline_smooth
        self.k_nn = k_nn
        
        # IRLS PCAモジュールを使用
        self.irls_pca = IRLSPCA(
            max_iterations=irls_iterations,
            convergence_threshold=1e-4,
            weight_function='huber'
        )
    
    def extract_centerline(
        self,
        cluster_points: np.ndarray,  # (N, 3)
        cluster_features: Optional[np.ndarray] = None,
        mask_w: Optional[np.ndarray] = None,
        z_axis: Optional[np.ndarray] = None
    ) -> Dict:
        """
        中心線を抽出
        
        Args:
            cluster_points: (N, 3) クラスタの点群座標
            cluster_features: (N, F) クラスタの特徴量（オプション）
            mask_w: (N,) 重みマスク（オプション）
            z_axis: (3,) 胴体軸方向（オプション）
        
        Returns:
            {
                'centerline': np.ndarray,  # (M, 3) 中心線の点列
                'spline': Optional[BSpline],  # Bスプライン（scipyが利用可能な場合）
                'direction_field': np.ndarray,  # (N, 3) 各点の方向ベクトル
                'linearity': float  # 線状度 [0, 1]
            }
        """
        if len(cluster_points) < 3:
            return self._empty_result()
        
        # 1. 方向場を計算（IRLSロバストPCA）
        direction_field, linearity = self._compute_direction_field(cluster_points)
        
        # 2. 方向重み付きkNNグラフを構築
        if not SCIPY_AVAILABLE:
            # scipyが利用できない場合は簡易版（主軸ベース）
            return self._simple_centerline(cluster_points, direction_field, linearity)
        
        graph = self._build_oriented_knn_graph(
            cluster_points, direction_field, mask_w
        )
        
        # 3. 最長経路を抽出
        path_indices = self._extract_longest_path(graph, cluster_points)
        
        if len(path_indices) < 2:
            # 経路が短すぎる場合は簡易版
            return self._simple_centerline(cluster_points, direction_field, linearity)
        
        path_points = cluster_points[path_indices]
        
        # 4. Bスプライン平滑化
        spline = None
        centerline = path_points
        
        if SCIPY_AVAILABLE and len(path_points) >= 3:
            try:
                # Bスプラインをフィット
                tck, u = splprep(
                    path_points.T,
                    s=self.bspline_smooth * len(path_points),
                    k=min(3, len(path_points) - 1)
                )
                u_new = np.linspace(0, 1, len(path_points))
                centerline = np.array(splev(u_new, tck)).T
                spline = BSpline(tck[0], tck[1], tck[2])
            except Exception:
                # Bスプラインフィットに失敗した場合は元の経路を使用
                centerline = path_points
        
        return {
            'centerline': centerline,
            'spline': spline,
            'direction_field': direction_field,
            'linearity': linearity
        }
    
    def _compute_direction_field(
        self,
        points: np.ndarray  # (N, 3)
    ) -> Tuple[np.ndarray, float]:
        """
        各点の方向場を計算（IRLSロバストPCA）
        
        Returns:
            direction_field: (N, 3) 各点の方向ベクトル
            linearity: float 線状度
        """
        n_points = len(points)
        direction_field = np.zeros((n_points, 3))
        
        if n_points < self.k_pca:
            # 点が少ない場合は全体の主軸を使用
            principal_axis, _, _ = self.irls_pca.compute_principal_axis(points)
            direction_field[:] = principal_axis
            linearity = 0.8  # デフォルト値
            return direction_field, linearity
        
        # k近傍を計算
        try:
            nbrs = NearestNeighbors(n_neighbors=min(self.k_pca + 1, n_points), algorithm='auto')
            nbrs.fit(points)
            distances, indices = nbrs.kneighbors(points)
        except Exception:
            # フォールバック: 全体の主軸を使用
            principal_axis, _, _ = self.irls_pca.compute_principal_axis(points)
            direction_field[:] = principal_axis
            linearity = 0.8
            return direction_field, linearity
        
        # 各点について、k近傍でIRLS PCAを実行
        linearities = []
        for i in range(n_points):
            neighbor_indices = indices[i, 1:]  # 自分自身を除外
            neighbor_points = points[neighbor_indices]
            
            if len(neighbor_points) < 3:
                # 点が少ない場合は全体の主軸を使用
                principal_axis, _, _ = self.irls_pca.compute_principal_axis(points)
                direction_field[i] = principal_axis
                linearities.append(0.8)
            else:
                # IRLS PCAで主軸を計算
                principal_axis, center, weights = self.irls_pca.compute_principal_axis(neighbor_points)
                direction_field[i] = principal_axis
                
                # 線状度を計算（最大固有値 / 全固有値の和）
                neighbor_centered = neighbor_points - center
                if len(neighbor_centered) >= 3:
                    cov = np.cov(neighbor_centered.T)
                    eigenvalues = np.linalg.eigvals(cov).real
                    eigenvalues = np.sort(eigenvalues)[::-1]
                    if eigenvalues[0] > 1e-6:
                        linearity_i = eigenvalues[0] / (np.sum(eigenvalues) + 1e-6)
                        linearities.append(linearity_i)
                    else:
                        linearities.append(0.5)
                else:
                    linearities.append(0.5)
        
        linearity = np.mean(linearities) if len(linearities) > 0 else 0.8
        
        return direction_field, linearity
    
    def _build_oriented_knn_graph(
        self,
        points: np.ndarray,  # (N, 3)
        direction_field: np.ndarray,  # (N, 3)
        mask_w: Optional[np.ndarray] = None
    ) -> csr_matrix:
        """
        方向重み付きkNNグラフを構築
        
        Returns:
            graph: scipy.sparse.csr_matrix (N, N) 重み付きグラフ
        """
        if not SCIPY_AVAILABLE:
            # scipyが利用できない場合は簡易版
            return None
        
        n = len(points)
        edges = []
        weights_list = []
        
        # k近傍を計算
        try:
            nbrs = NearestNeighbors(n_neighbors=min(self.k_nn + 1, n), algorithm='auto')
            nbrs.fit(points)
            distances, indices = nbrs.kneighbors(points)
        except Exception:
            return None
        
        # エッジと重みを計算
        for i in range(n):
            for j_idx, j in enumerate(indices[i, 1:], start=1):  # 自分自身を除外
                # 位置距離
                pos_dist_sq = distances[i, j_idx] ** 2
                w_pos = np.exp(-pos_dist_sq / (2 * self.sigma_x ** 2))
                
                # 方向類似度
                dir_i = direction_field[i]
                dir_j = direction_field[j]
                dir_similarity = np.abs(np.dot(dir_i, dir_j))  # 内積の絶対値
                w_dir = np.exp(-(1.0 - dir_similarity) / self.sigma_theta)
                
                # 重みマスク
                w_mask = mask_w[j] if mask_w is not None else 1.0
                
                # 総合重み
                w_total = w_pos * w_dir * w_mask
                
                edges.append((i, j))
                weights_list.append(w_total)
        
        # CSR行列を構築
        if len(edges) == 0:
            return None
        
        rows = [e[0] for e in edges]
        cols = [e[1] for e in edges]
        data = weights_list
        
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        
        # 対称化（有向グラフの場合は不要だが、ここでは対称化）
        graph = (graph + graph.T) / 2
        
        return graph
    
    def _extract_longest_path(
        self,
        graph: csr_matrix,
        points: np.ndarray  # (N, 3)
    ) -> np.ndarray:
        """
        最長経路を抽出
        
        Returns:
            path_indices: (M,) 経路の点インデックス
        """
        if graph is None or not SCIPY_AVAILABLE:
            # フォールバック: 主軸に沿った経路
            principal_axis, center, _ = self.irls_pca.compute_principal_axis(points)
            points_centered = points - center
            projections = np.dot(points_centered, principal_axis)
            sorted_indices = np.argsort(projections)
            return sorted_indices
        
        n = len(points)
        
        # 簡易版: 距離行列から最長経路を抽出
        # より高度な実装ではMST（最小全域木）を使用
        
        # 各点から最も遠い点への経路を探索
        best_path = []
        best_path_length = 0
        
        # 開始点を試す（端点候補）
        for start_idx in range(min(10, n)):  # 最初の10点のみ試す
            # 幅優先探索またはダイクストラで最長経路を探索
            visited = {start_idx}
            path = [start_idx]
            
            current = start_idx
            while len(path) < n:
                # 現在の点から未訪問の隣接点を探す
                neighbors = graph.getrow(current).indices
                unvisited_neighbors = [n for n in neighbors if n not in visited]
                
                if len(unvisited_neighbors) == 0:
                    break
                
                # 最も重みの高い隣接点を選択
                best_neighbor = None
                best_weight = -1
                for neighbor in unvisited_neighbors:
                    weight = graph[current, neighbor]
                    if weight > best_weight:
                        best_weight = weight
                        best_neighbor = neighbor
                
                if best_neighbor is None:
                    break
                
                path.append(best_neighbor)
                visited.add(best_neighbor)
                current = best_neighbor
            
            # 経路の長さを計算
            path_length = 0
            for i in range(len(path) - 1):
                path_length += np.linalg.norm(points[path[i]] - points[path[i+1]])
            
            if path_length > best_path_length:
                best_path_length = path_length
                best_path = path
        
        if len(best_path) < 2:
            # フォールバック: 主軸に沿った経路
            principal_axis, center, _ = self.irls_pca.compute_principal_axis(points)
            points_centered = points - center
            projections = np.dot(points_centered, principal_axis)
            sorted_indices = np.argsort(projections)
            return sorted_indices
        
        return np.array(best_path)
    
    def _simple_centerline(
        self,
        points: np.ndarray,
        direction_field: np.ndarray,
        linearity: float
    ) -> Dict:
        """scipyが利用できない場合の簡易版"""
        principal_axis, center, _ = self.irls_pca.compute_principal_axis(points)
        points_centered = points - center
        projections = np.dot(points_centered, principal_axis)
        sorted_indices = np.argsort(projections)
        centerline = points[sorted_indices]
        
        return {
            'centerline': centerline,
            'spline': None,
            'direction_field': direction_field,
            'linearity': linearity
        }
    
    def _empty_result(self) -> Dict:
        """空の結果を返す"""
        return {
            'centerline': np.array([]).reshape(0, 3),
            'spline': None,
            'direction_field': np.array([]).reshape(0, 3),
            'linearity': 0.0
        }


class TemporalIntegration:
    """
    Phase 3: 時間統合機能
    
    過去数フレームの点群を統合して、より安定した中心線抽出を可能にする
    """
    
    def __init__(
        self,
        time_window_sec: float = 0.4,  # 0.3-0.5秒
        fps: float = 15.0,
        decay_factor: float = 0.9  # 過去フレームの減衰係数
    ):
        """
        Args:
            time_window_sec: 時間窓の長さ（秒）
            fps: フレームレート
            decay_factor: 過去フレームの減衰係数（0-1）
        """
        self.time_window_sec = time_window_sec
        self.fps = fps
        self.time_window_frames = int(time_window_sec * fps)
        self.decay_factor = decay_factor
        
        # 時間窓のバッファ
        self.point_buffer: List[Tuple[np.ndarray, float]] = []  # [(points, timestamp), ...]
    
    def integrate(
        self,
        current_points: np.ndarray,  # (N, 6)
        timestamp: Optional[float] = None
    ) -> np.ndarray:
        """
        時間統合を実行
        
        Args:
            current_points: (N, 6) 現在フレームの点群
            timestamp: タイムスタンプ（オプション、Noneの場合は自動インクリメント）
        
        Returns:
            integrated_points: (M, 6) 統合された点群
        """
        if timestamp is None:
            # タイムスタンプを自動生成
            if len(self.point_buffer) > 0:
                last_timestamp = self.point_buffer[-1][1]
                timestamp = last_timestamp + (1.0 / self.fps)
            else:
                timestamp = 0.0
        
        # 現在フレームをバッファに追加
        self.point_buffer.append((current_points.copy(), timestamp))
        
        # 古いフレームを削除
        cutoff_time = timestamp - self.time_window_sec
        self.point_buffer = [
            (pts, ts) for pts, ts in self.point_buffer
            if ts >= cutoff_time
        ]
        
        if len(self.point_buffer) == 0:
            return current_points
        
        # 減衰重み付きで統合
        integrated_points_list = []
        for i, (pts, ts) in enumerate(self.point_buffer):
            age = timestamp - ts  # 経過時間
            weight = self.decay_factor ** age  # 減衰重み
            
            # 重みを点群に適用（特徴量のみ、座標はそのまま）
            weighted_pts = pts.copy()
            if len(weighted_pts) > 0 and weighted_pts.shape[1] >= 6:
                weighted_pts[:, 3:6] *= weight  # velocity, amplitude, energy_powerに重みを適用
            
            integrated_points_list.append(weighted_pts)
        
        # 統合
        if len(integrated_points_list) == 1:
            integrated_points = integrated_points_list[0]
        else:
            integrated_points = np.vstack(integrated_points_list)
        
        return integrated_points
    
    def reset(self):
        """バッファをリセット"""
        self.point_buffer = []


class TemporalStateManager:
    """
    Phase 3: HMM/FSM状態管理モジュール
    
    各部位（腕・脚）のActive/Inactive状態を管理し、
    非出現時でも追跡を維持する
    """
    
    def __init__(
        self,
        active_threshold: float = 0.6,
        deactive_threshold: float = 0.2,
        hold_seconds: float = 1.2,  # 1.0-1.5秒
        fps: float = 15.0
    ):
        """
        Args:
            active_threshold: Active状態への遷移閾値
            deactive_threshold: Inactive状態への遷移閾値
            hold_seconds: Inactive状態でも追跡を維持する時間（秒）
            fps: フレームレート
        """
        self.active_threshold = active_threshold
        self.deactive_threshold = deactive_threshold
        self.hold_seconds = hold_seconds
        self.hold_frames = int(hold_seconds * fps)
        
        # 各部位の状態を管理
        self.limb_states: Dict[str, str] = {}  # {'right_arm': 'Active', ...}
        self.limb_scores: Dict[str, float] = {}  # {'right_arm': 0.5, ...}
        self.limb_inactive_frames: Dict[str, int] = {}  # {'right_arm': 0, ...}
    
    def update_state(
        self,
        limb_id: str,
        observation_score: float,
        has_points: bool
    ) -> str:
        """
        状態を更新
        
        Args:
            limb_id: 部位ID（'right_arm', 'left_arm', 'right_leg', 'left_leg'）
            observation_score: 観測スコア [0, 1]
            has_points: 点群が存在するか
        
        Returns:
            state: 'Active' or 'Inactive'
        """
        if limb_id not in self.limb_states:
            # 初期状態
            self.limb_states[limb_id] = 'Inactive'
            self.limb_scores[limb_id] = 0.0
            self.limb_inactive_frames[limb_id] = 0
        
        current_state = self.limb_states[limb_id]
        current_score = self.limb_scores[limb_id]
        
        # スコアを更新（指数移動平均）
        alpha = 0.5  # 平滑化係数（より即座に反応するように）
        if has_points:
            self.limb_scores[limb_id] = alpha * observation_score + (1 - alpha) * current_score
        else:
            # 点群がない場合はスコアを減衰
            self.limb_scores[limb_id] = current_score * 0.9
        
        new_score = self.limb_scores[limb_id]
        
        # 状態遷移（ヒステリシス）
        if current_state == 'Active':
            if new_score < self.deactive_threshold:
                # Active -> Inactive
                self.limb_states[limb_id] = 'Inactive'
                self.limb_inactive_frames[limb_id] = 0
            else:
                # Activeのまま
                self.limb_inactive_frames[limb_id] = 0
        else:  # Inactive
            if new_score > self.active_threshold:
                # Inactive -> Active
                self.limb_states[limb_id] = 'Active'
                self.limb_inactive_frames[limb_id] = 0
            else:
                # Inactiveのまま
                self.limb_inactive_frames[limb_id] += 1
        
        return self.limb_states[limb_id]
    
    def should_track(
        self,
        limb_id: str
    ) -> bool:
        """
        Inactive状態でも追跡を維持するかどうか
        
        Args:
            limb_id: 部位ID
        
        Returns:
            should_track: 追跡を維持するか
        """
        if limb_id not in self.limb_states:
            return False
        
        state = self.limb_states[limb_id]
        inactive_frames = self.limb_inactive_frames.get(limb_id, 0)
        
        if state == 'Active':
            return True
        
        # Inactive状態でも、hold_frames以内なら追跡を維持
        return inactive_frames < self.hold_frames
    
    def reset(self):
        """状態をリセット"""
        self.limb_states = {}
        self.limb_scores = {}
        self.limb_inactive_frames = {}
