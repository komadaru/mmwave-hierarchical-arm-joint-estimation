#!/usr/bin/env python3
"""
Motion-Gated Online Tracking Layer (MG-OTL)

超スパースな3D点群から「部位ごとの粗クラスタ」を作り、
代表点（1点/クラスタ）を抽出し、末端部位は"動いた"ときのみ
カルマンフィルタ(KF)で更新する（胴体はKFなし、EMAのみ）。

参考文献：
- 提案仕様書（最終版）
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import deque
import json
import os
try:
    from scipy.stats import chi2
except ImportError:
    # scipyがインストールされていない場合のフォールバック
    chi2 = None


@dataclass
class TrackingResult:
    """追跡結果"""
    coords: np.ndarray  # (J, 3) - メートル単位の関節座標
    moved: np.ndarray  # (J,) - "動いた"フラグ（bool）
    missing: np.ndarray  # (J,) - 欠測フラグ（bool）
    # デバッグ用（オプション）
    kf_outputs: Optional[np.ndarray] = None  # (J, 3) - KF出力（末端関節のみ）
    correction_magnitudes: Optional[np.ndarray] = None  # (J,) - 補正ベクトルの大きさ


class KalmanFilter:
    """
    カルマンフィルタ（末端関節用）
    状態: 6D (x, y, z, vx, vy, vz)
    """
    
    def __init__(self, initial_pos: np.ndarray, Q_move: float = 0.01, Q_still: float = 0.001, 
                 R: float = 0.05, initial_velocity: Optional[np.ndarray] = None):
        """
        Args:
            initial_pos: 初期位置 (3,)
            Q_move: 移動時のプロセスノイズ（高い）
            Q_still: 静止時のプロセスノイズ（低い）
            R: 観測ノイズ
            initial_velocity: 初期速度 (3,)
        """
        # 状態: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6, dtype=np.float32)
        self.x[:3] = initial_pos.astype(np.float32)
        if initial_velocity is not None:
            self.x[3:] = initial_velocity.astype(np.float32)
        
        # 共分散行列
        self.P = np.eye(6, dtype=np.float32) * 0.1
        
        # プロセスノイズ（移動時と静止時で切替）
        self.Q_move = Q_move
        self.Q_still = Q_still
        self.Q = np.eye(6, dtype=np.float32) * Q_still
        self.Q[:3, :3] *= 0.01  # 位置ノイズ
        self.Q[3:, 3:] *= 0.01  # 速度ノイズ
        
        # 観測ノイズ
        self.R = R
        
        # 状態遷移行列（定速モデル）
        self.F = np.eye(6, dtype=np.float32)
        self.dt = 1.0 / 15.0  # 15 FPS（仮定）
        self.F[:3, 3:] = np.eye(3) * self.dt
        
        # 観測行列（位置のみ観測）
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[:3, :3] = np.eye(3)
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        予測ステップ
        
        Returns:
            predicted_state: (6,) - 予測状態
            predicted_cov: (6, 6) - 予測共分散
        """
        # 状態予測
        self.x = self.F @ self.x
        
        # 共分散予測
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy(), self.P.copy()
    
    def update(self, observation: np.ndarray, R_obs: Optional[float] = None, 
               gain_scale: float = 1.0, clip_norm: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新ステップ（"動いた"時のみ呼ばれる）
        
        Args:
            observation: 観測位置 (3,)
            R_obs: 観測ノイズ（Noneの場合はself.Rを使用）
            gain_scale: ゲインのスケール（0.5-0.8が推奨）
            clip_norm: イノベーションのクリップ半径（m）
        
        Returns:
            updated_state: (6,) - 更新後の状態
            updated_cov: (6, 6) - 更新後の共分散
        """
        if R_obs is None:
            R_obs = self.R
        
        # 観測予測誤差（イノベーション）
        y = observation - self.H @ self.x  # (3,)
        
        # イノベーション・クリップ
        if clip_norm is not None:
            norm = np.linalg.norm(y)
            if norm > clip_norm:
                y = y / norm * clip_norm
        
        # イノベーション共分散
        S = self.H @ self.P @ self.H.T + np.eye(3) * R_obs  # (3, 3)
        
        # カルマンゲイン（数値安定性のため solve を使用）
        PHt = self.P @ self.H.T  # (6, 3)
        K = np.linalg.solve(S.T, PHt.T).T  # (6, 3) - より数値的に安定
        K = K * gain_scale  # ゲイン縮小
        
        # 状態更新
        self.x = self.x + K @ y
        
        # 共分散更新（Joseph形式）
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ (np.eye(3) * R_obs) @ K.T
        
        return self.x.copy(), self.P.copy()
    
    def set_process_noise(self, moved: bool):
        """
        プロセスノイズの切替
        
        Args:
            moved: True=移動時、False=静止時
        """
        if moved:
            self.Q = np.eye(6, dtype=np.float32) * self.Q_move
        else:
            self.Q = np.eye(6, dtype=np.float32) * self.Q_still
        
        self.Q[:3, :3] *= 0.01
        self.Q[3:, 3:] *= 0.01
    
    def get_position(self) -> np.ndarray:
        """現在の位置を取得"""
        return self.x[:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """現在の速度を取得"""
        return self.x[3:].copy()


class MahalanobisGate:
    """
    マハラノビス距離によるゲート（統計的に適切な観測値の採否判定）
    """
    
    def __init__(self, confidence: float = 0.97):
        """
        Args:
            confidence: float - 信頼度（例: 0.97 = 97%）
        """
        self.confidence = confidence
        if chi2 is not None:
            # 自由度3（3D位置）でのカイ二乗閾値
            self.chi2_threshold = chi2.ppf(confidence, df=3)
        else:
            # scipyがない場合のフォールバック（近似値）
            # p=0.97, df=3 → chi2_threshold ≈ 9.35
            self.chi2_threshold = 9.35
    
    def filter_points(
        self,
        points: np.ndarray,
        predicted_pos: np.ndarray,
        covariance: np.ndarray,
        observation_noise: float = 0.03
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        マハラノビス距離でゲート内の点を抽出
        
        Args:
            points: (N, 3) - 候補点群
            predicted_pos: (3,) - KF予測位置
            covariance: (3, 3) - 予測共分散行列（位置部分）
            observation_noise: float - 観測ノイズ（標準偏差）
        
        Returns:
            filtered_points: (M, 3) - ゲート内の点
            mahalanobis_dists: (M,) - マハラノビス距離
        """
        if len(points) == 0:
            return np.zeros((0, 3)), np.array([])
        
        # 観測共分散を加算
        S = covariance + np.eye(3) * (observation_noise ** 2)
        
        # 数値安定性のため、Sの逆行列を計算（可能ならCholesky分解を使用）
        try:
            L = np.linalg.cholesky(S)
            S_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(3)))
        except np.linalg.LinAlgError:
            # 正定値でない場合は通常の逆行列を使用
            S_inv = np.linalg.inv(S)
        
        # 各点のマハラノビス距離を計算
        diff = points - predicted_pos  # (N, 3)
        mahal_dists_sq = np.sum((diff @ S_inv) * diff, axis=1)
        mahal_dists = np.sqrt(mahal_dists_sq)
        
        # ゲート内の点を抽出
        threshold = np.sqrt(self.chi2_threshold)
        in_gate = mahal_dists <= threshold
        filtered_points = points[in_gate]
        filtered_dists = mahal_dists[in_gate]
        
        return filtered_points, filtered_dists
    
    def check_gate(
        self,
        observation: np.ndarray,
        predicted_pos: np.ndarray,
        covariance: np.ndarray,
        observation_noise: float = 0.03
    ) -> Tuple[bool, float]:
        """
        単一観測値がゲート内にあるかチェック
        
        Args:
            observation: (3,) - 観測位置
            predicted_pos: (3,) - KF予測位置
            covariance: (3, 3) - 予測共分散行列（位置部分）
            observation_noise: float - 観測ノイズ（標準偏差）
        
        Returns:
            in_gate: bool - ゲート内にあるか
            mahalanobis_dist: float - マハラノビス距離
        """
        # 観測共分散を加算
        S = covariance + np.eye(3) * (observation_noise ** 2)
        
        # 数値安定性のため、Sの逆行列を計算
        try:
            L = np.linalg.cholesky(S)
            S_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(3)))
        except np.linalg.LinAlgError:
            S_inv = np.linalg.inv(S)
        
        # マハラノビス距離を計算
        diff = observation - predicted_pos
        mahal_dist_sq = diff @ S_inv @ diff
        mahal_dist = np.sqrt(mahal_dist_sq)
        
        # ゲート内かチェック
        threshold = np.sqrt(self.chi2_threshold)
        in_gate = mahal_dist <= threshold
        
        return in_gate, mahal_dist


class ConstantAccelerationKF:
    """
    定加速度カルマンフィルタ（9D状態: [x, y, z, vx, vy, vz, ax, ay, az]）
    """
    
    def __init__(self, initial_pos: np.ndarray, Q_move: float = 0.05, Q_still: float = 0.001, 
                 R: float = 0.03, initial_velocity: Optional[np.ndarray] = None,
                 initial_acceleration: Optional[np.ndarray] = None, dt: float = 1.0 / 15.0):
        """
        Args:
            initial_pos: 初期位置 (3,)
            Q_move: 移動時のプロセスノイズ（高い）
            Q_still: 静止時のプロセスノイズ（低い）
            R: 観測ノイズ
            initial_velocity: 初期速度 (3,)
            initial_acceleration: 初期加速度 (3,)
            dt: タイムステップ（秒）
        """
        # 状態: [x, y, z, vx, vy, vz, ax, ay, az] (9D)
        self.x = np.zeros(9, dtype=np.float32)
        self.x[:3] = initial_pos.astype(np.float32)
        if initial_velocity is not None:
            self.x[3:6] = initial_velocity.astype(np.float32)
        if initial_acceleration is not None:
            self.x[6:9] = initial_acceleration.astype(np.float32)
        
        # 共分散行列
        self.P = np.eye(9, dtype=np.float32) * 0.1
        
        # プロセスノイズ（移動時と静止時で切替）
        self.Q_move = Q_move
        self.Q_still = Q_still
        self.Q = np.eye(9, dtype=np.float32) * Q_still
        self.Q[:3, :3] *= 0.01  # 位置ノイズ
        self.Q[3:6, 3:6] *= 0.01  # 速度ノイズ
        self.Q[6:9, 6:9] *= 0.01  # 加速度ノイズ
        
        # 観測ノイズ
        self.R = R
        
        # 状態遷移行列（定加速度モデル）
        self.dt = dt
        self.F = self._create_transition_matrix(dt)
        
        # 観測行列（位置のみ観測）
        self.H = np.zeros((3, 9), dtype=np.float32)
        self.H[:3, :3] = np.eye(3)
    
    def _create_transition_matrix(self, dt: float) -> np.ndarray:
        """
        定加速度モデルの状態遷移行列（9D）
        
        状態: [x, y, z, vx, vy, vz, ax, ay, az]
        
        F = [I3  dt*I3  0.5*dt^2*I3]
            [0   I3     dt*I3      ]
            [0   0      I3          ]
        """
        F = np.eye(9, dtype=np.float32)
        
        # 位置 += 速度 * dt + 0.5 * 加速度 * dt^2
        F[:3, 3:6] = np.eye(3) * dt
        F[:3, 6:9] = np.eye(3) * (0.5 * dt * dt)
        
        # 速度 += 加速度 * dt
        F[3:6, 6:9] = np.eye(3) * dt
        
        return F
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        予測ステップ（定加速度モデル）
        
        Returns:
            predicted_state: (9,) - 予測状態
            predicted_cov: (9, 9) - 予測共分散
        """
        # 状態予測
        self.x = self.F @ self.x
        
        # 共分散予測
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy(), self.P.copy()
    
    def update(
        self,
        observation: np.ndarray,
        R_obs: Optional[float] = None,
        gain_scale: float = 1.0,
        clip_norm: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新ステップ（マハラノビス・ゲート通過後の観測値）
        
        Args:
            observation: (3,) - 観測位置
            R_obs: float - 観測ノイズ（Noneの場合はself.Rを使用）
            gain_scale: float - ゲインスケール（デフォルト1.0）
            clip_norm: Optional[float] - イノベーションクリップ半径（例: 0.08m）
        
        Returns:
            updated_state: (9,) - 更新後の状態
            updated_cov: (9, 9) - 更新後の共分散
        """
        if R_obs is None:
            R_obs = self.R
        
        # 観測予測誤差（イノベーション）
        y = observation - self.H @ self.x  # (3,)
        
        # イノベーション・クリップ
        if clip_norm is not None:
            norm = np.linalg.norm(y)
            if norm > clip_norm:
                y = y / norm * clip_norm
        
        # イノベーション共分散
        S = self.H @ self.P @ self.H.T + np.eye(3) * R_obs  # (3, 3)
        
        # カルマンゲイン（数値安定性のため solve を使用）
        PHt = self.P @ self.H.T  # (9, 3)
        K = np.linalg.solve(S.T, PHt.T).T  # (9, 3)
        K = K * gain_scale
        
        # 状態更新
        self.x = self.x + K @ y
        
        # 共分散更新（Joseph形式）
        I_KH = np.eye(9) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ (np.eye(3) * R_obs) @ K.T
        
        return self.x.copy(), self.P.copy()
    
    def set_process_noise(self, moved: bool):
        """
        プロセスノイズの切替
        
        Args:
            moved: True=移動時、False=静止時
        """
        if moved:
            self.Q = np.eye(9, dtype=np.float32) * self.Q_move
        else:
            self.Q = np.eye(9, dtype=np.float32) * self.Q_still
        
        self.Q[:3, :3] *= 0.01  # 位置ノイズ
        self.Q[3:6, 3:6] *= 0.01  # 速度ノイズ
        self.Q[6:9, 6:9] *= 0.01  # 加速度ノイズ
    
    def get_position(self) -> np.ndarray:
        """現在の位置を取得"""
        return self.x[:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """現在の速度を取得"""
        return self.x[3:6].copy()
    
    def get_acceleration(self) -> np.ndarray:
        """現在の加速度を取得"""
        return self.x[6:9].copy()
    
    def get_position_covariance(self) -> np.ndarray:
        """位置の共分散行列を取得（3x3）"""
        return self.P[:3, :3].copy()


class SpatialHash:
    """
    空間ハッシュ（ボクセルグリッド）による高速点群管理
    """
    
    def __init__(self, voxel_size: float = 0.12):
        """
        Args:
            voxel_size: ボクセルサイズ（m、例: 0.12m）
        """
        self.voxel_size = voxel_size
        self.hash_table: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
    
    def point_to_voxel(self, point: np.ndarray) -> Tuple[int, int, int]:
        """
        点をボクセルインデックスに変換
        
        Args:
            point: (3,) - 点の座標
        
        Returns:
            voxel_idx: (i, j, k) - ボクセルインデックス
        """
        i = int(np.floor(point[0] / self.voxel_size))
        j = int(np.floor(point[1] / self.voxel_size))
        k = int(np.floor(point[2] / self.voxel_size))
        return (i, j, k)
    
    def build_hash(self, points: np.ndarray) -> None:
        """
        点群からハッシュテーブルを構築
        
        Args:
            points: (N, 3) - 点群座標
        """
        self.hash_table.clear()
        for p in points:
            voxel_idx = self.point_to_voxel(p)
            if voxel_idx not in self.hash_table:
                self.hash_table[voxel_idx] = []
            self.hash_table[voxel_idx].append(p)
    
    def exists(self, voxel_idx: Tuple[int, int, int], check_neighbors: bool = True) -> bool:
        """
        ボクセル（および6近傍）に点が存在するかチェック
        
        Args:
            voxel_idx: (i, j, k) - ボクセルインデックス
            check_neighbors: bool - 6近傍セルもチェックするか
        
        Returns:
            exists: bool - 点が存在するか
        """
        # 現在のボクセルをチェック
        if voxel_idx in self.hash_table and len(self.hash_table[voxel_idx]) > 0:
            return True
        
        if not check_neighbors:
            return False
        
        # 6近傍セルをチェック（±1各軸）
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor_idx = (voxel_idx[0] + dx,
                                   voxel_idx[1] + dy,
                                   voxel_idx[2] + dz)
                    if neighbor_idx in self.hash_table and len(self.hash_table[neighbor_idx]) > 0:
                        return True
        
        return False
    
    def clear(self):
        """ハッシュテーブルをクリア"""
        self.hash_table.clear()


class NewPointDetector:
    """
    新規点検出器（前フレームに存在しなかった点を検出）
    """
    
    def __init__(self, voxel_size: float = 0.12, new_point_radius: float = 0.15, 
                 min_new_neighbors: int = 2):
        """
        Args:
            voxel_size: ボクセルサイズ（m）
            new_point_radius: 局所まとまり検査半径（m）
            min_new_neighbors: 最小近傍点数
        """
        self.voxel_size = voxel_size
        self.new_point_radius = new_point_radius
        self.min_new_neighbors = min_new_neighbors
        self.spatial_hash = SpatialHash(voxel_size)
        self.prev_hash: Optional[SpatialHash] = None
    
    def detect_new_points(
        self,
        curr_points: np.ndarray,
        prev_points: Optional[np.ndarray] = None,
        check_neighbors: bool = True
    ) -> np.ndarray:
        """
        現在フレームの点群から、前フレームに存在しなかった点を検出
        
        Args:
            curr_points: (N, 3) - 現在フレームの点群
            prev_points: (M, 3) - 前フレームの点群（Noneの場合は新規点を全て返す）
            check_neighbors: bool - 6近傍セルもチェックするか
        
        Returns:
            new_points: (K, 3) - 新規点の座標
        """
        if len(curr_points) == 0:
            return np.zeros((0, 3))
        
        # 前フレームがない場合は全て新規点
        if prev_points is None or len(prev_points) == 0:
            return curr_points
        
        # 前フレームのハッシュを構築（初回またはprev_pointsが変更された場合）
        if self.prev_hash is None or not hasattr(self.prev_hash, 'prev_points') or \
           not np.array_equal(self.prev_hash.prev_points, prev_points):
            self.prev_hash = SpatialHash(self.voxel_size)
            self.prev_hash.build_hash(prev_points)
            self.prev_hash.prev_points = prev_points.copy()
        
        # 現在フレームの各点をチェック
        new_points = []
        for p in curr_points:
            voxel_idx = self.spatial_hash.point_to_voxel(p)
            if not self.prev_hash.exists(voxel_idx, check_neighbors=check_neighbors):
                new_points.append(p)
        
        return np.array(new_points) if new_points else np.zeros((0, 3))
    
    def check_local_coherence(
        self,
        new_points: np.ndarray,
        point: np.ndarray
    ) -> bool:
        """
        新規点の周囲に十分な近傍点が存在するかチェック（ノイズ抑制）
        
        Args:
            new_points: (M, 3) - 新規点群
            point: (3,) - チェック対象の点
        
        Returns:
            has_coherence: bool - 局所まとまりがあるか
        """
        if len(new_points) == 0:
            return False
        
        distances = np.linalg.norm(new_points - point, axis=1)
        neighbors = np.sum(distances <= self.new_point_radius)
        return neighbors >= self.min_new_neighbors
    
    def update_prev_frame(self, prev_points: np.ndarray):
        """
        前フレームの点群を更新（次フレーム用）
        
        Args:
            prev_points: (M, 3) - 前フレームの点群
        """
        if self.prev_hash is None:
            self.prev_hash = SpatialHash(self.voxel_size)
        self.prev_hash.build_hash(prev_points)
        self.prev_hash.prev_points = prev_points.copy()


class RadiusClusterer:
    """
    半径クラスタリング（部位非依存）
    """
    
    def __init__(self, radius: float = 0.10, min_pts: int = 3):
        """
        Args:
            radius: クラスタ半径（m）
            min_pts: 最小点数
        """
        self.radius = radius
        self.min_pts = min_pts
    
    def cluster(self, points: np.ndarray, point_features: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        点群をクラスタリング
        
        Args:
            points: (N, 3) - 点群座標（メートル単位）
            point_features: (N, F) - オプション：点群特徴量
        
        Returns:
            clusters: List of (N_i, 3) - 各クラスタの点群
        """
        if len(points) == 0:
            return []
        
        # 空間ハッシュ（簡易版：固定グリッド）
        clusters = []
        visited = np.zeros(len(points), dtype=bool)
        
        for i in range(len(points)):
            if visited[i]:
                continue
            
            # 半径内の点を探索
            distances = np.linalg.norm(points - points[i], axis=1)
            neighbors = np.where((distances <= self.radius) & (~visited))[0]
            
            if len(neighbors) >= self.min_pts:
                # クラスタを形成
                cluster = points[neighbors]
                visited[neighbors] = True
                clusters.append(cluster)
        
        return clusters
    
    def get_cluster_representative(self, cluster: np.ndarray, 
                                   prev_rep: Optional[np.ndarray] = None) -> np.ndarray:
        """
        クラスタから代表点を抽出（メドイド）
        
        Args:
            cluster: (N, 3) - クラスタの点群
            prev_rep: (3,) - 前フレームの代表点（時間的一貫性に使用）
        
        Returns:
            representative: (3,) - 代表点
        """
        if len(cluster) == 0:
            return np.zeros(3, dtype=np.float32)
        
        if len(cluster) == 1:
            return cluster[0]
        
        # メドイド: 重心に最も近い点
        centroid = np.mean(cluster, axis=0)
        distances_to_centroid = np.linalg.norm(cluster - centroid, axis=1)
        medoid_idx = np.argmin(distances_to_centroid)
        
        rep = cluster[medoid_idx]
        
        # 時間的一貫性を考慮（前フレームの代表点が近い場合は優先）
        if prev_rep is not None:
            distances_to_prev = np.linalg.norm(cluster - prev_rep, axis=1)
            closest_to_prev_idx = np.argmin(distances_to_prev)
            
            # 前フレームとの距離が小さい場合はそちらを優先
            if distances_to_prev[closest_to_prev_idx] < 0.15:  # 15cm以内
                rep = cluster[closest_to_prev_idx]
        
        return rep


class MotionGatedTrackingLayer:
    """
    Motion-Gated Online Tracking Layer (MG-OTL)
    
    外付け後処理レイヤ：回帰出力を追跡して安定化
    """
    
    def __init__(self, 
                 num_joints: int = 22,
                 joint_names: Optional[List[str]] = None,
                 cluster_radius: float = 0.18,  # 18cm（補正適用率向上のため調整）
                 cluster_min_pts: int = 2,  # 2点（補正適用率向上のため調整）
                 gate_distance: float = 0.25,  # 25cm（補正適用率向上のため調整）
                 move_threshold_wrist: float = 0.06,  # 6cm（補正適用率向上のため調整）
                 move_threshold_ankle: float = 0.06,  # 6cm（補正適用率向上のため調整）
                 move_threshold_elbow: float = 0.08,  # 8cm（補正適用率向上のため調整）
                 move_threshold_knee: float = 0.08,  # 8cm（補正適用率向上のため調整）
                 ema_alpha_head: float = 0.25,
                 ema_alpha_torso: float = 0.20,
                 kf_Q_move: float = 0.05,  # 🔧 修正: 0.01 → 0.05 (5×) 更新力を向上
                 kf_Q_still: float = 0.001,
                 kf_R: float = 0.035,  # 🔧 修正: 0.05 → 0.035 (0.7×) 観測ノイズを下げて更新力を向上
                 kf_gate: float = 0.20,  # 🔧 修正: 0.18 → 0.20 (20cm) ゲートを少し拡大
                 kf_conflict: float = 0.15,  # 15cm
                 kf_clip: float = 0.06,  # 🔧 修正: 0.04 → 0.06 (6cm) イノベーションクリップを緩和
                 kf_gain_scale: float = 1.0,  # 🔧 修正: 0.7 → 1.0 ゲイン縮小を解除して更新力を向上
                 correction_weight: float = 0.3,  # KF補正の強さ（0.0-1.0）
                 max_correction: float = 0.15,  # 最大補正量（15cm）
                 use_motion_reliability: bool = True,  # 動きの信頼性計算を使用するか
                 energy_threshold: float = 100.0,  # energy_powerが低いと判定する閾値
                 velocity_threshold: float = 0.1):  # velocityが低いと判定する閾値
        """
        Args:
            num_joints: 関節数
            joint_names: 関節名リスト（Noneの場合はデフォルト使用）
            cluster_radius: クラスタ半径（m）
            cluster_min_pts: 最小点数
            gate_distance: 距離ゲート（m）
            move_threshold_*: 各部位の"動いた"判定閾値（m）
            ema_alpha_*: EMAの減衰率
            kf_*: カルマンフィルタのパラメータ
            correction_weight: KF補正の強さ（0.0=回帰のみ、1.0=KFのみ、デフォルト0.3）
            max_correction: 最大補正量（m、デフォルト0.15m）
            use_motion_reliability: 動きの信頼性計算を使用するか（デフォルトTrue）
            energy_threshold: energy_powerが低いと判定する閾値（デフォルト100.0）
            velocity_threshold: velocityが低いと判定する閾値（デフォルト0.1）
        """
        self.num_joints = num_joints
        
        if joint_names is None:
            joint_names = [
                "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
                "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
                "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
                "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
            ]
        self.joint_names = joint_names
        
        # 末端関節のインデックス（KFを使用）
        # 🔧 修正: Shoulder (16, 17) を除外してEMAに移行（観測が稀なため）
        self.distal_indices = [7, 8, 18, 19, 20, 21]  # Ankle, Elbow, Wrist (Shoulderは除外)
        
        # 胴体関節のインデックス（EMAのみ）
        self.torso_indices = [i for i in range(num_joints) if i not in self.distal_indices]
        
        # "動いた"判定閾値
        self.move_thresholds = np.zeros(num_joints, dtype=np.float32)
        for i in [20, 21]:  # Wrist
            if i < num_joints:
                self.move_thresholds[i] = move_threshold_wrist
        for i in [7, 8]:  # Ankle
            if i < num_joints:
                self.move_thresholds[i] = move_threshold_ankle
        for i in [18, 19]:  # Elbow
            if i < num_joints:
                self.move_thresholds[i] = move_threshold_elbow
        # 🔧 修正: Shoulder (16, 17) は末端から除外したため、ここでは設定しない
        
        # クラスタリング
        self.clusterer = RadiusClusterer(radius=cluster_radius, min_pts=cluster_min_pts)
        self.gate_distance = gate_distance
        
        # KFパラメータ
        self.kf_gate = kf_gate
        self.kf_conflict = kf_conflict
        self.kf_clip = kf_clip
        self.kf_gain_scale = kf_gain_scale
        self.kf_Q_move = kf_Q_move
        self.kf_Q_still = kf_Q_still
        self.kf_R = kf_R
        
        # ハイブリッド補正パラメータ
        self.correction_weight = correction_weight  # α: KF補正の強さ
        self.max_correction = max_correction  # 最大補正量（m）
        
        # 動きの信頼性計算パラメータ
        self.use_motion_reliability = use_motion_reliability
        self.energy_threshold = energy_threshold
        self.velocity_threshold = velocity_threshold
        
        # EMAパラメータ
        self.ema_alpha_head = ema_alpha_head  # Head, Neck
        self.ema_alpha_torso = ema_alpha_torso  # その他の胴体
        
        # 状態管理（シーケンスIDごとに分離）
        # key: (sequence_id, joint_idx)
        self.kalman_filters: Dict[Tuple[str, int], KalmanFilter] = {}
        self.ema_positions: Dict[Tuple[str, int], np.ndarray] = {}
        self.representative_points: Dict[Tuple[str, int], np.ndarray] = {}  # 前フレームの代表点
        
        # 新規実装: 前フレーム点群の管理（新規点検出用）
        self.prev_frame_points: Dict[str, np.ndarray] = {}  # シーケンスごとの前フレーム点群
        
        # 新規実装: ヒステリシス用のmoved状態履歴
        self.moved_history: Dict[Tuple[str, int], deque] = {}  # 最大2フレーム
        
        # 新規実装: LOSフレーム数（シーケンス×関節ごと）
        self.los_frames: Dict[Tuple[str, int], int] = {}
        
        # 新規実装: 前フレームで動きがあったか（シーケンス×関節ごと）
        # ノイズではないクラスタが見つかった場合にTrueになる
        self.prev_frame_had_motion: Dict[Tuple[str, int], bool] = {}
        
        # 新規実装: ソフト凍結用のクラスタリング（既存のRadiusClustererを使用）
        # ノイズフィルタリング用のクラスタリング器
        self.soft_freeze_clusterer = RadiusClusterer(radius=0.15, min_pts=2)  # ノイズ除去用
        
        # 新規実装: マハラノビス・ゲート（オプション）
        self.use_mahalanobis_gate = False  # デフォルトはFalse（既存コードを壊さない）
        self.mahalanobis_gate = MahalanobisGate(confidence=0.97)
        
        # 新規実装: 新規点検出器（オプション）
        self.use_new_point_detection = False  # デフォルトはFalse（既存コードを壊さない）
        self.new_point_detector = NewPointDetector(
            voxel_size=0.12,
            new_point_radius=0.15,
            min_new_neighbors=2
        )
        
        # 新規実装: CA-KFを使用するか（オプション）
        self.use_ca_kf = False  # デフォルトはFalse（既存のCV-KFを使用）
        
        # Phase 3: ソフト凍結とLOSモードのパラメータ
        self.use_soft_freeze = False  # デフォルトはFalse（既存コードを壊さない）
        self.soft_freeze_Q_factor = 0.7  # ソフト凍結時のQ倍率
        self.los_Q_expansion = 1.7  # LOS時のQ拡張率
        self.vel_decay_normal = 0.92  # 通常時の速度減衰
        self.vel_decay_los = 0.9  # LOS時の速度減衰
        self.max_step = 0.20  # 最大移動ステップ（m/フレーム）
        self.los_threshold = 2  # LOS移行閾値（連続フレーム数）
        
        # ロギング用
        self.frame_logs = []
    
    def reset(self, sequence_id: Optional[str] = None):
        """
        状態をリセット
        
        Args:
            sequence_id: 特定のシーケンスのみリセット（Noneの場合は全てリセット）
        """
        if sequence_id is None:
            self.kalman_filters.clear()
            self.ema_positions.clear()
            self.representative_points.clear()
            self.prev_frame_points.clear()
            self.moved_history.clear()
            self.los_frames.clear()
            self.prev_frame_had_motion.clear()
            self.frame_logs.clear()
        else:
            # 特定のシーケンスのみリセット
            keys_to_remove = [k for k in self.kalman_filters.keys() if k[0] == sequence_id]
            for k in keys_to_remove:
                del self.kalman_filters[k]
            
            keys_to_remove = [k for k in self.ema_positions.keys() if k[0] == sequence_id]
            for k in keys_to_remove:
                del self.ema_positions[k]
            
            keys_to_remove = [k for k in self.representative_points.keys() if k[0] == sequence_id]
            for k in keys_to_remove:
                del self.representative_points[k]
            
            # 新規実装: 前フレーム点群、moved履歴、LOSフレーム数をリセット
            if sequence_id in self.prev_frame_points:
                del self.prev_frame_points[sequence_id]
            
            keys_to_remove = [k for k in self.moved_history.keys() if k[0] == sequence_id]
            for k in keys_to_remove:
                del self.moved_history[k]
            
            keys_to_remove = [k for k in self.los_frames.keys() if k[0] == sequence_id]
            for k in keys_to_remove:
                del self.los_frames[k]
            
            # prev_frame_had_motionもリセット
            keys_to_remove = [k for k in self.prev_frame_had_motion.keys() if k[0] == sequence_id]
            for k in keys_to_remove:
                del self.prev_frame_had_motion[k]
    
    def initialize_joint(self, sequence_id: str, joint_idx: int, initial_pos: np.ndarray, 
                        initial_velocity: Optional[np.ndarray] = None):
        """
        関節の初期化
        
        Args:
            sequence_id: シーケンスID
            joint_idx: 関節インデックス
            initial_pos: 初期位置 (3,)
            initial_velocity: 初期速度 (3,)（オプション）
        """
        key = (sequence_id, joint_idx)
        if joint_idx in self.distal_indices:
            # 末端: KFを初期化（CA-KFまたはCV-KF）
            if self.use_ca_kf:
                self.kalman_filters[key] = ConstantAccelerationKF(
                    initial_pos=initial_pos,
                    Q_move=self.kf_Q_move,
                    Q_still=self.kf_Q_still,
                    R=self.kf_R,
                    initial_velocity=initial_velocity,
                    dt=1.0 / 15.0  # 15 FPS
                )
            else:
                self.kalman_filters[key] = KalmanFilter(
                    initial_pos=initial_pos,
                    Q_move=self.kf_Q_move,
                    Q_still=self.kf_Q_still,
                    R=self.kf_R,
                    initial_velocity=initial_velocity
                )
        else:
            # 胴体: EMAを初期化
            self.ema_positions[key] = initial_pos.copy()
    
    def process_frame(self, 
                     points: np.ndarray,
                     regression_output: np.ndarray,
                     sequence_id: Optional[str] = None,
                     frame_number: Optional[int] = None,
                     frame_idx: int = 0,
                     debug: bool = False) -> TrackingResult:
        """
        1フレームを処理
        
        Args:
            points: (N, 6) - 点群データ（メートル単位、[x,y,z,velocity,amplitude,energy_power]）
            regression_output: (J, 3) - 回帰モデルの出力（メートル単位）
            sequence_id: シーケンスID（時系列処理に必要）
            frame_number: フレーム番号（時系列処理に必要）
            frame_idx: フレームインデックス（デフォルト: 0）
            debug: デバッグモード
        
        Returns:
            result: TrackingResult
        """
        # シーケンスIDがない場合はデフォルト値を使用（独立フレームとして扱う）
        if sequence_id is None:
            sequence_id = "default"
        # 点群座標と特徴を分離
        point_coords = points[:, :3]  # (N, 3)
        point_features = points[:, 3:] if points.shape[1] >= 6 else None
        
        # Phase 1-3: 新規点検出とソフト凍結用に現在の点群を保存
        self.current_point_coords = point_coords
        self.current_point_features = point_features  # 適応的R計算用に保存
        
        # Phase 1: 新規点検出用に前フレーム点群を更新
        if self.use_new_point_detection:
            # 前フレーム点群を保存（次フレーム用）
            # 注意: 最初のフレームでは前フレームがないため、Noneのまま
            if sequence_id in self.prev_frame_points:
                # 既に前フレームがある場合は、新規点検出器も更新
                self.new_point_detector.update_prev_frame(self.prev_frame_points[sequence_id])
            
            # 現在フレームを前フレームとして保存（次フレーム処理後に更新）
            # 注意: この時点では前フレームとして使用され、処理後に更新される
        
        if debug:
            print(f"    [Tracking Layer] Processing frame {frame_idx}")
            print(f"      Point cloud: {len(point_coords)} points")
            print(f"      Point coords range: x=[{point_coords[:, 0].min():.3f}, {point_coords[:, 0].max():.3f}], "
                  f"y=[{point_coords[:, 1].min():.3f}, {point_coords[:, 1].max():.3f}], "
                  f"z=[{point_coords[:, 2].min():.3f}, {point_coords[:, 2].max():.3f}]")
        
        # 粗クラスタリング
        clusters = self.clusterer.cluster(point_coords, point_features)
        
        if debug:
            print(f"      Found {len(clusters)} clusters")
            for i, cluster in enumerate(clusters[:5]):  # 最初の5クラスタのみ
                print(f"        Cluster {i}: {len(cluster)} points, "
                      f"center=[{cluster.mean(axis=0)[0]:.3f}, {cluster.mean(axis=0)[1]:.3f}, {cluster.mean(axis=0)[2]:.3f}]")
        
        # 各クラスタから代表点を抽出
        representatives = []
        for cluster in clusters:
            # 前フレームの代表点を取得（該当する関節があれば）
            prev_rep = None
            for j in range(self.num_joints):
                key = (sequence_id, j)
                if key in self.representative_points:
                    prev_rep = self.representative_points[key]
                    break
            rep = self.clusterer.get_cluster_representative(cluster, prev_rep=prev_rep)
            representatives.append(rep)
        
        representatives = np.array(representatives) if representatives else np.zeros((0, 3))
        
        if debug:
            print(f"      Representatives: {len(representatives)} points")
            if len(representatives) > 0:
                print(f"      Representative range: x=[{representatives[:, 0].min():.3f}, {representatives[:, 0].max():.3f}], "
                      f"y=[{representatives[:, 1].min():.3f}, {representatives[:, 1].max():.3f}], "
                      f"z=[{representatives[:, 2].min():.3f}, {representatives[:, 2].max():.3f}]")
        
        # 各関節を処理
        tracked_coords = np.zeros((self.num_joints, 3), dtype=np.float32)
        moved_flags = np.zeros(self.num_joints, dtype=bool)
        missing_flags = np.zeros(self.num_joints, dtype=bool)
        
        # デバッグ用（オプション）
        kf_outputs = np.full((self.num_joints, 3), np.nan, dtype=np.float32)  # KF出力
        correction_magnitudes = np.zeros(self.num_joints, dtype=np.float32)  # 補正ベクトルの大きさ
        
        for j in range(self.num_joints):
            if j in self.distal_indices:
                # 末端: KFを使用
                coords, moved, missing, kf_out, corr_mag = self._process_distal_joint_with_debug(
                    sequence_id, j, representatives, regression_output[j], frame_idx
                )
                if not np.isnan(kf_out).any():
                    kf_outputs[j] = kf_out
                correction_magnitudes[j] = corr_mag
            else:
                # 🔧 修正: 胴体関節は回帰出力をそのまま返す（EMA不使用）
                # 修正前は_process_torso_jointを呼んでいたが、直接返すことで確実に回帰出力を使用
                coords = regression_output[j].copy()
                moved = False
                missing = False
                # デバッグ用: 胴体関節はKFを使用しないためNaNのまま
            
            tracked_coords[j] = coords
            moved_flags[j] = moved
            missing_flags[j] = missing
        
        # Phase 1: 新規点検出用に現在フレーム点群を前フレームとして保存
        if self.use_new_point_detection:
            self.prev_frame_points[sequence_id] = point_coords.copy()
        
        # ロギング（オプション）
        self.frame_logs.append({
            'frame': frame_idx,
            'num_clusters': len(clusters),
            'num_representatives': len(representatives)
        })
        
        return TrackingResult(
            coords=tracked_coords,
            moved=moved_flags,
            missing=missing_flags,
            kf_outputs=kf_outputs,
            correction_magnitudes=correction_magnitudes
        )
    
    def _process_distal_joint_with_debug(self, sequence_id: str, joint_idx: int, representatives: np.ndarray, 
                                         reg_output: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, bool, bool, np.ndarray, float]:
        """
        末端関節の処理（KFを使用、デバッグ情報付き）
        
        Returns:
            coords: (3,) - 最終座標
            moved: bool - "動いた"フラグ
            missing: bool - 欠測フラグ
            kf_output: (3,) - KF出力（NaNの場合はKF未使用）
            correction_magnitude: float - 補正ベクトルの大きさ
        """
        return self._process_distal_joint(sequence_id, joint_idx, representatives, reg_output, frame_idx)
    
    def _handle_loss_of_signal(
        self,
        joint_idx: int,
        sequence_id: str,
        has_points_near_pred: bool,
        predicted_pos: np.ndarray,
        predicted_vel: np.ndarray,
        kf: Union[KalmanFilter, ConstantAccelerationKF]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        ソフト凍結とLOSモードの処理
        
        Args:
            joint_idx: int - 関節インデックス
            sequence_id: str - シーケンスID
            has_points_near_pred: bool - 予測位置近くに点があるか
            predicted_pos: (3,) - KF予測位置
            predicted_vel: (3,) - KF予測速度
            kf: KalmanFilter or ConstantAccelerationKF - カルマンフィルタ
        
        Returns:
            updated_pos: (3,) - 更新後の位置
            updated_vel: (3,) - 更新後の速度（減衰適用）
            updated_Q_factor: float - 更新後のQ倍率
        """
        key = (sequence_id, joint_idx)
        
        if not has_points_near_pred:
            # 点がない → LOSフレーム数を増加
            self.los_frames[key] = self.los_frames.get(key, 0) + 1
            m = self.los_frames[key]
            
            # ソフト凍結: Qを低減、速度を減衰
            Q_factor = self.soft_freeze_Q_factor if m == 1 else (self.soft_freeze_Q_factor * (self.los_Q_expansion ** min(m - 1, 3)))
            Q_factor = min(Q_factor, 0.5)  # 上限: 0.5倍
            
            # 速度減衰
            vel_decay = self.vel_decay_normal if m == 1 else (self.vel_decay_normal * (self.vel_decay_los ** min(m - 1, 2)))
            updated_vel = predicted_vel * vel_decay
            
            # 最大移動制約（15-20cm/フレーム）
            dt = kf.dt if hasattr(kf, 'dt') else (1.0 / 15.0)
            step = np.linalg.norm(updated_vel * dt)
            if step > self.max_step:
                updated_vel = updated_vel / step * (self.max_step / dt)
            
            # LOSモード（m >= los_threshold）: Qを段階的に増やす
            if m >= self.los_threshold:
                # Qを段階的に増加（上限: 半径30-40cm等価）
                Q_factor = Q_factor * (self.los_Q_expansion ** min(m - self.los_threshold + 1, 2))
                Q_factor = min(Q_factor, 0.5)  # 上限
            
            updated_Q_factor = Q_factor
            updated_pos = predicted_pos + updated_vel * dt
            
            return updated_pos, updated_vel, updated_Q_factor
        else:
            # 点がある → LOSフレーム数をリセット
            self.los_frames[key] = 0
            return predicted_pos, predicted_vel, 1.0
    
    def _process_distal_joint(self, sequence_id: str, joint_idx: int, representatives: np.ndarray, 
                              reg_output: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, bool, bool, np.ndarray, float]:
        """
        末端関節の処理（KFを使用）
        
        修正: 代表点は"動いた"判定のためだけに使用
        観測値は回帰出力を直接使用（代表点は使用しない）
        
        Args:
            sequence_id: シーケンスID
            joint_idx: 関節インデックス
            representatives: (M, 3) - 代表点（"動いた"判定にのみ使用）
            reg_output: (3,) - 回帰出力（観測値として直接使用）
            frame_idx: フレームインデックス
        
        Returns:
            coords: (3,) - 最終座標
            moved: bool - "動いた"フラグ
            missing: bool - 欠測フラグ
            kf_output: (3,) - KF出力（NaNの場合はKF未使用）
            correction_magnitude: float - 補正ベクトルの大きさ
        """
        key = (sequence_id, joint_idx)
        # KFの初期化確認
        if key not in self.kalman_filters:
            self.initialize_joint(sequence_id, joint_idx, reg_output)
            if hasattr(self, '_debug') and self._debug:
                print(f"        [Distal Joint {joint_idx}] Initialized KF with position: {reg_output}")
        
        kf = self.kalman_filters[key]
        
        # 予測
        x_pred, P_pred = kf.predict()
        pred_pos = x_pred[:3]
        pred_vel = x_pred[3:6] if len(x_pred) >= 6 else np.zeros(3)
        
        if hasattr(self, '_debug') and self._debug and joint_idx in [0, 16, 20]:  # Pelvis, R_Shoulder, L_Wristのみ
            print(f"        [Distal Joint {joint_idx}] KF predicted position: {pred_pos}")
            print(f"        [Distal Joint {joint_idx}] Regression output: {reg_output}")
        
        # 新規点検出ベースの処理（オプション）- ソフト凍結の前に実行
        if self.use_new_point_detection:
            # 前フレーム点群を取得
            prev_points = self.prev_frame_points.get(sequence_id, None)
            
            # 現在フレームの点群座標を取得
            point_coords = None
            if hasattr(self, 'current_point_coords'):
                point_coords = self.current_point_coords
            elif hasattr(self, '_current_point_coords'):
                point_coords = self._current_point_coords
            
            if point_coords is not None and len(point_coords) > 0:
                # 新規点を検出
                new_points = self.new_point_detector.detect_new_points(
                    point_coords,
                    prev_points,
                    check_neighbors=True
                )
                
                if len(new_points) > 0:
                    # マハラノビス・ゲートで新規点をフィルタリング
                    # 位置の共分散行列（3x3）を取得
                    if isinstance(kf, ConstantAccelerationKF):
                        P_pred_pos = P_pred[:3, :3]  # CA-KF: (9, 9) -> (3, 3)
                    else:
                        P_pred_pos = P_pred[:3, :3]  # CV-KF: (6, 6) -> (3, 3)
                    if self.use_mahalanobis_gate:
                        filtered_new_points, mahal_dists = self.mahalanobis_gate.filter_points(
                            new_points,
                            pred_pos,
                            P_pred_pos,
                            observation_noise=self.kf_R
                        )
                    else:
                        # マハラノビス・ゲートが無効の場合は距離ゲートを使用
                        dists_to_pred = np.linalg.norm(new_points - pred_pos, axis=1)
                        in_gate = dists_to_pred <= self.gate_distance
                        filtered_new_points = new_points[in_gate]
                    
                    if len(filtered_new_points) > 0:
                        # 🔧 改善: クラスタリングでノイズフィルタリング
                        # DBSCANライクなクラスタリング（既存のRadiusClustererを使用）
                        clusters = self.soft_freeze_clusterer.cluster(filtered_new_points)
                        
                        # 有効なクラスタ（ノイズではない）が見つかったかチェック
                        has_valid_cluster = len(clusters) > 0
                        
                        # 前フレームで動きがあったか記録（有効なクラスタが見つかった場合）
                        if has_valid_cluster:
                            self.prev_frame_had_motion[key] = True
                        else:
                            # 有効なクラスタが見つからない場合、ノイズの可能性が高い
                            # ただし、前フレームの状態は保持（すぐにはFalseにしない）
                            if key not in self.prev_frame_had_motion:
                                self.prev_frame_had_motion[key] = False
                        
                        # 有効なクラスタが見つからない場合、処理をスキップ
                        if not has_valid_cluster:
                            # ノイズのみ → 回帰出力を返す
                            return reg_output, False, False, np.array([np.nan, np.nan, np.nan]), 0.0
                        
                        # 最大のクラスタを使用（最も信頼性が高い）
                        largest_cluster = max(clusters, key=len)
                        
                        # ロバスト重心を計算（最大クラスタから）
                        robust_center, n_points = compute_robust_center(
                            largest_cluster,
                            radius=0.08  # デフォルト値
                        )
                        
                        # 局所分散を計算（適応的R用）
                        local_dispersion = np.std(np.linalg.norm(largest_cluster - robust_center, axis=1))
                        if np.isnan(local_dispersion) or local_dispersion <= 0:
                            local_dispersion = 0.05  # デフォルト値
                        
                        # 点群特徴量から振幅とエネルギーの分散を計算
                        var_amplitude = 0.5  # デフォルト値
                        var_energy = 0.5  # デフォルト値
                        mean_energy_power = None
                        mean_velocity = None
                        
                        # 新規点に対応する点群特徴量を取得（可能な場合）
                        if hasattr(self, 'current_point_features') and self.current_point_features is not None:
                            # 新規点に対応する特徴量を抽出
                            # 簡易版: 新規点に最も近い点の特徴量を使用
                            if len(self.current_point_features) > 0:
                                # 新規点に対応するインデックスを取得（簡易版）
                                # 実際には、新規点検出時にインデックスも保持する必要があるが、
                                # ここでは簡易的に平均値を使用
                                if self.current_point_features.shape[1] >= 3:
                                    # [velocity, amplitude, energy_power] の順序
                                    var_amplitude = np.var(self.current_point_features[:, 1])  # amplitude
                                    var_energy = np.var(self.current_point_features[:, 2])  # energy_power
                                    
                                    # 動きの信頼性計算用の平均値を取得
                                    if self.use_motion_reliability:
                                        mean_velocity = np.mean(np.abs(self.current_point_features[:, 0]))  # velocity
                                        mean_energy_power = np.mean(self.current_point_features[:, 2])  # energy_power
                        
                        # 適応的Rを計算（動きの信頼性を考慮）
                        adaptive_R = compute_adaptive_R(
                            n_new_points=n_points,
                            local_dispersion=local_dispersion,
                            var_amplitude=var_amplitude,
                            var_energy=var_energy,
                            base_R=self.kf_R,
                            mean_energy_power=mean_energy_power,
                            mean_velocity=mean_velocity,
                            energy_threshold=self.energy_threshold,
                            velocity_threshold=self.velocity_threshold
                        )
                        
                        # "動いた"判定（距離ベース）
                        dist_to_robust = np.linalg.norm(robust_center - pred_pos)
                        moved = dist_to_robust >= self.move_thresholds[joint_idx]
                        
                        # ヒステリシスを適用
                        if key not in self.moved_history:
                            self.moved_history[key] = deque(maxlen=2)
                        moved = apply_hysteresis(moved, self.moved_history[key], require_two_hits=True)
                        
                        # プロセスノイズの切替
                        kf.set_process_noise(moved)
                        
                        if moved:
                            # マハラノビス・ゲートで観測値をチェック
                            if self.use_mahalanobis_gate:
                                in_gate, mahal_dist = self.mahalanobis_gate.check_gate(
                                    robust_center,
                                    pred_pos,
                                    P_pred_pos,
                                    observation_noise=adaptive_R
                                )
                                if not in_gate:
                                    # ゲート外 → 回帰出力を返す
                                    return reg_output, True, True, np.array([np.nan, np.nan, np.nan]), 0.0
                            
                            # KF更新（ロバスト重心を観測値として使用）
                            R_obs = adaptive_R
                            x_upd, P_upd = kf.update(
                                observation=robust_center,
                                R_obs=R_obs,
                                gain_scale=self.kf_gain_scale,
                                clip_norm=self.kf_clip
                            )
                            
                            kf_output = x_upd[:3]
                            
                            # 🔧 新規点検出ベースの処理: 設計書通りにKF出力を直接使用（αブレンドなし）
                            # 新規点検出が有効で新規点が検出された場合、correction_weightに関係なくKF出力を直接使用
                            # これにより、ロバスト重心と適応的Rによる高品質な観測値を最大限活用
                            final_coords = kf_output
                            correction_norm = 0.0
                            
                            return final_coords, True, False, kf_output, correction_norm
                        else:
                            # 静止時：回帰出力を返す
                            # ただし、前フレームで動きがあった場合は記録を保持
                            if key not in self.prev_frame_had_motion:
                                self.prev_frame_had_motion[key] = False
                            return reg_output, False, False, np.array([np.nan, np.nan, np.nan]), 0.0
                else:
                    # 新規点が検出されなかった場合
                    # 前フレームの状態を保持（すぐにはFalseにしない）
                    if key not in self.prev_frame_had_motion:
                        self.prev_frame_had_motion[key] = False
            
            # 新規点が検出されなかった場合、既存の処理にフォールバック
        
        # Phase 3: ソフト凍結とLOSモードのチェック（新規点検出の後に実行）
        if self.use_soft_freeze:
            # 予測位置近くに点があるかチェック
            point_coords = None
            if hasattr(self, 'current_point_coords'):
                point_coords = self.current_point_coords
            elif hasattr(self, '_current_point_coords'):
                point_coords = self._current_point_coords
            
            has_points_near_pred = False
            if point_coords is not None and len(point_coords) > 0:
                distances_to_pred = np.linalg.norm(point_coords - pred_pos, axis=1)
                has_points_near_pred = np.any(distances_to_pred <= self.gate_distance)
            
            # 🔧 改善: ソフト凍結の条件を変更
            # 「前フレームで動きがあった」かつ「今フレームで点がない」場合のみ適用
            prev_had_motion = self.prev_frame_had_motion.get(key, False)
            
            if not has_points_near_pred and prev_had_motion:
                # ソフト凍結を適用: 前フレームで動きがあったが、今フレームで点がない
                # → 停止したと推測
                updated_pos, updated_vel, Q_factor = self._handle_loss_of_signal(
                    joint_idx, sequence_id, has_points_near_pred, pred_pos, pred_vel, kf
                )
                
                # Qを一時的に変更
                original_Q = kf.Q.copy()
                if isinstance(kf, ConstantAccelerationKF):
                    kf.Q = np.eye(9, dtype=np.float32) * (self.kf_Q_still * Q_factor)
                    kf.Q[:3, :3] *= 0.01
                    kf.Q[3:6, 3:6] *= 0.01
                    kf.Q[6:9, 6:9] *= 0.01
                else:
                    kf.Q = np.eye(6, dtype=np.float32) * (self.kf_Q_still * Q_factor)
                    kf.Q[:3, :3] *= 0.01
                    kf.Q[3:, 3:] *= 0.01
                
                # 速度を更新（CA-KFの場合）
                if isinstance(kf, ConstantAccelerationKF) and len(kf.x) >= 6:
                    kf.x[3:6] = updated_vel
                
                # 位置を更新（予測のみ、更新なし）
                if isinstance(kf, ConstantAccelerationKF) and len(kf.x) >= 3:
                    kf.x[:3] = updated_pos
                
                # Qを元に戻す
                kf.Q = original_Q
                
                # ソフト凍結適用時: 予測位置を返す（KF予測ベース）
                return updated_pos, False, True, updated_pos, 0.0
            elif has_points_near_pred:
                # 点がある場合、前フレームの状態をリセット（動きがあった記録を保持）
                # 次のフレームでソフト凍結を適用できるように
                pass  # prev_frame_had_motionは維持（動きがあった記録を保持）
        
        # 既存の処理: 代表点で"動いた"判定（フォールバックまたは新規点検出が無効の場合）
        if len(representatives) > 0:
            # 距離ゲート内の代表点を探索
            dists_to_reps = np.linalg.norm(representatives - pred_pos, axis=1)
            within_gate = dists_to_reps <= self.gate_distance
            
            if np.any(within_gate):
                nearest_rep = representatives[np.argmin(dists_to_reps)]
                dist_to_rep = np.linalg.norm(nearest_rep - pred_pos)
                
                # "動いた"判定
                moved = dist_to_rep >= self.move_thresholds[joint_idx]
                rep_available = True
            else:
                moved = False
                rep_available = False
                nearest_rep = None
        else:
            moved = False
            rep_available = False
            nearest_rep = None
        
        # プロセスノイズの切替
        kf.set_process_noise(moved)
        
        if not moved:
            # 静止時：回帰出力をそのまま返す（KF予測は使用しない）
            return reg_output, False, not rep_available, np.array([np.nan, np.nan, np.nan]), 0.0
        
        # "動いた"場合：KF更新を実行し、その結果を使って回帰出力に補正を加える
        # 🔧 ハイブリッドアプローチ: 回帰出力をベースに、KFの補正を加える
        z_obs = reg_output  # (3,) - 回帰出力を直接観測値として使用
        
        # 安全策1: KF予測との距離ゲート
        dist_to_pred = np.linalg.norm(z_obs - pred_pos)
        if dist_to_pred > self.kf_gate:
            # ゲート外 → 回帰出力をそのまま返す（KFは使用しない）
            return reg_output, True, True, np.array([np.nan, np.nan, np.nan]), 0.0
        
        # KF更新（イノベーション・クリップ、ゲイン縮小付き）
        R_obs = self.kf_R
        x_upd, P_upd = kf.update(
            observation=z_obs,  # 回帰出力を直接使用
            R_obs=R_obs,
            gain_scale=self.kf_gain_scale,
            clip_norm=self.kf_clip
        )
        
        kf_output = x_upd[:3]  # KF更新後の位置
        
        # 🔥 ハイブリッド補正: 回帰出力 + α * (KF出力 - 回帰出力)
        correction = kf_output - reg_output  # 補正ベクトル
        
        # 補正ベクトルの大きさを制限（安全策）
        correction_norm = np.linalg.norm(correction)
        if correction_norm > self.max_correction:
            correction = correction * (self.max_correction / correction_norm)
        
        # 最終座標 = 回帰出力 + α * 補正ベクトル
        final_coords = reg_output + self.correction_weight * correction
        
        if hasattr(self, '_debug') and self._debug and joint_idx in [0, 16, 20]:
            print(f"        [Distal Joint {joint_idx}] Regression output: {reg_output}")
            print(f"        [Distal Joint {joint_idx}] KF updated position: {kf_output}")
            print(f"        [Distal Joint {joint_idx}] Correction vector: {correction}, norm: {np.linalg.norm(correction):.3f}m")
            print(f"        [Distal Joint {joint_idx}] Final coords (hybrid): {final_coords}")
        
        # 代表点を保存（次フレーム用）
        if rep_available:
            self.representative_points[key] = nearest_rep
        
        return final_coords, True, False, kf_output, correction_norm
    
    def _process_torso_joint(self, sequence_id: str, joint_idx: int, representatives: np.ndarray, 
                            reg_output: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        """
        胴体関節の処理（EMAなし、回帰出力をそのまま返す）
        
        修正: 代表点は使用せず、回帰出力を直接返す
        EMAは適用しない（胴体関節は既に高精度（8-10cm）のため、EMAによる遅延は不要）
        
        Args:
            sequence_id: シーケンスID
            joint_idx: 関節インデックス
            representatives: (M, 3) - 代表点（使用しない）
            reg_output: (3,) - 回帰出力（直接返す）
        
        Returns:
            coords: (3,) - 最終座標（回帰出力そのまま）
            moved: bool - "動いた"フラグ（胴体では常にFalse）
            missing: bool - 欠測フラグ（常にFalse）
        """
        # 🔧 修正: 胴体関節はEMAを適用せず、回帰出力をそのまま返す
        # （回帰出力が既に高精度（8-10cm）のため、EMAによる遅延は不要）
        return reg_output, False, False


def compute_robust_center(
    points: np.ndarray,
    radius: float = 0.08
) -> Tuple[np.ndarray, int]:
    """
    ロバスト重心を計算（外れ値に強い）
    
    手順:
    1. 各軸の中央値を計算
    2. 中央値周囲の半径内の点で平均を計算
    3. 点数が1-2の場合は最も近い点を使用
    
    Args:
        points: (N, 3) - 点群
        radius: float - 中央値周囲の検索半径（例: 0.08m）
    
    Returns:
        center: (3,) - ロバスト重心
        n_points: int - 使用した点数
    """
    if len(points) == 0:
        return np.zeros(3), 0
    
    if len(points) == 1:
        return points[0], 1
    
    # ステップ1: 各軸の中央値
    median = np.median(points, axis=0)
    
    # ステップ2: 中央値周囲の半径内の点
    distances_to_median = np.linalg.norm(points - median, axis=1)
    nearby_points = points[distances_to_median <= radius]
    
    if len(nearby_points) >= 2:
        # 平均を計算
        center = np.mean(nearby_points, axis=0)
        n_points = len(nearby_points)
    else:
        # 点数が少ない場合は最も近い点を使用
        closest_idx = np.argmin(distances_to_median)
        center = points[closest_idx]
        n_points = 1
    
    return center, n_points


def compute_point_reliability(
    energy_power: np.ndarray,  # (N,)
    velocity: np.ndarray,  # (N,)
    energy_threshold: float = 100.0,  # energy_powerが低い閾値
    velocity_threshold: float = 0.1  # velocityが低い閾値
) -> np.ndarray:
    """
    点群の信頼性を計算（energy_powerとvelocityが低い = 動いている = 信頼性低い）
    
    Args:
        energy_power: (N,) - エネルギー値
        velocity: (N,) - 速度値
        energy_threshold: energy_powerが低いと判定する閾値
        velocity_threshold: velocityが低いと判定する閾値
    
    Returns:
        reliability: (N,) - 信頼性スコア（0-1、低いほど信頼性が低い）
    """
    # 両方が低い場合（動いている可能性）は信頼性を下げる
    low_energy = energy_power < energy_threshold
    low_velocity = np.abs(velocity) < velocity_threshold
    
    # 両方が低い場合は信頼性を0.3-0.5に下げる
    reliability = np.ones(len(energy_power), dtype=np.float32)
    both_low = low_energy & low_velocity
    reliability[both_low] = 0.4  # 動いている部位は信頼性を下げる
    
    return reliability


def compute_adaptive_R(
    n_new_points: int,
    local_dispersion: float,
    var_amplitude: float,
    var_energy: float,
    base_R: float = 0.03,
    mean_energy_power: Optional[float] = None,  # 新規追加
    mean_velocity: Optional[float] = None,  # 新規追加
    energy_threshold: float = 100.0,  # 新規追加
    velocity_threshold: float = 0.1  # 新規追加
) -> float:
    """
    適応的観測ノイズを計算（energy_powerとvelocityの低さを考慮）
    
    R = base_R * f(n_new) * g(local_disp) * h(var_amp/power) * motion_factor
    
    Args:
        n_new_points: int - 新規点の数
        local_dispersion: float - 局所分散（新規点のばらつき）
        var_amplitude: float - 振幅の分散
        var_energy: float - エネルギーの分散
        base_R: float - ベース観測ノイズ（例: 0.03m）
        mean_energy_power: float - 平均エネルギー値（オプション）
        mean_velocity: float - 平均速度値（オプション）
        energy_threshold: float - energy_powerが低いと判定する閾値
        velocity_threshold: float - velocityが低いと判定する閾値
    
    Returns:
        R: float - 適応的観測ノイズ
    """
    # f(n_new): 点数が少ないほど観測ノイズが大きい
    # n_new >= 3: 1.0, n_new=2: 1.5, n_new=1: 2.5
    if n_new_points >= 3:
        f = 1.0
    elif n_new_points == 2:
        f = 1.5
    else:  # n_new_points == 1
        f = 2.5
    
    # g(local_disp): 分散が大きいほど観測ノイズが大きい
    # 分散が0.05m以下: 1.0, 0.10m: 1.5, 0.15m以上: 2.0
    if local_dispersion <= 0.05:
        g = 1.0
    elif local_dispersion <= 0.10:
        g = 1.5
    else:
        g = 2.0
    
    # h(var_amp/power): 信号強度が低いほど観測ノイズが大きい
    # 簡易版: 振幅とエネルギーの平均分散を使用
    avg_var = (var_amplitude + var_energy) / 2.0
    if avg_var >= 0.5:  # 強い信号
        h = 1.0
    elif avg_var >= 0.3:
        h = 1.5
    else:  # 弱い信号
        h = 2.0
    
    # 新規追加: energy_powerとvelocityが低い場合は観測ノイズを大きくする
    motion_factor = 1.0
    if mean_energy_power is not None and mean_velocity is not None:
        if mean_energy_power < energy_threshold and abs(mean_velocity) < velocity_threshold:
            motion_factor = 2.0  # 動いている部位は観測ノイズを2倍に
    
    R = base_R * f * g * h * motion_factor
    return min(R, base_R * 5.0)  # 上限: 5倍（motion_factorを考慮）


def apply_hysteresis(
    current_moved: bool,
    history: deque,
    require_two_hits: bool = True
) -> bool:
    """
    ヒステリシスを適用（動きの検出を安定化）
    
    ルール:
    - require_two_hits=True: 2連続でTrueで確定、1回でFalseに戻さない
    - require_two_hits=False: 1回でTrue/Falseを切り替え
    
    Args:
        current_moved: bool - 現在のmoved判定
        history: deque - 過去のmoved状態（最大2フレーム）
        require_two_hits: bool - 2連続Trueで確定するか
    
    Returns:
        final_moved: bool - ヒステリシス適用後のmoved状態
    """
    history.append(current_moved)
    
    if len(history) > 2:
        history.popleft()
    
    if require_two_hits:
        # 2連続Trueで確定
        if len(history) >= 2 and all(history):
            return True
        # 1回Falseでは戻さない（前回Trueなら維持）
        if len(history) >= 2 and history[-1] == False and history[-2] == True:
            return True  # 前回Trueを維持
        return False
    else:
        # 1回で切り替え
        return current_moved


def load_points_from_npz(npz_file: str, normalize: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    .npzファイルから点群データを読み込む（Motion-Gated Tracking用）
    
    Args:
        npz_file: .npzファイルのパス
        normalize: 正規化するか（通常はFalse、メートル単位のまま使用）
    
    Returns:
        points: (N, 6) - 点群データ（[x,y,z,velocity,amplitude,energy_power]、メートル単位）
        joints_3d: (22, 3) - GT関節データ（メートル単位、オプション）
    """
    data = np.load(npz_file)
    points = data['points'].astype(np.float32)  # (N, 6)
    
    if normalize:
        # 正規化が必要な場合（通常は不要）
        from mph.datasets.real_data_adapter import RealDataAdapter
        adapter = RealDataAdapter()
        points = adapter.normalize_points(points)
    
    # GT関節データも返す（オプション）
    joints_3d = None
    if 'joints_3d' in data:
        joints_55 = data['joints_3d']
        joints_3d = joints_55[:22].astype(np.float32)  # (22, 3)
    
    return points, joints_3d


def create_tracking_layer_from_joints_def(joints_def_path: Optional[str] = None, 
                                         tracking_params: Optional[Dict] = None,
                                         use_new_point_detection: bool = False,
                                         use_mahalanobis_gate: bool = False,
                                         use_ca_kf: bool = False,
                                         use_soft_freeze: bool = False) -> MotionGatedTrackingLayer:
    """
    関節定義ファイルからTrackingLayerを作成
    
    Args:
        joints_def_path: joints_def_22.jsonのパス
        tracking_params: Tracking Layerのハイパーパラメータ（Noneの場合はデフォルト値を使用）
    
    Returns:
        tracking_layer: MotionGatedTrackingLayerインスタンス
    """
    # 関節定義を読み込み
    possible_paths = [
        joints_def_path,
        "/home/users/grad/2024/24t0010/joints_def_22.json",
        "data_specs/joints_def_22.json",
        "../data_specs/joints_def_22.json",
    ]
    
    joint_names = None
    for path in possible_paths:
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    joints_def = json.load(f)
                joint_names = joints_def['joint_names']
                break
            except:
                continue
    
    # デフォルトパラメータ（補正適用率向上のため調整）
    default_params = {
        'cluster_radius': 0.18,  # 0.10 → 0.18: より大きな半径でクラスタ形成しやすく
        'cluster_min_pts': 2,  # 3 → 2: 最小点数を減らしてクラスタ形成しやすく
        'gate_distance': 0.25,  # 0.15 → 0.25: 距離ゲートを拡大
        'move_threshold_wrist': 0.06,  # 0.10 → 0.06: より敏感に検出
        'move_threshold_ankle': 0.06,  # 0.10 → 0.06: より敏感に検出（特に重要）
        'move_threshold_elbow': 0.08,  # 0.12 → 0.08: より敏感に検出
        'move_threshold_knee': 0.08,  # 0.12 → 0.08: より敏感に検出
        'ema_alpha_head': 0.25,
        'ema_alpha_torso': 0.20,
        'kf_Q_move': 0.05,  # 🔧 修正: 0.01 → 0.05 (5×) 更新力を向上
        'kf_Q_still': 0.001,
        'kf_R': 0.035,  # 🔧 修正: 0.05 → 0.035 (0.7×) 観測ノイズを下げて更新力を向上
        'kf_gate': 0.20,  # 🔧 修正: 0.18 → 0.20 (20cm) ゲートを少し拡大
        'kf_conflict': 0.15,
        'kf_clip': 0.06,  # 🔧 修正: 0.04 → 0.06 (6cm) イノベーションクリップを緩和
        'kf_gain_scale': 1.0,  # 🔧 修正: 0.7 → 1.0 ゲイン縮小を解除して更新力を向上
        'correction_weight': 0.3,  # KF補正の強さ（0.0=回帰のみ、1.0=KFのみ）
        'max_correction': 0.15,  # 最大補正量（15cm）
        'use_motion_reliability': True,  # 動きの信頼性計算を使用するか
        'energy_threshold': 100.0,  # energy_powerが低いと判定する閾値
        'velocity_threshold': 0.1  # velocityが低いと判定する閾値
    }
    
    # 新機能のフラグを事前に取得（tracking_paramsから除外するため）
    new_feature_flags = {}
    tracking_params_copy = None
    if tracking_params is not None:
        # tracking_paramsのコピーを作成（元の辞書を変更しないため）
        import copy
        tracking_params_copy = copy.deepcopy(tracking_params)
        
        # 新機能のフラグを抽出
        new_feature_flags['use_new_point_detection'] = tracking_params_copy.pop('use_new_point_detection', False)
        new_feature_flags['use_mahalanobis_gate'] = tracking_params_copy.pop('use_mahalanobis_gate', False)
        new_feature_flags['use_ca_kf'] = tracking_params_copy.pop('use_ca_kf', False)
        new_feature_flags['use_soft_freeze'] = tracking_params_copy.pop('use_soft_freeze', False)
        
        # 新機能のパラメータも除外
        new_feature_flags['voxel_size'] = tracking_params_copy.pop('voxel_size', 0.12)
        new_feature_flags['new_point_radius'] = tracking_params_copy.pop('new_point_radius', 0.15)
        new_feature_flags['min_new_neighbors'] = tracking_params_copy.pop('min_new_neighbors', 2)
        new_feature_flags['mahalanobis_confidence'] = tracking_params_copy.pop('mahalanobis_confidence', 0.97)
        new_feature_flags['soft_freeze_Q_factor'] = tracking_params_copy.pop('soft_freeze_Q_factor', 0.7)
        new_feature_flags['los_Q_expansion'] = tracking_params_copy.pop('los_Q_expansion', 1.7)
        new_feature_flags['vel_decay_normal'] = tracking_params_copy.pop('vel_decay_normal', 0.92)
        new_feature_flags['vel_decay_los'] = tracking_params_copy.pop('vel_decay_los', 0.9)
        new_feature_flags['max_step'] = tracking_params_copy.pop('max_step', 0.20)
        new_feature_flags['los_threshold'] = tracking_params_copy.pop('los_threshold', 2)
    else:
        # 個別引数が指定されている場合はそれを使用
        new_feature_flags['use_new_point_detection'] = use_new_point_detection
        new_feature_flags['use_mahalanobis_gate'] = use_mahalanobis_gate
        new_feature_flags['use_ca_kf'] = use_ca_kf
        new_feature_flags['use_soft_freeze'] = use_soft_freeze
    
    # default_paramsからも新機能のパラメータを除外（念のため）
    excluded_params = [
        'use_new_point_detection', 'use_mahalanobis_gate', 'use_ca_kf', 'use_soft_freeze',
        'voxel_size', 'new_point_radius', 'min_new_neighbors', 'mahalanobis_confidence',
        'soft_freeze_Q_factor', 'los_Q_expansion', 'vel_decay_normal', 'vel_decay_los',
        'max_step', 'los_threshold'
    ]
    for param in excluded_params:
        default_params.pop(param, None)
    
    # tracking_paramsが指定されている場合はデフォルト値を上書き（コピーを使用）
    if tracking_params_copy is not None:
        default_params.update(tracking_params_copy)
    
    # Tracking Layerを作成
    tracking_layer = MotionGatedTrackingLayer(
        num_joints=22,
        joint_names=joint_names,
        **default_params
    )
    
    # 新機能のフラグを設定
    tracking_layer.use_new_point_detection = new_feature_flags['use_new_point_detection']
    tracking_layer.use_mahalanobis_gate = new_feature_flags['use_mahalanobis_gate']
    tracking_layer.use_ca_kf = new_feature_flags['use_ca_kf']
    tracking_layer.use_soft_freeze = new_feature_flags['use_soft_freeze']
    
    # 新機能のパラメータを設定
    if new_feature_flags['use_new_point_detection']:
        tracking_layer.new_point_detector = NewPointDetector(
            voxel_size=new_feature_flags['voxel_size'],
            new_point_radius=new_feature_flags['new_point_radius'],
            min_new_neighbors=new_feature_flags['min_new_neighbors']
        )
    
    if new_feature_flags['use_mahalanobis_gate']:
        tracking_layer.mahalanobis_gate = MahalanobisGate(
            confidence=new_feature_flags['mahalanobis_confidence']
        )
    
    if new_feature_flags['use_soft_freeze']:
        tracking_layer.soft_freeze_Q_factor = new_feature_flags['soft_freeze_Q_factor']
        tracking_layer.los_Q_expansion = new_feature_flags['los_Q_expansion']
        tracking_layer.vel_decay_normal = new_feature_flags['vel_decay_normal']
        tracking_layer.vel_decay_los = new_feature_flags['vel_decay_los']
        tracking_layer.max_step = new_feature_flags['max_step']
        tracking_layer.los_threshold = new_feature_flags['los_threshold']
    
    return tracking_layer

