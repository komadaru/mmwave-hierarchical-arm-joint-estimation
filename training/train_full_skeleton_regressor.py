#!/usr/bin/env python3
"""
xzヒートマップから全骨格（22関節）座標を回帰するモデルの学習スクリプト
標準的なMSE Lossを使用（末端部位への特殊処理なし）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heatmap_distal_detection.dataset_full_skeleton import FullSkeletonDataset, collate_fn
from heatmap_distal_detection.dataset_full_skeleton_xy import FullSkeletonDatasetXY
from heatmap_distal_detection.models.full_skeleton_regressor import create_full_skeleton_regressor
from heatmap_distal_detection.models.full_skeleton_regressor_xy import create_full_skeleton_regressor_xy


class FullSkeletonTrainer:
    """
    xzヒートマップから全骨格（22関節）座標を回帰するモデルの学習器
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 損失関数: 標準的なMSE Loss（重み付けなし）
        self.criterion = nn.MSELoss(reduction='mean')
        
        # オプティマイザ
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # スケジューラ
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('epochs', 50)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # 出力ディレクトリ（/mnt/d/に保存）
        output_dir = config.get('output_dir', '/mnt/d/full_skeleton_models')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 学習履歴
        self.train_history = {
            'loss': [],
            'val_loss': [],
            'lr': []
        }
        
        # ベストモデル
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.early_stopping_min_delta = config.get('early_stopping_min_delta', 1e-6)
        self.patience_counter = 0
        
        # FP16対応（AMP）
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            from torch.cuda.amp import GradScaler, autocast
            self.scaler = GradScaler()
            self.autocast = autocast
            print("FP16 (AMP) enabled")
        else:
            self.scaler = None
            self.autocast = None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポックの学習"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            # データをデバイスに移動
            heatmaps = batch['heatmap'].to(self.device)  # (B, 1, H, W)
            joint_coords = batch['joint_coords'].to(self.device)  # (B, num_joints*2)
            
            # NaNをマスク
            mask = ~torch.isnan(joint_coords)  # (B, num_joints*2)
            
            # フォワードパス
            self.optimizer.zero_grad()
            
            # FP16対応
            if self.use_amp:
                with self.autocast():
                    pred_coords = self.model(heatmaps)  # (B, num_joints*2)
                    
                    # NaNをマスクしてから損失を計算
                    pred_masked = pred_coords * mask.float()
                    target_masked = torch.nan_to_num(joint_coords, nan=0.0)
                    loss = self.criterion(pred_masked, target_masked)
                
                # バックワードパス（FP16）
                self.scaler.scale(loss).backward()
                
                # 勾配クリッピング
                if self.config.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )
                
                # パラメータ更新
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_coords = self.model(heatmaps)  # (B, num_joints*2)
                
                # NaNをマスクしてから損失を計算
                pred_masked = pred_coords * mask.float()
                target_masked = torch.nan_to_num(joint_coords, nan=0.0)
                loss = self.criterion(pred_masked, target_masked)
                
                # バックワードパス
                loss.backward()
                
                # 勾配クリッピング
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )
                
                # パラメータ更新
                self.optimizer.step()
            
            # 統計更新
            total_loss += loss.item()
            num_batches += 1
            
            # プログレスバー更新
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """検証"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for batch in pbar:
                # データをデバイスに移動
                heatmaps = batch['heatmap'].to(self.device)
                joint_coords = batch['joint_coords'].to(self.device)
                
                # NaNをマスク
                mask = ~torch.isnan(joint_coords)
                
                # フォワードパス（FP16対応）
                if self.use_amp:
                    with self.autocast():
                        pred_coords = self.model(heatmaps)
                        pred_masked = pred_coords * mask.float()
                        target_masked = torch.nan_to_num(joint_coords, nan=0.0)
                        loss = self.criterion(pred_masked, target_masked)
                else:
                    pred_coords = self.model(heatmaps)
                    pred_masked = pred_coords * mask.float()
                    target_masked = torch.nan_to_num(joint_coords, nan=0.0)
                    loss = self.criterion(pred_masked, target_masked)
                
                # 統計更新
                total_loss += loss.item()
                num_batches += 1
                
                # プログレスバー更新
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'val_loss': avg_loss
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """チェックポイントを保存"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'train_history': self.train_history
        }
        
        # 通常のチェックポイント
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # ベストモデル
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def plot_learning_curves(self):
        """学習曲線をプロット"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 損失曲線
        axes[0].plot(self.train_history['loss'], label='Train Loss')
        axes[0].plot(self.train_history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 学習率曲線
        if self.train_history['lr']:
            axes[1].plot(self.train_history['lr'], label='Learning Rate')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_curves.png', dpi=150)
        plt.close()
    
    def train(self, epochs: int):
        """学習ループ"""
        print(f"Starting training for {epochs} epochs...")
        print(f"Model parameters: {self.model.get_num_parameters():,}")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(1, epochs + 1):
            # 学習
            train_metrics = self.train_epoch(epoch)
            
            # 検証
            val_metrics = self.validate(epoch)
            
            # 学習率更新
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # 学習履歴を更新
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['val_loss'].append(val_metrics['val_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['lr'].append(current_lr)
            
            # ベストモデルを保存
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # チェックポイントを保存
            metrics = {**train_metrics, **val_metrics}
            self.save_checkpoint(epoch, metrics, is_best=is_best)
            
            # 学習曲線をプロット
            self.plot_learning_curves()
            
            # ログ出力
            print(f"Epoch {epoch}/{epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
            print(f"  LR: {current_lr:.6f}")
            if is_best:
                print(f"  ✓ New best model! (Val Loss: {val_metrics['val_loss']:.6f})")
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best model was at epoch {self.best_epoch} with Val Loss: {self.best_val_loss:.6f}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best model: epoch {self.best_epoch}, Val Loss: {self.best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train full skeleton (22 joints) regressor from heatmap')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Training data path (JSONL)')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Validation data path (JSONL)')
    parser.add_argument('--output_dir', type=str, default='/mnt/d/full_skeleton_models',
                       help='Output directory for models (default: /mnt/d/full_skeleton_models)')
    parser.add_argument('--plane', type=str, default='xz',
                       choices=['xz', 'xy'],
                       help='Plane to use: xz or xy (default: xz)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--base_channels', type=int, default=32,
                       help='Base number of channels')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping (0 to disable)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use Automatic Mixed Precision (FP16)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--x_range', type=float, nargs=2, default=[-1.0, 1.0],
                       metavar=('X_MIN', 'X_MAX'),
                       help='Fixed x-axis range')
    parser.add_argument('--z_range', type=float, nargs=2, default=[-1.0, 0.75],
                       metavar=('Z_MIN', 'Z_MAX'),
                       help='Fixed z-axis range (for xz plane)')
    parser.add_argument('--y_range', type=float, nargs=2, default=[3.5, 5.0],
                       metavar=('Y_MIN', 'Y_MAX'),
                       help='Fixed y-axis range (for xy plane, default: 3.5 5.0)')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of bins for heatmap')
    parser.add_argument('--feature', type=str, default='energy_power',
                       choices=['velocity', 'energy_power'],
                       help='Feature to use for heatmap')
    parser.add_argument('--normalize_heatmap', action='store_true',
                       help='Normalize heatmap values to 0-1 range (default: False)')
    
    args = parser.parse_args()
    
    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using plane: {args.plane}")
    
    # データセット
    x_range = tuple(args.x_range)
    
    # データファイルの存在確認
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data file not found: {args.train_data}")
    if not os.path.exists(args.val_data):
        raise FileNotFoundError(f"Validation data file not found: {args.val_data}")
    
    # 平面に応じてデータセットとモデルを切り替え
    if args.plane == 'xz':
        z_range = tuple(args.z_range)
        train_dataset = FullSkeletonDataset(
            data_path=args.train_data,
            x_range=x_range,
            z_range=z_range,
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=args.normalize_heatmap
        )
        val_dataset = FullSkeletonDataset(
            data_path=args.val_data,
            x_range=x_range,
            z_range=z_range,
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=args.normalize_heatmap
        )
        create_model_fn = create_full_skeleton_regressor
        range_key = 'z_range'
        range_value = z_range
    else:  # xy
        y_range = tuple(args.y_range)
        train_dataset = FullSkeletonDatasetXY(
            data_path=args.train_data,
            x_range=x_range,
            y_range=y_range,
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=args.normalize_heatmap
        )
        val_dataset = FullSkeletonDatasetXY(
            data_path=args.val_data,
            x_range=x_range,
            y_range=y_range,
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=args.normalize_heatmap
        )
        create_model_fn = create_full_skeleton_regressor_xy
        range_key = 'y_range'
        range_value = y_range
    
    # データセットサイズの確認
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty! Check if {args.train_data} contains valid data.")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty! Check if {args.val_data} contains valid data.")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # データローダー
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # モデル
    model = create_model_fn(
        heatmap_size=args.bins,
        num_joints=22,
        base_channels=args.base_channels,
        dropout=args.dropout
    )
    
    # 設定
    config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'epochs': args.epochs,
        'grad_clip': args.grad_clip,
        'early_stopping_patience': args.early_stopping_patience,
        'use_amp': args.use_amp,
        'output_dir': args.output_dir,
        'x_range': x_range,
        range_key: range_value,
        'bins': args.bins,
        'feature': args.feature,
        'num_joints': 22,
        'plane': args.plane,
        'normalize_heatmap': args.normalize_heatmap
    }
    
    # 学習器
    trainer = FullSkeletonTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # 学習開始
    trainer.train(epochs=args.epochs)


if __name__ == '__main__':
    main()

