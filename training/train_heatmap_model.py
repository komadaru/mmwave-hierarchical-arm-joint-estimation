#!/usr/bin/env python3
"""
グリッドヒートマップベース末端部位検出モデルの学習スクリプト
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

from heatmap_distal_detection.dataset import HeatmapDistalDataset, collate_fn
from heatmap_distal_detection.models.cnn_2d import create_cnn_2d_model
from heatmap_distal_detection.models.cnn_3d import create_cnn_3d_model, create_multiscale_cnn_3d_model
from heatmap_distal_detection.models.multi_axis_multi_feature import create_multi_axis_model
from heatmap_distal_detection.losses import create_loss_function


class HeatmapTrainer:
    """
    ヒートマップモデルの学習器
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
        
        # 損失関数
        self.criterion = create_loss_function(
            loss_type=config.get('loss_type', 'focal_dice'),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            dice_smooth=config.get('dice_smooth', 1.0),
            dice_lambda=config.get('dice_lambda', 1.0),
            class_weights=config.get('class_weights', None)
        )
        
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
        
        # 出力ディレクトリ
        self.output_dir = Path(config.get('output_dir', 'outputs/heatmap_distal_models'))
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
            inputs = batch['input'].to(self.device)  # (B, C, H, W)
            labels = batch['label'].to(self.device)  # (B, H, W)
            
            # フォワードパス
            self.optimizer.zero_grad()
            
            # FP16対応
            if self.use_amp:
                with self.autocast():
                    outputs = self.model(inputs)
                    segmentation = outputs['segmentation']  # (B, num_classes, H, W)
                    loss = self.criterion(segmentation, labels)
                
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
                outputs = self.model(inputs)
                segmentation = outputs['segmentation']  # (B, num_classes, H, W)
                loss = self.criterion(segmentation, labels)
                
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
                inputs = batch['input'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # フォワードパス（FP16対応）
                if self.use_amp:
                    with self.autocast():
                        outputs = self.model(inputs)
                        segmentation = outputs['segmentation']
                        loss = self.criterion(segmentation, labels)
                else:
                    outputs = self.model(inputs)
                    segmentation = outputs['segmentation']
                    loss = self.criterion(segmentation, labels)
                
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


def create_model(
    model_type: str,
    input_mode: str,
    spatial_bins: int,
    feature_bins: int,
    num_classes: int,
    base_channels: int,
    in_channels: Optional[int] = None
) -> nn.Module:
    """モデルを作成"""
    if model_type == 'multi_axis':
        # 多軸・多特徴量モデル
        if in_channels is None:
            # input_modeから自動計算
            in_channels = {'diff': 6, 'concat': 12, 'both': 18}[input_mode]
        return create_multi_axis_model(
            in_channels=in_channels,
            spatial_bins=spatial_bins,
            feature_bins=feature_bins,
            num_classes=num_classes,
            base_channels=base_channels
        )
    elif model_type == 'cnn_2d':
        return create_cnn_2d_model(
            input_mode=input_mode,
            spatial_bins=spatial_bins,
            feature_bins=feature_bins,
            num_classes=num_classes,
            base_channels=base_channels
        )
    elif model_type == 'cnn_3d':
        return create_cnn_3d_model(
            in_channels={'diff': 1, 'concat': 2, 'both': 3}[input_mode],
            spatial_bins=spatial_bins,
            feature_bins=feature_bins,
            num_classes=num_classes,
            base_channels=base_channels
        )
    elif model_type == 'multiscale_3d':
        return create_multiscale_cnn_3d_model(
            input_mode=input_mode,
            spatial_bins=spatial_bins,
            feature_bins=feature_bins,
            num_classes=num_classes,
            base_channels=base_channels
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Train heatmap-based distal detection model')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data (JSONL)')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation data (JSONL)')
    parser.add_argument('--model_type', type=str, default='cnn_2d',
                       choices=['cnn_2d', 'cnn_3d', 'multiscale_3d', 'multi_axis'],
                       help='Model type')
    parser.add_argument('--input_mode', type=str, default='diff',
                       choices=['diff', 'concat', 'both'],
                       help='Input mode')
    parser.add_argument('--spatial_bins', type=int, default=50,
                       help='Number of spatial bins')
    parser.add_argument('--feature_bins', type=int, default=50,
                       help='Number of feature bins')
    parser.add_argument('--base_channels', type=int, default=32,
                       help='Base number of channels')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience (0 to disable)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-6,
                       help='Minimum delta for early stopping')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--loss_type', type=str, default='focal_dice',
                       choices=['focal', 'dice', 'focal_dice', 'ce'],
                       help='Loss function type')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma')
    parser.add_argument('--dice_lambda', type=float, default=1.0,
                       help='Dice loss weight')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping norm')
    parser.add_argument('--output_dir', type=str, default='outputs/heatmap_distal_models',
                       help='Output directory')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use FP16 (AMP) for training')
    
    args = parser.parse_args()
    
    # デバイス
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # データセットを読み込み（最初のサンプルでパラメータを確認）
    print("Loading datasets...")
    train_dataset = HeatmapDistalDataset(args.train_data)
    val_dataset = HeatmapDistalDataset(args.val_data)
    
    # 最初のサンプルからパラメータを取得
    sample = train_dataset[0]
    metadata = sample['metadata']
    spatial_bins = metadata.get('spatial_bins', args.spatial_bins)
    feature_bins = metadata.get('feature_bins', args.feature_bins)
    input_mode = metadata.get('input_mode', args.input_mode)
    use_multi_axis = metadata.get('use_multi_axis', False)
    
    # 入力チャネル数を取得
    input_shape = sample['input'].shape
    in_channels = input_shape[0] if len(input_shape) == 3 else input_shape[1]
    
    print(f"  Spatial bins: {spatial_bins}, Feature bins: {feature_bins}")
    print(f"  Input mode: {input_mode}")
    print(f"  Input channels: {in_channels}")
    if use_multi_axis:
        print(f"  Mode: Multi-axis, Multi-feature")
    else:
        print(f"  Mode: Single-axis, Single-feature (legacy)")
    
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
    
    # モデルを作成
    print(f"Creating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        input_mode=input_mode,
        spatial_bins=spatial_bins,
        feature_bins=feature_bins,
        num_classes=2,
        base_channels=args.base_channels,
        in_channels=in_channels if args.model_type == 'multi_axis' else None
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")
    
    # 設定
    config = {
        'model_type': args.model_type,
        'input_mode': input_mode,
        'spatial_bins': spatial_bins,
        'feature_bins': feature_bins,
        'base_channels': args.base_channels,
        'in_channels': in_channels if args.model_type == 'multi_axis' else None,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'loss_type': args.loss_type,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'dice_lambda': args.dice_lambda,
        'scheduler': args.scheduler,
        'grad_clip': args.grad_clip,
        'output_dir': args.output_dir,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_min_delta': args.early_stopping_min_delta,
        'use_amp': args.use_amp
    }
    
    # 学習器を作成
    trainer = HeatmapTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # 学習ループ
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # 学習
        train_metrics = trainer.train_epoch(epoch)
        
        # 検証
        val_metrics = trainer.validate(epoch)
        
        # スケジューラを更新
        if trainer.scheduler:
            if args.scheduler == 'plateau':
                trainer.scheduler.step(val_metrics['val_loss'])
            else:
                trainer.scheduler.step()
        
        # 学習履歴を更新
        trainer.train_history['loss'].append(train_metrics['loss'])
        trainer.train_history['val_loss'].append(val_metrics['val_loss'])
        if trainer.scheduler:
            trainer.train_history['lr'].append(trainer.optimizer.param_groups[0]['lr'])
        
        # ログ出力
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        if trainer.scheduler:
            print(f"  LR: {trainer.optimizer.param_groups[0]['lr']:.6f}")
        
        # ベストモデルを保存
        is_best = val_metrics['val_loss'] < trainer.best_val_loss
        if is_best:
            trainer.best_val_loss = val_metrics['val_loss']
            trainer.best_epoch = epoch
            trainer.patience_counter = 0  # 改善したらカウンターをリセット
        else:
            # Early stoppingのカウンターを更新
            if trainer.early_stopping_patience > 0:
                improvement = trainer.best_val_loss - val_metrics['val_loss']
                if improvement < trainer.early_stopping_min_delta:
                    trainer.patience_counter += 1
                else:
                    trainer.patience_counter = 0
        
        # チェックポイントを保存
        if epoch % 10 == 0 or is_best:
            trainer.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best=is_best)
        
        # Early stoppingチェック
        if trainer.early_stopping_patience > 0 and trainer.patience_counter >= trainer.early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"  Best epoch: {trainer.best_epoch}")
            print(f"  Best val loss: {trainer.best_val_loss:.4f}")
            print(f"  Patience: {trainer.early_stopping_patience} epochs without improvement")
            break
    
    # 学習曲線をプロット
    trainer.plot_learning_curves()
    
    # 設定を保存
    config_path = Path(args.output_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"  Best epoch: {trainer.best_epoch}")
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()

