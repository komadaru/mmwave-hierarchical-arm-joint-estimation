#!/usr/bin/env python3
"""
Vision Transformer (ViT) を使用した腕関節回帰モデルの学習スクリプト
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

from heatmap_distal_detection.dataset_arm_joints import ArmJointDataset, collate_fn
from heatmap_distal_detection.dataset_arm_joints_xy import ArmJointDatasetXY
from heatmap_distal_detection.models.vit_arm_joint_regressor import create_vit_arm_joint_regressor


class WeightedMSELoss(nn.Module):
    """
    関節ごとに重み付けされたMSE Loss
    末端関節（Wrist）により大きな重みを付ける
    """
    
    def __init__(
        self,
        joint_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        if joint_weights is None:
            # デフォルト: 末端関節（Wrist）に2倍の重み
            # [L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist]
            weights = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 2.0])
            # 各関節は(x, z/y)の2座標なので、重みを2倍に拡張
            self.joint_weights = weights.repeat_interleave(2).unsqueeze(0)  # (1, 12)
        else:
            self.joint_weights = joint_weights.unsqueeze(0)  # (1, num_joints*2)
        
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, num_joints*2) - 予測関節座標
            target: (B, num_joints*2) - GT関節座標
        
        Returns:
            loss: 重み付けMSE Loss
        """
        # NaNをマスク
        mask = ~torch.isnan(target)  # (B, num_joints*2)
        
        # モデルの出力にNaNやInfがないか確認
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print(f"[WeightedMSELoss] Warning: pred contains NaN or Inf!")
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # MSEを計算
        mse = (pred - target) ** 2  # (B, num_joints*2)
        
        # 重みを適用
        weights = self.joint_weights.to(pred.device)  # (1, num_joints*2)
        weighted_mse = mse * weights * mask.float()  # (B, num_joints*2)
        
        # 有効な要素のみで平均
        if self.reduction == 'mean':
            denominator = (mask.float() * weights).sum()
            if denominator == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            loss = weighted_mse.sum() / denominator
        elif self.reduction == 'sum':
            loss = weighted_mse.sum()
        else:
            loss = weighted_mse
        
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return loss


class VITArmJointTrainer:
    """
    Vision Transformerを使用した腕関節回帰モデルの学習器
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
        loss_type = config.get('loss_type', 'mse')
        if loss_type == 'weighted_mse':
            joint_weights = config.get('joint_weights', None)
            if joint_weights is not None:
                joint_weights = torch.tensor(joint_weights)
            self.criterion = WeightedMSELoss(joint_weights=joint_weights)
        else:
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
        
        # 出力ディレクトリ
        self.output_dir = Path(config.get('output_dir', 'outputs/vit_arm_joint_models'))
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
            heatmaps = batch['heatmap'].to(self.device)  # (B, 1, H, W)
            joint_coords = batch['joint_coords'].to(self.device)  # (B, num_joints*2)
            
            # NaNをマスク
            mask = ~torch.isnan(joint_coords)
            
            # フォワードパス
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with self.autocast():
                    pred_coords = self.model(heatmaps)
                    if isinstance(self.criterion, WeightedMSELoss):
                        loss = self.criterion(pred_coords, joint_coords)
                    else:
                        pred_masked = pred_coords * mask.float()
                        target_masked = torch.nan_to_num(joint_coords, nan=0.0)
                        loss = self.criterion(pred_masked, target_masked)
                
                skip_batch = (loss.item() == 0.0 or torch.isnan(loss) or torch.isinf(loss))
                if skip_batch:
                    total_loss += 0.0
                    num_batches += 1
                    continue
                
                self.scaler.scale(loss).backward()
                if self.config.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_coords = self.model(heatmaps)
                if isinstance(self.criterion, WeightedMSELoss):
                    loss = self.criterion(pred_coords, joint_coords)
                else:
                    pred_masked = pred_coords * mask.float()
                    target_masked = torch.nan_to_num(joint_coords, nan=0.0)
                    loss = self.criterion(pred_masked, target_masked)
                
                skip_batch = (loss.item() == 0.0 or torch.isnan(loss) or torch.isinf(loss))
                if skip_batch:
                    total_loss += 0.0
                    num_batches += 1
                    continue
                
                loss.backward()
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """検証"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for batch in pbar:
                heatmaps = batch['heatmap'].to(self.device)
                joint_coords = batch['joint_coords'].to(self.device)
                
                mask = ~torch.isnan(joint_coords)
                
                if self.use_amp:
                    with self.autocast():
                        pred_coords = self.model(heatmaps)
                        if isinstance(self.criterion, WeightedMSELoss):
                            loss = self.criterion(pred_coords, joint_coords)
                        else:
                            pred_masked = pred_coords * mask.float()
                            target_masked = torch.nan_to_num(joint_coords, nan=0.0)
                            loss = self.criterion(pred_masked, target_masked)
                else:
                    pred_coords = self.model(heatmaps)
                    if isinstance(self.criterion, WeightedMSELoss):
                        loss = self.criterion(pred_coords, joint_coords)
                    else:
                        pred_masked = pred_coords * mask.float()
                        target_masked = torch.nan_to_num(joint_coords, nan=0.0)
                        loss = self.criterion(pred_masked, target_masked)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1
                pbar.set_postfix({'loss': loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else 0.0})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """チェックポイントを保存"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # エポックごとのチェックポイント
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # ベストモデル
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def plot_learning_curves(self):
        """学習曲線をプロット"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.train_history['loss'], label='Train Loss')
        axes[0].plot(self.train_history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
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
        
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['val_loss'].append(val_metrics['val_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['lr'].append(current_lr)
            
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            metrics = {**train_metrics, **val_metrics}
            self.save_checkpoint(epoch, metrics, is_best=is_best)
            self.plot_learning_curves()
            
            print(f"Epoch {epoch}/{epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
            print(f"  LR: {current_lr:.6f}")
            if is_best:
                print(f"  ✓ New best model! (Val Loss: {val_metrics['val_loss']:.6f})")
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best model was at epoch {self.best_epoch} with Val Loss: {self.best_val_loss:.6f}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best model: epoch {self.best_epoch}, Val Loss: {self.best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train ViT arm joint regressor from heatmap')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Training data path (JSONL)')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Validation data path (JSONL)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for models')
    parser.add_argument('--plane', type=str, default='xz',
                       choices=['xz', 'xy'],
                       help='Plane to use: xz or xy (default: xz)')
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Model size: small, medium, or large')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--loss_type', type=str, default='weighted_mse',
                       choices=['mse', 'weighted_mse'],
                       help='Loss function type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--drop_rate', type=float, default=0.0,
                       help='Dropout rate')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0,
                       help='Attention dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.0,
                       help='Drop path rate')
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
    parser.add_argument('--y_range', type=float, nargs=2, default=[2.0, 5.0],
                       metavar=('Y_MIN', 'Y_MAX'),
                       help='Fixed y-axis range (for xy plane)')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of bins for heatmap')
    parser.add_argument('--feature', type=str, default='energy_power',
                       choices=['velocity', 'energy_power'],
                       help='Feature to use for heatmap')
    parser.add_argument('--normalize_heatmap', action='store_true',
                       help='Normalize heatmap values to 0-1 range')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using plane: {args.plane}")
    print(f"Model size: {args.model_size}")
    
    x_range = tuple(args.x_range)
    
    if args.plane == 'xz':
        z_range = tuple(args.z_range)
        train_dataset = ArmJointDataset(
            data_path=args.train_data,
            x_range=x_range,
            z_range=z_range,
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=args.normalize_heatmap
        )
        val_dataset = ArmJointDataset(
            data_path=args.val_data,
            x_range=x_range,
            z_range=z_range,
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=args.normalize_heatmap
        )
        range_key = 'z_range'
        range_value = z_range
    else:  # xy
        y_range = tuple(args.y_range)
        train_dataset = ArmJointDatasetXY(
            data_path=args.train_data,
            x_range=x_range,
            y_range=y_range,
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=args.normalize_heatmap
        )
        val_dataset = ArmJointDatasetXY(
            data_path=args.val_data,
            x_range=x_range,
            y_range=y_range,
            bins=args.bins,
            feature=args.feature,
            normalize_heatmap=args.normalize_heatmap
        )
        range_key = 'y_range'
        range_value = y_range
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # モデルを作成
    model = create_vit_arm_joint_regressor(
        img_size=args.bins,
        patch_size=5,  # 50×50を5×5パッチに分割（10×10=100パッチ）
        num_joints=6,
        model_size=args.model_size,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate
    )
    
    # 設定
    config = {
        'model_type': 'vit',
        'model_size': args.model_size,
        'plane': args.plane,
        'x_range': x_range,
        range_key: range_value,
        'bins': args.bins,
        'feature': args.feature,
        'normalize_heatmap': args.normalize_heatmap,
        'num_joints': 6,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'loss_type': args.loss_type,
        'scheduler': args.scheduler,
        'drop_rate': args.drop_rate,
        'attn_drop_rate': args.attn_drop_rate,
        'drop_path_rate': args.drop_path_rate,
        'grad_clip': args.grad_clip,
        'early_stopping_patience': args.early_stopping_patience,
        'use_amp': args.use_amp,
        'output_dir': args.output_dir
    }
    
    # 学習器を作成
    trainer = VITArmJointTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # 学習開始
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()

