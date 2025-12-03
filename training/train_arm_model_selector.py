#!/usr/bin/env python3
"""
モデル選択器の学習スクリプト

Attentionマップから腕特化モデルと全身モデルのどちらを使うかを判定する分類器を学習
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heatmap_distal_detection.models.arm_model_selector import create_arm_model_selector


class ModelSelectionDataset(Dataset):
    """
    モデル選択器の学習データセット
    """
    
    def __init__(
        self,
        data_path: str,
        plane: str = 'xz'  # 'xz' or 'xy'
    ):
        """
        Args:
            data_path: NPZファイルのパス（train_selection_data_xz.npz など）
            plane: 'xz' or 'xy'
        """
        data = np.load(data_path)
        
        self.attention_left = data['attention_left']  # (N, 3, 6, 6)
        self.attention_right = data['attention_right']  # (N, 3, 6, 6)
        self.labels = data['labels']  # (N,)
        self.arm_mpjpe = data['arm_mpjpe']  # (N,)
        self.full_mpjpe = data['full_mpjpe']  # (N,)
        
        # NaNを除外
        valid_mask = ~np.isnan(self.labels) & ~np.isnan(self.arm_mpjpe) & ~np.isnan(self.full_mpjpe)
        self.attention_left = self.attention_left[valid_mask]
        self.attention_right = self.attention_right[valid_mask]
        self.labels = self.labels[valid_mask]
        self.arm_mpjpe = self.arm_mpjpe[valid_mask]
        self.full_mpjpe = self.full_mpjpe[valid_mask]
        
        print(f"Loaded {len(self.labels)} samples from {data_path}")
        print(f"  Labels: {np.sum(self.labels)} arm model, {np.sum(self.labels==0)} full model")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'attention_left': torch.FloatTensor(self.attention_left[idx]),  # (3, 6, 6)
            'attention_right': torch.FloatTensor(self.attention_right[idx]),  # (3, 6, 6)
            'label': torch.LongTensor([self.labels[idx]])[0],  # scalar
            'arm_mpjpe': self.arm_mpjpe[idx],
            'full_mpjpe': self.full_mpjpe[idx]
        }


class ModelSelectionTrainer:
    """
    モデル選択器の学習器
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
        self.criterion = nn.BCELoss()
        
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
        else:
            self.scheduler = None
        
        # 学習履歴
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1_scores = []
        
        # ベストモデル
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
        # Early stopping
        self.patience = config.get('patience', 10)
        self.early_stopping_counter = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポックの学習"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch in pbar:
            attention_left = batch['attention_left'].to(self.device)  # (B, 3, 6, 6)
            attention_right = batch['attention_right'].to(self.device)  # (B, 3, 6, 6)
            labels = batch['label'].float().to(self.device)  # (B,)
            
            # 予測
            self.optimizer.zero_grad()
            probs = self.model(attention_left, attention_right).squeeze()  # (B,)
            
            # 損失
            loss = self.criterion(probs, labels)
            
            # 逆伝播
            loss.backward()
            self.optimizer.step()
            
            # 統計
            total_loss += loss.item()
            all_preds.extend((probs > 0.5).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # メトリクス計算
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """検証"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch in pbar:
                attention_left = batch['attention_left'].to(self.device)  # (B, 3, 6, 6)
                attention_right = batch['attention_right'].to(self.device)  # (B, 3, 6, 6)
                labels = batch['label'].float().to(self.device)  # (B,)
                
                # 予測
                probs = self.model(attention_left, attention_right).squeeze()  # (B,)
                
                # 損失
                loss = self.criterion(probs, labels)
                
                # 統計
                total_loss += loss.item()
                all_preds.extend((probs > 0.5).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        # メトリクス計算
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # 混同行列
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def train(self, num_epochs: int, output_dir: str):
        """学習ループ"""
        print(f"\n{'='*60}")
        print("Training Model Selector")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            # 学習
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['loss'])
            
            # 検証
            val_metrics = self.validate(epoch)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.val_precisions.append(val_metrics['precision'])
            self.val_recalls.append(val_metrics['recall'])
            self.val_f1_scores.append(val_metrics['f1'])
            
            # スケジューラ更新
            if self.scheduler is not None:
                self.scheduler.step()
            
            # ログ出力
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.6f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.6f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    {val_metrics['confusion_matrix']}")
            
            # ベストモデルを保存
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
                self.early_stopping_counter = 0
                print(f"  ✓ New best model (val_loss: {val_metrics['loss']:.6f})")
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if self.early_stopping_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # ベストモデルを復元
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nBest model restored (val_loss: {self.best_val_loss:.6f}, "
                  f"val_accuracy: {self.best_val_accuracy:.4f})")
        
        # 学習履歴を保存
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_accuracy': self.val_accuracies,
            'val_precision': self.val_precisions,
            'val_recall': self.val_recalls,
            'val_f1': self.val_f1_scores,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy
        }
        
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to: {history_path}")
        
        # モデルを保存
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy
        }
        
        checkpoint_path = os.path.join(output_dir, 'best_model.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Model saved to: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train model selector')
    
    # データパス
    parser.add_argument('--train_data_xz', type=str, required=True,
                        help='Path to train data (xz plane)')
    parser.add_argument('--val_data_xz', type=str, required=True,
                        help='Path to val data (xz plane)')
    parser.add_argument('--train_data_xy', type=str, required=True,
                        help='Path to train data (xy plane)')
    parser.add_argument('--val_data_xy', type=str, required=True,
                        help='Path to val data (xy plane)')
    
    # 出力
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')
    
    # 学習設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    # モデル設定
    parser.add_argument('--attention_size', type=int, nargs=2, default=[6, 6],
                        help='Attention map size (H, W)')
    parser.add_argument('--num_joints_per_arm', type=int, default=3,
                        help='Number of joints per arm')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 設定を保存
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'patience': args.patience,
        'attention_size': args.attention_size,
        'num_joints_per_arm': args.num_joints_per_arm,
        'hidden_dim': args.hidden_dim
    }
    
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # データセットを読み込み
    print("\nLoading datasets...")
    train_dataset_xz = ModelSelectionDataset(args.train_data_xz, plane='xz')
    val_dataset_xz = ModelSelectionDataset(args.val_data_xz, plane='xz')
    train_dataset_xy = ModelSelectionDataset(args.train_data_xy, plane='xy')
    val_dataset_xy = ModelSelectionDataset(args.val_data_xy, plane='xy')
    
    # データローダーを作成
    train_loader_xz = DataLoader(
        train_dataset_xz, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader_xz = DataLoader(
        val_dataset_xz, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    train_loader_xy = DataLoader(
        train_dataset_xy, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader_xy = DataLoader(
        val_dataset_xy, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # xz平面のモデルを学習
    print(f"\n{'='*60}")
    print("Training model selector for XZ plane")
    print(f"{'='*60}")
    
    model_xz = create_arm_model_selector(
        attention_size=tuple(args.attention_size),
        num_joints_per_arm=args.num_joints_per_arm,
        hidden_dim=args.hidden_dim
    )
    
    trainer_xz = ModelSelectionTrainer(
        model=model_xz,
        train_loader=train_loader_xz,
        val_loader=val_loader_xz,
        device=device,
        config=config
    )
    
    output_dir_xz = os.path.join(args.output_dir, 'xz')
    os.makedirs(output_dir_xz, exist_ok=True)
    trainer_xz.train(args.epochs, output_dir_xz)
    
    # xy平面のモデルを学習
    print(f"\n{'='*60}")
    print("Training model selector for XY plane")
    print(f"{'='*60}")
    
    model_xy = create_arm_model_selector(
        attention_size=tuple(args.attention_size),
        num_joints_per_arm=args.num_joints_per_arm,
        hidden_dim=args.hidden_dim
    )
    
    trainer_xy = ModelSelectionTrainer(
        model=model_xy,
        train_loader=train_loader_xy,
        val_loader=val_loader_xy,
        device=device,
        config=config
    )
    
    output_dir_xy = os.path.join(args.output_dir, 'xy')
    os.makedirs(output_dir_xy, exist_ok=True)
    trainer_xy.train(args.epochs, output_dir_xy)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

