#!/usr/bin/env python3
"""
Figure 1: アーキテクチャ図の作成スクリプト

matplotlibを使用してアーキテクチャ図を作成します。
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from pathlib import Path

# 出力ディレクトリ
OUTPUT_DIR = Path("outputs/paper_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_architecture_diagram():
    """Figure 1: アーキテクチャ図を作成"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 色の定義
    color_input = '#E8F4F8'
    color_encoder = '#B3D9E6'
    color_attention = '#FFE6CC'
    color_regressor = '#D4EDDA'
    color_output = '#F8D7DA'
    color_full_body = '#E2E3E5'
    
    # 1. 入力（ヒートマップ）
    input_box = FancyBboxPatch((0.5, 8.5), 1.5, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_input, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 9, 'Heatmap\n(50×50)', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # 2. Encoder
    encoder_box = FancyBboxPatch((0.5, 6.5), 1.5, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=color_encoder, 
                                edgecolor='black', linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(1.25, 7.25, 'CNN Encoder\n(4 Blocks)', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # 3. 特徴マップ
    feature_box = FancyBboxPatch((0.5, 4.5), 1.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=color_encoder, 
                                edgecolor='black', linewidth=2)
    ax.add_patch(feature_box)
    ax.text(1.25, 5, 'Features\n(256×6×6)', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # 4. Joint-Specific Attention（左腕・右腕）
    # 左腕
    attn_left_box = FancyBboxPatch((3, 5.5), 1.2, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=color_attention, 
                                  edgecolor='black', linewidth=2)
    ax.add_patch(attn_left_box)
    ax.text(3.6, 6.25, 'Joint-Specific\nAttention\n(Left Arm)', ha='center', va='center', 
           fontsize=9, fontweight='bold')
    
    # 右腕
    attn_right_box = FancyBboxPatch((3, 3.5), 1.2, 1.5, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=color_attention, 
                                   edgecolor='black', linewidth=2)
    ax.add_patch(attn_right_box)
    ax.text(3.6, 4.25, 'Joint-Specific\nAttention\n(Right Arm)', ha='center', va='center', 
           fontsize=9, fontweight='bold')
    
    # 5. 階層的回帰ヘッド
    # 肩
    shoulder_box = FancyBboxPatch((5, 6.5), 1.2, 0.8, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=color_regressor, 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(shoulder_box)
    ax.text(5.6, 6.9, 'Shoulder\nRegressor', ha='center', va='center', 
           fontsize=9, fontweight='bold')
    
    # 肘
    elbow_box = FancyBboxPatch((5, 5.2), 1.2, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=color_regressor, 
                              edgecolor='black', linewidth=2)
    ax.add_patch(elbow_box)
    ax.text(5.6, 5.6, 'Elbow\nRegressor', ha='center', va='center', 
           fontsize=9, fontweight='bold')
    
    # 手首
    wrist_box = FancyBboxPatch((5, 3.9), 1.2, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=color_regressor, 
                              edgecolor='black', linewidth=2)
    ax.add_patch(wrist_box)
    ax.text(5.6, 4.3, 'Wrist\nRegressor', ha='center', va='center', 
           fontsize=9, fontweight='bold')
    
    # 6. 全身モデル統合（オプション）
    full_body_box = FancyBboxPatch((7, 4.5), 1.5, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=color_full_body, 
                                   edgecolor='black', linewidth=2, 
                                   linestyle='--')
    ax.add_patch(full_body_box)
    ax.text(7.75, 5, 'Full-Body\nModel\n(Optional)', ha='center', va='center', 
           fontsize=9, fontweight='bold', style='italic')
    
    # 7. 出力
    output_box = FancyBboxPatch((8.5, 4.5), 1, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_output, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(9, 5, 'Joint\nCoordinates\n(6 joints)', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # 矢印
    # 入力 → Encoder
    arrow1 = FancyArrowPatch((2, 9), (2, 8), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Encoder → 特徴マップ
    arrow2 = FancyArrowPatch((2, 7.25), (2, 5.5), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # 特徴マップ → Attention
    arrow3 = FancyArrowPatch((2, 5), (3, 6.25), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    arrow4 = FancyArrowPatch((2, 5), (3, 4.25), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # Attention → 回帰ヘッド
    arrow5 = FancyArrowPatch((4.2, 6.25), (5, 6.9), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow5)
    
    arrow6 = FancyArrowPatch((4.2, 5.5), (5, 5.6), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow6)
    
    arrow7 = FancyArrowPatch((4.2, 4.25), (5, 4.3), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow7)
    
    # 肩 → 肘
    arrow8 = FancyArrowPatch((5.6, 6.5), (5.6, 6), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='green', linestyle='--')
    ax.add_patch(arrow8)
    
    # 肘 → 手首
    arrow9 = FancyArrowPatch((5.6, 5.2), (5.6, 4.7), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='green', linestyle='--')
    ax.add_patch(arrow9)
    
    # 全身モデル → 回帰ヘッド
    arrow10 = FancyArrowPatch((7, 5), (6.2, 6.9), 
                             arrowstyle='->', mutation_scale=20, 
                             linewidth=2, color='blue', linestyle='--')
    ax.add_patch(arrow10)
    
    # 回帰ヘッド → 出力
    arrow11 = FancyArrowPatch((6.2, 5), (8.5, 5), 
                             arrowstyle='->', mutation_scale=20, 
                             linewidth=2, color='black')
    ax.add_patch(arrow11)
    
    # ラベル
    ax.text(5.6, 6.1, 'Hierarchical', ha='center', va='center', 
           fontsize=8, color='green', style='italic', rotation=90)
    ax.text(6.6, 6.5, 'Full-Body\nIntegration', ha='center', va='center', 
           fontsize=8, color='blue', style='italic')
    
    # タイトル
    ax.text(5, 9.5, 'Hierarchical Arm Joint Estimation Architecture', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 凡例
    legend_elements = [
        mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input'),
        mpatches.Patch(facecolor=color_encoder, edgecolor='black', label='Encoder'),
        mpatches.Patch(facecolor=color_attention, edgecolor='black', label='Attention'),
        mpatches.Patch(facecolor=color_regressor, edgecolor='black', label='Regressor'),
        mpatches.Patch(facecolor=color_full_body, edgecolor='black', linestyle='--', label='Full-Body Model'),
        mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_1_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure_1_architecture.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Created Figure 1: {OUTPUT_DIR / 'figure_1_architecture.png'}")


if __name__ == '__main__':
    create_architecture_diagram()

