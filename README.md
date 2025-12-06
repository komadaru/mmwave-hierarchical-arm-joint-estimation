# Hierarchical Arm Joint Estimation with Joint-Specific Attention for Real-time mmWave Radar Pose Estimation

This repository contains the official implementation of the paper "Hierarchical Arm Joint Estimation with Joint-Specific Attention for Real-time mmWave Radar Pose Estimation".

## Overview

This work proposes a hierarchical framework for estimating arm joint positions from mmWave radar heatmaps. The method combines:
- **Joint-Specific Attention**: Dedicated attention maps for each joint (shoulder, elbow, wrist)
- **Hierarchical Regression**: Sequential prediction from shoulder → elbow → wrist
- **Full-Body Integration**: Integration of full-body model predictions for improved accuracy

## Key Features

- **Real-time Performance**: Achieves 30+ FPS inference speed
- **High Accuracy**: Reduces MPJPE by ~30% for wrist joints compared to baseline methods
- **Lightweight**: Model size of 1-2M parameters, memory usage of 50-100MB

## Results

- **Overall MPJPE (22 joints)**: 8.96 cm
- **Overall MPJPE (6 arm joints)**: 13.15 cm
- **Inference Speed**: 30+ FPS

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA (for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mmwave-pose-estimation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This work uses the [mmBody Benchmark](https://github.com/radar-lab/mmBody) dataset. Please refer to the mmBody repository for dataset download and setup instructions.

**Note**: The dataset files are not included in this repository. You need to download them separately from the mmBody Benchmark website.

## Usage

### Training

#### Train Arm Joint Regressor (xz plane)
```bash
python training/train_arm_joint_regressor.py \
    --train_data <path_to_train_data> \
    --val_data <path_to_val_data> \
    --output_dir <output_directory> \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3
```

#### Train Arm Joint Regressor (xy plane)
```bash
python training/train_arm_joint_regressor_xy.py \
    --train_data <path_to_train_data> \
    --val_data <path_to_val_data> \
    --output_dir <output_directory> \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3
```

#### Train Full Skeleton Regressor
```bash
python training/train_full_skeleton_regressor.py \
    --train_data <path_to_train_data> \
    --val_data <path_to_val_data> \
    --output_dir <output_directory> \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3
```

### Evaluation

#### Evaluate Hybrid Skeleton (3D)
```bash
python evaluation/evaluate_hybrid_skeleton_3d.py \
    --xz_model <path_to_xz_model> \
    --xy_model <path_to_xy_model> \
    --full_skeleton_xz_model <path_to_full_skeleton_xz_model> \
    --full_skeleton_xy_model <path_to_full_skeleton_xy_model> \
    --test_data <path_to_test_data> \
    --output_dir <output_directory>
```

#### Evaluate Arm Joints (3D)
```bash
python evaluation/evaluate_arm_joint_3d.py \
    --xz_model <path_to_xz_model> \
    --xy_model <path_to_xy_model> \
    --test_data <path_to_test_data> \
    --output_dir <output_directory>
```

### Ablation Study

Run the ablation study to evaluate the contribution of each component:
```bash
python ablation/run_ablation_study.py \
    --train_data <path_to_train_data> \
    --val_data <path_to_val_data> \
    --test_data <path_to_test_data> \
    --output_dir <output_directory>
```

### Visualization

#### Create Paper Figures
```bash
python visualization/create_paper_figures.py \
    --evaluation_results <path_to_evaluation_results> \
    --output_dir <output_directory>
```

#### Visualize Architecture
```bash
python visualization/visualize_architecture.py \
    --output_path <output_path>
```

#### Measure Inference Speed
```bash
python visualization/measure_realtime_inference_speed.py \
    --xz_model <path_to_xz_model> \
    --xy_model <path_to_xy_model> \
    --test_data <path_to_test_data> \
    --num_iterations 1000
```

## Project Structure

```
mmwave-pose-estimation/
├── models/              # Model implementations
│   ├── arm_joint_regressor_hierarchical.py
│   ├── full_skeleton_regressor.py
│   └── ...
├── datasets/            # Dataset classes
│   ├── dataset_arm_joints.py
│   ├── dataset_full_skeleton.py
│   └── ...
├── training/            # Training scripts
│   ├── train_arm_joint_regressor.py
│   ├── train_full_skeleton_regressor.py
│   └── ...
├── evaluation/          # Evaluation scripts
│   ├── evaluate_arm_joint_3d.py
│   ├── evaluate_hybrid_skeleton_3d.py
│   └── ...
├── ablation/            # Ablation study scripts
│   └── run_ablation_study.py
├── visualization/       # Visualization scripts
│   ├── create_paper_figures.py
│   ├── visualize_architecture.py
│   └── ...
├── utils/              # Utility functions
│   ├── losses.py
│   ├── evaluation.py
│   └── data_preprocessing.py
├── mph/                # Internal modules
│   └── postprocessing/
└── requirements.txt
```

## Model Architecture

The proposed method consists of four main components:

1. **CNN Encoder**: Extracts features from 50×50 heatmaps
2. **Joint-Specific Attention**: Generates dedicated attention maps for each joint
3. **Hierarchical Regressor**: Predicts joints sequentially (shoulder → elbow → wrist)
4. **Full-Body Integration**: Integrates full-body model predictions

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Hierarchical Arm Joint Estimation with Joint-Specific Attention for Real-time mmWave Radar Pose Estimation},
  author={Your Name and Co-authors},
  journal={Conference/Journal Name},
  year={2024}
}
```

## Acknowledgments

This work uses the [mmBody Benchmark](https://github.com/radar-lab/mmBody) dataset. We thank the authors for providing this valuable dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact [masaharu.kodama.6j@stu.hosei.ac.jp].

