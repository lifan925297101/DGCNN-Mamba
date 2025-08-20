# DGCNN-Mamba: Robust Few-Shot 3D Point Cloud Classification via Cross-Modal Feature Fusion

![Architecture Overview](https://user-images.githubusercontent.com/63827451/195998009-4463e9c3-af93-4ae1-b47c-6f5472c7e0a3.png)

## Introduction

Few-shot classification for 3D point clouds remains a challenging task, primarily because real-world data is often noisy and incomplete, complicating robust feature learning from limited samples. Real-world point clouds are inherently irregular and often contain missing points and noise, posing difficulties for robust classification from scarce data. 

To address these challenges, we propose a novel framework that fuses Mamba and DGCNN for robust few-shot 3D point cloud classification. To overcome challenges from data sparsity and noise in few-shot scenarios, our model creates a holistic object representation. It moves beyond purely local feature extraction by fusing the detailed geometric information from a DGCNN with the global, long-range context provided efficiently by a Mamba architecture. This fusion of local and global features enhances the model's resilience to partial artifacts and sensor noise. 

Furthermore, we introduce a multi-bin Cross-Instance Adaptation (bCIA) module to more comprehensively model the correlations between support and query sets, mitigating the problem of large intra-class variance and small inter-class variance that arises from data scarcity. Rigorous evaluation across ModelNet40, ModelNet40-C, and ScanObjectNN benchmarks validates the efficacy of our approach. The proposed framework consistently surpasses established baselines, achieving a 1.46% improvement on the ScanObjectNN dataset in the 5-way 5-shot setting.

## Architecture Overview

Our proposed framework consists of two main branches:

**Point Cloud Branch (Bottom):**
- **Inputs:** Support Point Clouds (P^S) and Query Point Clouds (P^Q)
- **DGCNN:** Processes input point clouds to extract local geometric features
- **Hilbert & Trans-Hilbert:** Dual-path processing for enhanced feature representation
- **Mamba Block:** Captures global, long-range dependencies efficiently
- **Feature Fusion:** Combines local DGCNN features with global Mamba context

**Cross-Modal Fusion:**
- **bCIA Module:** Multi-bin Cross-Instance Adaptation for comprehensive support-query correlation modeling
- **Feature Integration:** Fuses geometric and contextual information for robust classification

## Installation

This project is built upon the following environment:
* Install Python 3.6+
* Install CUDA 11.0+
* Install PyTorch 1.10.2+

The package requirements include:
* pytorch==1.10.2
* tqdm==4.63.1
* tensorboard==2.8.0

## Datasets

* Download [ModelNet40](https://modelnet.cs.princeton.edu/)
* Download [ModelNet40-C from Google Drive](https://drive.google.com/drive/folders/10YeQRh92r_WdL-Dnog2zQfFr03UW4qXX)
* Download [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)

All data used in this project is in .npy format.

## Training

Train a model on the ModelNet40 dataset by:
```bash
python main.py --dataset modelnet40 --fs_head crossAtt_mixmodel
```

## Evaluation

```bash
python main.py --train False
```

## Results

Our framework achieves significant improvements over baseline methods:
- **ModelNet40:** Consistent performance improvement in few-shot scenarios
- **ModelNet40-C:** Robust classification under corruption and noise
- **ScanObjectNN:** **1.46% improvement** in 5-way 5-shot setting

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{dgcnn-mamba-2024,
  title={DGCNN-Mamba: Robust Few-Shot 3D Point Cloud Classification via Cross-Modal Feature Fusion},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
