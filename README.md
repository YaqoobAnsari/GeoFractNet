# GeoFractNet: A Dilated U-Net with Edge-Aware Skip Connections for the Semantic Edge Detection of Natural Fractures in Outcrop

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **GeoFractNet**, a novel deep learning architecture for automated semantic edge detection of geological fractures in outcrop imagery.

---

## Overview

Natural fracture networks govern subsurface fluid flow, rock mass stability, and strain accommodation in the brittle crust. However, automated delineation of fracture traces from outcrop imagery remains challenging due to noise, scale variation, and visual clutter. **GeoFractNet** addresses these challenges through:

- **ConvNeXt encoders** with multi-scale dilated convolutions for capturing fractal-like fracture scaling
- **Gated edge-aware skip connections** fusing Scharr and Gabor filter responses
- **Gradient-Induced Edge-Aware Loss (GIEA)** for boundary-sensitive optimization
- **State-of-the-art performance**: mIoU of 0.91, Dice score of 0.92, and Boundary F1 of 0.90

<p align="center">
  <img src="docs/arch.png" alt="GeoFractNet Architecture" width="800"/>
</p>

---

## Key Features

✅ **Purpose-built architecture** for geological fracture edge detection  
✅ **Multi-scale context aggregation** via dilated convolutions (d = 1, 2, 4, 8)  
✅ **Edge-aware feature fusion** with traditional edge filters (Scharr + Gabor)  
✅ **Superior boundary localization** through gradient-induced loss function  
✅ **Robust to scene artifacts**: suppresses vegetation, shadows, blast holes, and topographic occlusions  
✅ **38% reduction in false positives** compared to YOLACT++  
✅ **94% recovery of long-range fractures** missed by classical edge detection  

---

## Repository Structure

```
GeoFractNet/
├── Code/                          # Training, evaluation, and utility scripts
│   ├── train.py                   # Main training script
│   ├── evaluate.py                # Model evaluation
│   ├── model.py                   # GeoFractNet architecture
│   ├── losses.py                  # GIEA loss implementation
│   └── utils.py                   # Helper functions
├── Dataset/                       # Dataset split specifications
│   ├── patch_pairs.csv            # Image-mask pairs mapping
│   ├── train.csv                  # Training set indices (70%)
│   ├── validation.csv             # Validation set indices (15%)
│   └── test.csv                   # Test set indices (15%)
├── Original Images/               # Input outcrop images (from GeoCrack)
├── Edge Binary Masks/             # Ground truth fracture masks (from GeoCrack)
├── Test Images/                   # Held-out test images (from GeoCrack)
├── Results/
│   └── Result Images/             # Sample predictions (10 examples included)
├── images.txt                     # Image-to-split mapping file
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

> **Note**: The folders `Original Images/`, `Edge Binary Masks/`, and `Test Images/` are empty by default to maintain repository size. Download the complete dataset from GeoCrack (see Dataset section).

---

## Dataset

### GeoCrack Dataset

GeoFractNet is trained and validated on **[GeoCrack](https://doi.org/10.7910/DVN/GeoCrack)**, the first open-access, high-resolution dataset for fracture edge detection in geological outcrops.

**Dataset Specifications:**
- **49 high-quality outcrop images** from 11 study sites
- **Geographic coverage**: Europe (Greece, Malta, Italy) and Middle East (Oman, UAE)
- **12,158 annotated patches** (224×224 px, non-overlapping)
- **Data split**: 8,501 training / 1,824 validation / 1,833 test (70/15/15%)
- **Diverse lithologies**: limestone, sandstone, carbonates, and more
- **Realistic scene artifacts**: shadows, vegetation, blast holes, topographic occlusions

**Dataset Access:**
1. Download GeoCrack dataset: [Harvard Dataverse](https://doi.org/10.7910/DVN/GeoCrack)
2. Extract images and masks into respective folders
3. Use `images.txt` to map training/validation/test splits

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: NVIDIA RTX A6000 or equivalent)
- 16GB+ VRAM for training

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YaqoobAnsari/GeoFractNet.git
cd GeoFractNet
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download GeoCrack dataset** and organize files:
```
GeoFractNet/
├── Original Images/        # Place RGB images here
├── Edge Binary Masks/      # Place binary masks here
└── Test Images/            # Place test images here
```

---

## Usage

### Training

Train GeoFractNet from scratch:

```bash
python Code/train.py --config configs/default.yaml \
                     --batch_size 16 \
                     --epochs 200 \
                     --lr 1e-4
```

**Key hyperparameters:**
- Optimizer: Adam (β₁=0.9, β₂=0.999)
- Learning rate: 10⁻⁴ with cosine annealing
- Batch size: 16
- Early stopping: patience=10 epochs
- Training time: ~7.5 hours per run (2× NVIDIA RTX A6000)

### Evaluation

Evaluate trained model on test set:

```bash
python Code/evaluate.py --checkpoint weights/geofractnet_best.pth \
                        --test_dir Dataset/test.csv
```

**Evaluation metrics:**
- Region overlap: mIoU, Dice coefficient
- Pixel-level: Precision, Recall
- Boundary alignment: Boundary IoU, Boundary F1

### Inference

Run inference on new outcrop images:

```bash
python Code/inference.py --image path/to/outcrop.jpg \
                         --checkpoint weights/geofractnet_best.pth \
                         --output results/
```

---

## Results

### Quantitative Performance

| Method | mIoU | Dice | Precision | Recall | B-IoU | BF1 |
|--------|------|------|-----------|--------|-------|-----|
| **GeoFractNet** | **0.91** | **0.92** | **0.91** | **0.93** | **0.89** | **0.90** |
| YOLACT++ | 0.89 | 0.91 | 0.90 | 0.92 | 0.86 | 0.89 |
| DeepLabv3+ | 0.87 | 0.90 | 0.89 | 0.91 | 0.83 | 0.87 |
| U-Net | 0.85 | 0.88 | 0.87 | 0.89 | 0.80 | 0.85 |
| Mask R-CNN | 0.84 | 0.87 | 0.86 | 0.88 | 0.81 | 0.84 |
| Canny | 0.60 | 0.70 | 0.68 | 0.72 | 0.56 | 0.66 |

### Ablation Study

| Model Variant | mIoU | BF1 |
|--------------|------|-----|
| Baseline U-Net | 0.85 | 0.85 |
| + ConvNeXt + Dilated Conv. | 0.87 | 0.86 |
| + Gated Edge-aware Skip | 0.89 | 0.88 |
| + Scharr + Gabor Filters | 0.90 | 0.89 |
| + GIEA Loss (Full Model) | **0.91** | **0.90** |

### Qualitative Results

Sample predictions on diverse geological settings:

<p align="center">
  <img src="Docs/results.png" alt="Result 1" width="800"/>
  <br>
  <em>Left: Original outcrop. Middle: Ground truth. Right: GeoFractNet prediction.</em>
</p>

**Key achievements:**
- Continuous fracture trace reconstruction across patch boundaries
- Effective suppression of non-geological edges (vegetation, shadows, blast holes)
- Accurate capture of branching geometries, terminations, and subtle offsets
- Scale-invariant performance from patch (224×224 px) to outcrop (5472×3648 px) scale

---

## Model Architecture

### Core Components

1. **ConvNeXt Encoder with Dilated Convolutions**
   - Multi-scale feature extraction with dilation rates [1, 2, 4, 8]
   - Expanded receptive field without spatial resolution loss
   - Residual connections for training stability

2. **Edge-Aware Skip Connections**
   - Fusion of encoder features with Scharr and Gabor filter responses
   - Gated convolution mechanism for selective feature propagation
   - Enhanced edge localization at multiple resolutions

3. **Gradient-Induced Edge-Aware Loss (GIEA)**
   ```
   L_GIEA = L_Dice + L_IoU + λ · Σ|∇Ŷᵢ - ∇Yᵢ|
   ```
   - Combines region-based (Dice, IoU) and boundary-sensitive (gradient) terms
   - Explicit penalization of edge misalignment
   - Superior boundary localization compared to standard losses

### Technical Specifications

- **Input**: 224×224×3 RGB patches
- **Output**: 224×224×1 binary fracture masks
- **Parameters**: ~23M trainable parameters
- **Inference time**: ~45ms per patch (NVIDIA RTX A6000)
- **Memory footprint**: ~2.1GB during inference

---

## Citation

If you use GeoFractNet in your research, please cite:

```bibtex
@article{yaqoob2024geofractnet,
  title={GeoFractNet: A Dilated U-Net with Edge-Aware Skip Connections for the Semantic Edge Detection of Natural Fractures in Outcrop},
  author={Yaqoob, Mohammed and Ishaq, Mohammed and Ansari, Mohammed Yusuf and Seers, Thomas Daniel},
  journal={[Journal Name]},
  year={2024},
  publisher={[Publisher]},
  doi={[DOI]}
}
```

**GeoCrack Dataset:**
```bibtex
@article{yaqoob2024geocrack,
  title={GeoCrack: A High-Resolution Dataset for Segmentation of Fracture Edges in Geological Outcrops},
  author={Yaqoob, Mohammed and Ansari, Mohammed Yusuf and Ishaq, Mohammed and Seers, Thomas Daniel},
  journal={Scientific Data},
  volume={11},
  pages={1--13},
  year={2024},
  doi={10.7910/DVN/GeoCrack}
}
```

---

## Applications

GeoFractNet is designed for:

- **Naturally fractured reservoir characterization** for hydrocarbon exploration and geothermal energy
- **Rock mass stability assessment** for geotechnical engineering and mining
- **Paleostress field reconstruction** from fracture orientation analysis
- **Discrete fracture network (DFN) modeling** for fluid flow simulation
- **Carbon sequestration site evaluation** via fracture network characterization
- **Structural geology mapping** from UAV and terrestrial photogrammetry

### Extensibility

The edge-aware architecture can be adapted to:
- Core and borehole image analysis
- Sedimentary structure detection
- Lithologic contact extraction
- Concrete crack detection
- Remote sensing lineament analysis

---

## Model Weights

Pre-trained model weights will be released upon paper acceptance. Check back for updates or contact the authors.

**Temporary access**: Contact [yansari@tamu.edu](mailto:yansari@tamu.edu) for research purposes.

---

## Acknowledgments

This work was supported by:
- **Texas A&M University at Qatar**
- **Electrical and Computer Engineering Department**
- **Petroleum Engineering Department**

We thank the GeoCrack dataset contributors and the reviewers for their valuable feedback.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Corresponding Author**: Mohammed Yaqoob  
📧 Email: [yansari@tamu.edu](mailto:yansari@tamu.edu)  
🏛️ Affiliation: Texas A&M University, Electrical and Computer Engineering, Doha, Qatar

**Project Contributors:**
- Mohammed Ishaq (Texas A&M University, Qatar)
- Mohammed Yusuf Ansari (Texas A&M University, USA)
- Thomas Daniel Seers (Texas A&M University, Qatar)

---

## Related Projects

- **[GeoCrack Dataset](https://doi.org/10.7910/DVN/GeoCrack)** - High-resolution fracture edge dataset
- **[FraSegNet](https://github.com/...)** - Alternative fracture segmentation approach
- **[Virtual Outcrop Models](https://github.com/...)** - 3D fracture analysis tools

---

## Roadmap

- [ ] Release pre-trained model weights
- [ ] Instance segmentation extension for discrete fracture trace extraction
- [ ] Integration with 3D virtual outcrop models
- [ ] Real-time inference optimization for edge devices
- [ ] Multi-resolution inference for large-scale orthophotos
- [ ] Transfer learning toolkit for related geological tasks

---

**Star ⭐ this repository if you find it useful!**

For bug reports and feature requests, please open an issue on GitHub.
