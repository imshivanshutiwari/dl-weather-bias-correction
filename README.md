<div align="center">

# 🌧️ Deep Learning-Based Weather Bias Correction & Spatial Downscaling

### *Dual-Phase AI Framework for Precision Precipitation Forecasting over India*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)](LICENSE)

**IIT Madras (Pune Campus) · M.Tech Thesis · 2025**

---

*Correcting systematic biases in GFS forecasts using 3D Residual CNNs, then enhancing spatial resolution up to **25×** using Super-Resolution GANs — validated against IMDAA reanalysis over the Indian Monsoon (JJAS 2019–2023).*

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Architecture](#-solution-architecture)
- [Pipeline Overview](#-pipeline-overview)
- [Phase 1 — CNN Bias Correction](#-phase-1--cnn-bias-correction-cnnbc)
- [Phase 2 — SRGAN Spatial Downscaling](#-phase-2--srgan-spatial-downscaling)
- [Dataset Engineering](#-dataset-engineering)
- [Evaluation Metrics](#-evaluation-metrics)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Future Scope](#-future-scope)
- [Author](#-author)

---

## ❓ Problem Statement

The **Global Forecast System (GFS)** is one of the world's most widely used numerical weather prediction models. However, when applied to the **Indian subcontinent** — a region dominated by the Himalayas, extensive coastlines, and the chaotic Indian Monsoon — GFS suffers from two critical limitations:

| Problem | Impact |
|---------|--------|
| **Systematic Bias** | GFS consistently over/under-estimates rainfall due to unresolved sub-grid topography (mountains, valleys). A 70 km grid cannot capture orographic lifting. |
| **Coarse Resolution** | At 0.625°, one pixel covers ~70×70 km. Entire cities fit in a single data point — useless for localized flood prediction or district-level planning. |

> **Goal:** Build an AI post-processing pipeline that takes raw, biased, coarse GFS output and produces **bias-free, hyper-local (0.025°)** precipitation forecasts.

---

## 🏗️ Solution Architecture

```mermaid
graph LR
    A["Raw GFS Forecast 0.625deg biased"] --> B["Phase 1: CNNBC Bias Correction"]
    B --> C["Phase 2: SRGAN Spatial Downscaling"]
    C --> D["Final Output 0.025deg unbiased"]

    E["IMDAA Reanalysis Ground Truth"] -.->|Training Target| B
    E -.->|Training Target| C

    style A fill:#ff6b6b,color:#fff
    style B fill:#4ecdc4,color:#fff
    style C fill:#45b7d1,color:#fff
    style D fill:#96ceb4,color:#fff
    style E fill:#ffeaa7,color:#333
```

---

## 🔄 Pipeline Overview

```mermaid
flowchart TB
    subgraph DATA["Data Engineering"]
        direction LR
        G1["GFS GRIB Files"] -->|pygrib| G2["Extract and Mask"]
        G2 -->|accumulate| G3["4D Tensor: Time Ensemble Lat Lon"]
        I1["IMDAA NetCDF"] -->|xarray| I2["Nearest-Neighbor Interpolation"]
        I2 --> I3["Aligned Ground Truth 269x257"]
    end

    subgraph PHASE1["Phase 1: Bias Correction"]
        direction LR
        P1["10 GFS Ensembles"] --> P2["ResCNNv4: 9x9 Conv, PReLU, GroupedConv, 8 ResBlocks"]
        P2 --> P3["Bias-Free Output 0.625deg"]
    end

    subgraph PHASE2["Phase 2: Downscaling"]
        direction LR
        S1["Coarse Input"] --> S2["SRGAN Generator: 8 ResBlocks, 5x PixelShuffle, GELU"]
        S2 --> S3["High-Res Output 0.025deg"]
        S4["Discriminator: StridedConv, LeakyReLU"] -.->|adversarial loss| S2
    end

    DATA --> PHASE1 --> PHASE2

    style DATA fill:#f8f9fa,stroke:#dee2e6
    style PHASE1 fill:#d4edda,stroke:#28a745
    style PHASE2 fill:#cce5ff,stroke:#0d6efd
```

---

## 🧠 Phase 1 — CNN Bias Correction (CNNBC)

The **ResCNNv4** architecture (`model_4.py`) removes systematic errors from GFS precipitation forecasts.

```mermaid
graph LR
    A["Input: 10 Ensembles"] --> B["9x9 Conv2D + PReLU"]
    B --> C["Grouped Conv groups=10"]
    C --> D["8x Residual Blocks"]
    D --> E["Output Conv 1 Channel"]

    D -.->|"skip: x + F of x"| D

    style A fill:#e8f5e9
    style E fill:#c8e6c9
```

**Key Design Decisions:**

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Initial Kernel | 9×9 | Captures synoptic-scale (100s of km) weather patterns |
| Activation | PReLU | Learnable negative slope prevents dead neurons during dry periods |
| Groups=10 | Grouped Conv | Independent ensemble processing → implicit regularization |
| Skip Connections | Additive | Preserves baseline weather map topology through 8 deep blocks |
| Optimizer | Adam (lr=1e-8) | Ultra-conservative for late-stage fine-tuning |
| Precision | AMP (FP16) | 2× speed, 0.5× VRAM — critical for multi-GPU training |

---

## 🔬 Phase 2 — SRGAN Spatial Downscaling

After bias correction, the **SRGAN** upscales data from 0.625° → 0.025° (**25× enhancement**).

```mermaid
graph TB
    subgraph GEN["Generator"]
        direction LR
        G1["9x9 Conv"] --> G2["8x ResBlocks + BN + GELU"]
        G2 --> G3["5x PixelShuffle 32x upscale"]
        G3 --> G4["High-Res Output"]
    end

    subgraph DISC["Discriminator"]
        direction LR
        D1["HR Input"] --> D2["Strided Conv 64 to 512"]
        D2 --> D3["AdaptiveAvgPool + Dense"]
        D3 --> D4["Real or Fake Probability"]
    end

    G4 -->|"fake sample"| DISC
    DISC -.->|"adversarial gradient"| GEN

    style GEN fill:#e3f2fd,stroke:#1976d2
    style DISC fill:#fce4ec,stroke:#c62828
```

**Training Strategy:**
1. **Pre-train** Generator on pure MSE loss (learn basic mapping)
2. **Adversarial phase** — Discriminator trained with BCE, Generator with composite loss:
   - `L_total = λ_MSE × L_MSE + λ_adv × L_adversarial`
3. PixelShuffle used instead of Transpose Conv → **zero checkerboard artifacts**

---

## 📊 Dataset Engineering

| Dataset | Role | Resolution | Source |
|---------|------|------------|--------|
| **GFS** | Predictor (input) | 0.625° (~70 km) | NCEP/NOAA |
| **IMDAA** | Ground Truth (target) | 0.125° (~12 km) | IMD-NCMRWF |
| **IMD Gridded** | Validation | 0.25° (~25 km) | IMD |

**Domain:** India (6.5°N–38.5°N, 66.5°E–100.1°E)  
**Period:** JJAS 2019–2023 (Indian Monsoon — peak volatility)

```mermaid
flowchart LR
    A["Raw GRIB"] -->|pygrib| B["Variable Extraction"]
    B -->|Mask_125deg| C["Geographic Masking"]
    C -->|accumulate| D["3-hourly Rainfall Tensor"]
    D --> E["4D NetCDF"]

    style A fill:#fff3e0
    style E fill:#e8f5e9
```

---

## 📈 Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **PSNR** | `10 × log₁₀(MAX² / MSE)` | Pixel-level reconstruction quality (>40 dB = exceptional) |
| **MSE** | `Σ(ŷ - y)² / n` | Penalizes large errors exponentially — critical for cloudburst detection |
| **MAE** | `Σ|ŷ - y| / n` | Linear error for seasonal rainfall volume |
| **Pearson r** | `cov(ŷ,y) / (σ_ŷ × σ_y)` | Statistical distribution alignment (target: r → 1.0) |

---

## 📁 Repository Structure

```
├── 📂 Bias_Correction_2025/       # Latest bias correction pipeline
│   ├── model_4.py                 # ResCNNv4 architecture (main)
│   ├── train_cnn.py               # Multi-GPU training script
│   ├── dataset.py                 # Custom PyTorch Dataset
│   ├── config.py                  # Hyperparameters
│   └── robust/                    # Advanced loss functions (SSIM, focal, spectral)
│
├── 📂 CNNBC/                      # Original CNN Bias Correction
│   ├── cnnbc_final.py             # Production model
│   ├── train_cnn.py               # Training loop
│   └── *.pbs                      # HPC job scripts
│
├── 📂 Downscaling/                # SRGAN Spatial Downscaling
│   ├── srgan_pytorch/             # Main SRGAN implementation
│   │   ├── models.py              # Generator + Discriminator
│   │   ├── train_gen.py           # Generator training
│   │   ├── train_disc.py          # Discriminator training
│   │   └── data_loader.py         # Dataset pipeline
│   ├── GFS/                       # GFS-specific downscaling
│   └── vgg/                       # VGG baseline comparison
│
├── 📂 Preprocess_datasets/        # Data engineering scripts
│   ├── Preprocess_GFS_data.py     # GRIB → NetCDF conversion
│   ├── Preprocess_IMDAA.py        # Ground truth alignment
│   ├── Make_Timestamps.py         # Temporal synchronization
│   └── calculate_climatology.py   # Physical upper-bound ceiling
│
├── 📂 bias_correction/            # Legacy bias correction code
├── 📂 pygrib_cfs/                 # GRIB file utilities
├── claude_code.py                 # Automated analysis pipeline
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision xarray pygrib numpy matplotlib cartopy pandas pillow wandb
```

### Training Bias Correction (CNNBC)

```bash
cd Bias_Correction_2025
python train_cnn.py --config config.py
```

### Training SRGAN Downscaling

```bash
cd Downscaling/srgan_pytorch

# Step 1: Pre-train Generator
python train_gen.py

# Step 2: Train Discriminator
python train_disc.py
```

### HPC (Supercomputer) Submission

```bash
qsub CNNBC/train_cnnbc.pbs
qsub Downscaling/srgan_pytorch/srgan_job1.pbs
```

---

## 🔮 Future Scope

```mermaid
mindmap
  root((Future Roadmap))
    Multi-Modal Input
      Temperature at 2m
      Wind U and V vectors
      Geopotential 500hPa
      Relative Humidity
    Real-Time MLOps
      Apache Airflow DAGs
      Auto-ingest GFS GRIB
      Live dashboard API
      Edge deployment
    Next-Gen Architectures
      Vision Transformers
      Swin-Transformer
      Diffusion Models
      Replace GAN instability
    Model Optimization
      ONNX export
      TensorRT compilation
      Cpp inference
      Edge IoT devices
    Continuous Learning
      Concept drift detection
      Auto-retrain pipeline
      Transfer learning
      Climate adaptation
```

---

## 👤 Author

**Shivanshu Tiwari**  
M.Tech, IIT Madras (Pune Campus)  

---

<div align="center">

*Built with ❤️ for precision meteorology*

**If this work helped your research, please ⭐ this repository.**

</div>
