# An Ensemble-Based Self-Taught Learning Approach for Parking Space Classification Under Limited Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.03258-b31b1b.svg)](https://arxiv.org/pdf/2609.03258)
[![Conference](https://img.shields.io/badge/IEEE%20SMC-2026-darkblue.svg)](https://arxiv.org/pdf/2609.03258)
[![Weights](https://img.shields.io/badge/Pretrained%20Weights-Google%20Drive-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1hvT4jwJ5oAgfegz5Of59-zBIVtMXY_Ui/view?usp=sharing)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official PyTorch implementation and experimental benchmark for the paper:
> **An Ensemble-Based Self-Taught Learning Approach for Parking Space Classification Under Limited Data**  
> *Accepted for presentation at the IEEE International Conference on Systems, Man, and Cybernetics (SMC 2026)*  
> 📄 **Paper (arXiv):** [https://arxiv.org/pdf/2609.03258](https://arxiv.org/pdf/2609.03258) | 💾 **Pretrained Weights:** [Google Drive](https://drive.google.com/file/d/1hvT4jwJ5oAgfegz5Of59-zBIVtMXY_Ui/view?usp=sharing)

---

## Abstract & Scientific Overview

Parking spot classification is a fundamental task in intelligent transportation systems, yet most deep learning approaches rely on large amounts of scenario-specific annotated data and exhibit limited generalization across heterogeneous environments.

To address these limitations, this work introduces a **Self-Taught Learning (STL)** framework based on unsupervised representation learning with an ensemble of heterogeneous convolutional autoencoders:

- **Step 1 — Unsupervised Representation Learning**:
  - A stochastic architecture generator instantiates $K=10$ heterogeneous **Convolutional Autoencoders (CAEs)** with varying depth ($L \sim U\{2, 5\}$), convolutional filters ($f_i \in \{8, 16, 32, 64, 128\}$), spatial pooling schedules ($p \le 4$), and latent dimensions ($d \sim U[128, 2048]$) for $64 \times 64$ inputs.
  - Each CAE is trained independently using Mean Squared Error (MSE) reconstruction loss on unlabeled source-domain images (`PKLot` or `CNRPark-EXT`), enforcing strict separation from target-domain deployment cameras.
- **Step 2 — Supervised Task Adaptation under Limited Data**:
  - Learned encoders $E_{\theta_k}$ are frozen and reused as fixed feature extractors.
  - Lightweight task-specific classifier heads (`Dropout(0.2) -> FC(256, ReLU) -> FC(128, ReLU) -> Softmax(2)`) are trained on top of the frozen latent representations using a chronologically sampled budget of $R \in \{64, 128, 256, 512, 1024\}$ labeled target samples.
- **Step 3 — Ensemble Decision Fusion**:
  - At inference time, posterior predictions across all $K=10$ encoder–classifier pairs are combined via ensemble fusion rules (**Majority Voting**, **Sum**, **Mean**, **Product**, and **Maximum** of posterior probabilities).

> [!NOTE]
> **Extended Experiments Note (\*)**: *This repository contains additional experiments beyond those reported in the IEEE SMC 2026 paper.* Specifically, in addition to the 10 baseline Convolutional Autoencoders (`ae`), the codebase implements and benchmarks **Skip-Connection Autoencoders** (`skip`), **Variational Autoencoders** (`vae`), **Jointly-Trained Autoencoders with Orthogonality Constraints** (`joint`, `jointskip`), **Mixture-of-Experts (MoE)** gating (`src/models/moe.py`), and auxiliary unlabeled representation learning on the **Kyoto** natural image dataset (`CSV/Kyoto/`).

---

## Repository Architecture

```text
├── CSV/                          # Official dataset split manifests and chronological batches
│   ├── CNR/                      # Unsupervised representation splits for CNRPark-EXT
│   ├── PKLot/                    # Unsupervised representation splits for PKLot
│   ├── Kyoto/                    # Auxiliary representation splits (extended experiments)
│   ├── PUC/                      # Target splits + chronological batches (64, 128, 256, 512, 1024)
│   ├── UFPR04/                   # Target splits + chronological batches (64..1024)
│   ├── UFPR05/                   # Target splits + chronological batches (64..1024)
│   └── camera1..camera9/         # CNRPark-EXT target splits + chronological batches (64..1024)
├── notebooks/                    # Exploratory analysis and latent space visualization notebooks
│   ├── analises.ipynb            # Quantitative analysis and reconstruction inspection
│   ├── clusters.ipynb            # Latent representation clustering analysis
│   └── testes.ipynb              # Model sanity checks and visual diagnostics
├── paper/                        # Camera-ready paper PDF
│   └── SMC2026_Ensemble_STL_Parking.pdf
├── results/                      # Consolidated experimental metrics and evaluation CSVs
│   ├── encoders_metrics.csv      # Autoencoder reconstruction metrics (MSE, PSNR, SSIM)
│   ├── fusion_metrics2.csv       # Ensemble fusion accuracy across target scenarios
│   ├── grid_search_results.csv   # Hyperparameter exploration summary
│   ├── combinacoes.csv           # Exhaustive subset combination evaluations (K=1..10)
│   ├── fusao_tudo.csv            # Consolidated cross-scenario fusion results
│   └── classifiers/              # Individual encoder-classifier pair evaluation CSVs
├── scripts/                      # Data preparation, path normalization, and legacy utilities
│   ├── _bootstrap.py             # Root path bootstrapper
│   ├── fix_csv_paths.py          # Rewrites CSV image paths to match local dataset directories
│   ├── make_kyoto_dataaug.py     # Generates augmented Kyoto images for unsupervised training
│   ├── verify_test_alignment.py  # Verifies and canonizes prediction-to-ground-truth alignment
│   └── legacy/                   # Standalone scripts from earlier experimental iterations
├── src/                          # Core modular PyTorch framework
│   ├── config/                   # Dataset paths, model family registries, and training configs
│   ├── models/                   # CAEs, Skip-CAEs, VAEs, Joint-AEs, MoE, and Classifier heads
│   ├── pipeline/                 # End-to-end training & ensemble fusion pipelines
│   │   ├── train_autoencoders_64.py       # Step 1: 64x64 CAE representation learning
│   │   ├── train_classifier_64.py         # Step 2: 64x64 frozen-encoder classifier training
│   │   ├── fusion_64.py                   # Step 3: 64x64 ensemble probability & voting fusion
│   │   ├── train_autoencoders_families.py # Extended multi-family autoencoder trainer
│   │   ├── train_classifiers_families.py  # Extended multi-family classifier trainer
│   │   ├── fusion_families.py             # Extended intra-family and cross-family fusion
│   │   └── join_10_models.py              # Joint multi-autoencoder training with orthogonality loss
│   └── utils/                    # Dataset loaders, image metrics (SSIM/PSNR/VIF), and plotting
├── run_64.py                     # CLI entry point for reproducing the SMC 2026 paper pipeline
├── run_families.py               # CLI entry point for the extended multi-family benchmark
├── run_all.sh                    # Idempotent shell script for full multi-family execution
├── main.py                       # CLI entry point for joint 10-model autoencoder training
├── pyproject.toml                # Project metadata and uv dependency lock configuration
└── requirements.txt              # Pip requirements specification
```

---

## Pretrained Weights & Checkpoints

All trained autoencoder backbones (`*_encoder.pth`, `*_decoder.pth`), task-specific classifier heads, and precomputed probability outputs are publicly available on Google Drive:

- **Download Link:** [Google Drive — Pretrained Models & Weights (`Modelos.zip`)](https://drive.google.com/file/d/1hvT4jwJ5oAgfegz5Of59-zBIVtMXY_Ui/view?usp=sharing)

After downloading, extract the archive in the repository root so that checkpoints follow the expected directory structure:

```text
Modelos/
├── Modelo_AE/
│   ├── Modelo-Base/
│   │   └── Pesos/                # Unsupervised encoder & decoder checkpoints (.pth)
│   └── Classificador-0..9/
│       ├── Pesos/                # Task-specific classifier checkpoints (.pth)
│       └── Resultados/           # Predicted posterior probabilities (.npy / .npz)
├── Modelo_Skip/                  # Extended experiment: Skip-Connection Autoencoders
├── Modelo_VAE/                   # Extended experiment: Variational Autoencoders
├── Modelo_Joint/                 # Extended experiment: Joint Autoencoders
└── Modelo_JointSkip/             # Extended experiment: Joint Skip Autoencoders
```

---

## Environment Setup

### 1. Requirements

- Python 3.10+
- Linux (Ubuntu 20.04+ recommended) or Windows
- NVIDIA GPU with CUDA support

### 2. Installation via `uv` (Recommended)

Using [`uv`](https://docs.astral.sh/uv/) guarantees reproducible dependency resolution via `uv.lock`:

```bash
# Clone the repository
git clone https://github.com/lucasdocunha/Representation-Fusion.git
cd Representation-Fusion

# Synchronize virtual environment with lockfile
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### 3. Installation via `pip` (Alternative)

```bash
pip install -r requirements.txt
```

---

## Data Preparation & Environment Variables

The pipeline resolves dataset locations dynamically through environment variables (`src/config/paths.py`):

| Variable | Description | Default |
| :--- | :--- | :--- |
| `DATASETS_ROOT` | Root directory containing `CNRPark-EXT`, `PKLot`, and `kyoto` folders | `~/DeepLearning/datasets` |
| `PKLOT_DIR` | Override path for the PKLot dataset root | `$DATASETS_ROOT/PKLot` |
| `CNR_DIR` | Override path for the CNRPark-EXT dataset root | `$DATASETS_ROOT/CNRPark-EXT` |
| `KYOTO_DIR` | Override path for the Kyoto dataset root (extended experiments) | `$DATASETS_ROOT/kyoto` |

Before running training on a new machine, normalize the image paths in `CSV/` to match your local filesystem:

```bash
export DATASETS_ROOT="/path/to/your/datasets"
uv run python run_families.py --stage csv-fix
```

---

## Step-by-Step Reproduction Guide

### 1. Reproducing the IEEE SMC 2026 Paper Pipeline (`run_64.py`)

Execute the 3-stage $64 \times 64$ pipeline (Unsupervised CAE Training $\rightarrow$ Supervised Classifier Adaptation across $R \in \{64, 128, 256, 512, 1024\}$ $\rightarrow$ Ensemble Fusion):

```bash
# Run the complete 3-stage pipeline (50 epochs for CAEs, 10 epochs for classifiers)
uv run python run_64.py --epochs 50 --models all

# Run only Stage 2 (Classifier Training) and Stage 3 (Ensemble Fusion) using pretrained encoders
uv run python run_64.py --skip-ae --epochs 10

# Run only Stage 3 (Ensemble Fusion evaluation) from saved predictions
uv run python run_64.py --skip-ae --skip-cls
```

### 2. Running the Extended Multi-Family Benchmark (`run_families.py` / `run_all.sh`)

To train and evaluate all five architectural families (`ae`, `skip`, `vae`, `joint`, `jointskip`) across `CNR`, `PKLot`, and `Kyoto`:

```bash
# Option A: Execute the idempotent shell orchestrator
EPOCHS=10 FAMILIES=all FRAC=1.0 ./run_all.sh

# Option B: Run individual stages via run_families.py
uv run python run_families.py --stage ae --epochs 50 --families ae,skip,vae
uv run python run_families.py --stage cls --epochs 10 --families ae,skip,vae
uv run python run_families.py --stage fusion --frac 1.0

# Run cross-family ensemble fusion (combining 1 model from each family)
uv run python -m src.pipeline.fusion_families --cross --frac 1.0
```

---

## Main Experimental Results (IEEE SMC 2026)

### 1. Cross-Dataset Adaptation under Limited Target Data (Table II)

Classification accuracy (%) across all 12 target deployment scenarios as a function of annotated target samples $R \in \{64, 128, 256, 512, 1024\}$ (unsupervised representations learned strictly on the opposite dataset):

| Representation Domain | Target Domain | 64 Samples | 128 Samples | 256 Samples | 512 Samples | 1024 Samples |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNRPark-EXT** | **PUC** | 95.1% | 97.4% | 98.2% | 98.4% | **98.7%** |
| **CNRPark-EXT** | **UFPR04** | 94.0% | 96.3% | 96.5% | 97.0% | **97.4%** |
| **CNRPark-EXT** | **UFPR05** | 97.0% | 97.6% | 98.1% | **98.4%** | 98.3% |
| **PKLot** | **CNR-1** | 90.7% | 92.6% | 93.7% | 94.3% | **95.8%** |
| **PKLot** | **CNR-2** | 97.3% | 97.1% | 97.4% | 97.6% | **98.4%** |
| **PKLot** | **CNR-3** | 95.8% | 94.7% | 96.3% | 97.0% | **97.7%** |
| **PKLot** | **CNR-4** | 94.7% | 94.5% | 95.1% | 95.0% | **95.4%** |
| **PKLot** | **CNR-5** | 86.2% | 89.7% | 92.3% | 93.5% | **94.7%** |
| **PKLot** | **CNR-6** | 92.2% | 92.5% | 93.7% | 93.4% | **95.1%** |
| **PKLot** | **CNR-7** | 92.3% | 91.8% | 91.1% | 92.0% | **92.9%** |
| **PKLot** | **CNR-8** | 92.9% | 95.0% | 95.3% | 96.0% | **96.6%** |
| **PKLot** | **CNR-9** | 91.2% | 91.0% | 92.5% | 94.3% | **95.8%** |
| **Average** | **All 12 Scenarios** | **93.3%** | **94.2%** | **95.0%** | **95.6%** | **96.4%** |

---

### 2. Cross-Scenario Generalization without Weight Optimization (Table III)

Accuracy (%) when task-specific ensemble classifiers trained with $R=1024$ samples on a single target camera are directly evaluated across all 12 scenarios without further fine-tuning:

| Task-Specific Classifier | CNR-1 | CNR-2 | CNR-3 | CNR-4 | CNR-5 | CNR-6 | CNR-7 | CNR-8 | CNR-9 | PUC | UFPR04 | UFPR05 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNR-1** | **95.8** | 98.1 | 96.5 | 96.1 | 92.0 | 95.7 | 90.9 | 95.7 | 93.7 | 89.2 | 76.9 | 76.6 |
| **CNR-2** | 90.5 | **98.4** | 95.2 | 96.3 | 91.4 | 95.8 | 90.3 | 96.2 | 93.8 | 91.2 | 85.4 | 84.1 |
| **CNR-3** | 85.9 | 98.0 | **97.7** | 96.0 | 89.0 | 95.1 | 87.3 | 95.9 | 91.5 | 92.3 | 83.0 | 87.4 |
| **CNR-4** | 86.3 | 97.2 | 94.1 | **95.4** | 90.8 | 95.3 | 87.7 | 96.5 | 92.7 | 93.7 | 88.3 | 87.4 |
| **CNR-5** | 88.5 | 98.3 | 96.3 | 97.2 | **94.7** | 96.1 | 90.7 | 96.8 | 93.8 | 96.6 | 88.4 | 85.4 |
| **CNR-6** | 83.1 | 96.5 | 93.0 | 94.5 | 90.1 | **95.1** | 86.0 | 95.5 | 90.1 | 93.8 | 85.5 | 84.2 |
| **CNR-7** | 91.1 | 98.1 | 97.7 | 97.6 | 95.4 | 97.2 | **92.9** | 97.3 | 95.4 | 93.3 | 90.1 | 79.4 |
| **CNR-8** | 83.0 | 93.9 | 94.6 | 93.7 | 89.5 | 95.3 | 87.3 | **96.6** | 92.0 | 94.8 | 84.7 | 85.7 |
| **CNR-9** | 91.1 | 98.0 | 95.4 | 97.3 | 94.0 | 97.0 | 91.0 | 97.6 | **95.8** | 91.8 | 90.1 | 88.2 |
| **PUC** | 91.4 | 95.0 | 92.0 | 95.4 | 92.4 | 94.4 | 90.2 | 93.4 | 91.4 | **98.7** | 91.5 | 81.3 |
| **UFPR04** | 89.0 | 94.2 | 92.8 | 92.6 | 88.5 | 93.6 | 90.6 | 91.8 | 88.5 | 93.2 | **97.4** | 84.5 |
| **UFPR05** | 71.7 | 83.0 | 85.1 | 84.9 | 74.9 | 87.6 | 76.9 | 85.8 | 77.4 | 82.6 | 87.4 | **98.3** |

---

### 3. State-of-the-Art Comparison (Table IV)

| Author | Approach | Limited Target Annotated Data | Target Training | Average Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| Almeida et al. (2015) | LBP + SVM | — | No | ~87% |
| Amato et al. (2017) | Custom CNN (mAlexNet) | — | No | ~88% |
| Nurullayev & Lee (2019) | Dilated CNN (CarNet) | — | No | ~96% |
| Almeida et al. (2022) | Cross-Dataset Survey | — | — | ~92% |
| Hochuli et al. (2023) | MobileNetV3 | — | No | ~92.8% |
| Alves et al. (2024) | Teacher-Student Distillation | Pseudo-labels (7 days) | Yes | ~96.6% |
| Santos et al. (2026) | GANs + MobileNetV3 | 256–1024 samples | Yes | ~97–98% |
| **Ours (SMC 2026)** | **Ensemble CAE-ANN (STL)** | **64 samples** | **Yes (Head Only)** | **~93.3%** |
| **Ours (SMC 2026)** | **Ensemble CAE-ANN (STL)** | **1024 samples** | **Yes (Head Only)** | **~96.4%** |

---

## Authors

*Pontifical Catholic University of Paraná (PUCPR) & Federal University of Paraná (UFPR), Curitiba, PR, Brazil*

| Author | Affiliation | Email | GitHub |
| :--- | :--- | :--- | :---: |
| **Lucas de Oliveira Cunha** | Computer Science / PPGIa, PUCPR | [`c.oliveira25@pucpr.br`](mailto:c.oliveira25@pucpr.br) | [![GitHub](https://img.shields.io/badge/GitHub-lucasdocunha-181717?logo=github)](https://github.com/lucasdocunha) |
| **Joelton Deonei Gotz** | Computer Science, PUCPR | [`joelton.gotz@pucpr.br`](mailto:joelton.gotz@pucpr.br) | — |
| **Paulo Lisboa de Almeida** | Dept. of Informatics, UFPR | [`paulorla@ufpr.br`](mailto:paulorla@ufpr.br) | — |
| **Andre Gustavo Hochuli** | Graduate Program in Informatics (PPGIa), PUCPR | [`aghochuli@ppgia.pucpr.br`](mailto:aghochuli@ppgia.pucpr.br) | — |

---

## Citation

If you find this repository or our self-taught representation learning framework useful in your research, please cite our IEEE SMC 2026 paper:

```bibtex
@inproceedings{cunha2026ensemble,
  title     = {An Ensemble-Based Self-Taught Learning Approach for Parking Space Classification Under Limited Data},
  author    = {Cunha, Lucas de Oliveira and Gotz, Joelton Deonei and de Almeida, Paulo Lisboa and Hochuli, Andre Gustavo},
  booktitle = {IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  year      = {2026},
  eprint    = {2609.03258},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2609.03258}
}
```

---

## Acknowledgments

The authors acknowledge the financial support of the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES), Financiadora de Estudos e Projetos (FINEP), and the Fundação Araucária, in partnership with the Secretaria da Ciência, Tecnologia e Ensino Superior do Estado do Paraná (SETI-PR), under Grant No. 653/2025.
