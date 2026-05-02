# Decentralized-crop-disease-diagnosis-fedratedlearning-xai
Real-time, lightweight crop disease diagnosis for edge devices — built with Federated Learning, VGG-ViT hybrid architecture, Grad-CAM heatmaps, and a custom SLM for actionable treatment suggestions.
# 🌿 Federated Crop Disease Detection using VGG-ViT and Explainable AI

> Privacy-preserving crop disease detection using Federated Learning, a hybrid VGG16-ViT model, and Explainable AI with an integrated Small Language Model for treatment recommendations.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![Flower](https://img.shields.io/badge/FL-Flower%20Framework-green)
![Accuracy](https://img.shields.io/badge/Accuracy-97--98%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Federated Learning Setup](#federated-learning-setup)
- [Explainable AI & SLM](#explainable-ai--slm)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

---

## 📖 Overview

Crop diseases are a leading cause of agricultural yield loss, yet most current deep learning solutions rely on **centralized data sharing**, which raises serious concerns about **data privacy** and is impractical in rural areas with limited internet bandwidth.

This project proposes a **decentralized, lightweight crop disease diagnosis system** that:
- Trains a hybrid deep learning model across distributed clients using **Federated Learning (FL)**
- Ensures **data privacy** — raw images never leave the client devices
- Provides **visual explanations** of predictions using **Grad-CAM**
- Generates **natural language disease descriptions and treatment suggestions** using a custom **Small Language Model (SLM)**

The system was evaluated on **17 disease classes** across apple, corn, and tomato crops, achieving a high accuracy of **97–98%** across all clients.

---

## ✨ Key Features

- 🔒 **Privacy-preserving** — only model weights are shared, never raw image data
- 🌐 **Real distributed setup** — three physical laptops as clients, connected via Tailscale VPN
- 🤖 **Hybrid VGG16-ViT architecture** — combines local feature extraction (CNN) with global context (Transformer)
- 🧠 **Explainable AI** — Grad-CAM++ heatmaps highlight disease-affected leaf regions
- 💬 **Domain-specific SLM** — custom-built Small Language Model generates disease descriptions and treatment advice
- ⚖️ **Non-IID data handling** — FedProx algorithm manages heterogeneous data distributions across clients
- 📦 **Lightweight & edge-deployable** — significantly smaller model size compared to standalone VGG16

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CENTRAL SERVER                        │
│              (Global Model Aggregation - FedProx)            │
└──────────────┬──────────────┬──────────────┬────────────────┘
               │              │              │
        Step 1 │ Distribute   │              │ Distribute
               ▼              ▼              ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   Client 1   │  │   Client 2   │  │   Client 3   │
    │    Apple     │  │     Corn     │  │    Tomato    │
    │  (RTX 2050)  │  │  (RTX 3050)  │  │  (RTX 4060)  │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
           │   Local Train   │                 │
           │   (3 epochs)    │                 │
           │                 │                 │
           └─────────────────┴─────────────────┘
                     Step 3: Share only weights
                     (No raw image data shared)
```

**FL Training Configuration:**
- 40 rounds, 3 epochs per round
- FedProx (µ = 0.01) for non-IID data handling
- AdamW optimizer (LR = 0.0002), batch size 64
- Networking: Tailscale VPN + Flower Framework

---

## 📊 Dataset

The dataset consists of **13,613 crop leaf images** across **17 disease classes** from 3 crop types.

| Client | Crop   | Total Images | Train | Test |
|--------|--------|-------------|-------|------|
| 1      | Apple  | 3,698       | 2,958 | 740  |
| 2      | Corn   | 3,852       | 3,081 | 771  |
| 3      | Tomato | 6,063       | 4,850 | 1,213|

<details>
<summary>Click to see all 17 disease classes</summary>

| Crop   | Disease Class            | Images |
|--------|--------------------------|--------|
| Apple  | Apple-Brown Spot         | 411    |
| Apple  | Apple-Frogeye Leaf Spot  | 635    |
| Apple  | Apple-Grey Spot          | 339    |
| Apple  | Apple-Healthy            | 516    |
| Apple  | Apple-Leaf Spot          | 417    |
| Apple  | Apple-Mosaic             | 371    |
| Apple  | Apple-Powdery Mildew     | 488    |
| Apple  | Apple-Rust               | 521    |
| Corn   | Corn-Common Rust         | 1,192  |
| Corn   | Corn-Gray Leaf Spot      | 513    |
| Corn   | Corn-Healthy             | 1,162  |
| Corn   | Corn-Leaf Blight         | 985    |
| Tomato | Tomato-Bacterial Spot    | 1,702  |
| Tomato | Tomato-Early Blight      | 800    |
| Tomato | Tomato-Healthy           | 1,273  |
| Tomato | Tomato-Late Blight       | 1,527  |
| Tomato | Tomato-Leaf Mold         | 761    |

</details>

> **Note:** The dataset is split in a non-IID fashion, where each client holds data for only one crop type, simulating real-world agricultural scenarios.

---

## 🧠 Model Architecture

### Hybrid VGG16-ViT Model

```
Input Image (224×224×3)
        │
        ▼
┌───────────────┐
│     VGG16     │  ← Blocks 1-3 frozen (pre-trained low-level features)
│  (CNN Branch) │  ← Blocks 4-5 fine-tuned on crop diseases
└───────┬───────┘
        │  7×7×512 feature map
        │  Flattened → 49 spatial tokens
        │  Projected: 512 → 768 dims (Linear + LayerNorm + Dropout)
        ▼
┌───────────────┐
│  ViT (Last 2  │  ← 49 tokens + CLS token
│    Layers)    │  ← 12-head multi-head self-attention × 2
└───────┬───────┘
        │  CLS token output
        │  Dropout + Linear
        ▼
   17-class Output
```

### CNN Benchmarking Results

| Model          | Test Accuracy |
|----------------|--------------|
| VGG16          | 93%          |
| ResNet50       | 75%          |
| EfficientNetB0 | 75%          |
| DenseNet121    | 74%          |
| **VGG-ViT (Ours)** | **97–98%** |

VGG16 was selected as the CNN backbone based on benchmarking results before building the hybrid architecture.

---

## 🔍 Explainable AI & SLM

### Grad-CAM++ Visualization
Grad-CAM++ generates spatial heatmaps by computing gradients of the target class score with respect to the final convolutional layer's activation maps.
- 🔴 **Red regions** → highest probability of disease
- 🔵 **Blue regions** → least contribution to prediction

### Small Language Model (SLM)
A custom transformer-based SLM built from scratch using PyTorch:
- Architecture: 4-layer encoder-decoder, 8 attention heads, hidden size 256
- Parameters: ~5.47 million trainable parameters
- Training: 80 epochs, AdamW with warmup LR schedule, label smoothing 0.1
- Output: Disease name, symptoms, probable cause, and treatment recommendations

---

## 📈 Results

### VGG-ViT Hybrid Model Performance

| Client   | Crop   | Train Accuracy | Test Accuracy |
|----------|--------|---------------|--------------|
| Client 1 | Apple  | ~99%          | ~97%         |
| Client 2 | Corn   | ~99%          | ~98–99%      |
| Client 3 | Tomato | ~100%         | ~99%         |

**Overall accuracy: 98% | Macro Avg F1: 0.97 | Weighted Avg F1: 0.98**

Key highlights:
- Apple-Brown Spot, Corn-Healthy, Tomato-Healthy → Perfect F1 score of **1.00**
- Lowest F1 score across all 17 classes: **0.89** (Apple-Grey Spot)
- Macro average AUC: near-perfect across all disease classes

---

## 📁 Project Structure

```
federated-crop-disease-detection-vgg-vit-xai/
│
├── server/
│   └── server.py               # Flower FL server, FedProx aggregation
│
├── clients/
│   ├── client1_apple.py        # Client 1 - Apple dataset
│   ├── client2_corn.py         # Client 2 - Corn dataset
│   └── client3_tomato.py       # Client 3 - Tomato dataset
│
├── models/
│   ├── vgg_vit_hybrid.py       # Hybrid VGG16-ViT architecture
│   └── benchmarks/             # VGG16, ResNet50, EfficientNetB0, DenseNet121
│
├── xai/
│   ├── gradcam.py              # Grad-CAM++ implementation
│   └── visualize.py            # Heatmap overlay utilities
│
├── slm/
│   ├── model.py                # SLM transformer architecture
│   ├── train.py                # SLM training script
│   ├── inference.py            # Disease explanation generator
│   └── knowledge_base/         # Crop disease knowledge base
│
├── data/
│   └── README.md               # Dataset description & download link
│
├── results/
│   ├── accuracy_plots/         # Client-wise accuracy graphs
│   ├── confusion_matrices/     # Per-model confusion matrices
│   └── roc_curves/             # ROC curves for all models
│
├── report/
│   └── MP_report.pdf           # Full B.Tech thesis report
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/letmebesai/federated-crop-disease-detection-vgg-vit-xai.git
cd federated-crop-disease-detection-vgg-vit-xai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**
```
torch
torchvision
flwr (Flower)
timm
opencv-python
matplotlib
scikit-learn
numpy
Pillow
```

---

## 🚀 How to Run

### 1. Prepare the Dataset
Download the dataset and place it in the `data/` folder following the structure described in `data/README.md`.

### 2. Start the Flower Server
```bash
python server/server.py
```

### 3. Start Each Client (on separate machines or terminals)
```bash
# Client 1 - Apple
python clients/client1_apple.py --server_address <SERVER_IP>:8080

# Client 2 - Corn
python clients/client2_corn.py --server_address <SERVER_IP>:8080

# Client 3 - Tomato
python clients/client3_tomato.py --server_address <SERVER_IP>:8080
```

> **Note:** For a real distributed setup across different networks, use [Tailscale](https://tailscale.com/) to create a secure VPN mesh between all machines.

### 4. Run Explainable AI (Grad-CAM++)
```bash
python xai/visualize.py --image_path <path_to_leaf_image> --model_path <path_to_saved_model>
```

### 5. Run SLM Inference
```bash
python slm/inference.py --disease "Corn-Leaf Blight"
```

---

## 👨‍💻 Authors

| Name | Reg. No |
|------|---------|
| Chintapenta Sai Srinivas |
| Prashant Kumar |
| Utkarsh Singh | 

**B.Tech — Electronics and Communication Engineering**  
School of Electrical & Electronics Engineering  
SASTRA Deemed to be University, Thanjavur, Tamil Nadu, India  
May 2026

**Project Supervisor:** Dr. Ghousiya Begum K, AP-III, SEEE, SASTRA

---

## 🙏 Acknowledgements

We thank our project supervisor **Dr. Ghousiya Begum K** for her guidance throughout this work. We also acknowledge the support of the School of Electrical and Electronics Engineering, SASTRA Deemed to be University.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
