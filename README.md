<div align="center">

# 🔬 DermaScan AI

### AI-Powered Skin Cancer Detection & Explainability

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Vercel](https://img.shields.io/badge/Deployed_on-Vercel-000000?style=flat-square&logo=vercel&logoColor=white)](https://vercel.com)
[![License](https://img.shields.io/badge/License-MIT-00D2B4?style=flat-square)](LICENSE)
[![BE Project](https://img.shields.io/badge/B.E._Project-Computer_Engineering_2025--26-f5a623?style=flat-square)](#)

<br/>

> ResNet18 fine-tuned on the HAM10000 dermoscopy dataset for 7-class skin lesion classification,  
> with Grad-CAM explainability, dual skin-pixel validation, and a Gemini-powered AI assistant.

**Built by [Ammar Gazi](https://github.com/ammargazi) · B.E. Computer Engineering · 2025–26**

<br/>

![DermaScan AI Banner](https://img.shields.io/badge/DermaScan_AI-Neural_Dermatology-00D2B4?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMiAxNWwtNS01IDEuNDEtMS40MUwxMCAxNC4xN2w3LjU5LTcuNTlMMTkgOGwtOSA5eiIvPjwvc3ZnPg==)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Training](#-training)
- [Web App (Streamlit)](#-web-app-streamlit)
- [Web App (Vercel Deployment)](#-web-app-vercel-deployment)
- [Grad-CAM Explainability](#-grad-cam-explainability)
- [Skin Validation](#-skin-pixel-validation)
- [AI Chatbot](#-ai-chatbot)
- [Results](#-results)
- [Disclaimer](#-disclaimer)

---

## 🧬 Overview

**DermaScan AI** is a deep learning web application for skin lesion classification built as a 3rd-Year B.E. Computer Engineering project. It classifies dermoscopy images into **7 diagnostic categories** from the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection), with explainability via **Grad-CAM** heatmaps and a scoped **Gemini AI assistant** for result interpretation.

The system addresses the severe class imbalance in HAM10000 (Melanocytic Nevi account for ~67% of samples) using `WeightedRandomSampler`, and exports model weights in **FP16** to stay under GitHub's 25 MB file limit while retaining inference accuracy.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **7-Class Classification** | Benign, pre-cancerous, and malignant lesion detection |
| 🛡️ **Skin-Pixel Gatekeeper** | Dual Kovac RGB + YCbCr validation before inference |
| 🌡️ **Grad-CAM Heatmaps** | Visual explainability on `model.layer4[-1]` |
| ⚖️ **Class Imbalance Handling** | `WeightedRandomSampler` for balanced training |
| ⚡ **AMP Training** | Automatic Mixed Precision on CUDA |
| 💾 **FP16 Weights** | ~22 MB export (< GitHub's 25 MB limit) |
| 🤖 **Gemini AI Assistant** | Scoped chatbot for result interpretation |
| 🔒 **Secure API Proxy** | Vercel serverless function — key never in browser |
| 📱 **Responsive UI** | Mobile, tablet, and desktop layouts |

---

## 🛠 Tech Stack

**Machine Learning**
- [PyTorch](https://pytorch.org/) — model training, inference, AMP
- [torchvision](https://pytorch.org/vision/) — ResNet18, transforms
- [scikit-learn](https://scikit-learn.org/) — stratified train/val split
- [OpenCV](https://opencv.org/) — Grad-CAM overlay blending
- [Pillow](https://pillow.readthedocs.io/) — image preprocessing

**Web Application**
- [Streamlit](https://streamlit.io/) — Python web UI (`app.py`)
- Vanilla HTML/CSS/JS — futuristic frontend (`index.html`)
- [Vercel](https://vercel.com/) — static hosting + serverless functions
- [Gemini 2.0 Flash](https://aistudio.google.com/) — AI assistant backend

**Dataset**
- [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — Human Against Machine with 10000 training images

---

## 📊 Dataset

The **HAM10000** (*Human Against Machine with 10,000 training images*) dataset is a large collection of multi-source dermatoscopic images of common pigmented skin lesions.

| Class | Code | Type | Count (approx.) |
|---|---|---|---|
| Melanocytic Nevi | `nv` | Benign | ~6,705 (67%) |
| Melanoma | `mel` | **Malignant** | ~1,113 (11%) |
| Benign Keratosis-like | `bkl` | Benign | ~1,099 (11%) |
| Basal Cell Carcinoma | `bcc` | **Malignant** | ~514 (5%) |
| Actinic Keratoses | `akiec` | Pre-cancerous | ~327 (3%) |
| Vascular Lesions | `vasc` | Benign | ~142 (1.4%) |
| Dermatofibroma | `df` | Benign | ~115 (1.1%) |

> **Class imbalance** is handled via `WeightedRandomSampler` — each class gets equal
> representation per training batch without modifying the underlying dataset.

**Setup:**
```
dataset/
├── HAM10000_metadata.csv
└── images/
    ├── ISIC_0024306.jpg
    ├── ISIC_0024307.jpg
    └── ...
```

Download from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) and place in the `dataset/` folder.

---

## 🏗 Model Architecture

```
Input (224×224×3)
      │
      ▼
ResNet18 Backbone (ImageNet pretrained)
  ├── Conv1 → BN → ReLU → MaxPool
  ├── Layer1 (64 channels)
  ├── Layer2 (128 channels)
  ├── Layer3 (256 channels)
  └── Layer4 (512 channels)  ◄── Grad-CAM hooks here
      │
      ▼
GlobalAvgPool → Dropout(0.4) → Linear(512 → 7)
      │
      ▼
Softmax → 7-class probabilities
```

**Training configuration:**

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | `1e-4` |
| Weight Decay | `1e-4` |
| Scheduler | CosineAnnealingLR |
| Epochs | 25 |
| Batch Size | 64 |
| Image Size | 224 × 224 |
| Dropout | 0.4 |
| AMP | Enabled (CUDA only) |
| Export | FP16 (~22 MB) |

---

## 📁 Project Structure

```
dermascan-ai/
│
├── 📄 main.py                  # Training script
├── 📄 app.py                   # Streamlit web app
├── 📄 model_final.pth          # Trained FP16 weights (generated after training)
│
├── 📁 dataset/
│   ├── HAM10000_metadata.csv
│   └── images/
│
├── 📁 public/                  # Vercel frontend
│   └── index.html              # Futuristic web UI
│
├── 📁 api/                     # Vercel serverless functions
│   └── chat.js                 # Gemini API proxy (key stays server-side)
│
├── 📄 vercel.json              # Vercel routing config
├── 📄 requirements.txt         # Python dependencies
└── 📄 README.md
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/ammargazi/dermascan-ai.git
cd dermascan-ai
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

**4. Download the dataset**

Place HAM10000 images and metadata CSV in `dataset/` as shown in the structure above.

---

## 🏋️ Training

Run the training script:

```bash
python main.py
```

**What happens:**
1. Reads `dataset/HAM10000_metadata.csv`
2. Deduplicates by `lesion_id` (prevents data leakage)
3. Stratified 80/20 train/val split
4. Trains ResNet18 with AMP for 25 epochs
5. Saves best epoch weights as `model_final.pth` in **FP16**

**Expected output:**
```
══════════════════════════════════════════════════════════════════════
  DermaScan AI — Training Configuration
══════════════════════════════════════════════════════════════════════
  Device      : cuda (NVIDIA GeForce RTX ...)
  AMP enabled : True
  Epochs      : 25  |  Batch: 64  |  LR: 0.0001
══════════════════════════════════════════════════════════════════════
 Epoch   Tr Loss    Tr Acc    Vl Loss    Vl Acc    Time
----------------------------------------------------------------------
   1     1.2340     62.30     1.1890     65.40     48s
   2     0.9820     70.15     0.9540     71.20     46s ✔
  ...
  25     0.4210     84.60     0.5120     81.30     44s ✔
══════════════════════════════════════════════════════════════════════
  Training complete.  Best validation accuracy : 81.30%
  FP16 weights saved to                        : model_final.pth
  Saved model file size                        : 21.8 MB
══════════════════════════════════════════════════════════════════════
```

> ⏱ Training time: ~20 min on GPU · ~4–6 hours on CPU

---

## 🌐 Web App (Streamlit)

After training, run the Streamlit app locally:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

**App flow:**
1. Upload a JPG/JPEG/PNG skin lesion image
2. Dual skin-pixel validation (Kovac RGB + YCbCr)
3. ResNet18 inference → softmax probabilities
4. Grad-CAM heatmap generation on `layer4`
5. Results displayed with confidence bars and risk level

---

## 🚀 Web App (Vercel Deployment)

The `public/index.html` + `api/chat.js` version is deployable as a **zero-config static site** on Vercel with a secure serverless Gemini proxy.

### Deploy steps

**1. Push to GitHub**
```bash
git add .
git commit -m "initial commit"
git push origin main
```

**2. Import on Vercel**
- Go to [vercel.com](https://vercel.com) → New Project → Import repo
- Framework Preset: **Other**
- Output Directory: `public`
- Build Command: *(leave empty)*

**3. Set environment variable**

In Vercel dashboard → Settings → Environment Variables:

| Name | Value |
|---|---|
| `GEMINI_API_KEY` | `AIzaSy...your_key...` |

**4. Deploy** — your app is live at `https://your-app.vercel.app`

### How the proxy works

```
Browser                    Vercel Edge              Google Gemini
   │                           │                         │
   │  POST /api/chat           │                         │
   │  { history, context }     │                         │
   │ ─────────────────────────►│                         │
   │                           │  POST generateContent   │
   │                           │  + GEMINI_API_KEY       │
   │                           │ ───────────────────────►│
   │                           │                         │
   │                           │◄── { reply } ───────────│
   │◄── { reply } ─────────────│                         │
   │                           │                         │
```

> 🔒 The API key **never reaches the browser** — it lives only in Vercel's encrypted environment variable store.

---

## 🌡️ Grad-CAM Explainability

DermaScan AI uses **Gradient-weighted Class Activation Mapping** (Grad-CAM) to produce visual explanations.

**Reference:** Selvaraju et al., 2017 — *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*

**How it works:**

```
1. Register forward hook on model.layer4[-1]  → capture activations A^k
2. Register backward hook on model.layer4[-1] → capture gradients ∂y^c/∂A^k
3. Compute importance weights:  α^c_k = GlobalAvgPool(∂y^c/∂A^k)
4. Generate CAM:  L^c = ReLU(Σ_k  α^c_k ⊙ A^k)
5. Upsample 7×7 → 224×224, apply Jet colormap, alpha-blend over original
```

**Heatmap color interpretation:**

| Color | Meaning |
|---|---|
| 🔴 Red / Yellow | High activation — model focused here strongly |
| 🟢 Green | Moderate activation |
| 🔵 Blue | Low activation — little influence on prediction |

---

## 🛡️ Skin-Pixel Validation

Before inference, every uploaded image passes a **dual-rule skin segmentation** check to reject non-skin images (screenshots, objects, animals, etc.).

**Rule 1 — Kovac et al. (2002) explicit RGB:**
```python
R > 95  AND  G > 40  AND  B > 20
max(R,G,B) - min(R,G,B) > 15
|R - G| > 15  AND  R > G  AND  R > B
```

**Rule 2 — YCbCr chrominance locus:**
```python
Cb ∈ [77, 127]  AND  Cr ∈ [133, 173]
```

A pixel must satisfy **both rules simultaneously**. Images with fewer than **10% qualifying pixels** are rejected.

---

## 🤖 AI Chatbot

The embedded **DermaScan Assistant** is powered by `gemini-2.0-flash` with a strict system prompt that enforces domain scope.

**Allowed topics:**
- The 7 HAM10000 lesion classes
- Interpreting confidence scores and Grad-CAM heatmaps
- ABCDE rule for skin lesion awareness
- When to consult a dermatologist
- HAM10000 dataset and ResNet18 architecture (conceptual)

**Hard blocked:**
- ❌ Inventing skin diseases outside the 7 HAM10000 classes
- ❌ Providing diagnoses or clinical opinions
- ❌ Treatment advice or medication names
- ❌ Any off-topic question (politics, coding, general knowledge)

**Scan-context awareness:** After a scan completes, the chatbot automatically receives the predicted class, confidence score, and risk level so it can answer *"What does my result mean?"* accurately.

---

## 📈 Results

| Metric | Value |
|---|---|
| Best Validation Accuracy | ~81% |
| Model Size (FP16) | ~22 MB |
| Classes | 7 |
| Training Images | ~8,012 (deduplicated) |
| Validation Images | ~2,003 |
| Training Epochs | 25 |
| Device | CUDA (AMP) |

> Results vary by GPU and random seed. The model intentionally uses a conservative architecture (ResNet18 + Dropout) to avoid overfitting on the relatively small HAM10000 dataset.

---

## ⚠️ Disclaimer

> **DermaScan AI is an academic research and educational tool.**
>
> It does **NOT** constitute a clinical medical diagnosis and should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment.
>
> Always consult a **licensed dermatologist** for evaluation of any skin lesion or concern. Early professional diagnosis is critical for conditions like melanoma.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**DermaScan AI** · HAM10000 · ResNet18 · Grad-CAM XAI

Built by **Ammar Gazi** · B.E. Computer Engineering · 2025–26

</div>
