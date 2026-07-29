# 🎗️ Breast Cancer Classification with Neural Network (NN)
> **An Enterprise Multi-Agent AI Ecosystem for Precision Oncology & Biopsy Classification**
<img width="1894" height="914" alt="Screenshot 2026-07-30 014655" src="https://github.com/user-attachments/assets/516e0c94-8d62-4082-a14f-459dc5bc22e3" />


[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Overview

**Breast Cancer Classification with Neural Network (NN)** is an AI-powered clinician decision-support platform engineered for real-time classification of breast tissue biopsies into **Malignant** or **Benign** categories.

Built on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, the system integrates an **Optuna-tuned Multi-Layer Perceptron (MLP) Neural Network** alongside an autonomous **6-Agent Consensus Swarm** to deliver explainable, zero-trust, and highly accurate diagnostic predictions with a **0% missed cancer rate** (100% Sensitivity).

---

## 🔬 Key Architecture & Innovations

```
                               ┌────────────────────────────────────────┐
                               │       Clinician Portal (Port 8000)     │
                               └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │     FastAPI Microservice Gateway       │
                               └───────────────────┬────────────────────┘
                                                   │
                ┌──────────────────────────────────┴──────────────────────────────────┐
                ▼                                                                     ▼
┌───────────────────────────────┐                                   ┌───────────────────────────────────┐
│     Multi-Agent Swarm         │                                   │    Neural Network & ML Pipeline    │
├───────────────────────────────┤                                   ├───────────────────────────────────┤
│ 🛡️ SecurityShield             │                                   │ 🧠 MLP Neural Network (Primary)   │
│ 🔬 BiophysicMonitor           │ ──────── Consensus Protocol ────► │ 🌲 Random Forest                  │
│ 🎯 PredictiveEstimator        │                                   │ ⚡ XGBoost                         │
│ 📈 PrognosisRisk              │                                   │ 📊 Logistic Regression            │
│ 💊 TherapyOptimizer           │                                   └───────────────────────────────────┘
│ 👨‍⚕️ ChiefConsensusLead        │
└───────────────────────────────┘
```

### 🤖 Autonomous 6-Agent Consensus Swarm
1. **SecurityShield Agent**: Validates access credentials and sanitizes PHI telemetry in accordance with HIPAA standards.
2. **BiophysicMonitor Agent**: Performs sanity checks on 30 biophysical cell parameters to detect out-of-range sensor readings.
3. **PredictiveEstimator Agent**: Executes model inference using the trained MLP Neural Network.
4. **PrognosisRisk Agent**: Calculates hazard index scores and estimates 60-month disease-free survival probability.
5. **TherapyOptimizer Agent**: Generates BSA-based chemotherapy dosage estimates (e.g., AC-T, TC, TCH-P regimens).
6. **ChiefConsensusLead Agent**: Aggregates multi-agent votes to authorize the final clinical sign-off.

---

## 📊 Model Performance Benchmarks

All models were evaluated on 114 unseen test biopsies from the WDBC dataset using Optuna hyperparameter optimization:

| Model Algorithm | Malignant Recall (Sensitivity) | Overall Accuracy | ROC-AUC Score | Missed Cancer Rate |
| :--- | :---: | :---: | :---: | :---: |
| **🧠 Neural Network (MLP)** *(Primary)* | **100.0%** | **96.5%** | **99.1%** | **0.0%** |
| **⚡ XGBoost** | 97.6% | 95.6% | 98.8% | 2.4% |
| **🌲 Random Forest** | 95.2% | 94.7% | 98.1% | 4.8% |
| **📈 Logistic Regression** | 95.2% | 93.8% | 97.4% | 4.8% |

---

## 💻 Clinician Portal Features

* **🔬 Live 2D Cell Nucleus Morphological Projection**: HTML5 Canvas widget rendering real-time cell nuclear contours based on slider parameters.
* **🔍 SHAP Global Feature Importance**: Ranks top biophysical drivers influencing malignancy decisions (`worst area`, `worst concavity`, `mean radius`).
* **📉 2D PCA Cluster Projections**: Dimensionality reduction plot demonstrating clear separability between malignant and benign clusters.
* **📈 Kaplan-Meier Survival Curves**: Estimates 60-month disease-free survival probabilities.
* **📋 MDT Tumor Board Note Generator**: Auto-compiles multi-disciplinary discharge summaries with 1-click clipboard export.

---

## ⚡ Quick Start & Setup

### Prerequisites
* Python 3.10 or higher
* Git

### 1. Clone Repository
```bash
git clone https://github.com/17shravani/OncoNode-AI-Intelligent-Breast-Cancer-Classification-using-Neural-Networks.git
cd OncoNode-AI-Intelligent-Breast-Cancer-Classification-using-Neural-Networks
```

### 2. Install Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r onconode_system/backend/requirements.txt
```

### 3. Launch Clinician Portal
```bash
python -m uvicorn onconode_system.backend.app.main:app --host 127.0.0.1 --port 8000
```
Open **`http://localhost:8000`** in your browser.

---

## 🧪 Testing Presets

| Preset Profile | Key Inputs | Expected Classification | Malignancy Risk |
| :--- | :--- | :---: | :---: |
| **⚡ Malignant Profile** | Radius: `17.99`, Area: `1001.0`, Worst Area: `2019.0` | `WDBC-MALIGNANT` | **95.0%** |
| **🌱 Benign Profile** | Radius: `12.05`, Area: `449.3`, Worst Area: `582.6` | `WDBC-BENIGN` | **8.0%** |

---

## 📁 Repository Directory Structure

```text
.
├── Cancer Classification using neural Networks June 2026/
│   └── Breast_Cancer_Classification_with_Neural_Network.ipynb
├── breast_cancer_system/
│   ├── Breast_Cancer_Classification_Professional.ipynb
│   ├── models/           # Pre-trained MLP & Scaler Joblib artifacts
│   ├── plots/            # SHAP & ROC Curve visualizations
│   └── src/              # Feature preprocessing & trainer logic
├── onconode_system/
│   ├── backend/          # FastAPI REST endpoints & Multi-Agent Swarm
│   │   └── app/static/index.html   # Glassmorphic Clinician Portal UI
│   └── devops/           # Kubernetes manifests & CI/CD workflows
└── README.md
```

---

## 🛡️ License & Disclaimer

Distributed under the **MIT License**.

> **Clinical Disclaimer**: This software is intended for research and educational clinical decision-support demonstration purposes. It should not be used as a sole diagnostic instrument without physician sign-off.
