# 🧠 ML From Scratch: K-Nearest Neighbors & Radius Nearest Neighbors

This project demonstrates a complete machine learning pipeline for implementing and evaluating K-Nearest Neighbors (KNN) and Radius Nearest Neighbors (rNN) classifiers **from scratch**, using `numpy` only. Additionally, it includes comparisons with `scikit-learn` implementations for benchmarking, and explores hyperparameter tuning using cross-validation and performance visualizations.

## 🚀 Project Overview

The goal is to build and understand nearest neighbor classification methods without relying on machine learning libraries for the core logic.

### Datasets Used:

- **MNIST Handwritten Digits** – for KNN implementation:
  - Image size: 28x28 (flattened to 784 features)
  - Task: Multi-class classification (digits 0–9)

- **Breast Cancer Wisconsin Dataset** – for Radius-NN:
  - Features: 30 numerical diagnostic measurements
  - Task: Binary classification (Malignant vs. Benign)

---

## 📦 Features

### ✅ K-Nearest Neighbors (KNN) from Scratch
- Euclidean & Manhattan distance metrics
- Tie-breaking logic
- Vectorized computation
- Custom `k` hyperparameter
- Cross-validation (k-fold) for hyperparameter tuning
- Performance metrics:
  - Confusion matrix
  - Accuracy
  - Macro-averaged F1 score
- Visualization of accuracy and F1-score vs. `k`
- Heatmap plots for confusion matrices

### ✅ Radius Nearest Neighbors (rNN) from Scratch
- Radius-based neighbor selection
- Tie-breaking and fallback logic for empty neighborhoods
- Distance options: Euclidean and Manhattan
- Visual analysis of overfitting/underfitting based on radius
- Cross-validation across a range of `r` values
- Visualization of model performance over different radii

### ✅ Benchmarking with Scikit-learn
- `KNeighborsClassifier`
- Evaluation using:
  - `accuracy_score`
  - `classification_report`
  - `confusion_matrix`
- Heatmaps for scikit-learn confusion matrices

---

## 📊 Visuals & Insights

Included visual outputs:
- Accuracy & F1-score comparisons for different `k` and distance metrics
- Cross-validation heatmaps for model performance
- Confusion matrices (custom + scikit-learn)
- Performance over radius (`r`) for rNN

Example charts:
- ![Accuracy vs K](./images/knn_accuracy_k.png)
- ![F1 vs K](./images/knn_f1_k.png)
- ![Confusion Matrix](./images/cm_euclidean.png)

---

## 🛠️ Tech Stack

- **Language**: Python 3
- **Libraries**:
  - `numpy` – for data manipulation & vectorization
  - `matplotlib` & `seaborn` – for data visualization
  - `pandas` – for dataset handling
  - `idx2numpy` – to parse MNIST dataset
  - `scikit-learn` – for benchmarking

---

## 📈 Results Summary

| Model           | Dataset         | Accuracy | Macro F1 Score |
|-----------------|------------------|----------|----------------|
| KNN (Euclidean) | MNIST            | 96.1%    | 95.9%          |
| KNN (Manhattan) | MNIST            | 95.8%    | 95.7%          |
| rNN (Euclidean) | Breast Cancer    | 91.0%    | 90.1%          |
| rNN (Manhattan) | Breast Cancer    | 90.5%    | 89.7%          |

> *Note: Results may vary slightly due to random splits.*

---

## 🧠 Key Learnings

- Building a classifier from scratch improves understanding of vectorization and distance metrics.
- KNN performs well on low-dimensional structured data (like MNIST).
- Radius-based methods are sensitive to the curse of dimensionality, as seen with rNN on high-dimensional data.
- Visualizations help interpret overfitting/underfitting behavior with hyperparameter tuning.

---
