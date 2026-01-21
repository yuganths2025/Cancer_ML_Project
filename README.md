# Breast Cancer Classification using ANN

This repository contains an enhanced Python implementation of an Artificial Neural Network (MLP Classifier) to predict breast cancer malignancy using the Scikit-Learn Wisconsin dataset.

## 🚀 Key Features
- **Automated Hyperparameter Tuning:** Uses `GridSearchCV` to optimize hidden layer sizes and learning rates.
- **Robust Preprocessing:** Implements `StandardScaler` for feature normalization and `stratify` splits to handle class imbalance.
- **Visual Analytics:** Generates a Loss Curve to track model convergence and a Seaborn heatmap for Confusion Matrix analysis.
- **Python 3.13 Ready:** Includes multiprocessing guards and serial processing configurations to prevent `TerminatedWorkerError` on Windows systems.

## 🛠️ Installation
Ensure you have Python 3.8+ installed. Install the required libraries using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
