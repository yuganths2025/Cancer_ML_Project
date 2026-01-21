import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Wrapping in the main block is required for Windows multiprocessing
if __name__ == "__main__":
    
    # 1. Load and Prepare Data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Stratify ensures the cancer/no-cancer ratio is preserved in the split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Hyperparameter Tuning
    param_grid = {
        'hidden_layer_sizes': [(30,), (50, 20)],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01],
    }

    # Added early_stopping to prevent overfitting during long runs
    mlp = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)
    
    print("Starting Grid Search... (this may take a moment)")
    grid = GridSearchCV(mlp, param_grid, cv=5, scoring='f1', n_jobs=1)
    grid.fit(X_train_scaled, y_train)

    # 3. Final Model Evaluation
    best_mlp = grid.best_estimator_
    y_pred = best_mlp.predict(X_test_scaled)

    print(f"\nBest Parameters Found: {grid.best_params_}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # 4. Visualizations
    plt.figure(figsize=(12, 5))

    # Loss Curve Plot
    plt.subplot(1, 2, 1)
    plt.plot(best_mlp.loss_curve_)
    plt.title("Neural Network Learning (Loss Curve)")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    # Confusion Matrix Heatmap
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data.target_names, yticklabels=data.target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()
    plt.show()

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
