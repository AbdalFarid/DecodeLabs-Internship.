"""
DecodeLabs AI Internship - Project 2
Data Classification Using AI — KNN on Iris Dataset
Author: AbdalFarid
Batch: 2026
"""

# ============================================================
# STEP 1: IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)

# ============================================================
# STEP 2: LOAD & UNDERSTAND THE DATASET (IPO: INPUT)
# ============================================================
iris = load_iris()
X = iris.data           # Features: 4 dimensions (sepal/petal length/width)
y = iris.target         # Labels: 3 species (0, 1, 2)

# Convert to DataFrame for easy reading
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

print("=" * 60)
print("  DecodeLabs Project 2 — Data Classification Using AI")
print("=" * 60)
print(f"\nDataset Shape : {df.shape}  (150 samples × 4 features)")
print(f"Classes       : {list(iris.target_names)}")
print(f"Features      : {list(iris.feature_names)}")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================
# STEP 3: TRAIN-TEST SPLIT (80/20) — Shuffle to remove order bias
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,      # 20% for testing
    random_state=42,     # Reproducibility
    shuffle=True         # Randomize before splitting
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ============================================================
# STEP 4: FEATURE SCALING (StandardScaler: Mean=0, Variance=1)
# ============================================================
# Why? Raw features have different scales (cm vs mm). KNN uses
# distance, so unscaled features bias the model. Scaling fixes it.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit on train ONLY
X_test_scaled = scaler.transform(X_test)          # Apply same scale to test

# ============================================================
# STEP 5: TRAIN KNN MODEL (Instantiate → Fit → Predict)
# ============================================================
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# ============================================================
# STEP 6: EVALUATION (Confusion Matrix, F1 Score, Accuracy)
# ============================================================
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')
cm = confusion_matrix(y_test, predictions)

print("\n" + "=" * 60)
print("  MODEL EVALUATION (K=5)")
print("=" * 60)
print(f"\nAccuracy  : {accuracy:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"                Setosa  Versi  Virgin")
print(f"{cm}")
print(f"\nFull Classification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# ============================================================
# STEP 7: ELBOW METHOD — Find the Optimal K
# ============================================================
print("\n" + "=" * 60)
print("  ELBOW METHOD — Finding Optimal K")
print("=" * 60)

error_rates = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred_k = knn.predict(X_test_scaled)
    error_rates.append(1 - accuracy_score(y_test, pred_k))

best_k = error_rates.index(min(error_rates)) + 1
print(f"Best K (lowest error) : {best_k}")
print(f"Best Error Rate       : {min(error_rates):.4f}")

# Train final model with best K
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_scaled, y_train)
final_predictions = final_model.predict(X_test_scaled)
final_acc = accuracy_score(y_test, final_predictions)
print(f"Final Model (K={best_k}) Accuracy: {final_acc:.4f}")

# ============================================================
# STEP 8: VISUALIZE THE ELBOW CURVE
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(range(1, 31), error_rates, marker='o', linestyle='-', color='steelblue')
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Optimal K = {best_k}')
plt.title('Elbow Method — Optimal K for KNN')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=100)
print("\n✅ Elbow curve saved as 'elbow_curve.png'")
plt.show()

# ============================================================
# STEP 9: PREDICT A NEW FLOWER (Interactive Demo)
# ============================================================
print("\n" + "=" * 60)
print("  PREDICT A NEW FLOWER")
print("=" * 60)
print("Enter 4 measurements (in cm) to classify a flower.")

sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # Default sample
sample_scaled = scaler.transform(sample)
prediction = final_model.predict(sample_scaled)
print(f"\nSample Input       : {sample[0]}")
print(f"Predicted Species  : {iris.target_names[prediction[0]]}")
print("\nDone! ✅")
