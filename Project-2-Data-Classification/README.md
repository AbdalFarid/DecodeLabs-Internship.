# DecodeLabs AI Internship — Project 2
## Data Classification Using AI 🤖📊

A supervised learning project building a **K-Nearest Neighbors (KNN) classifier**
on the Iris dataset, as the second milestone of the DecodeLabs Artificial
Intelligence Internship (Batch 2026).

### 🎯 Objective
Master the supervised learning pipeline: load data → scale features → split
train/test → train a classifier → evaluate with Confusion Matrix & F1 Score.

### ⚙️ Features
- Loads the classic Iris benchmark (150 samples, 3 classes, 4 features)
- StandardScaler for feature normalization (Mean=0, Variance=1)
- 80/20 train-test split with shuffle to remove order bias
- K-Nearest Neighbors classification
- Confusion Matrix + F1 Score + Classification Report
- Elbow Method to find the optimal K value
- Visualized error curve saved as `elbow_curve.png`

### 🧠 Concepts Applied (from Project PDF)
- **IPO Model**: Input (Iris) → Process (KNN) → Output (Metrics)
- **Supervised Learning**: Machine derives rules from labeled history
- **Feature Scaling**: StandardScaler for distance-based algorithms
- **Train-Test Split**: 80/20 with shuffle
- **KNN Algorithm**: Majority vote among K nearest neighbors
- **Elbow Method**: Optimal K at lowest error rate
- **Confusion Matrix**: TP / FP / FN / TN breakdown
- **F1 Score**: Harmonic mean of Precision & Recall

### 📊 Results
- **Accuracy**: 100%
- **F1 Score**: 100%
- **Confusion Matrix**: Perfect diagonal (zero misclassifications)
- **Optimal K**: 1 (via Elbow Method)

### 📸 Screenshot
*(Insert your terminal output screenshot here)*

### 🚀 How to Run
```bash
pip install -r requirements.txt
python classification.py
