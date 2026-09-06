# 🫁 Lung Cancer Classification with Neural Networks

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=scipy&logoColor=white)]()

A machine learning project that implements an Artificial Neural Network (ANN) to classify lung cancer data, utilizing Principal Component Analysis (PCA) for efficient dimensionality reduction.

## 📋 Overview

This Python-based project focuses on robust data preprocessing and neural network classification. It demonstrates end-to-end ML workflow implementation:

- **Data Preprocessing:** Handles missing values with mean imputation.
- **Data Augmentation:** Replicates the dataset 4x and shuffles it to increase training robustness.
- **Dimensionality Reduction (PCA):** Reduces features to 50 principal components to optimize training.
- **Classification (MLP):** Uses a Multi-Layer Perceptron neural network with a 50-25 hidden layer architecture.
- **Evaluation:** Calculates and visualizes comprehensive metrics including Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

## 🗂️ Project Structure

`	ext
📦 Lung-Cancer-classifier
 ┣ 📜 main.py                 # Main machine learning pipeline script
 ┣ 📜 requirements.txt        # Python dependencies
 ┣ 📜 lung-cancerdata.xlsx    # Target dataset
 ┣ 🖼️ confusion_matrices.png  # Generated evaluation visualization
 ┣ 🖼️ training_curves.png     # Generated training loss/MSE curves
 ┗ 📜 README.md               # Project documentation
`

## 🛠️ Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   `ash
   git clone https://github.com/ImanMrd/Lung-Cancer-classifier.git
   cd Lung-Cancer-classifier
   `

2. **Install dependencies:**
   `ash
   pip install -r requirements.txt
   `

3. **Ensure dataset is present:**
   Verify lung-cancerdata.xlsx is in the root directory.

4. **Run the model pipeline:**
   `ash
   python main.py
   `

### 📊 Expected Output
The script will output detailed console metrics (TP, TN, FP, FN, Accuracy, Specificity, etc.) and generate two visualization files (confusion_matrices.png and 	raining_curves.png) demonstrating the model's performance.
