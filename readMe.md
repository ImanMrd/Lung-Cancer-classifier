# Lung Cancer Classification using Artificial Neural Network

A machine learning project that implements an Artificial Neural Network (ANN) for lung cancer classification using Principal Component Analysis (PCA) for dimensionality reduction.

## 📋 Overview

This project converts a MATLAB-based lung cancer classification system to Python, implementing:

-   Data preprocessing and cleaning
-   Principal Component Analysis (PCA) for feature reduction
-   Multi-layer Perceptron (MLP) neural network for classification
-   Performance evaluation on training and testing datasets

## 🚀 Features

-   **Data Augmentation**: Replicates dataset 4 times to increase training samples
-   **Data Shuffling**: Randomizes dataset order for better training
-   **PCA Dimensionality Reduction**: Reduces features to 50 components (or available features)
-   **Missing Value Handling**: Robust imputation using mean strategy
-   **Neural Network**: 3-layer MLP with 50 and 25 hidden neurons
-   **Performance Metrics**: Training and testing accuracy calculation

## 📁 Project Structure

```
LungCancerANN/
├── main.py                 # Main script
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── lung-cancerdata.xlsx   # Dataset (not included)
```

## 🛠️ Installation

### Prerequisites

-   Python 3.7 or higher
-   pip package manager

### Setup

1. **Clone or download this repository**

    ```bash
    git clone <repository-url>
    cd LungCancerANN
    ```

2. **Create a virtual environment (recommended)**

    ```bash
    python -m venv classifier
    ```

3. **Activate the virtual environment**

    - Windows:
        ```bash
        classifier\Scripts\activate
        ```
    - macOS/Linux:
        ```bash
        source classifier/bin/activate
        ```

4. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Dataset Requirements

The project expects an Excel file named `lung-cancerdata.xlsx` with:

-   Features in all columns except the last one
-   Target labels in the last column
-   Missing values can be represented as '?', empty cells, or NaN
-   Target labels should be binary (typically 1 and 2, where 2 gets converted to 0)

## 🏃‍♂️ Usage

1. **Place your dataset**

    - Ensure `lung-cancerdata.xlsx` is in the project directory
    - Dataset should contain lung cancer data with appropriate features and labels

2. **Run the classification**

    ```bash
    python main.py
    ```

3. **View results**
   The script will output:
    - Data loading and preprocessing information
    - Training progress details
    - Training accuracy percentage
    - Testing accuracy percentage
    - Additional model statistics

## 🔧 Configuration

You can modify these parameters in `main.py`:

```python
# PCA components (default: 50 or number of available features)
K = min(50, input_data.shape[1])

# Neural network architecture
hidden_layer_sizes=(50, 25)  # Two hidden layers with 50 and 25 neurons

# Training parameters
learning_rate_init=0.01      # Learning rate
max_iter=2000               # Maximum training iterations
tol=1e-6                   # Tolerance for stopping criterion

# Data split ratio
train_ratio = 0.7          # 70% training, 30% testing
```

## 📈 Model Architecture

The neural network consists of:

-   **Input Layer**: Number of features after PCA (≤50)
-   **Hidden Layer 1**: 50 neurons with tanh activation
-   **Hidden Layer 2**: 25 neurons with tanh activation
-   **Output Layer**: 1 neuron for binary classification

## 🔍 Algorithm Steps

1. **Data Loading**: Read Excel dataset with robust missing value handling
2. **Data Augmentation**: Replicate dataset 4 times
3. **Data Shuffling**: Randomize sample order
4. **Preprocessing**: Handle missing values using mean imputation
5. **Feature Reduction**: Apply PCA to reduce dimensionality
6. **Normalization**: Scale features to appropriate range
7. **Train-Test Split**: 70% training, 30% testing
8. **Model Training**: Train MLP with specified architecture
9. **Evaluation**: Calculate and display accuracy metrics

## 📋 Dependencies

-   **pandas (≥1.3.0)**: Data manipulation and Excel file reading
-   **numpy (≥1.21.0)**: Numerical operations and array handling
-   **scikit-learn (≥1.0.0)**: Machine learning algorithms (PCA, MLP, preprocessing)
-   **openpyxl (≥3.0.0)**: Excel file reading support

## 🐛 Troubleshooting

### Common Issues

1. **"could not convert string to float" Error**

    - Ensure dataset doesn't contain unexpected string values
    - Check that missing values are properly handled
    - Verify Excel file format and structure

2. **File Not Found Error**

    - Confirm `lung-cancerdata.xlsx` is in the project directory
    - Check file name spelling and extension

3. **Memory Issues**

    - Reduce dataset replication factor (change from 4 to 2 or 1)
    - Decrease PCA components (K parameter)

4. **Poor Accuracy**
    - Adjust neural network parameters
    - Try different train-test split ratios
    - Experiment with preprocessing techniques

## 📊 Expected Output

```
Dataset loaded successfully. Shape: (xxx, yyy)
Cleaning data...
Training neural network...
Training completed successfully!
The percent of accuracy train is = XX.XX %
The percent of accuracy test is = XX.XX %

Training set size: XXX
Test set size: XXX
Number of features after PCA: XX
Original feature dimensions: XX
PCA explained variance ratio: X.XXXX
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review your dataset format and structure
3. Ensure all dependencies are properly installed
4. Create an issue in the repository with error details

## 🎯 Future Enhancements

-   [ ] Cross-validation implementation
-   [ ] Hyperparameter tuning with GridSearch
-   [ ] Support for multiple file formats (CSV, JSON)
-   [ ] Advanced preprocessing techniques
-   [ ] Model performance visualization
-   [ ] Feature importance analysis
-   [ ] Web interface for easy usage

---

**Note**: This project is converted from MATLAB implementation to Python for better accessibility and integration with modern ML workflows.
