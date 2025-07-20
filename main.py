import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def calculate_detailed_metrics(y_true, y_pred):
  
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
   
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
       
        if len(np.unique(y_true)) == 1:
            if np.unique(y_true)[0] == 0:
                tn, fp, fn, tp = len(y_true), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(y_true)
        else:
            tn = fp = fn = tp = 0
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'specificity': specificity, 'sensitivity': sensitivity
    }

def print_metrics(metrics, dataset_name):
    
    print(f"\n{'='*50}")
    print(f"{dataset_name.upper()} DATASET METRICS")
    print(f"{'='*50}")
    print(f"Confusion Matrix Components:")
    print(f"  True Positives (TP):  {metrics['tp']}")
    print(f"  True Negatives (TN):  {metrics['tn']}")
    print(f"  False Positives (FP): {metrics['fp']}")
    print(f"  False Negatives (FN): {metrics['fn']}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")

def plot_training_curves(net, train_mse, test_mse):
    
    
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
   
    if hasattr(net, 'loss_curve_'):
        ax1.plot(net.loss_curve_, 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax1.set_title('Neural Network Training Loss Curve')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        
        final_loss = net.loss_curve_[-1]
        ax1.annotate(f'Final Loss: {final_loss:.6f}', 
                    xy=(len(net.loss_curve_)-1, final_loss),
                    xytext=(len(net.loss_curve_)*0.7, final_loss*1.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    else:
        ax1.text(0.5, 0.5, 'Loss curve not available\n(Early stopping may have occurred)', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Training Loss Curve')
    
    
    mse_values = [train_mse, test_mse]
    labels = ['Training MSE', 'Testing MSE']
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax2.bar(labels, mse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('MSE Comparison: Training vs Testing')
    ax2.grid(True, alpha=0.3, axis='y')
    
   
    for bar, value in zip(bars, mse_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.6f}', ha='center', va='bottom', fontweight='bold')
    
    
    ax2.set_ylim(0, max(mse_values) * 1.15)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training curves plot saved as 'training_curves.png'")

def plot_confusion_matrices(train_cm, test_cm, target_train, target_test, train_pred, test_pred):
   
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training Confusion Matrix
    im1 = ax1.imshow(train_cm, interpolation='nearest', cmap='Blues')
    ax1.set_title('Training Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
   
    thresh = train_cm.max() / 2.
    for i in range(train_cm.shape[0]):
        for j in range(train_cm.shape[1]):
            ax1.text(j, i, format(train_cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if train_cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    # Testing Confusion Matrix  
    im2 = ax2.imshow(test_cm, interpolation='nearest', cmap='Blues')
    ax2.set_title('Testing Confusion Matrix')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
  
    thresh = test_cm.max() / 2.
    for i in range(test_cm.shape[0]):
        for j in range(test_cm.shape[1]):
            ax2.text(j, i, format(test_cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if test_cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    
    classes = ['Negative', 'Positive'] if len(np.unique(np.concatenate([target_train, target_test]))) == 2 else [str(i) for i in np.unique(np.concatenate([target_train, target_test]))]
    ax1.set_xticks(range(len(classes)))
    ax1.set_yticks(range(len(classes)))
    ax1.set_xticklabels(classes)
    ax1.set_yticklabels(classes)
    
    ax2.set_xticks(range(len(classes)))
    ax2.set_yticks(range(len(classes)))
    ax2.set_xticklabels(classes)
    ax2.set_yticklabels(classes)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Confusion matrices plot saved as 'confusion_matrices.png'")


try:
    
    data = pd.read_excel('lung-cancerdata.xlsx', na_values=['?', '', ' ', 'nan', 'NaN', 'null'])
    print(f"Dataset loaded successfully. Shape: {data.shape}")
    
except Exception as e:
    print(f"Error loading Excel file: {e}")
    print("Please check if the file 'lung-cancerdata.xlsx' exists and is accessible.")
    exit()


print("\nCleaning data...")  ##for missed data replacement


for col in data.columns:
    data[col] = data[col].astype(str) 
    data[col] = data[col].replace(['?', '', ' ', 'nan', 'NaN', 'null'], np.nan)
    

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

print(f"After cleaning - Data types:\n{data.dtypes}")
print(f"Missing values per column:\n{data.isnull().sum()}")


dataset = data.values
print(f"Dataset shape after conversion: {dataset.shape}")


if not np.issubdtype(dataset.dtype, np.number):
    print("Warning: Dataset still contains non-numeric values")
    print(f"Dataset dtype: {dataset.dtype}")
    
    dataset = dataset.astype(float)


dataset = np.vstack([dataset, dataset, dataset, dataset])  ##copied to have better traning
print(f"Dataset shape after replication: {dataset.shape}")

# Shuffle the dataset
np.random.seed(42)
r = np.random.permutation(dataset.shape[0])
dataset = dataset[r, :]

# Separate input and target
input_data = dataset[:, :-1]
target = dataset[:, -1]

print(f"Input data shape: {input_data.shape}")
print(f"Target shape: {target.shape}")

# Handle missing values in input data using mean imputation
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
input_data = imputer.fit_transform(input_data)

# Handle missing values in target
target_imputer = SimpleImputer(strategy='most_frequent')
target = target_imputer.fit_transform(target.reshape(-1, 1)).ravel()

target = np.where(target == 2, 0, target)
target = target.astype(int)

print(f"Unique target values: {np.unique(target)}")

# Apply PCA with appropriate number of components
K = min(50, input_data.shape[1])
print(f"Using {K} PCA components (original features: {input_data.shape[1]})")

pca = PCA(n_components=K)
input_data = pca.fit_transform(input_data)

# Normalize input data
input_max = np.max(np.abs(input_data))
if input_max > 0:
    input_data = input_data / input_max

print(f"Input data shape after PCA: {input_data.shape}")
print(f"Input data range: [{np.min(input_data):.4f}, {np.max(input_data):.4f}]")

# Split data into train and test (70% train, 30% test)
s1, s2 = input_data.shape
s = int(s1 * 0.7)

data_train = input_data[:s, :]
data_test = input_data[s:, :]
target_train = target[:s]
target_test = target[s:]

print(f"Training set: {data_train.shape}, Test set: {data_test.shape}")

# Create and configure neural network
net = MLPClassifier(
    hidden_layer_sizes=(50, 25),
    activation='tanh',
    solver='adam',
    learning_rate_init=0.01,
    max_iter=200,
    tol=1e-6,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

# Train the network
print("Training neural network...")
try:
    net.fit(data_train, target_train)
    print("Training completed successfully!")
    
   
    train_pred_proba = net.predict_proba(data_train)[:, 1]  
    train_pred_binary = net.predict(data_train)
    
    
    train_mse = mean_squared_error(target_train, train_pred_proba)
    print(f"\nTraining Phase Mean Squared Error (MSE): {train_mse:.6f}")
    
   
    train_metrics = calculate_detailed_metrics(target_train, train_pred_binary)
    print_metrics(train_metrics, "TRAINING")
    
   
    test_pred_proba = net.predict_proba(data_test)[:, 1]
    test_pred_binary = net.predict(data_test)
    
    
    test_mse = mean_squared_error(target_test, test_pred_proba)
    print(f"\nTesting Phase Mean Squared Error (MSE): {test_mse:.6f}")
    
   
    test_metrics = calculate_detailed_metrics(target_test, test_pred_binary)
    print_metrics(test_metrics, "TESTING")
    
    # Plot training curves and MSE comparison
    print(f"\n{'='*50}")
    print("GENERATING PLOTS")
    print(f"{'='*50}")
    plot_training_curves(net, train_mse, test_mse)
    
    # Plot confusion matrices
    train_cm = confusion_matrix(target_train, train_pred_binary)
    test_cm = confusion_matrix(target_test, test_pred_binary)
    plot_confusion_matrices(train_cm, test_cm, target_train, target_test, train_pred_binary, test_pred_binary)
    
    # Summary comparison
    print(f"\n{'='*50}")
    print("SUMMARY COMPARISON")
    print(f"{'='*50}")
    print(f"Training Accuracy: {train_metrics['accuracy']*100:.2f}%")
    print(f"Testing Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Training MSE:      {train_mse:.6f}")
    print(f"Testing MSE:       {test_mse:.6f}")
    
  
    print(f"\n{'='*50}")
    print("MODEL INFORMATION")
    print(f"{'='*50}")
    print(f"Training set size: {len(target_train)}")
    print(f"Test set size: {len(target_test)}")
    print(f"Number of features after PCA: {K}")
    print(f"Original feature dimensions: {dataset.shape[1]-1}")
    print(f"PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    print(f"Number of training iterations: {net.n_iter_}")
    
    
    if hasattr(net, 'loss_curve_'):
        print(f"Final training loss: {net.loss_curve_[-1]:.6f}")
        print(f"Loss curve length: {len(net.loss_curve_)} iterations")
    
    
    print(f"\n{'='*50}")
    print("DETAILED CLASSIFICATION REPORT - TRAINING")
    print(f"{'='*50}")
    print(classification_report(target_train, train_pred_binary, zero_division=0))
    
    print(f"\n{'='*50}")
    print("DETAILED CLASSIFICATION REPORT - TESTING")
    print(f"{'='*50}")
    print(classification_report(target_test, test_pred_binary, zero_division=0))
    
    
    print(f"\n{'='*50}")
    print("CONFUSION MATRICES")
    print(f"{'='*50}")
    print("Training Confusion Matrix:")
    print(confusion_matrix(target_train, train_pred_binary))
    print("\nTesting Confusion Matrix:")
    print(confusion_matrix(target_test, test_pred_binary))
    
except Exception as e:
    print(f"Error during training: {e}")
    print("Check your data for any remaining issues.")