# Import necessary libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Import SMOTE for oversampling
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
np.random.seed(42)

# Define filenames
X_filename = "X_Reduced_PCA_set.csv"
Y_filename = "Y_Reduced_set.csv"

# Read in provided CSVs
print('Reading in data...')
X = pd.read_csv(X_filename)
Y = pd.read_csv(Y_filename)

# Ensure that the target variable is correctly formatted
Y = Y['isFraud']

# Split data into training and testing sets
print('Splitting data...')
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Feature Scaling
scaler_full = StandardScaler()
X_train_scaled_full = scaler_full.fit_transform(X_train_full)
X_test_scaled = scaler_full.transform(X_test)

# Use SMOTE to oversample the minority class in the training data
print('Applying SMOTE to balance the classes...')
#smote = SMOTE(random_state=42)
smote = SMOTE(sampling_strategy=0.2, random_state=42)
print(f'Before SMOTE, counts of label \'1\': {sum(y_train_full==1)}')
print(f'Before SMOTE, counts of label \'0\': {sum(y_train_full==0)}')
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled_full, y_train_full)
print(f'After SMOTE, counts of label \'1\': {sum(y_train_balanced==1)}')
print(f'After SMOTE, counts of label \'0\': {sum(y_train_balanced==0)}')

# Define the parameter combination (without 'class_weight' or 'sample_weight')
params = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.001,
    'batch_size': 64,
    'learning_rate': 'adaptive',
    'max_iter': 400,
    'random_state': 42,
    'early_stopping': True,
    'n_iter_no_change': 5,
    'verbose': True
}

# Initialize the final model
final_mlp = MLPClassifier(**params)

# Train the model on the balanced data
print('Training the final model on balanced data...')
final_mlp.fit(X_train_balanced, y_train_balanced)

# Evaluate the final model on the test set
print('\nEvaluating the final model on the test set...')
y_test_pred = final_mlp.predict(X_test_scaled)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Accuracy on test set: {test_accuracy:.4f}')

# Classification Report
print('\nClassification Report:')
print(classification_report(y_test, y_test_pred))

# Calculate ROC AUC
y_test_pred_proba = final_mlp.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_test_pred_proba)
print(f'ROC AUC Score on test set: {roc_auc:.4f}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print('Confusion Matrix:')
print(cm)

# Plot Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix Heatmap (SMOTE Oversampling)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random chance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (SMOTE Oversampling)')
plt.legend(loc='lower right')
plt.show()

# Save the final model and the scaler
model_filename = 'best_mlp_model_smote.pkl'
scaler_filename = 'scaler_smote.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(final_mlp, file)
print(f'Model saved as {model_filename}')

with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler_full, file)
print(f'Scaler saved as {scaler_filename}')
