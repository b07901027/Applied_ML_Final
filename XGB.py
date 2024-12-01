# Import necessary libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Import SMOTE for oversampling
from imblearn.over_sampling import SMOTE

# Import XGBoost
from xgboost import XGBClassifier

# Set random seed for reproducibility
np.random.seed(42)

# Define filenames
X_filename = "X_Reduced_set.csv"
Y_filename = "Y_Reduced_set.csv"

# Read in provided CSVs
print('Reading in data...')
X = pd.read_csv(X_filename)
Y = pd.read_csv(Y_filename)

print('Dropping columns...')
dropped_columns = ['id_33', 'id_34']
X = X.drop(columns=dropped_columns, errors='ignore')  # Drops 'id_33' and 'id_34' if they exist

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
X_train_balanced, y_train_balanced = (X_train_scaled_full, y_train_full)


# Define XGBoost parameters
params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'scale_pos_weight': 1,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# Initialize the XGBoost model
final_xgb = XGBClassifier(**params)

# Train the model on the balanced data
print('Training the final model on balanced data...')
final_xgb.fit(X_train_balanced, y_train_balanced)

# Evaluate the final model on the test set
print('\nEvaluating the final model on the test set...')
y_test_pred = final_xgb.predict(X_test_scaled)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Accuracy on test set: {test_accuracy:.4f}')

# Classification Report
print('\nClassification Report:')
print(classification_report(y_test, y_test_pred))

# Calculate ROC AUC
y_test_pred_proba = final_xgb.predict_proba(X_test_scaled)[:, 1]
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
model_filename = 'best_xgb_model_smote.pkl'
scaler_filename = 'scaler_smote.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(final_xgb, file)
print(f'Model saved as {model_filename}')

with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler_full, file)
print(f'Scaler saved as {scaler_filename}')
