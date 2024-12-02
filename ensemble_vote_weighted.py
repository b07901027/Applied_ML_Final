# Import necessary libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns


from imblearn.over_sampling import SMOTE

# Import classifiers
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Define filenames
X_filename = "X_Reduced_set.csv"
Y_filename = "Y_Reduced_set.csv"

# Read in provided CSVs
print('Reading in data...')
X = pd.read_csv(X_filename)
Y = pd.read_csv(Y_filename)

print('Dropping columns...')
dropped_columns = ['id_33', 'id_34']
X = X.drop(columns=dropped_columns, errors='ignore')

# Ensure that the target variable is correctly formatted
Y = Y['isFraud']

# Split data into training and testing sets
print('Splitting data...')
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_STATE, stratify=Y
)

# Feature Scaling
scaler_full = StandardScaler()
X_train_scaled_full = scaler_full.fit_transform(X_train_full)
X_test_scaled = scaler_full.transform(X_test)

# Use SMOTE to oversample the minority class in the training data
# Apply SMOTE correctly
print('Applying SMOTE to balance the training data...')
smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.5)
X_train_balanced, y_train_balanced = (X_train_scaled_full, y_train_full)
print(f'Balanced training set shape: {X_train_balanced.shape}, {y_train_balanced.shape}')

# Update individual models with class weights or scale_pos_weight
print('Defining individual models...')
catboost_model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=RANDOM_STATE,
    verbose=100,
    early_stopping_rounds=50,
    class_weights=[1, 10]  # Adjust the weight for the minority class
)

xgboost_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=RANDOM_STATE,
    scale_pos_weight=10,  # Adjust this based on the imbalance ratio
    use_label_encoder=False,
    eval_metric='logloss'
)

lightgbm_model = LGBMClassifier(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary',
    random_state=RANDOM_STATE,
    class_weight={0: 1, 1: 10}  # Adjust the weight for the minority class
)


# Create the voting ensemble
print('Creating voting ensemble...')
voting_clf = VotingClassifier(
    estimators=[
        ('catboost', catboost_model),
        ('xgboost', xgboost_model),
        ('lightgbm', lightgbm_model)
    ],
    voting='soft',
    n_jobs=-1
)

# Train the voting ensemble
print('Training the voting ensemble...')
voting_clf.fit(X_train_balanced, y_train_balanced)

# Evaluate the ensemble on the test set
print('\nEvaluating the voting ensemble on the test set...')
y_test_pred = voting_clf.predict(X_test_scaled)
y_test_pred_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Accuracy on test set: {test_accuracy:.4f}')

# Classification Report
print('\nClassification Report:')
print(classification_report(y_test, y_test_pred))

# Calculate ROC AUC
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
plt.title('Confusion Matrix Heatmap (Voting Ensemble)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random chance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Voting Ensemble)')
plt.legend(loc='lower right')
plt.show()

# Save the ensemble model and the scaler
ensemble_model_filename = 'voting_ensemble_model.pkl'
scaler_filename = 'scaler.pkl'

with open(ensemble_model_filename, 'wb') as file:
    pickle.dump(voting_clf, file)
print(f'Ensemble model saved as {ensemble_model_filename}')

with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler_full, file)
print(f'Scaler saved as {scaler_filename}')
