# Import necessary libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

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

# Define stratified k-fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Define individual models
print('Defining individual models...')
catboost_model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=RANDOM_STATE,
    verbose=False,
    early_stopping_rounds=50,
    class_weights=[1, 10],
    thread_count=2
)

xgboost_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=RANDOM_STATE,
    scale_pos_weight=10,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=2
)

lightgbm_model = LGBMClassifier(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary',
    random_state=RANDOM_STATE,
    class_weight={0: 1, 1: 10},
    n_jobs=2
)

# Create stacking ensemble
print('Creating stacking ensemble...')
estimators = [
    ('catboost', catboost_model),
    ('xgboost', xgboost_model),
    ('lightgbm', lightgbm_model)
]

final_estimator = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_seed=RANDOM_STATE
)

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=skf,
    n_jobs=2,
    passthrough=False
)

# Train the stacking ensemble using cross-validation
print('Training and evaluating the stacking ensemble with cross-validation...')
cv_scores = cross_val_score(
    stacking_clf,
    X_train_scaled_full,
    y_train_full,
    cv=skf,
    scoring='roc_auc',
    n_jobs=2
)
print(f'Cross-validation ROC AUC scores: {cv_scores}')
print(f'Mean ROC AUC score: {cv_scores.mean():.4f}')

# Fit the stacking ensemble on the full training data
print('Fitting the stacking ensemble on the full training data...')
stacking_clf.fit(X_train_scaled_full, y_train_full)

# Evaluate the ensemble on the test set
print('\nEvaluating the stacking ensemble on the test set...')
y_test_pred = stacking_clf.predict(X_test_scaled)
y_test_pred_proba = stacking_clf.predict_proba(X_test_scaled)[:, 1]

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
plt.title('Confusion Matrix Heatmap (Stacking Ensemble)')
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
plt.title('Receiver Operating Characteristic (Stacking Ensemble)')
plt.legend(loc='lower right')
plt.show()

# Save the ensemble model and the scaler
ensemble_model_filename = 'stacking_ensemble_model.pkl'
scaler_filename = 'scaler.pkl'

with open(ensemble_model_filename, 'wb') as file:
    pickle.dump(stacking_clf, file)
print(f'Ensemble model saved as {ensemble_model_filename}')

with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler_full, file)
print(f'Scaler saved as {scaler_filename}')
