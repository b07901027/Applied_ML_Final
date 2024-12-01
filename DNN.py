# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
import json
import itertools

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns

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
print('Scaling features...')
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (50, 25, 12)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'batch_size': [64],
    'learning_rate': ['adaptive']
}

# Prepare to store the results
results = []
best_score = -np.inf
best_params = None
best_model = None

# Number of cross-validation folds
n_splits = 3  # Adjust as needed

# Generate all combinations of parameters
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]


# Custom MLPClassifier to record per-iteration metrics
class CustomMLPClassifier(MLPClassifier):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, batch_size='auto', learning_rate='constant',
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4, verbose=False,
                 warm_start=False, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10,
                 max_fun=15000):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
            alpha=alpha, batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter,
            shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping, validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change,
            max_fun=max_fun
        )
        self._train_loss_curve = []  # Initialize loss curve

    def fit(self, X, y):
        self._train_loss_curve = []  # Reset loss curve on each fit
        return super().fit(X, y)

    def partial_fit(self, X, y, classes=None):
        result = super().partial_fit(X, y, classes)
        if hasattr(self, 'loss_'):
            self._train_loss_curve.append(self.loss_)  # Append current loss
        return result

# Early stopping parameters
tol = 0.0001  # Tolerance for improvement in loss
patience = 5  # Number of epochs to wait for improvement
max_epochs = 200

# Loop over each parameter combination
for params in param_combinations:
    print(f"\nTraining with parameters: {params}")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_full_scaled, y_train_full)):
        print(f"\nStarting fold {fold_idx + 1}/{n_splits}")
        X_train, X_val = X_train_full_scaled[train_idx], X_train_full_scaled[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        # Initialize the CustomMLPClassifier with current parameters
        mlp = CustomMLPClassifier(
            **params,
            max_iter=1,
            random_state=42,
            warm_start=True,
            verbose=False
        )

        train_accuracies = []
        val_accuracies = []
        losses = []
        no_improvement_count = 0  # Counter for early stopping

        for iteration in range(max_epochs):
            mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))

            # Record training loss
            if hasattr(mlp, '_train_loss_curve'):
                losses.append(mlp._train_loss_curve[-1])
            else:
                losses.append(mlp.loss_)

            # Predict on training data
            y_train_pred = mlp.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)
            train_accuracies.append(train_acc)

            # Predict on validation data
            y_val_pred = mlp.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_accuracies.append(val_acc)

            # Check for early stopping
            if iteration > 0 and abs(losses[-1] - losses[-2]) < tol:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Early stopping at iteration {iteration + 1} due to no improvement in loss.")
                    break
            else:
                no_improvement_count = 0  # Reset counter if improvement occurs

            # Optional: Print progress every 10 iterations
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"Iteration {iteration + 1}/{max_epochs} - "
                      f"Loss: {losses[-1]:.4f} - "
                      f"Train Acc: {train_acc:.4f} - "
                      f"Val Acc: {val_acc:.4f}")

        # Store the results for this fold
        fold_results.append({
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'losses': losses
        })

    # Average metrics across folds
    avg_train_acc = np.mean([fr['train_accuracies'][-1] for fr in fold_results])
    avg_val_acc = np.mean([fr['val_accuracies'][-1] for fr in fold_results])

    print(f"\nAverage Training Accuracy: {avg_train_acc:.4f}")
    print(f"Average Validation Accuracy: {avg_val_acc:.4f}")

    # If this is the best model so far, save it
    if avg_val_acc > best_score:
        best_score = avg_val_acc
        best_params = params
        best_fold_results = fold_results
        # Retrain the best model on the full training set
        best_model = CustomMLPClassifier(
            **params,
            max_iter=max_epochs,
            random_state=42,
            verbose=False
        )
        best_model.fit(X_train_full_scaled, y_train_full)

    # Store the results for this parameter combination
    results.append({
        'params': params,
        'fold_results': fold_results
    })

# Save the per-iteration metrics to a file for future plotting
with open('training_results.json', 'w') as f:
    json.dump(results, f)
print("\nTraining results saved to 'training_results.json'")

# Evaluate the best model on the test set
print('\nEvaluating the best model on the test set...')
y_pred = best_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {accuracy:.4f}')

# Classification Report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Calculate ROC AUC
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC Score on test set: {roc_auc:.4f}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Plot Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random chance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Save the best model and the scaler
model_filename = 'best_mlp_model.pkl'
scaler_filename = 'scaler.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)
print(f'Model saved as {model_filename}')

with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)
print(f'Scaler saved as {scaler_filename}')
