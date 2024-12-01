from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import sys
import pandas as pd
import pickle
import matplotlib.pyplot as plt

max_iter = 250

# read cli arguments
improper_input_reason = ''
X_filename = "X_Reduced_PCA_set.csv"
Y_filename = "Y_Reduced_set.csv"


# Read in provided csvs
print('Reading in...')
sys.stdout.flush()
X = pd.read_csv(X_filename)
Y = pd.read_csv(Y_filename)

# Do logistic regression sklearn
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y['isFraud'], test_size=0.2, random_state=42)

# Train model
print('Training model...')
model = LogisticRegression(max_iter=max_iter)
model.fit(X_train, y_train)

# Predict
print('Predicting...')
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Confusion Matrix
print('Classification Report:')
print(classification_report(y_test, y_pred))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC Score: {roc_auc}')

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# Save model
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
print(f'Model saved as {filename}')
