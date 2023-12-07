# Logistic Regression Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'datasets/dataset_preprocessed.csv'
data = pd.read_csv(file_path)

X = data.drop('Species', axis=1)  # Features
y = pd.get_dummies(data['Species'], drop_first=True)  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train_scaled, y_train.values.ravel())

y_pred_lr = logistic_regression_model.predict(X_test_scaled)

accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f'Logistic Regression Accuracy: {accuracy_lr:.4f}')

cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Mouse', 'Human'], yticklabels=['Mouse', 'Human'])
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
confusion_matrix_path = 'results/logistic_regression_confusion_matrix.png'
plt.savefig(confusion_matrix_path, bbox_inches='tight')
