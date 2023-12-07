# SVM Classifier

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'datasets/dataset_preprocessed.csv'
data = pd.read_csv(file_path)

X = data.drop('Species', axis=1)  # Features
y = pd.get_dummies(data['Species'], drop_first=True)  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train.values.ravel())

# Testing
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluattion
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Calculating the confusion matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)

# Plotting the confusion matrix
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['Mouse', 'Human'], yticklabels=['Mouse', 'Human'])
plt.title('Confusion Matrix for SVM Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('results/svm_confusion_matrix.png', bbox_inches='tight')

# Outputting the accuracy
print(f'SVM Classifier Accuracy: {accuracy_svm:.4f}')
