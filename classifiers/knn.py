import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'datasets/dataset_pca.csv'
data = pd.read_csv(file_path)

X = data.drop('Species', axis=1)  # Features
y = pd.get_dummies(data['Species'], drop_first=True)  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train.values.ravel())

y_pred_knn = knn_model.predict(X_test_scaled)

accuracy_knn = accuracy_score(y_test, y_pred_knn)

cm_knn = confusion_matrix(y_test, y_pred_knn)

# Plot the confusion matrix
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Mouse', 'Human'], yticklabels=['Mouse', 'Human'])
plt.title('Confusion Matrix for KNN Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('results/knn_confusion_matrix.png', bbox_inches='tight')

print(f'KNN Classifier Accuracy: {accuracy_knn:.4f}')
