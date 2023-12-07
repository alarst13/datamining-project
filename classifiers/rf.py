# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'datasets/dataset_pca.csv'  # Update with the actual path
data = pd.read_csv(file_path)

X = data.drop('Species', axis=1)  # Features
y = pd.get_dummies(data['Species'], drop_first=True)  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_scaled, y_train.values.ravel())

y_pred_rf = random_forest_model.predict(X_test_scaled)

accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Plot the confusion matrix
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Mouse', 'Human'], yticklabels=['Mouse', 'Human'])
plt.title('Confusion Matrix for Random Forest Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('results/random_forest_confusion_matrix.png', bbox_inches='tight')

print(f'Random Forest Classifier Accuracy: {accuracy_rf:.4f}')
