import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural network architecture
class GeneExpressionNN(nn.Module):
    def __init__(self, input_size):
        super(GeneExpressionNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# class GeneExpressionNN(nn.Module):
#     def __init__(self, input_size):
#         super(GeneExpressionNN, self).__init__()
#         self.layer1 = nn.Linear(input_size, 128)
#         self.layer2 = nn.Linear(128, 64)
#         self.layer3 = nn.Linear(64, 32)
#         self.output_layer = nn.Linear(32, 1)

#         # Additional layers for matching dimensions for residual connections
#         self.skip1 = nn.Linear(input_size, 64) # To match dimension of layer2
#         self.skip2 = nn.Linear(64, 32) # To match dimension of layer3

#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         identity = x

#         # Layer 1
#         x = self.relu(self.layer1(x))
#         x = self.dropout(x)

#         # Layer 2 with residual connection
#         x = self.relu(self.layer2(x)) + self.skip1(identity)
        
#         identity = x # Update identity to output of layer2

#         # Layer 3 with residual connection
#         x = self.relu(self.layer3(x)) + self.skip2(identity)

#         # Output layer
#         x = self.sigmoid(self.output_layer(x))

#         return x

# Load the dataset
file_path = 'datasets/dataset_preprocessed.csv'
data = pd.read_csv(file_path)

X = data.drop('Species', axis=1).values
y = pd.get_dummies(data['Species'], drop_first=True).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled.astype(np.float32)).to(device)
y_train_tensor = torch.tensor(y_train.astype(np.float32)).to(device)
X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32)).to(device)
y_test_tensor = torch.tensor(y_test.astype(np.float32)).to(device)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Instantiate the model, loss function, and optimizer
model = GeneExpressionNN(X_train_scaled.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.float().view_as(output))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_predicted = (output > 0.5).float()
        train_total += target.size(0)
        train_correct += (train_predicted == target).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = train_correct / train_total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# Evaluate the model
model.eval()
y_pred_list = []
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        predicted = (outputs > 0.5).float()
        y_pred_list.append(predicted.cpu().numpy())
        total += target.size(0)
        correct += (predicted == target).sum().item()


accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

# Flatten the list of predictions
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
y_pred_flat = [item for sublist in y_pred_list for item in sublist]

cm_nn = confusion_matrix(y_test, y_pred_flat)

# Plot the confusion matrix
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', xticklabels=['Mouse', 'Human'], yticklabels=['Mouse', 'Human'])
plt.title('Confusion Matrix for Neural Network Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('results/nn_confusion_matrix.png', bbox_inches='tight')
