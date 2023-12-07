import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'datasets/dataset_preprocessed.csv'
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop('Species', axis=1)
y = data['Species']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

# Convert the PCA result and target into a DataFrame
pca_df = pd.DataFrame(data=X_pca)
pca_df['Species'] = y

# Save the new dataset with PCA applied
pca_file_path = 'datasets/dataset_pca.csv'
pca_df.to_csv(pca_file_path, index=False)