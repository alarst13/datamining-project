import pandas as pd

# Load the dataset
data = pd.read_csv('datasets/dataset_preprocessed.csv')

# Check the distribution of species
species_distribution = data['Species'].value_counts()
print(species_distribution)
