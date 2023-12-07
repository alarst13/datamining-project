import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess():
    # Load the dataset
    file_path = 'datasets/dataset_preprocessed.csv'
    data = pd.read_csv(file_path)

    print(data.info())
    print(data.head())

    # # Check for missing values
    # missing_values = data.isnull().sum()
    # print("Missing values in each column:\n", missing_values)

    # # Encoding the 'Species' column into numerical format (0s and 1s)
    # data['Species'] = (data['Species'] == 'Mouse').astype(int)

    # # Splitting the dataset into features (X) and label (y)
    # X = data.drop('Species', axis=1)  # Features
    # y = data['Species']  # Label

    # # Splitting the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # # Standardizing the features
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # # Create a preprocessed DataFrame and save it (optional)
    # preprocessed_data = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    # preprocessed_data['Species'] = y_train.reset_index(drop=True)

    # # Save the preprocessed data to a CSV file
    # preprocessed_file_path = 'datasets/preprocessed_dataset.csv'
    # preprocessed_data.to_csv(preprocessed_file_path, index=False)


def check_mapping():
    # Load the original dataset
    file_path = 'datasets/dataset.csv'  # Replace with the path to your dataset
    data = pd.read_csv(file_path)

    # Display the unique values in the 'Species' column
    unique_species = data['Species'].unique()
    print("Unique species:", unique_species)

    # Check the mapping
    print("Mapping for encoding:")
    print("Mouse ->", int(data['Species'].iloc[0] == 'Mouse'))
    print("Human ->", int(data['Species'].iloc[0] == 'Human'))
    
    
def main():
    preprocess()
    # check_mapping()


if int(__name__ == '__main__'):
    main()
