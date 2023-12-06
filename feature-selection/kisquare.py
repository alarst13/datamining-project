import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import table

# Load and transpose the dataset to have genes as features
data_imputed = pd.read_csv('datasets/GSE142209/Human-Counts-imputed.csv')
data_transposed = data_imputed.transpose()
data_transposed.columns = data_transposed.iloc[0]  # Set gene names as column headers
data_transposed = data_transposed.drop(data_transposed.index[0])  # Remove the first row as it's now headers

# Mapping of sample names to human and mouse brain categories
tissue_mapping = {
    'Microvessels 1': 'Human Brain',
    'Microvessels 2': 'Human Brain',
    'Microvessels 3': 'Human Brain',
    'Whole brain 1': 'Human Brain',
    'Whole brain 2': 'Human Brain',
    'Whole brain 3': 'Human Brain',
}


# Clean sample names and map to tissue types
data_transposed.index = data_transposed.index.str.strip()
data_transposed['TissueType'] = data_transposed.index.to_series().map(tissue_mapping)

# Separate features (X) from labels (y)
X = data_transposed.drop(['TissueType'], axis=1)
y = data_transposed['TissueType']

# Adjust features to ensure non-negativity for chi-squared test
X += X.min().min()

# Feature selection with chi-squared test
k = 20
chi2_selector = SelectKBest(chi2, k=k)
X_kbest_features = chi2_selector.fit_transform(X, y)

# Identify and retain selected features
selected_features_mask = chi2_selector.get_support()
selected_features = X.columns[selected_features_mask]
selected_data = data_transposed[selected_features.tolist() + ['TissueType']]

# Save selected features and labels to a new file
output_file_path = 'datasets/GSE142209/Human-Counts-transformed.csv'
selected_data.to_csv(output_file_path, index=False)

# # Copy the DataFrame to avoid SettingWithCopyWarning
# selected_data_copy = selected_data.copy()

# # Select a subset of columns for visualization
# start_columns = selected_data_copy.columns.tolist()[:2]  # First 2 columns
# end_columns = selected_data_copy.columns.tolist()[-2:]  # Last 2 columns

# # Select the rows you want to display
# top_rows = selected_data_copy.head(5).reset_index(drop=True)
# bottom_rows = selected_data_copy.tail(5).reset_index(drop=True)

# # Create an ellipsis row with the correct shape
# ellipsis_row = pd.DataFrame([['...'] * (len(start_columns) + len(end_columns))], columns=start_columns + end_columns)

# # Create a DataFrame to indicate omitted rows
# omitted_rows = pd.DataFrame([['...'] * len(start_columns + end_columns)], columns=start_columns + end_columns)

# # Combine the top, omitted placeholder, and bottom DataFrames
# visual_df_rows = pd.concat([top_rows, omitted_rows, bottom_rows], ignore_index=True)

# # Add the ellipsis column by reindexing with the new column set
# visual_df = visual_df_rows.reindex(columns=start_columns + ['...'] + end_columns, fill_value='...')

# fig, ax = plt.subplots(figsize=(12, 4))  # Adjust the size as needed
# ax.axis('off')

# # Create the table with the visual DataFrame
# tbl = table(ax, visual_df, loc='center', cellLoc='center', colWidths=[0.1]*(len(start_columns) + 1 + len(end_columns)))

# # Improve the visual layout
# tbl.auto_set_font_size(False)
# tbl.set_fontsize(10)
# tbl.auto_set_column_width(col=list(range(len(visual_df.columns))))

# # Add the dimensions of the original DataFrame as text below the table
# ax.text(0.5, -0.05, f'DataFrame shape: {selected_data_copy.shape}', transform=ax.transAxes, ha='center')

# # Save the figure
# plt.savefig('/home/ala22014/datamining/datamining-project/figures/dataframe_visualization.png', bbox_inches='tight', dpi=300)