import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import table

# Load and transpose the dataset to have genes as features
data_imputed = pd.read_csv('/home/ala22014/datamining/datamining-project/datasets/GSE73721_Human_and_mouse_table_imputed.csv')
data_transposed = data_imputed.transpose()
data_transposed.columns = data_transposed.iloc[0]  # Set gene names as column headers
data_transposed = data_transposed.drop(data_transposed.index[0])  # Remove the first row as it's now headers

# Mapping of sample names to tissue types
tissue_mapping = {
    # Tumor tissues
    '59yo tumor periphery astro': 'Tumor Astrocyte',
    '59yo tumor core astro': 'Tumor Astrocyte',
    '64yo tumor core astro': 'Tumor Astrocyte',
    '65yo tumor core astro': 'Tumor Astrocyte',

    # Hippocampus tissues
    '21yo Hippocampus astro': 'Hippocampus Astrocyte',
    '22yo Hippocampus astro': 'Hippocampus Astrocyte',
    '53yo A Hippocampus astro': 'Hippocampus Astrocyte',
    '53yo B Hippocampus astro': 'Hippocampus Astrocyte',

    # Cortex astrocytes
    'Fetal ctx 1 astro': 'Cortex Astrocyte',
    'Fetal ctx 2 astro': 'Cortex Astrocyte',
    'Fetal ctx 3 astro': 'Cortex Astrocyte',
    'Fetal ctx 4 astro': 'Cortex Astrocyte',
    'Fetal ctx 5 astro': 'Cortex Astrocyte',
    'Fetal ctx 6 astro': 'Cortex Astrocyte',
    '8yo ctx astro': 'Cortex Astrocyte',
    '13yo ctx astro': 'Cortex Astrocyte',
    '16yo ctx astro': 'Cortex Astrocyte',
    '21yo ctx astro': 'Cortex Astrocyte',
    '22yo ctx astro': 'Cortex Astrocyte',
    '35yo ctx astro': 'Cortex Astrocyte',
    '47yo ctx astro': 'Cortex Astrocyte',
    '51yo ctx astro': 'Cortex Astrocyte',
    '53yo ctx astro': 'Cortex Astrocyte',
    '60yo ctx astro': 'Cortex Astrocyte',
    '63yo ctx 1 astro': 'Cortex Astrocyte',
    '63yo ctx 2 astro': 'Cortex Astrocyte',

    # Neurons
    '25yo ctx neuron': 'Cortex Neuron',

    # Oligodendrocytes
    '22yo ctx oligo': 'Cortex Oligodendrocyte',
    '63yo ctx A oligo': 'Cortex Oligodendrocyte',
    '63yo ctx B oligo': 'Cortex Oligodendrocyte',
    '47yo ctx oligo': 'Cortex Oligodendrocyte',
    '63yo ctx 3 oligo': 'Cortex Oligodendrocyte',

    # Myeloid cells
    '45yo ctx myeloid': 'Cortex Myeloid',
    '51yo ctx myeloid': 'Cortex Myeloid',
    '63 yo ctx myeloid': 'Cortex Myeloid',

    # Endothelial cells
    '47 yo ctx endo': 'Cortex Endothelial',
    '13yo ctx endo': 'Cortex Endothelial',

    # Whole cortex
    '45yo whole cortex': 'Whole Cortex',
    '63yo whole cortex': 'Whole Cortex',
    '25yo whole cortex': 'Whole Cortex',
    '53yo whole cortex': 'Whole Cortex',

    # Mouse astrocytes
    'Mouse 1 month astro': 'Mouse Astrocyte',
    'Mouse 4 month astro': 'Mouse Astrocyte',
    'Mouse 7 month astro': 'Mouse Astrocyte',
    'Mouse 9 month astro': 'Mouse Astrocyte',
    'Mouse 12 month astro': 'Mouse Astrocyte',
    'Mouse 18 month astro': 'Mouse Astrocyte',
    'Mouse 24 month astro': 'Mouse Astrocyte',
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
output_file_path = '/home/ala22014/datamining/datamining-project/datasets/transformed_data.csv'
selected_data.to_csv(output_file_path, index=False)

# Copy the DataFrame to avoid SettingWithCopyWarning
selected_data_copy = selected_data.copy()

# Select a subset of columns for visualization
start_columns = selected_data_copy.columns.tolist()[:2]  # First 2 columns
end_columns = selected_data_copy.columns.tolist()[-2:]  # Last 2 columns

# Select the rows you want to display
top_rows = selected_data_copy.head(5).reset_index(drop=True)
bottom_rows = selected_data_copy.tail(5).reset_index(drop=True)

# Create an ellipsis row with the correct shape
ellipsis_row = pd.DataFrame([['...'] * (len(start_columns) + len(end_columns))], columns=start_columns + end_columns)

# Create a DataFrame to indicate omitted rows
omitted_rows = pd.DataFrame([['...'] * len(start_columns + end_columns)], columns=start_columns + end_columns)

# Combine the top, omitted placeholder, and bottom DataFrames
visual_df_rows = pd.concat([top_rows, omitted_rows, bottom_rows], ignore_index=True)

# Add the ellipsis column by reindexing with the new column set
visual_df = visual_df_rows.reindex(columns=start_columns + ['...'] + end_columns, fill_value='...')

fig, ax = plt.subplots(figsize=(12, 4))  # Adjust the size as needed
ax.axis('off')

# Create the table with the visual DataFrame
tbl = table(ax, visual_df, loc='center', cellLoc='center', colWidths=[0.1]*(len(start_columns) + 1 + len(end_columns)))

# Improve the visual layout
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.auto_set_column_width(col=list(range(len(visual_df.columns))))

# Add the dimensions of the original DataFrame as text below the table
ax.text(0.5, -0.05, f'DataFrame shape: {selected_data_copy.shape}', transform=ax.transAxes, ha='center')

# Save the figure
plt.savefig('/home/ala22014/datamining/datamining-project/figures/dataframe_visualization.png', bbox_inches='tight', dpi=300)