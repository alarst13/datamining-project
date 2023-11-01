import pandas as pd
import random
import matplotlib.pyplot as plt


def replace_outliers_with_mean_modified_zscore(column):
    """
    Replace outliers in a column using the Modified Z-Score method.
    """
    median = column.median()
    mad = abs(column - median).median()
    modified_z_score = 0.6745 * (column - median) / mad

    # Typically, a threshold of 3.5 for the modified z-score is considered to identify outliers
    outliers = modified_z_score > 3.5

    # Replace outliers with the mean of non-outlier values
    column[outliers] = column[~outliers].mean()

    return column


def plot_boxplots_before_after_modified_zscore(original_data, imputed_data):
    """
    Plot boxplots for a randomly selected column with outliers, showing data before and after outlier imputation using Modified Z-Score.

    Parameters:
    - original_data (DataFrame): The original dataset with outliers.
    - imputed_data (DataFrame): The dataset after outlier imputation.
    """

    # Identify columns with outliers using Modified Z-Score
    outlier_columns = []
    for col in original_data.columns[1:]:
        median = original_data[col].median()
        mad = abs(original_data[col] - median).median()
        modified_z_score = 0.6745 * (original_data[col] - median) / mad
        
        # Outliers are those with a Modified Z-Score > 3.5
        if original_data[modified_z_score > 3.5].shape[0] > 0:
            outlier_columns.append(col)

    # Randomly select one of these columns
    selected_column = random.choice(outlier_columns)
    
    # Calculate number of outliers for the selected column using Modified Z-Score
    median = original_data[selected_column].median()
    mad = abs(original_data[selected_column] - median).median()
    modified_z_score = 0.6745 * (original_data[selected_column] - median) / mad
    num_outliers = original_data[modified_z_score > 3.5].shape[0]
    total_values = original_data[selected_column].shape[0]
    
    plt.figure(figsize=(10, 6))
    
    # Data for boxplots
    data_to_plot = [original_data[selected_column], imputed_data[selected_column]]
    
    plt.boxplot(data_to_plot, vert=True, patch_artist=True, labels=['Before Imputation', 'After Imputation'])
    
    # Adjust y-limits to focus on the distribution after imputation
    after_imputation_max = imputed_data[selected_column].max()
    plt.ylim(0, 14)
    
    plt.title(f'Boxplot of {selected_column} Before and After Imputation using Modified Z-Score')
    plt.ylabel('Expression Level')
    plt.grid(axis='y')
    
    # Adding the comment about the number of outliers
    plt.annotate(f'Outliers replaced: {num_outliers} out of {total_values}', 
                 xy=(0.5, 0.01), xycoords='axes fraction', ha='center', va='bottom', color='blue')
    
    # Save the figure
    plt.savefig(f'figures/boxplot_{selected_column}_before_after_imputation_modified_zscore.png')


def main():
    # Load the dataset
    data = pd.read_csv('datasets/GSE73721_Human_and_mouse_table.csv')
    
    # Apply the function to each column (excluding the 'Gene' column)
    data_imputed = data.copy()
    data_imputed.iloc[:, 1:] = data_imputed.iloc[:, 1:].apply(replace_outliers_with_mean_modified_zscore)

    # Save the imputed dataset
    data_imputed.to_csv('datasets/GSE73721_Human_and_mouse_table_imputed.csv', index=False)
    
    # Load the imputed dataset
    data_imputed = pd.read_csv('datasets/GSE73721_Human_and_mouse_table_imputed.csv')
    
    # Plot boxplots for a randomly selected column with outliers, showing data before and after outlier imputation
    plot_boxplots_before_after_modified_zscore(data, data_imputed)


if __name__ == "__main__":
    main()
