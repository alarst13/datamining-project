import pandas as pd
import random
import matplotlib.pyplot as plt


# Function to replace outliers with the mean of non-outlier values
def replace_outliers_with_mean(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = (column < lower_bound) | (column > upper_bound)
    
    # Replace outliers with mean of non-outlier values
    column[outliers] = column[~outliers].mean()
    return column


def plot_boxplots_before_after(original_data, imputed_data):
    """
    Plot boxplots for a randomly selected column with outliers, showing data before and after outlier imputation.

    Parameters:
    - original_data (DataFrame): The original dataset with outliers.
    - imputed_data (DataFrame): The dataset after outlier imputation.
    """

    # Identify columns with outliers
    outlier_columns = []
    for col in original_data.columns[1:]:
        Q1 = original_data[col].quantile(0.25)
        Q3 = original_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        if original_data[(original_data[col] < lower_bound) | (original_data[col] > upper_bound)].shape[0] > 0:
            outlier_columns.append(col)

    # Randomly select one of these columns
    selected_column = random.choice(outlier_columns)
    
    # Calculate number of outliers for the selected column
    Q1 = original_data[selected_column].quantile(0.25)
    Q3 = original_data[selected_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    num_outliers = original_data[(original_data[selected_column] < lower_bound) | 
                                 (original_data[selected_column] > upper_bound)].shape[0]
    total_values = original_data[selected_column].shape[0]
    
    plt.figure(figsize=(10, 6))
    
    # Data for boxplots
    data_to_plot = [original_data[selected_column], imputed_data[selected_column]]
    
    plt.boxplot(data_to_plot, vert=True, patch_artist=True, labels=['Before Imputation', 'After Imputation'])
    
    # Adjust y-limits to focus on the distribution after imputation
    after_imputation_max = imputed_data[selected_column].max()
    plt.ylim(0, after_imputation_max + 0.5 * after_imputation_max)
    
    plt.title(f'Boxplot of {selected_column} Before and After Imputation')
    plt.ylabel('Expression Level')
    plt.grid(axis='y')
    
    # Adding the comment about the number of outliers
    plt.annotate(f'Outliers replaced: {num_outliers} out of {total_values}', 
                 xy=(0.5, 0.01), xycoords='axes fraction', ha='center', va='bottom', color='blue')
    
    # Save the figure
    plt.savefig(f'figures/boxplot_{selected_column}_before_after_imputation.png')


def main():
    # Load the dataset
    data = pd.read_csv('datasets/GSE142209/Human-Counts.csv')
    
    # Apply the function to each column (excluding the 'Gene' column)
    data_imputed = data.copy()
    data_imputed.iloc[:, 1:] = data_imputed.iloc[:, 1:].apply(replace_outliers_with_mean)

    # Save the imputed dataset
    data_imputed.to_csv('datasets/GSE142209/Human-Counts-imputed.csv', index=False)
    
    # # Load the imputed dataset
    # data_imputed = pd.read_csv('datasets/GSE73721_Human_and_mouse_table_imputed.csv')
    
    # # Plot boxplots for a randomly selected column with outliers, showing data before and after outlier imputation
    # plot_boxplots_before_after(data, data_imputed)


if __name__ == "__main__":
    main()