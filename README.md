## Removing Outliers in Gene Expression Data:

### 1. Why Address Outliers?
- **Definition**: Outliers are extreme values.
- **Causes**:
  - Technical errors during data collection.
  - Genuine biological variations.
- **Impact**: While genuine variations offer insights, technical errors can distort analyses.

### 2. Why Remove Outliers Early?
- **Benefits of early removal**:
  - Ensures accurate statistical measures.
  - Provides clear visualizations.
  - Optimizes model performance.

### 3. Approach to Outlier Removal:

#### 3.1 IQR Method:

- **Method**: Employed the IQR to identify outliers, defined by values outside the range `[Q1 - 1.5 x IQR, Q3 + 1.5 x IQR]`.
  
- **Strategy**: Imputed outliers with the mean of non-outlier values, rather than deletion.

- **Observations**:
   - Visualizations showed pronounced changes in data distribution post-imputation.
   - This method concentrated data around the mean, which might overshadow genuine biological variations.
  
- **Considerations**: While effective in handling extreme values, it's essential to recognize the method's impact on data distribution. Alternatives or additional analyses might be necessary, especially when considering the biological significance of detected outliers.

- **Visualization**: Below is a boxplot showcasing the distribution of a randomly selected gene column before and after outlier imputation using the IQR method.
![Boxplot Before and After Imputation](./figures/boxplot_Fetal%20ctx%203%20astro_before_after_imputation.png)

#### 3.2 Modified Z-Score Method:

- **Method**: Utilized the Modified Z-Score to detect outliers. The formula for the Modified Z-Score is:
   $$
   M_i = 0.6745 \times \frac{(X_i - \text{Med})}{\text{MAD}}
   $$
  where $ X_i $ is the data point, $ \text{Med} $ is the median of the data, and $ \text{MAD} $ is the Median Absolute Deviation.

- **Strategy**: Imputed outliers, identified as those with a Modified Z-Score greater than 3.5, with the mean of non-outlier values.

- **Observations**:
   - **Outlier Detection**: The Modified Z-Score method identified more outliers compared to the IQR method. This could be because the Modified Z-Score is more sensitive, especially in datasets that don't strictly follow a normal distribution.
   - **Data Distribution**: Post-imputation visualizations show a more natural spread of data values, unlike the concentration around the mean observed with the IQR method.
- **Considerations**: While this method was effective in detecting and handling outliers, it's crucial to exercise caution. Given the sensitivity of the Modified Z-Score method, there's a risk of misclassifying genuine extreme values as outliers. It's essential to ensure that we are not discarding biologically significant variations.
- **Visualization**: Below is a boxplot showcasing the distribution of a randomly selected gene column before and after outlier imputation using the Modified Z-Score method.
![Boxplot Before and After Imputation](./figures/boxplot_45yo%20whole%20cortex_before_after_imputation_modified_zscore.png)

For the remainder of the project, we will utilize the IQR method for outlier handling.

