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

