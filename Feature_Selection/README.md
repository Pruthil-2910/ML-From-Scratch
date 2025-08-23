## 📊 Feature Selection Cheatsheet  

| **Method Type** | **Method** | **Description** |
|-----------------|------------|-----------------|
| **Filter Methods** | Variance Threshold | Removes features whose variance doesn’t meet a set threshold. Good for eliminating constants or near constants. |
| | Correlation Coefficient | Finds correlation between feature pairs; remove highly correlated features to avoid redundancy. |
| | Chi-Square Test | Tests if there’s a significant association between two variables. Commonly for categorical data. |
| | Mutual Information | Measures dependency between variables, capturing both linear and non-linear relationships. |
| | ANOVA (Analysis of Variance) | Compares means of samples to test the impact of factors on a continuous variable. |
| **Wrapper Methods** | Recursive Feature Elimination (RFE) | Recursively removes features using model accuracy to rank importance. |
| | Sequential Feature Selection (SFS) | Adds/removes one feature at a time to find the optimal subset. |
| | Exhaustive Feature Selection | Brute-force search of all feature combinations to find the best subset; expensive for large datasets. |
| **Embedded Methods** | Lasso Regression | Performs variable selection and regularization for a simple, interpretable model. |
| | Ridge Regression | Minimizes complexity of regression model but does not select features. |
| | Elastic Net | Combines Lasso & Ridge penalties; effective with highly correlated features. |
| | Random Forest Importance | Uses impurity decrease to rank feature importance. |
