# Regularization Techniques

This repository contains my implementations and experiments with **Regularization methods** in Machine Learning.  
Regularization helps to **reduce overfitting** by penalizing large coefficients and improving generalization on unseen data.

---

## 📘 Ridge Regression (L2 Regularization)
###  1. Simple Linear Ridge Regression
- Implemented using the closed-form solution.  
- Works on a single feature dataset.  
- Shows how ridge regression prevents overfitting by shrinking coefficients.  


Ridge Regression adds an **L2 penalty** (squared magnitude of coefficients) to the loss function: 

<img src = "https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}L=\sum(y_{i}-\hat{y_{i}})^{2}&plus;\lambda(m)^{2}">

#### Coefficients m and b for 2D data :
<img src = "https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}&space;m=\frac{\sum((y_{i}-y_{mean})(x_{i}-x_{mean}))}{\sum(x_{i}-x_{mean})^{2}&plus;\lambda}">

<img src= "https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}b=\bar{y}-m\bar{x}">

### 🔹 2. Multiple Linear Ridge Regression
- Implemented using the **OLS closed-form solution** with Ridge penalty.  
- Will soon add **Gradient Descent** solution in the coming days.  

#### 📊 Steps Explanation:
To understand the derivation and intuition, check the diagrams below:

![Step 1](https://drive.google.com/uc?id=1M-zppzw9IO-rtkTWJyb_y_JXJsoQpgw8)
![Step 2](https://drive.google.com/uc?id=1LvT6rOYXrTu-klzX29CjD6ammtKZUh_p)


*(These images explain the mathematical derivation of Multiple Ridge Regression)*
#### Coefficients m and b for ND data:

here the only difference in the OLS Solution of linear regression and ridge regression is that Lambda is added into denominator because while differenciating the loss function with penalty term <strong>λ · m²</stromg>

- **Why use it?**
  - Reduces model complexity
  - Shrinks coefficients but does not eliminate them
  - Works well when we have multicollinearity

- **Hyperparameter**:  
  - `alpha` → Controls strength of penalty  
    - High `alpha` → more shrinkage, simpler model  
    - Low `alpha` → behaves like Linear Regression  

---
## 📂 Upcoming Additions
- Gradient Descent implementation for Ridge Regression  
- Lasso Regression  
- Elastic Net Regression  

---
## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/Pruthil-2910/ML-From-Scratch.git
   cd ML-From-Scratch
