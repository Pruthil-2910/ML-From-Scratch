# Regularization Techniques

This repository contains my implementations and experiments with **Regularization methods** in Machine Learning.  
Regularization helps to **reduce overfitting** by penalizing large coefficients and improving generalization on unseen data.

---

## 📂 Implementations
- ✅ Ridge Regression (L2 Regularization) – *Completed*
- ⬜ Lasso Regression (L1 Regularization) – *Upcoming*
- ⬜ Elastic Net (Combination of L1 & L2) – *Upcoming*

---

## 📘 Ridge Regression (L2 Regularization)

Ridge Regression adds an **L2 penalty** (squared magnitude of coefficients) to the loss function:

<img src = "https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}&space;m=\frac{\sum((y_{i}-y_{mean})(x_{i}-x_{mean}))}{\sum(x_{i}-x_{mean})^{2}&plus;\lambda}">

here the only difference in the OLS Solution of linear regression and ridge regression is that Lambda is added into denominator because while differenciating the loss function with penalty term * lambda ( m ^ 2)
- **Why use it?**
  - Reduces model complexity
  - Shrinks coefficients but does not eliminate them
  - Works well when we have multicollinearity

- **Hyperparameter**:  
  - `alpha` → Controls strength of penalty  
    - High `alpha` → more shrinkage, simpler model  
    - Low `alpha` → behaves like Linear Regression  

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
