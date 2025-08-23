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

\[
\text{Loss} = \sum (y_i - \hat{y}_i)^2 + \alpha \sum \beta_j^2
\]

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
