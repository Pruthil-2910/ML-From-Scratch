# 📈 Linear Regression (From Scratch)

This folder contains two Python implementations of Linear Regression built **completely from scratch** to deeply understand the math behind them.

- `simple_linear_regression.py` → Simple Linear Regression (1 feature)
- `multiple_linear_regression.py` → Multiple Linear Regression (multiple features)

---

## 📐 **Mathematical Explanation (with handwritten notes)**

To show how the equations and derivations come from first principles, I’ve attached my own handwritten notes:

| Topic                            | Image                                                                 |
|----------------------------------|:----------------------------------------------------------------------:|
| <h2> Multiple Linear Regression (matrix form, normal equation derivation)</h2> |<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*zz8elaqpcp5Yt6ZcdUcagA.jpeg" width="400"> |
| <h2> Continue: Matrix form & error minimization </h2>|<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*CBlLyDn4oAMPVkXUfMM7zw.jpeg" width="400"> |
| <h2> Matrix differentiation & finding coefficients </h2>|<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*SnsVh-Buoe-XoJIhX5MByw.jpeg" width="400"> |
| <h2> Simple Linear Regression derivation (calculating slope & intercept)</h2>|<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*l1v2dZsaPq186Bvsn-VAXw.jpeg" width="400"> |

These pages explain:
- How to get from the basic hypothesis to **matrix form**: $\hat{y} = X \beta$
- How to derive the **Normal Equation**: $\beta = (X^T X)^{-1} X^T y$
- For simple LR: how to find $m$ and $b$ by minimizing squared error.

---

## 💻 **Code Explanation**

### ✅ `simple_linear_regression.py`
- Implements Linear Regression for a single input feature.
- Calculates mean of X and y.
- Finds slope `m` and intercept `c` manually.
- Predicts values and shows $R^2$ score.

### ✅ `multiple_linear_regression.py`
- Works for multiple features using the **Normal Equation**.
- Adds bias term to X.
- Uses numpy to compute: $\beta = (X^T X)^{-1} X^T y$
- Predicts values and shows $R^2$ score.

---
# ⚡ Linear Regression using Gradient Descent

A minimal implementation of Linear Regression trained via **Gradient Descent**, built from scratch using only NumPy.
Includes three classes:
| Variant                    | Class name                | Method                                     |
|--------------------------:|--------------------------:|-------------------------------------------:|
| Batch Gradient Descent    | `BatchGDRegressor`        | Updates on full dataset every epoch       |
| Stochastic Gradient Descent | `StochasticGDRegressor` | Updates on each single sample             |
| Mini-Batch Gradient Descent | `MiniBatchGDRegressor`  | Updates on small random batches           |


---

## ✏️ **Why three variants?**
Gradient Descent isn't just one algorithm:
- **Batch GD**: stable, but slower on large data
- **Stochastic GD**: faster & can escape local minima, but noisier
- **Mini-Batch GD**: balance of speed and stability

---
## Polynomial Regression  
<p>
Implements polynomial regression from scratch, extending linear regression to capture non-linear relationships in data.  
Includes flexible degree selection for fitting complex curves while maintaining full control over the algorithm’s inner workings.
</p>


## ⚙️ **How to use**
```python
from linear_regression_gd import GDRegressor

## ⚙️ **Run**

```bash
python simple_linear_regression.py
python multiple_linear_regression.py
