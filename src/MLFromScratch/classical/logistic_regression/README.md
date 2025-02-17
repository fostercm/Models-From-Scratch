# Logistic Regression  

## Overview  

- Logistic regression is a **_supervised_** statistical model that attempts to map a set of input variables to an output class
- It is one of the most basic statistical models, built off of linear regression
- It is nearly fully linear, but a nonlinear sigmoid/softmax applied to linear logits
- Similar to linear regression, its simplicity and interpretability are its greatest strengths
- However, logistic regression has slightly fewer assumptions for use
- For my implementation, I used gradient descent rather than a numerical solver

## Mathematical Derivation

### Definition

Logistic regression is a statistical model used to predict the probability of a binary outcome based on one or more predictor variables.
It models the relationship between the probability of a binary outcome $`\Huge Y \in \{0, 1\}`$ and the independent variables $\Huge X$.

$$
\Huge Y = \sigma(X\beta)
$$

where:

- $\Huge Y \in \mathbb{R}^{n \times 1}$ is the vector of observed values (with n samples and 1 output)
- $\Huge X \in \mathbb{R}^{n \times d}$ is the matrix of input features (with n samples and d-1 independent variables with a column of 1s to represent a bias)
- $\Huge \beta \in \mathbb{R}^{d \times 1}$ is the vector of regression coefficients (d independent variables/bias and 1 output)
- $\Huge \sigma$ is the sigmoid function $\Huge \sigma(X) = \frac{1}{1 + e^{-X}}$

### Multiclass definition

Logistic regression is natively for binary classification, but can be modified for multiclass with a few differences:

$$
\Huge Y = S(X\beta)
$$

- $\Huge Y \in \mathbb{R}^{n \times c}$ a one-hot matrix (row-wise) of observed values (with n samples and c classes)
- $\Huge \beta \in \mathbb{R}^{d \times c}$ is the vector of regression coefficients (d independent variables/bias and c classes)
- $\Huge S$ is the softmax function $\Huge S(X_i) = \frac{e^{x_i}}{\sum_{j=1}^{c}e^{x_j}}$

### Loss function

To estimate $\Huge \beta$, we minimize the cross entropy loss:

- **Binary:**

$$
\Huge \mathcal{L}(\beta) = -\frac{1}{n} \sum_{i=1}^{n} (\log_{ }Y_i - (X\beta)_{i})^2
$$

- **Multiclass:**
or, in matrix form where $\\Vert \cdot \\Vert_F$ denotes the frobenius norm:

$$
\Huge \mathcal{L}(\beta) = \frac{1}{n}\\Vert Y - X \beta \\Vert_F^2
$$

This is referred to as mean squared error, or MSE.

### Closed-Form Solution

Miraculously, since linear regression is an entirely linear model, there exists a closed-form solution for $$\Huge \beta$$ which I will derive below:  

$$
\Huge \mathcal{L}(\beta) = \frac{1}{n}\\Vert Y - X \beta \\Vert_F^2  
$$

Without getting too deep into the linear algebra:

$$
\Huge \nabla_\beta\mathcal{L}(\beta) = \frac{2}{n}X^T(Y - X\beta)
$$

Now we set equal to 0 to minimize the loss:

$$
\Huge 0 = \frac{2}{n}X^T(Y - X\beta)
$$

$$
\Huge 0 = X^T(Y - X\beta)
$$

$$
\Huge 0 = X^TY - X^TX\beta
$$

$$
\Huge X^TX\beta = X^TY
$$

$$
\Huge \beta = (X^TX)^{-1}X^TY
$$

It is worth noting that if $\Huge X^TX$ is not invertible, then the Moore-Penrose pseudoinverse should be used.

### Pseudoinverse

For matrices that aren't invertible (correlated columns being the most likely reason here), the Moore-Penrose pseudoinverse is applicable, defined as:

$$
\Huge A^+ = V\Sigma^+U^T
$$

Where $\Huge V$, $\Huge \Sigma$, and $\Huge U^T$ are computed by taking the singular value decomposition (SVD) of A:

$$
\Huge A = U \Sigma V^T
$$

To get $\Huge \Sigma^+$ simply tranpose $\Huge \Sigma$ and invert the singular values.

### Predictions

Once we have $\Huge \hat{\beta}$, predictions for new data $\Huge X_{\text{new}}$ can be made as:

$$
\Huge \hat{Y} = X_{\text{new}} \hat{\beta}
$$

## Assumptions  

Linear regression has a few notable assumptions that should be checked before use:

- **Linearity of relationships:** A linear relationship should exist (Lack-of-Fit Test)
- **Normally Distributed Variables:** Inputs should be multivariate normal (Q-Q plot)
- **Independence of errors:** Model errors should not correlate (Durbin-Watson statistic)
- **Homoscedasticity:** Model errors should be constant (Levene’s test)
- **No multicollinearity:** Input variables should not correlate (r statistic)

## Strengths and Weaknesses  

### Strengths  

- Simple and interpretable  
- Computationally efficient  
- Works well with small datasets

### Weaknesses  

- Sensitive to outliers  
- Assumes a linear relationship  
- Can suffer from overfitting with high-dimensional data  

## Implementation Details  

Some specifics to keep in mind for this implementation:

- Floats were used as the data type for a balance between speed and precision (float in C/C++/CUDA, np.float32 in Python)
- I only stepped one layer of abstraction below linear regression, I used C libraries like CBLAS and LAPACKE and CUDA libraries like cuBLAS and cuSOLVER
- For modularity, host-side data is handled in C++ and device-side data is handled in CUDA alongisde kernels I write
- I used ctypes to create wrapper classes around the C/C++/CUDA functions
- Type and value checking is handled in the base classes

## Benchmarking & Performance  

<img src="../../../../benchmarks/linear_regression/execution_time.png" width="600">
<img src="../../../../benchmarks/linear_regression/memory_usage.png" width="600">

## Thoughts & Ideas  

- I found it fascinating that linear regression has a closed-form solution and I'm happy I was able to derive it.
- It is worth noting that the C class is clearly slower than the others at scale, I found this to be because of SVD computation and lack of multiprocessing.
- Coding this up has been a good exercise in created shared C/C++/CUDA libraries, which is likely what I spent the most time on.
- I will code up logistic regression next, it should work well as a child class of this.
