import numpy as np
from .linear_regression_base import LinearRegressionBase

class LinearRegressionPython(LinearRegressionBase):
    """
    A simple implementation of Linear Regression using Ordinary Least Squares (OLS)
    """
    
    def __init__(self) -> None:
        """Initialize the model parameters"""
        super().__init__()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the Linear Regression model to the training data

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            Y (np.ndarray): Target matrix of shape (n_samples, n_targets)
        """
        # Validate the input arrays and pad X with ones
        X, Y = super().fit(X, Y)
        
        # Evaluate the model parameters using OLS
        self.params['beta'] = np.linalg.pinv(X.T @ X) @ X.T @ Y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the learned model

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values of shape (n_samples, n_targets)
        """
        # Validate the input array and pad X with ones
        X = super().predict(X)
        
        # Return the predicted values
        return X @ self.params['beta']

    def cost(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the Mean Squared Error cost

        Args:
            Y_pred (np.ndarray): Predicted target values
            Y (np.ndarray): True target values

        Returns:
            float: Mean Squared Error
        """
        Y_pred, Y = super().cost(Y_pred, Y)
        SCALE_FACTOR = 0.5 / Y.shape[0]
        return SCALE_FACTOR * np.linalg.norm(Y_pred - Y)**2