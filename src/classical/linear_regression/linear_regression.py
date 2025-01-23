import numpy as np
from classical.classical_model import ClassicalModel

class LinearRegression(ClassicalModel):
    """
    A simple implementation of Linear Regression using Ordinary Least Squares (OLS)
    """
    def __init__(self):
        """Initialize the model parameters"""
        self.params = None
    
    def _validate_input(self, array: np.ndarray) -> None:
        """
        Validate the input numpy array

        Args:
            array (np.ndarray): Input numpy array
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if len(array.shape) != 2:
            raise ValueError("Input must be a 2D array")
        
        if array.size == 0:
            raise ValueError("Input must not be empty")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the Linear Regression model to the training data

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            Y (np.ndarray): Target matrix of shape (n_samples, 1)
        """
        # Validate the input arrays
        self._validate_input(X)
        self._validate_input(Y)
        
        # Check if the number of rows in X and Y are equal
        if X.shape[0] != Y.shape[0]:
            raise ValueError("The number of rows in X and Y must be equal")
        
        # Evaluate the model parameters using OLS
        self.params = {'beta': np.linalg.pinv(X.T @ X) @ X.T @ Y}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the learned model

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values of shape (n_samples, n_targets)
        """
        if self.params is None:
            raise ValueError("Fit the model before making predictions")
        
        if X.shape[1] != self.params['beta'].shape[0]:
            raise ValueError("The number of columns in X must be equal to the number of features in the model")
        
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
        self._validate_input(Y_pred)
        self._validate_input(Y)
        SCALE_FACTOR = 0.5 / Y.shape[0]
        return SCALE_FACTOR * np.linalg.norm(Y_pred - Y)**2

    def get_params(self) -> dict:
        """
        Retrieve model parameters

        Returns:
            dict: Model parameters {'beta': np.ndarray}
        """
        return self.params

    