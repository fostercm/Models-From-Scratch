import numpy as np
np.seterr(all='ignore')
from .linear_regression_base import LinearRegressionBase

class LinearRegressionPython(LinearRegressionBase):
    """
    A simple implementation of Linear Regression using Ordinary Least Squares (OLS).

    This class implements linear regression using the OLS method to compute the model parameters.
    It inherits from the LinearRegressionBase class and provides the basic functionality for
    fitting the model, making predictions, and calculating the cost (Mean Squared Error).
    """
    
    def __init__(self) -> None:
        """
        Initialize the LinearRegressionPython model.

        This method initializes the model by calling the parent class constructor
        and setting up the necessary model parameters.
        """
        super().__init__()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the Linear Regression model to the training data using Ordinary Least Squares (OLS).

        This method validates the input data and computes the model parameters (beta)
        using the OLS formula: β = (X^T * X)^(-1) * X^T * Y.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            Y (np.ndarray): Target matrix of shape (n_samples, n_targets).

        Raises:
            ValueError: If the dimensions of X and Y are not compatible.
        """
        # Validate the input arrays and pad X with ones
        X, Y = super().fit(X, Y)
        
        # Evaluate the model parameters using OLS
        gram_matrix = X.T @ X
        # self.params['beta'] = np.linalg.pinv(gram_matrix) @ X.T @ Y
        if np.isclose(np.linalg.det(gram_matrix), 0.0):
            self.params['beta'] = np.linalg.pinv(gram_matrix) @ X.T @ Y
        else:
            self.params['beta'] = np.linalg.inv(X.T @ X) @ X.T @ Y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the learned model.

        This method makes predictions by computing the dot product of the feature matrix (X)
        and the learned model parameters (beta).

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values of shape (n_samples, n_targets).

        Raises:
            ValueError: If the model is not fitted before calling predict.
        """
        # Validate the input array and pad X with ones
        X = super().predict(X)
        
        # Return the predicted values
        return X @ self.params['beta']

    def cost(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the Mean Squared Error (MSE) cost between the predicted and true target values.

        This method calculates the MSE using the formula: 
        MSE = 0.5 * (1/n_samples) * ||Y_pred - Y||^2.

        Args:
            Y_pred (np.ndarray): Predicted target values of shape (n_samples, n_targets).
            Y (np.ndarray): True target values of shape (n_samples, n_targets).

        Returns:
            float: The Mean Squared Error (MSE) between the predicted and true values.

        Raises:
            ValueError: If the dimensions of Y_pred and Y do not match.
        """
        Y_pred, Y = super().cost(Y_pred, Y)
        SCALE_FACTOR = 0.5 / Y.shape[0]
        return SCALE_FACTOR * np.linalg.norm(Y_pred - Y)**2