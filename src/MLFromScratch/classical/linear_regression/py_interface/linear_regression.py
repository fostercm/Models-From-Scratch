from ...base.model.linear_model_base import LinearModelBase
from ...base.mixin.regression_mixin import RegressionMixin
import numpy as np
from typing import Literal

np.seterr(all="ignore")


class LinearRegression(LinearModelBase, RegressionMixin):
    """
    Linear Regression model using Ordinary Least Squares (OLS).

    This class implements a linear regression model with support for multiple backends
    ('Python', 'C', or 'CUDA'). It provides methods for training (fitting) the model
    and making predictions.

    Attributes:
        params (dict): A dictionary storing model parameters, including the fitted coefficients (`beta`)
                       and a flag (`fitted`) indicating if the model has been trained.
    """
    
    def __init__(self, language: Literal["Python", "C", "CUDA"] = "Python") -> None:
        """
        Initializes the Linear Regression model.

        Args:
            language (Literal["Python", "C", "CUDA"], optional): The implementation language. Default is 'Python'.

        Raises:
            ValueError: If the specified language is not valid.
        """
        # Validate the language parameter
        self._validate_language(language)

        # Initialize the parameters
        super().__init__(language=language)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the linear regression model using Ordinary Least Squares (OLS).

        Args:
            X (np.ndarray): The input feature matrix of shape (n_samples, n_features).
            Y (np.ndarray): The target values of shape (n_samples,).

        Raises:
            ValueError: If the dimensions of `X` and `Y` are incompatible.

        Notes:
            - If the Gram matrix (X^T X) is singular, the pseudoinverse is used to compute `beta`.
            - Otherwise, the standard matrix inversion method is used.
        """
        # Validate the input arrays
        X = self._validateInput(X)
        Y = self._validateTarget(Y)
        self._validateInputPair(X, Y)

        # Pad the feature matrix with ones for the bias term
        X = self._add_bias(X)

        # Evaluate the model parameters using OLS
        gram_matrix = X.T @ X
        if np.isclose(np.linalg.det(gram_matrix), 0.0):
            self.params["beta"] = np.linalg.pinv(gram_matrix) @ X.T @ Y
        else:
            self.params["beta"] = np.linalg.inv(X.T @ X) @ X.T @ Y

        # Set the fitted flag
        self.params["fitted"] = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained linear regression model.

        Args:
            X (np.ndarray): The input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted values of shape (n_samples,).

        Raises:
            ValueError: If the model has not been fitted before calling `predict`.
        """
        # Check if the model is fitted
        if self.params["fitted"] is False:
            raise ValueError("Model is not fitted")

        # Validate the input array
        X = self._validateInput(X)

        # Pad the feature matrix with ones for the bias term
        X = self._add_bias(X)

        # Compute the logits
        return self._compute_logits(X)
