from ...base.model.linear_model_base import LinearModelBase
from ...base.mixin.regression_mixin import RegressionMixin
import numpy as np
from typing import Literal

np.seterr(all="ignore")


class LinearRegression(LinearModelBase, RegressionMixin):
    def __init__(self, language: Literal["Python", "C", "CUDA"] = "Python") -> None:
        # Validate the language parameter
        self._validate_language(language)

        # Initialize the parameters
        super().__init__(language=language)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
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
        # Check if the model is fitted
        if self.params["fitted"] is False:
            raise ValueError("Model is not fitted")

        # Validate the input array
        X = self._validateInput(X)

        # Pad the feature matrix with ones for the bias term
        X = self._add_bias(X)

        # Compute the logits
        return self._compute_logits(X)
