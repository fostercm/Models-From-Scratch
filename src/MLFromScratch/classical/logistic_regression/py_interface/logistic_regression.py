from typing import Literal

import numpy as np

from ....utils.python_utils.activation_functions import sigmoid, softmax
from ...base.mixin.classification_mixin import ClassificationMixin
from ...base.model.linear_model_base import LinearModelBase


class LogisticRegression(LinearModelBase, ClassificationMixin):
    def __init__(
        self,
        language: Literal["Python", "C", "CUDA"] = "Python",
        lr: float = 0.01,
        iterations: int = 10000,
        tolerance: float = 0.1,
    ) -> None:
        # Validate the language parameter
        self._validate_language(language)
        
        # Validate the learning rate parameter
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("Learning rate must be a positive number")
        
        # Validate the iterations parameter
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("Iterations must be a positive integer")
        
        # Validate the tolerance parameter
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise ValueError("Tolerance must be a positive number")

        # Initialize the parameters
        super().__init__(
            language=language, lr=lr, iterations=iterations, tolerance=tolerance
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Validate the input arrays
        X = super()._validateInput(X)
        y = super()._validateTarget(y)
        super()._validateInputPair(X, y)

        # Pad the feature matrix with ones for the bias term
        X = super()._add_bias(X)

        # Get the number of classes
        unique_y = np.unique(y)
        self.params["num_classes"] = len(unique_y)

        # Transform the target values to one-hot encoding
        if self.params["num_classes"] > 2:
            Y = super()._oneHotEncode(y, self.params["num_classes"])
        else:
            Y = y.reshape(-1, 1)

        # Initialize the model parameters
        if self.params["num_classes"] == 2:
            # Initialize beta for binary classification
            self.params["beta"] = np.zeros((X.shape[1], 1), dtype=np.float32)
        else:
            # Initialize beta for multi-class classification
            self.params["beta"] = np.zeros(
                (X.shape[1], self.params["num_classes"]), dtype=np.float32
            )

        # Gradient Descent
        for i in range(self.params["iterations"]):
            # Compute the predicted values
            Y_pred = self._predict(X)

            # Compute the gradient of the cost function
            gradient = X.T @ (Y_pred - Y) / X.shape[0]

            # Check for convergence
            if i % 100 == 0 and np.linalg.norm(gradient) < self.params["tolerance"]:
                break

            # Update the model parameters
            self.params["beta"] -= self.params["lr"] * gradient

        # Set the fitted flag
        self.params["fitted"] = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Validate the input array
        X = super()._validateInput(X)

        # Validate the model
        if self.params["fitted"] is False:
            raise ValueError("Model is not fitted")

        # Pad the feature matrix with ones for the bias term
        X = super()._add_bias(X)

        # Compute the predictions
        return self._predict(X)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        # Compute the logits
        logits = self._compute_logits(X)

        # Return the predicted values
        if self.params["num_classes"] == 2:
            return sigmoid(logits)
        return softmax(logits)
