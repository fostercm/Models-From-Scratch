from .unsupervised_model import UnsupervisedModel
import numpy as np


class SupervisedModel(UnsupervisedModel):
    """
    A base class for classical machine learning models.

    This class provides common functionality for validating input data,
    storing model parameters, and providing placeholder methods for model
    fitting, prediction, and cost calculation. Subclasses should implement
    the fit, predict, and cost methods to define the specific behavior of
    different classical models (e.g., Linear Regression, Logistic Regression).

    Attributes:
        params (dict): A dictionary that holds the parameters of the model,
                       such as coefficients or weights after fitting.

    Methods:
        fit(X, Y): Fit the model to the training data (to be implemented in subclass).
        predict(X): Predict the target variable for the given data (to be implemented in subclass).
        cost(Y_pred, Y): Calculate the cost (e.g., loss function) for the model's predictions.
        get_params(): Get the parameters of the model.
        load_params(params): Load the model parameters.
    """

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit the model to the training data (to be implemented in subclass)."""
        pass

    def cost(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """Calculate the cost of the model (to be implemented in subclass)."""
        pass

    def get_params(self) -> dict:
        """Get the parameters of the model"""
        return self.params

    def load_params(self, params: dict) -> None:
        """Load the parameters of the model"""
        self.params = params
        for key, value in self.params.items():
            self.params[key] = np.array(value, dtype=np.float32)
