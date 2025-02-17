import numpy as np
from ...supervised_model import SupervisedModel


class LinearRegressionBase(SupervisedModel):
    """
    A base class for Linear Regression model.

    This class provides methods for fitting the linear regression model,
    predicting target values, and calculating the cost (Mean Squared Error).

    Attributes:
        params (dict): A dictionary storing the model parameters, including 'beta'.
    """

    def __init__(self) -> None:
        """
        Initialize the Linear Regression model parameters.

        Initializes the 'beta' parameter to None. This will be populated
        when the model is fitted to the training data.
        """
        self.params = {"beta": None}

    def fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fit the Linear Regression model to the training data.

        This method fits a linear regression model to the provided training data
        (X and Y) using the normal equation approach. It pads the feature matrix
        with a bias term (a column of ones) and validates the input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features), where
                            n_samples is the number of training samples and
                            n_features is the number of features per sample.
            Y (np.ndarray): Target matrix of shape (n_samples, n_targets), where
                            n_samples is the number of training samples and
                            n_targets is the number of targets per sample.

        Returns:
            np.ndarray: The modified feature matrix (with the bias term added) and the target values (Y).

        Raises:
            ValueError: If the number of samples in X does not match the number of samples in Y.
        """
        # Validate the input arrays
        X, Y = super()._validate_input(X, Y)

        # Pad the feature matrix with ones for the bias term
        X = np.hstack((np.ones((X.shape[0], 1)), X), dtype=np.float32)

        return X, Y

        # To be implemented in derived classes

    def predict(self, X: np.ndarray, pad: bool = True) -> np.ndarray:
        """
        Predict target values using the learned model.

        This method predicts the target values based on the input features X using
        the learned model parameters. It ensures the model is fitted and that the
        dimensions of X match the model parameters.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features), where
                            n_samples is the number of test samples and
                            n_features is the number of features per sample.

        Returns:
            np.ndarray: Predicted target values of shape (n_samples, n_targets).

        Raises:
            ValueError: If the model is not fitted or if the dimensions of X do not match
                        the expected shape.
        """
        # Validate the input array
        X, _ = super()._validate_input(X)

        # Check if the model is fitted
        if self.params["beta"] is None:
            raise ValueError("Model is not fitted")

        if pad:
            # Pad the feature matrix with ones for the bias term
            X = np.hstack((np.ones((X.shape[0], 1)), X), dtype=np.float32)

        # Check if the input dimensions match the model parameters
        if X.shape[1] != self.params["beta"].shape[0]:
            raise ValueError(
                "The number of columns in X must be equal to the number of features in the model"
            )

        return X

        # To be implemented in derived classes

    def cost(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the Mean Squared Error (MSE) cost.

        This method calculates the Mean Squared Error between the predicted values
        (Y_pred) and the true target values (Y). This is a common cost function
        used in regression tasks.

        Args:
            Y_pred (np.ndarray): Predicted target values of shape (n_samples, n_targets).
            Y (np.ndarray): True target values of shape (n_samples, n_targets).

        Returns:
            float: The Mean Squared Error between Y_pred and Y.

        Raises:
            ValueError: If the dimensions of Y_pred and Y do not match.
        """
        # Validate the input arrays
        Y_pred, Y = super()._validate_input(Y_pred, Y)

        # Check if the dimensions of Y_pred and Y match
        if Y_pred.shape != Y.shape:
            raise ValueError(
                f"Dimensions of Y_pred {Y_pred.shape} and Y {Y.shape} do not match"
            )

        return Y_pred, Y

        # To be implemented in derived classes
