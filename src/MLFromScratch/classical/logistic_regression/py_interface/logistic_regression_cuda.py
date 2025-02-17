from .logistic_regression_base import LogisticRegressionBase
import ctypes
import numpy as np
import os


class LogisticRegressionCUDA(LogisticRegressionBase):
    """
    Logistic Regression model using a C implementation for performance optimization.

    This class extends the LogisticRegressionBase class and utilizes a C-based
    library for model fitting, prediction, and cost computation to improve
    performance over pure Python implementations.

    Attributes:
        lib (ctypes.CDLL): The C shared library for logistic regression operations.
    """

    def __init__(self):
        """
        Initialize the LogisticRegressionC model.

        This method loads the C shared library and sets up the argument types
        for the C functions used for fitting, predicting, and calculating cost.
        """
        super().__init__()

        # Load the C library
        package_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(
            package_dir, "../../../lib/liblogistic_regression_cuda.so"
        )
        lib_path = os.path.normpath(lib_path)
        self.lib = ctypes.CDLL(lib_path)

        # Define the types of the arguments
        self.lib.fit.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
        ]

        self.lib.predict.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.lib.cost.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cost.restype = ctypes.c_float

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the Logistic Regression model to the training data using the C implementation.

        This method validates the input data, flattens the arrays, and calls the
        C function to fit the model. The learned parameters (beta) are then
        reshaped and stored in the model's parameters.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            Y (np.ndarray): Target matrix of shape (n_samples, n_targets).

        Raises:
            ValueError: If the dimensions of X and Y do not match.
        """
        X, Y = super().fit(X, Y)
        num_classes = (
            self.params["num_classes"] if self.params["num_classes"] > 2 else 1
        )

        # Get the dimensions of the input and output
        num_samples, num_input_features = X.shape

        # Flatten arrays
        X = X.flatten()
        Y = Y.flatten()

        # Allocate memory for the model parameters
        Beta = np.zeros((num_input_features * num_classes), dtype=np.float32).flatten()

        # Fit the model
        self.lib.fit(
            X,
            Y,
            Beta,
            num_samples,
            num_input_features,
            num_classes,
            self.iterations,
            self.learning_rate,
            self.tolerance,
        )

        # Reshape the Beta array and store it
        self.params["beta"] = Beta.reshape((num_input_features, num_classes))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the fitted model and the C implementation.

        This method validates the input data, flattens the arrays, and calls the
        C function to make predictions. The predictions are then reshaped to
        the appropriate dimensions.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted target values of shape (n_samples, n_targets).

        Raises:
            ValueError: If the model is not fitted or the dimensions of X do not match.
        """
        X = super().predict(X)

        # Get the dimensions of the input and output
        num_samples, num_input_features = X.shape
        num_classes = self.params["beta"].shape[1]
        num_classes = 1 if num_classes == 2 else num_classes

        # Allocate memory for the prediction and flatten
        prediction = np.zeros((num_samples, num_classes), dtype=np.float32).flatten()
        X = X.flatten()
        Beta = self.params["beta"].flatten()

        # Predict
        self.lib.predict(
            X, Beta, prediction, num_samples, num_input_features, num_classes
        )

        # Reshape the prediction array and return it
        return prediction.reshape((num_samples, num_classes))

    def cost(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the Mean Squared Error (MSE) cost using the C implementation.

        This method validates the input data, flattens the arrays, and calls the
        C function to compute the cost between predicted and true values.

        Args:
            Y_pred (np.ndarray): Predicted target values of shape (n_samples, n_targets).
            Y (np.ndarray): True target values of shape (n_samples, n_targets).

        Returns:
            float: The Mean Squared Error between Y_pred and Y.

        Raises:
            ValueError: If the dimensions of Y_pred and Y do not match.
        """

        Y_pred, Y = super().cost(Y_pred, Y)

        # Get array dimensions
        num_samples, num_classes = Y_pred.shape
        num_classes = 1 if num_classes == 2 else num_classes

        # Flatten arrays
        Y_pred = Y_pred.flatten()
        Y = Y.flatten()

        # Return the cost
        return self.lib.cost(Y_pred, Y, num_samples, num_classes)
