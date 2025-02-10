from .linear_regression_base import LinearRegressionBase
import ctypes
import numpy as np
import os

class LinearRegressionCUDA(LinearRegressionBase):
    """
    Linear Regression model using a CUDA implementation for performance optimization.

    This class extends the LinearRegressionBase class and utilizes a CUDA-based
    library for model fitting, prediction, and cost computation to take advantage
    of GPU acceleration for large-scale computations.

    Attributes:
        lib (ctypes.CDLL): The CUDA shared library for linear regression operations.
    """
    
    def __init__(self):
        """
        Initialize the LinearRegressionCUDA model.

        This method loads the CUDA shared library and sets up the argument types
        for the CUDA functions used for fitting, predicting, and calculating cost.
        """
        super().__init__()
        
        # Load the CUDA library
        package_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(package_dir, "../../../lib/liblinear_regression_cuda.so")
        lib_path = os.path.normpath(lib_path)
        self.lib = ctypes.CDLL(lib_path)
        
        # Define the types of the arguments
        self.lib.fit.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            ctypes.c_int, 
            ctypes.c_int,
            ctypes.c_int
        ]
        
        self.lib.predict.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            ctypes.c_int, 
            ctypes.c_int,
            ctypes.c_int
        ]
        
        self.lib.cost.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            ctypes.c_int,
            ctypes.c_int
        ]
        self.lib.cost.restype = ctypes.c_float

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the Linear Regression model to the training data using the CUDA implementation.

        This method validates the input data, flattens the arrays, and calls the
        CUDA function to fit the model. The learned parameters (beta) are then
        reshaped and stored in the model's parameters.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            Y (np.ndarray): Target matrix of shape (n_samples, n_targets).

        Raises:
            ValueError: If the dimensions of X and Y do not match.
        """
        X, Y = super().fit(X, Y)
        
        # Get the dimensions of the input and output
        num_samples, num_input_features = X.shape
        num_output_features = Y.shape[1]
        
        # Flatten arrays
        X = X.flatten()
        Y = Y.flatten()
        Beta = np.zeros((num_input_features * num_output_features), dtype=np.float32).flatten()
        
        # Fit the model
        self.lib.fit(X, Y, Beta, num_samples, num_input_features, num_output_features)
        
        # Reshape the Beta array and store it
        self.params['beta'] = Beta.reshape((num_input_features, num_output_features))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the fitted model and the CUDA implementation.

        This method validates the input data, flattens the arrays, and calls the
        CUDA function to make predictions. The predictions are then reshaped to
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
        num_output_features = self.params['beta'].shape[1]
        
        # Allocate memory for the prediction and flatten
        prediction = np.zeros((num_samples, num_output_features), dtype=np.float32).flatten()
        X = X.flatten()
        Beta = self.params['beta'].flatten()
        
        # Predict
        self.lib.predict(X, Beta, prediction, num_samples, num_input_features, num_output_features)
        
        # Reshape the prediction array and return it
        return prediction.reshape((num_samples, num_output_features))
    
    def cost(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the Mean Squared Error (MSE) cost using the CUDA implementation.

        This method validates the input data, flattens the arrays, and calls the
        CUDA function to compute the cost between predicted and true values.

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
        num_samples, num_output_features = Y_pred.shape
        
        # Flatten arrays
        Y_pred = Y_pred.flatten()
        Y = Y.flatten()
        
        # Return the cost
        return self.lib.cost(Y_pred, Y, num_samples, num_output_features)