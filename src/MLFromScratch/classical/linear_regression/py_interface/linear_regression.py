from ...base.model.linear_model_base import LinearModelBase
from ...base.mixin.regression_mixin import RegressionMixin
import numpy as np
from typing import Literal
import ctypes
import os

np.seterr(all="ignore")


class LinearRegression(LinearModelBase, RegressionMixin):

    def __init__(self, language: Literal['Python', 'C', 'CUDA'] = 'Python') -> None:
        super().__init__(language=language)
        
        # If the language is not Python, load the relevant shared library
        if language != 'Python':
            
            # Get the file directory
            package_dir = os.path.dirname(os.path.abspath(__file__))
            
            if language == 'C':
                lib_path = os.path.join(package_dir, "../../../lib/liblinear_regression_c.so")
                self.lib = ctypes.CDLL(lib_path)
                
            elif language == 'CUDA':
                lib_path = os.path.join(package_dir, "../../../lib/liblinear_regression_cuda.so")
                self.lib = ctypes.CDLL(lib_path)
                
            else:
                raise ValueError("Invalid language. Please choose from 'Python', 'C', or 'CUDA'")
            
            # Define the function argument types
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


    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        # Validate the input arrays and pad X with ones
        Y = super()._validateTarget(Y)
        X, Y = super().fit(X, Y)
        
        # If the language is Python, evaluate the model parameters using OLS
        if self.params["language"] == 'Python':

            # Evaluate the model parameters using OLS
            gram_matrix = X.T @ X
            if np.isclose(np.linalg.det(gram_matrix), 0.0):
                self.params["beta"] = np.linalg.pinv(gram_matrix) @ X.T @ Y
            else:
                self.params["beta"] = np.linalg.inv(X.T @ X) @ X.T @ Y
        
        else:
            
            # Get the dimensions of the input and output
            num_samples, num_input_features = X.shape
            num_output_features = Y.shape[1]
            
            # Allocate memory for the model parameters
            Beta = np.zeros((num_input_features, num_output_features), dtype=np.float32)
            
            # Flatten arrays
            X = X.flatten()
            Y = Y.flatten()
            Beta = Beta.flatten()
            
            # Call the relevant fit function
            self.lib.fit(X, Y, Beta, num_samples, num_input_features, num_output_features)
            
            # Reshape the model parameters
            self.params["beta"] = Beta.reshape((num_input_features, num_output_features))
            
        # Set the fitted flag
        self.params["fitted"] = True

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
        
        if self.params["language"] == 'Python':
            # Return the predicted values
            return X @ self.params["beta"]
        
        else:
            # Get the dimensions of the input and output
            num_samples, num_input_features = X.shape
            num_output_features = self.params["beta"].shape[1]

            # Allocate memory for the prediction
            Prediction = np.zeros((num_samples, num_output_features), dtype=np.float32)
            
            # Flatten arrays
            X = X.flatten()
            Beta = self.params["beta"].flatten()
            Prediction = Prediction.flatten()
            
            # Call the relevant predict function
            self.lib.predict(X, Beta, Prediction, num_samples, num_input_features, num_output_features)
            
            # Reshape the prediction
            return Prediction.reshape((num_samples, num_output_features))
