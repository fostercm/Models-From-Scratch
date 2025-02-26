from ...base.model.linear_model_base import LinearModelBase
from ...base.mixin.classification_mixin import ClassificationMixin
from ....utils.python_utils.activation_functions import sigmoid, softmax
import numpy as np
from typing import Literal
import ctypes
import os

class LogisticRegression(LinearModelBase, ClassificationMixin):

    def __init__(
        self,
        language: Literal["Python", "C", "CUDA"] = "Python",
        lr: float = 0.01,
        iterations: int = 10000,
        tolerance: float = 0.1,
    ) -> None:
        super().__init__(language=language, lr=lr, iterations=iterations, tolerance=tolerance)
        
        # If the language is not Python, load the relevant shared library
        if language != 'Python':
            
            # Get the file directory
            package_dir = os.path.dirname(os.path.abspath(__file__))
            
            if language == 'C':
                lib_path = os.path.join(package_dir, "../../../lib/liblogistic_regression_c.so")
                self.lib = ctypes.CDLL(lib_path)
                
            elif language == 'CUDA':
                lib_path = os.path.join(package_dir, "../../../lib/liblogistic_regression_cuda.so")
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Validate the input array
        y = super()._validateTarget(y)
        X, y = super().fit(X, y)

        # Get the number of classes
        unique_y = np.unique(y)
        self.params["num_classes"] = len(unique_y)

        # Transform the target values to one-hot encoding
        if self.params["num_classes"] > 2:
            Y = super()._oneHotEncode(y, self.params["num_classes"])
        else:
            Y = y.reshape(-1,1)
            
        # Initialize the model parameters
        if self.params["num_classes"] == 2:
            # Initialize beta for binary classification
            self.params["beta"] = np.zeros((X.shape[1], 1), dtype=np.float32)
        else:
            # Initialize beta for multi-class classification
            self.params["beta"] = np.zeros((X.shape[1], self.params["num_classes"]), dtype=np.float32)
            
        # Set model as fitted for predict method
        self.params["fitted"] = True
            
        if self.params["language"] == 'Python':
            # Gradient Descent
            for i in range(self.params["iterations"]):
                # Compute the predicted values
                Y_pred = self.predict(X, padded=True)                    

                # Compute the gradient of the cost function
                gradient = X.T @ (Y_pred - Y) / X.shape[0]
                
                # Check for convergence
                if i % 100 == 0 and np.linalg.norm(gradient) < self.params["tolerance"]:
                    break

                # Update the model parameters
                self.params["beta"] -= self.params["lr"] * gradient
            
        else:
            
            # Get the dimensions of the input and output
            num_samples, num_input_features = X.shape
            num_output_columns = Y.shape[1]
            
            # Allocate memory for the model parameters
            Beta = np.zeros((num_input_features, num_output_columns), dtype=np.float32)
            
            # Flatten arrays
            X = X.flatten()
            Y = Y.flatten()
            Beta = Beta.flatten()
            
            # Call the relevant fit function
            self.lib.fit(X, Y, Beta, num_samples, num_input_features, num_output_columns, self.params["iterations"], self.params["lr"], self.params["tolerance"])
            
            # Reshape the model parameters
            self.params["beta"] = Beta.reshape((num_input_features, num_output_columns))
    
        self.params["fitted"] = True

    def predict(self, X: np.ndarray, padded=False) -> np.ndarray:
        """
        Predict target values using the learned model.

        This method predicts the target values based on the input features X using
        the learned model parameters.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features), where
                            n_samples is the number of samples and n_features is the number of features per sample.

        Returns:
            np.ndarray: Predicted target values.
        """
        # Validate the input array and pad X with ones
        X = super().predict(X, padded=padded)
        
        if self.params["language"] == 'Python':
            if self.params["num_classes"] == 2:
                return sigmoid(X @ self.params["beta"])
            else:
                return softmax(X @ self.params["beta"])
        
        else:
            
            # Get the dimensions of the input and output
            num_samples, num_input_features = X.shape
            num_classes = self.params["num_classes"]
            num_classes = num_classes if num_classes > 2 else 1

            # Allocate memory for the prediction
            Prediction = np.zeros((num_samples, num_classes), dtype=np.float32)
            
            # Flatten arrays
            X = X.flatten()
            Beta = self.params["beta"].flatten()
            Prediction.flatten()

            # Predict
            self.lib.predict(X, Beta, Prediction, num_samples, num_input_features, num_classes)

            # Reshape the prediction array and return it
            return Prediction.reshape((num_samples, num_classes))