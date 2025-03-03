from ..base.model.k_nearest_neighbors_model_base import KNNBase
from ..base.mixin.regression_mixin import RegressionMixin
from typing import Literal
import numpy as np


class KNNRegression(KNNBase, RegressionMixin):
    """
    K-Nearest Neighbors (KNN) regressor.

    This class implements the K-Nearest Neighbors regression algorithm. 
    It supports different distance metrics, including Euclidean, Manhattan, and Cosine similarity. 
    The model fits to the data by storing the training samples and makes predictions by finding the 
    k-nearest neighbors of a given test sample and averaging their target values for regression.

    Inherits from:
        - KNNBase: Base class providing foundational KNN functionality.
        - RegressionMixin: Mixin providing regression-specific utilities.

    Attributes:
        params (dict): Dictionary storing model parameters such as training data, number of samples, etc.
    """
    
    def __init__(self,**kwargs) -> None:
        """
        Initialize the KNNRegression model with the provided parameters.

        Args:
            **kwargs: Parameters for configuring the KNN model, passed to the parent classes.
        """
        super().__init__(**kwargs)
        
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the KNN model to the given training data for regression.

        Args:
            X (np.ndarray): The feature matrix (samples x features).
            Y (np.ndarray): The target values (samples, target_dim).

        Raises:
            ValueError: If the input data is invalid or the dimensions do not match.
        """
        # Validate the target vector
        Y = self._validateTarget(Y)
        
        # Call the base fit method
        super().fit(X, Y)
    
    def predict(self, X: np.ndarray, k: int = 5, distance_type: Literal["euclidean", "manhattan", "cosine"] = "euclidean") -> np.ndarray:
        """
        Predict the target values for the given test samples using K-Nearest Neighbors for regression.

        Args:
            X (np.ndarray): The test feature matrix (samples x features).
            k (int, optional): The number of nearest neighbors to consider for regression. Default is 5.
            distance_type (Literal["euclidean", "manhattan", "cosine"], optional): The distance metric to use for 
                finding nearest neighbors. Default is "euclidean".

        Returns:
            np.ndarray: The predicted target values for each test sample.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        # Validate the input array and get the indices of the k-nearest neighbors
        indices = super().predict(X, k, distance_type)
        
        # Get the number of samples
        n_samples = X.shape[0]
        
        # Get the corresponding target values
        targets = self.params["Y"][indices]
        
        # Initialize the predicted values
        Y_pred = np.zeros((n_samples, self.params["Y"].shape[1]))
        
        # Average the target values for regression
        for i in range(n_samples):
            Y_pred[i] = np.mean(targets[i], axis=0)
        
        # Return the predicted values
        return Y_pred