from ..abstract.supervised_model import SupervisedModel
import numpy as np
from typing import Literal


class KNNBase(SupervisedModel):
    """
    A base class for the k-Nearest Neighbors (KNN) algorithm that extends from the SupervisedModel class.
    This class provides methods to fit a KNN model to training data and make predictions based on nearest neighbors.

    Attributes:
        params (dict): A dictionary to store model parameters such as training data, fitted status, 
                        k-value, and the distance metric used.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the KNN model by calling the constructor of the parent class (SupervisedModel).

        Args:
            **kwargs: Arbitrary keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the KNN model to the provided training data.

        This method stores the feature matrix (X) and target values (Y) and marks the model as fitted.

        Args:
            X (np.ndarray): The feature matrix of the training data.
            Y (np.ndarray): The target values corresponding to the training data.

        Raises:
            ValueError: If the input arrays X and Y are not compatible.
        """
        # Validate the input array
        X = super()._validateInput(X)
        super()._validateInputPair(X, Y)
        
        # Store the input feature matrix and target values
        self.params["X"] = X
        self.params["Y"] = Y
        
        # Set the model as fitted
        self.params["fitted"] = True

    def predict(self, X: np.ndarray, k: int = 5, distance_type: Literal["euclidean", "manhattan", "cosine"] = "euclidean") -> np.ndarray:
        """
        Predicts the k-nearest neighbors for each input sample in X and returns the indices of the k nearest neighbors.

        Args:
            X (np.ndarray): The input data to predict for, must have the same number of features as the training data.
            k (int): The number of nearest neighbors to consider (default is 5).
            distance_type (Literal): The distance metric to use for neighbor selection, can be 'euclidean', 
                                      'manhattan', or 'cosine' (default is 'euclidean').

        Returns:
            np.ndarray: Indices of the k-nearest neighbors for each sample in X.

        Raises:
            ValueError: If the model is not fitted, k is invalid, or the dimensions of X do not match the training data.
        """
        # Check if the model is fitted
        if self.params["fitted"] is False:
            raise ValueError("Model is not fitted")
        
        # Check k is a positive integer
        if not isinstance(k, int) or k <= 0 or k > self.params["X"].shape[0]:
            raise ValueError("k must be a positive integer less than the number of training samples")
        self.params["k"] = k
        
        # Validate the input array
        X = super()._validateInput(X)
        
        # Check if the dimensions of the input array match the stored feature matrix
        if X.shape[1] != self.params["X"].shape[1]:
            raise ValueError("The number of columns in X must be equal to the number of features in the model")
        
        # Calculate the distance between the input samples and the training samples
        distances = self._calculateDistance(X, distance_type)
        
        # Get the indices of the k-nearest neighbors
        indices = np.argsort(distances, axis=1)[:, :k]
        
        return indices
    
    def _calculateDistance(self, X: np.ndarray, distance_type: Literal["euclidean", "manhattan", "cosine"] = "euclidean") -> np.ndarray:
        """
        Computes the distance between input samples (X) and the training samples using the specified distance metric.

        Args:
            X (np.ndarray): The input data to compute distances for.
            distance_type (Literal): The type of distance metric to use, can be 'euclidean', 'manhattan', or 'cosine'.
                                     Default is 'euclidean'.

        Returns:
            np.ndarray: A 2D array where each row contains the distances between a sample in X and all training samples.

        Raises:
            ValueError: If an invalid distance_type is provided.
        """
        # Calculate the distance between the input samples and the training samples
        if distance_type == "euclidean":
            # Record the distance type
            self.params["distance"] = "euclidean"
            
            # Calculate the Euclidean distance (L2 norm)
            distances = np.linalg.norm(X[:, np.newaxis] - self.params["X"], axis=2)
            
        elif distance_type == "manhattan":
            
            # Record the distance type
            self.params["distance"] = "manhattan"
            
            # Calculate the Manhattan distance (L1 norm)
            distances = np.sum(np.abs(X[:, np.newaxis] - self.params["X"]), axis=2)
            
        elif distance_type == "cosine":
            
            # Record the distance type
            self.params["distance"] = "cosine"
            
            # Calculate the cosine distance
            dot_product = np.dot(X, self.params["X"].T)
            norm_X = np.linalg.norm(X, axis=1)
            norm_Y = np.linalg.norm(self.params["X"], axis=1)
            distances = 1 - dot_product / np.outer(norm_X, norm_Y)
        
        else:
            raise ValueError("Invalid distance type, must be 'euclidean', 'manhattan' or 'cosine'")
        
        return distances