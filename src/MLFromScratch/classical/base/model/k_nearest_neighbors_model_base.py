from ..abstract.supervised_model import SupervisedModel
import numpy as np
from typing import Literal


class KNNModelBase(SupervisedModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        # Validate the input array
        X = super()._validateInput(X)
        super()._validateInputPair(X, Y)
        
        # Store the input feature matrix and target values
        self.params["X"] = X
        self.params["Y"] = Y
        
        # Set the model as fitted
        self.params["fitted"] = True

    def predict(self, X: np.ndarray, k: int = 5, distance_type: Literal["euclidean", "manhattan", "cosine"] = "euclidean") -> np.ndarray:
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