from ...base.model.k_nearest_neighbors_model_base import KNNBase
from ...base.mixin.classification_mixin import ClassificationMixin
from typing import Literal
import numpy as np


class KNNClassification(KNNBase, ClassificationMixin):
    """
    K-Nearest Neighbors (KNN) classifier.

    This class implements the K-Nearest Neighbors classification algorithm. 
    It supports different distance metrics, including Euclidean, Manhattan, and Cosine similarity. 
    The model fits to the data by storing the training samples and makes predictions by finding the 
    k-nearest neighbors of a given test sample.

    Inherits from:
        - KNNBase: Base class providing foundational KNN functionality.
        - ClassificationMixin: Mixin providing classification-specific utilities.

    Attributes:
        params (dict): Dictionary storing model parameters such as training data, number of samples, etc.
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Initialize the KNNClassification model with the provided parameters.

        Args:
            **kwargs: Parameters for configuring the KNN model, passed to the parent classes.
        """
        super().__init__(**kwargs)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the KNN model to the given training data.

        Args:
            X (np.ndarray): The feature matrix (samples x features).
            y (np.ndarray): The target labels (samples,).

        Raises:
            ValueError: If the input data is invalid or the dimensions do not match.
        """
        # Validate the input arrays
        X = self._validateInput(X)
        y = self._validateTarget(y)
        self._validateInputPair(X, y)
        
        # Reshape the target vector
        Y = y.reshape(-1, 1)
        
        # Call the base fit method
        super().fit(X, Y)
    
    def predict(self, X: np.ndarray, k: int = 5, distance_type: Literal["euclidean", "manhattan", "cosine"] = "euclidean") -> np.ndarray:
        """
        Predict the class labels for the given test samples using K-Nearest Neighbors.

        Args:
            X (np.ndarray): The test feature matrix (samples x features).
            k (int, optional): The number of nearest neighbors to consider for classification. Default is 5.
            distance_type (Literal["euclidean", "manhattan", "cosine"], optional): The distance metric to use for 
                finding nearest neighbors. Default is "euclidean".

        Returns:
            np.ndarray: The predicted class labels for each test sample.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        # Validate the input array and get the indices of the k-nearest neighbors
        indices = super().predict(X, k, distance_type)
        
        # Get the number of samples
        n_samples = X.shape[0]
        
        # Get the corresponding target values
        targets = self.params["Y"][indices]
        
        # Initialize the predicted classes
        y_pred = np.zeros(n_samples)
        
        # Get the predicted classes
        for i in range(n_samples):
            y_pred[i] = np.argmax(np.bincount(targets[i].flatten()))
        
        # Return the predicted classes
        return y_pred