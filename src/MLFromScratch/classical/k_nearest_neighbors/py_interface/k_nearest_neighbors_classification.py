from ...base.model.k_nearest_neighbors_model_base import KNNModelBase
from ...base.mixin.classification_mixin import ClassificationMixin
from typing import Literal
import numpy as np


class KNNClassification(KNNModelBase, ClassificationMixin):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Validate the target vector
        y = super()._validateTarget(y)
        
        # Reshape the target vector
        Y = y.reshape(-1, 1)
        
        # Call the base fit method
        super().fit(X, Y)
    
    def predict(self, X: np.ndarray, k: int = 5, distance_type: Literal["euclidean", "manhattan", "cosine"] = "euclidean") -> np.ndarray:
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