from ...base.model.k_nearest_neighbors_model_base import KNNBase
from ...base.mixin.regression_mixin import RegressionMixin
from typing import Literal
import numpy as np


class KNNRegression(KNNBase, RegressionMixin):
    
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        # Validate the target vector
        Y = self._validateTarget(Y)
        
        # Call the base fit method
        super().fit(X, Y)
    
    def predict(self, X: np.ndarray, k: int = 5, distance_type: Literal["euclidean", "manhattan", "cosine"] = "euclidean") -> np.ndarray:
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