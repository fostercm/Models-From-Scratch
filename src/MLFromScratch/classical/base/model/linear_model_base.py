from ..abstract.supervised_model import SupervisedModel
import numpy as np


class LinearModelBase(SupervisedModel):

    def __init__(self, **kwargs) -> None:
        # Initialize the parameters
        super().__init__(beta=None, **kwargs)
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        # Add a bias term to the feature matrix
        return np.hstack((np.ones((X.shape[0], 1)), X), dtype=np.float32)
    
    def _compute_logits(self, X: np.ndarray) -> np.ndarray:
        # Compute the logits
        return X @ self.params["beta"]