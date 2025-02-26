from ..abstract.supervised_model import SupervisedModel
import numpy as np


class LinearModelBase(SupervisedModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(beta=None, **kwargs)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # Validate the input array
        X = super()._validateInput(X)
        super()._validateInputPair(X, Y)
        
        # Pad the feature matrix with ones for the bias term
        X = self._addBias(X)
        
        # Return the modified feature matrix and target values
        return X, Y
        

    def predict(self, X: np.ndarray, padded: bool = False) -> np.ndarray:
        # Check if the model is fitted
        if self.params["fitted"] is False:
            raise ValueError("Model is not fitted")
        
        # Validate the input array
        X = super()._validateInput(X)

        # Pad the feature matrix with ones for the bias term if required
        if not padded:
            X = self._addBias(X)

        # Check if the input dimensions match the model parameters
        if X.shape[1] != self.params["beta"].shape[0]:
            raise ValueError("The number of columns in X must be equal to the number of features in the model")

        return X
    
    def _addBias(self, X: np.ndarray) -> np.ndarray:
        """
        Add a bias term to the input feature matrix
        
        Parameters
        ----------
        X : np.ndarray
            The input feature matrix
        
        Returns
        -------
        np.ndarray
            The input feature matrix with a bias term added
        """
        return np.hstack((np.ones((X.shape[0], 1)), X), dtype=np.float32)