from ..abstract.supervised_model import SupervisedModel
import numpy as np


class LinearModelBase(SupervisedModel):
    """
    A base class for linear models that extends from the SupervisedModel class.
    This class provides methods for adding a bias term to the feature matrix and computing the logits 
    for a linear model, using learned parameters.

    Attributes:
        params (dict): A dictionary to store model parameters such as 'beta' (the coefficients).
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the LinearModelBase by calling the constructor of the parent class (SupervisedModel).
        
        The constructor optionally accepts the model coefficients (beta) and other parameters via kwargs.
        
        Args:
            **kwargs: Arbitrary keyword arguments passed to the parent class. Typically includes 
                      parameters for model configuration.
        """
        # Initialize the parameters
        super().__init__(beta=None, **kwargs)
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """
        Adds a bias term to the feature matrix X by appending a column of ones to the input data.

        This method is commonly used in linear models to account for the intercept term in the model.

        Args:
            X (np.ndarray): The feature matrix, where each row is a sample and each column is a feature.

        Returns:
            np.ndarray: The feature matrix with an additional column of ones for the bias term.
        """
        # Add a bias term to the feature matrix
        return np.hstack((np.ones((X.shape[0], 1)), X), dtype=np.float32)
    
    def _compute_logits(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the logits (linear predictions) for the input feature matrix X by performing 
        a matrix multiplication with the model's coefficients (beta).

        Args:
            X (np.ndarray): The feature matrix, where each row is a sample and each column is a feature.

        Returns:
            np.ndarray: The computed logits for each sample in X, which is the result of X @ beta.
        """
        # Compute the logits
        return X @ self.params["beta"]