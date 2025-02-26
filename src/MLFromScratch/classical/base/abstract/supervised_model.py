from .base_model import BaseModel
import numpy as np
from abc import abstractmethod


class SupervisedModel(BaseModel):
    """
    An abstract base class for supervised learning models

    This class serves as a blueprint for all supervised models, ensuring they implement
    `fit` and `predict` methods. It inherits from `BaseModel` and provides common validation 
    utilities for input data
    
    Subclasses must implement:
    - `fit(X,Y)`: A method that fits model parameters to the training data
    - `predict(X)`: A method that predicts the target variable for new data

    Methods
    --------
    fit(X: np.ndarray, Y: np.ndarray) -> None
        Fit the model to the training data
    predict(X: np.ndarray) -> np.ndarray
        Predict the target variable for the given data

    Raises
    -------
    NotImplementedError
        If a subclass does not implement the `fit` or `predict` methods
    """
    def __init__(self, **kwargs) -> None:
        """
        Initializes the model parameter dictionary
        """
        super().__init__(fitted=False, **kwargs)
        

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Train the model using input data and corresponding target values

        This method must be implemented in subclasses to define the learning process

        Parameters:
        -----------
        X : np.ndarray
            A 2D NumPy array representing the input features with shape (n_samples, n_features)
        Y : np.ndarray
            A NumPy array representing the target values with shape (n_samples, n_classes/n_targets/1)

        Raises:
        -------
        NotImplementedError
            If the method is not implemented in a subclass
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions based on the trained model

        This method must be implemented in subclasses to define how the model produces predictions

        Parameters:
        -----------
        X : np.ndarray
            A 2D NumPy array representing the input features with shape (n_samples, n_features)

        Returns:
        --------
        np.ndarray
            A NumPy array containing the predicted target values with shape (n_samples, n_classes/n_targets/1) 

        Raises:
        -------
        NotImplementedError
            If the method is not implemented in a subclass
        """
        pass