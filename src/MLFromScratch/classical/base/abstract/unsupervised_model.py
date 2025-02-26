from .base_model import BaseModel
import numpy as np
from abc import abstractmethod


class UnsupervisedModel(BaseModel):
    """
    An abstract base class for unsupervised learning models.

    This class serves as a blueprint for all unsupervised models, ensuring they implement
    a `transform` method. It inherits from `BaseModel` and provides common validation 
    utilities for input data

    Subclasses must implement:
    - `transform(X)`: A method that applies the model transformation to the input data

    Methods
    --------
    transform(X: np.ndarray) -> np.ndarray
        Applies the model transformation to the input data

    Raises
    -------
    NotImplementedError
        If a subclass does not implement the `transform` method
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the learned transformation to the input data.

        This method should be implemented by subclasses to perform transformations
        such as dimensionality reduction (e.g., PCA) or clustering assignments (e.g., K-Means)

        Parameters
        -----------
        X : np.ndarray
            A 2D NumPy array where rows represent samples and columns represent features

        Returns
        --------
        np.ndarray
            The transformed data, typically of shape (n_samples, n_new_features/1) depending on the specific unsupervised model

        Raises
        -------
        NotImplementedError
            If the method is not implemented in a subclass
        """
        pass