from ..base.abstract.unsupervised_model import UnsupervisedModel
from ...utils.python_utils.matrix_functions import standardize
import numpy as np
from typing import Literal

class PCA(UnsupervisedModel):
    """
    Principal Component Analysis (PCA) implementation.
    
    This class performs PCA using Singular Value Decomposition (SVD) to reduce 
    the dimensionality of input data while preserving as much variance as possible.
    
    Attributes:
        language (Literal['Python', 'C', 'CUDA']): Specifies the implementation backend.
    """
    
    def __init__(self, language: Literal['Python', 'C', 'CUDA'] = 'Python'):
        """
        Initializes the PCA model with the specified implementation backend.
        
        Args:
            language (Literal['Python', 'C', 'CUDA'], optional): Specifies the backend 
                implementation to use. Defaults to 'Python'.
        """
        # Validate the language parameter
        self._validate_language(language)
        
        # Initialize the model parameters
        super().__init__(language=language)
    
    
    def transform(self, X: np.ndarray, n_components: int=None, explained_variance_ratio: float=None) -> np.ndarray:
        """
        Transforms the input data using PCA, reducing its dimensionality.
        
        Args:
            X (np.ndarray): The input data matrix of shape (n_samples, n_features).
            n_components (int, optional): The number of principal components to retain. 
                Must be between 1 and the number of features. Defaults to None.
            explained_variance_ratio (float, optional): The minimum cumulative explained 
                variance ratio to retain. Must be between 0 and 1. Defaults to None.
                If provided, it determines n_components dynamically.
        
        Returns:
            np.ndarray: The transformed data matrix with reduced dimensions.
        
        Raises:
            TypeError: If n_components is not an integer or explained_variance_ratio is not a float.
            ValueError: If n_components or explained_variance_ratio is out of valid range,
                or if both parameters are provided simultaneously.
        """
        # Validate the input
        X = self._validateInput(X, n_components, explained_variance_ratio)
        
        # Get the number of components
        n_components = X.shape[1] if n_components is None else n_components
        
        # Standardize the data
        X = standardize(X)
        
        # Perform SVD
        _, S, V = np.linalg.svd(X, full_matrices=False)
        
        # If explained_variance is specified, use the components that explain the variance
        if explained_variance_ratio:
            # Calculate the explained variance
            S = S**2
            total_variance = np.sum(S)
            
            # Calculate the cumulative explained variance
            cumulative_explained_variance = 0
            for i in range(len(S)):
                cumulative_explained_variance += S[i] / total_variance
                if cumulative_explained_variance >= explained_variance_ratio:
                    n_components = i + 1
                    break
        
        # Return the transformed data
        return X @ V[:n_components].T
    
    def _validateInput(self, X: np.ndarray, n_components: int, explained_variance_ratio: float) -> np.ndarray:
        """
        Validates the input parameters and ensures they meet expected constraints.
        
        Args:
            X (np.ndarray): The input data matrix.
            n_components (int, optional): The number of principal components.
            explained_variance_ratio (float, optional): The explained variance threshold.
        
        Returns:
            np.ndarray: The validated input data.
        
        Raises:
            TypeError: If n_components is not an integer or explained_variance_ratio is not a float.
            ValueError: If n_components or explained_variance_ratio is out of valid range,
                or if both parameters are provided simultaneously.
        """
        # Validate input array
        X = super()._validateInput(X)
        
        # Check if n_components is an integer
        if n_components is not None and not isinstance(n_components, int):
            raise TypeError("n_components must be an integer.")

        # Check if explained_variance is a float
        if explained_variance_ratio is not None and not isinstance(explained_variance_ratio, float):
            raise TypeError("explained_variance_ratio must be a float.")
        
        # Check if n_components is in the range of the number of features
        if n_components is not None and (n_components <= 0 or n_components > X.shape[1]):
            raise ValueError("n_components must be between 1 and the number of features.")
        
        # Check if explained_variance is between 0 and 1 (exclusive)
        if explained_variance_ratio is not None and (explained_variance_ratio <= 0 or explained_variance_ratio >= 1):
            raise ValueError("explained_variance_ratio must be between 0 and 1.")
        
        # Check if n_components and explained_variance_ratio are both set
        if n_components is not None and explained_variance_ratio is not None:
            raise ValueError("Only one of n_components and explained_variance_ratio can be set.")
        
        return X