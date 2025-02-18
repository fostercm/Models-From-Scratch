from .principal_component_analysis_base import PCABase
from ....utils.python_utils.matrix_functions import standardize
import numpy as np

class PCAPython(PCABase):
    
    def transform(self, X: np.ndarray, N_components: int=None, explained_variance_ratio: float=None) -> np.ndarray:
        # Validate the input
        X = super().transform(X)
        
        # Standardize the data
        X = standardize(X)
        
        # Perform SVD
        _, S, V = np.linalg.svd(X, full_matrices=False)
        
        # Calculate the explained variance
        total_variance = np.sum(S**2)
        explained_variance = (S**2) / total_variance
        
        # Calculate the cumulative explained variance
        cumulative_explained_variance = np.cumsum(explained_variance)
        
        # If N_components is specified, use the first N_components
        if N_components:
            return X @ V[:N_components].T
        
        # If explained_variance is specified, use the components that explain the variance
        if explained_variance_ratio:
            for i in range(len(cumulative_explained_variance)):
                if cumulative_explained_variance[i] >= explained_variance_ratio:
                    N_components = i + 1
                    break
            return X @ V[:N_components].T
        
        # If N_components is not specified, use all components
        return X @ V.T