from .principal_component_analysis_base import PCABase
from ....utils.python_utils.matrix_functions import standardize
import numpy as np

class PCAPython(PCABase):
    
    def transform(self, X: np.ndarray, n_components: int=None, explained_variance_ratio: float=None) -> np.ndarray:
        # Validate the input
        X = super().transform(X)
        
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