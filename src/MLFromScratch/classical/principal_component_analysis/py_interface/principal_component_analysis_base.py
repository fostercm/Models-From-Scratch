# from ...base.abstract.unsupervised_model import UnsupervisedModel
# import numpy as np

# class PCABase(UnsupervisedModel):
    
#     def transform(self, X: np.ndarray, N_components: int=None, explained_variance_ratio: float=None) -> np.ndarray:
#         X, _ = super()._validate_input(X)
        
#         # Check if N_components is an integer
#         if N_components is not None and not isinstance(N_components, int):
#             raise TypeError("N_components must be an integer.")

#         # Check if explained_variance is a float
#         if explained_variance_ratio is not None and not isinstance(explained_variance_ratio, float):
#             raise TypeError("explained_variance_ratio must be a float.")
        
#         # Check if N_components is in the range of the number of features
#         if N_components is not None and (N_components <= 0 or N_components > X.shape[1]):
#             raise ValueError("N_components must be between 1 and the number of features.")
        
#         # Check if explained_variance is between 0 and 1 (exclusive)
#         if explained_variance_ratio is not None and (explained_variance_ratio <= 0 or explained_variance_ratio >= 1):
#             raise ValueError("explained_variance_ratio must be between 0 and 1.")
        
#         # Check if N_components and explained_variance_ratio are both set
#         if N_components is not None and explained_variance_ratio is not None:
#             raise ValueError("Only one of N_components and explained_variance_ratio can be set.")
                
#         return X