# from .principal_component_analysis_base import PCABase
# import numpy as np
# import os
# import ctypes

# class PCAC(PCABase):
    
#     def __init__(self):
#         super().__init__()
        
#         # Load the C library
#         package_dir = os.path.dirname(os.path.abspath(__file__))
#         lib_path = os.path.join(package_dir, "../../../lib/libprincipal_component_analysis_c.so")
#         lib_path = os.path.normpath(lib_path)
#         self.lib = ctypes.CDLL(lib_path)
        
#         # Define the types of the arguments
#         self.lib.transform.argtypes = [
#             np.ctypeslib.ndpointer(dtype=np.float32),
#             np.ctypeslib.ndpointer(dtype=np.float32),
#             ctypes.c_int,
#             ctypes.c_int,
#             ctypes.c_int,
#             ctypes.c_float
#         ]
#         self.lib.transform.restype = ctypes.c_int
    
#     def transform(self, X: np.ndarray, n_components: int=None, explained_variance_ratio: float=None) -> np.ndarray:
#         # Validate the input
#         X = super().transform(X)
        
#         # Get the number of of samples and features
#         n_samples, n_features = X.shape
        
#         # If N_components or explained_variance_ratio None, set it to 0
#         n_components = 0 if n_components is None else n_components
#         explained_variance_ratio = 0 if explained_variance_ratio is None else explained_variance_ratio
        
#         # Allocate memory for the transformed data
#         X_transformed = np.zeros(n_samples * n_features, dtype=np.float32)
        
#         # Flatten the data
#         X = X.flatten()
        
#         # Call the C function to transform the data
#         n_output_components = self.lib.transform(X, X_transformed, n_samples, n_features, n_components, explained_variance_ratio)
        
#         # Reshape the transformed data
#         X_transformed = X_transformed[:n_output_components * n_samples]
#         X_transformed = X_transformed.reshape(n_samples, n_output_components)
        
#         return X_transformed