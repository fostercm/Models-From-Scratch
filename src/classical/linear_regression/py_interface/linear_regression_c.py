from .linear_regression_base import LinearRegressionBase
import ctypes
import numpy as np

class LinearRegressionC(LinearRegressionBase):
    
    def __init__(self):
        super().__init__()
        
        # Load the CUDA library
        self.lib = ctypes.CDLL("./c_backend/linear_regression_c.so")
        
        # Define the types of the arguments
        self.lib.fit.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            ctypes.c_int, 
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64)
        ]
        
        
        self.lib.predict_kernel.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64)
        ]

    def fit(self, X: np.ndarray, y):
        n, m = X.shape
        weights = np.zeros(m, dtype=np.float64)
        self.lib.fit_kernel(X, y, n, m, weights)
        self.weights = weights

    def predict(self, X):
        n, m = X.shape
        predictions = np.zeros(n, dtype=np.float64)
        self.lib.predict_kernel(X, self.weights, n, m, predictions)
        return predictions