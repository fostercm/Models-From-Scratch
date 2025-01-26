from .linear_regression_base import LinearRegressionBase
import ctypes
import numpy as np
from pathlib import Path

class LinearRegressionC(LinearRegressionBase):
    
    def __init__(self):
        super().__init__()
        
        # Load the C library
        # lib_path = Path(__file__).resolve().parents[4] / 'build' / 'lib' / 'liblinear_regression_c.so'
        lib_path = Path(__file__).resolve().parents[4] / 'linear_regression.so'
        self.lib = ctypes.CDLL(str(lib_path))
        
        # Define the types of the arguments
        self.lib.fit.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            ctypes.c_int, 
            ctypes.c_int,
            ctypes.c_int
        ]
        
        self.lib.predict.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            ctypes.c_int, 
            ctypes.c_int,
            ctypes.c_int
        ]
        
        self.lib.cost.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            ctypes.c_int,
            ctypes.c_int
        ]
        self.lib.cost.restype = ctypes.c_double

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        X = super().fit(X, Y)
        
        # Get the dimensions of the input and output
        num_samples, num_input_features = X.shape
        num_output_features = Y.shape[1]
        
        # Flatten arrays
        X = X.flatten()
        Y = Y.flatten()
        Beta = np.zeros((num_input_features * num_output_features), dtype=np.float64).flatten()
        
        # Fit the model
        self.lib.fit(X, Y, Beta, num_samples, num_input_features, num_output_features)
        
        # Reshape the Beta array and store it
        self.params['beta'] = Beta.reshape((num_input_features, num_output_features))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = super().predict(X)
        
        # Get the dimensions of the input and output
        num_samples, num_input_features = X.shape
        num_output_features = self.params['beta'].shape[1]
        
        # Allocate memory for the prediction and flatten
        prediction = np.zeros((num_samples, num_output_features), dtype=np.float64).flatten()
        X = X.flatten()
        Beta = self.params['beta'].flatten()
        
        # Predict
        self.lib.predict(X, Beta, prediction, num_samples, num_input_features, num_output_features)
        
        # Reshape the prediction array and return it
        return prediction.reshape((num_samples, num_output_features))
    
    def cost(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        super().cost(Y_pred, Y)
        
        # Get array dimensions
        num_samples, num_output_features = Y_pred.shape
        
        # Flatten arrays
        Y_pred = Y_pred.flatten()
        Y = Y.flatten()
        
        # Return the cost
        return self.lib.cost(Y_pred, Y, num_samples, num_output_features)
        
        
# gcc -shared -o linear_regression.so src/classical/linear_regression/c_backend/linear_regression.c -I./src/utils/c_utils ./build/src/utils/c_utils/libc_utils.a -fPIC -lm -lgsl -lgslcblas -llapacke