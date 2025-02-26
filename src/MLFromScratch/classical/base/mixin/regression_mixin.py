import numpy as np
from ....utils.python_utils.loss_functions import meanSquaredError

class RegressionMixin:
    
    def _validateTarget(self, Y: np.ndarray) -> np.ndarray:
        # Check if the input data is a numpy array
        if not isinstance(Y, np.ndarray):
            raise ValueError("Input data must be of type numpy.ndarray")
        
        # Check if the input data is 2D
        if len(Y.shape) != 2:
            raise ValueError("Input data must be a 2D array")
        
        # Check for at least one sample in the input data
        if Y.shape[0] == 0 or Y.shape[1] == 0:
            raise ValueError("Input data must have at least 1 sample and 1 feature")
        
        # Check if the input data is numerical
        if not np.issubdtype(Y.dtype, np.number):
            raise ValueError("Input data must be numerical")
        
        # Check for NaN or infinite values in the input data
        if not np.isfinite(Y).all():
            raise ValueError("Input data contains NaN or infinite values")
        
        # Ensure the input data is C-contiguous and float32
        if not Y.flags['C_CONTIGUOUS']:
            Y = np.ascontiguousarray(Y, dtype=np.float32)
        else:
            Y = Y.astype(np.float32, copy=False)
        
        return Y
    
    def loss(self, Y_true, Y_pred, language: str = 'python') -> float:
        return meanSquaredError(Y_true, Y_pred, language)
    
    def RSquared(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        return 1 - self.loss(Y_true, Y_pred) / np.var(Y_true)
    
    def adjustedRSquared(self, Y_true: np.ndarray, Y_pred: np.ndarray, n_features: int) -> float:
        n = len(Y_true)
        return 1 - (1 - self.RSquared(Y_true, Y_pred)) * (n - 1) / (n - n_features - 1)
    
    def MSE(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        return self.loss(Y_true, Y_pred)
    
    def RMSE(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        return np.sqrt(self.loss(Y_true, Y_pred))
    
    def MAE(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        return np.mean(np.abs(Y_true - Y_pred))
    
    def MAPE(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        return np.mean(np.abs((Y_true - Y_pred) / Y_true)) * 100
    
    def explainedVariance(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        return 1 - np.var(Y_true - Y_pred) / np.var(Y_true)
    
    def getStats(self, Y_true: np.ndarray, Y_pred: np.ndarray, n_features: int) -> dict:
        stats_dict = {}
        stats_dict['R-Squared'] = self.RSquared(Y_true, Y_pred)
        stats_dict['Adjusted R-Squared'] = self.adjustedRSquared(Y_true, Y_pred, n_features)
        stats_dict['MSE'] = self.MSE(Y_true, Y_pred)
        stats_dict['RMSE'] = self.RMSE(Y_true, Y_pred)
        stats_dict['MAE'] = self.MAE(Y_true, Y_pred)
        stats_dict['MAPE'] = self.MAPE(Y_true, Y_pred)
        stats_dict['Explained Variance'] = self.explainedVariance(Y_true, Y_pred)
        return stats_dict