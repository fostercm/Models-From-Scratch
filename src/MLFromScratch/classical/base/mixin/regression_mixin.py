import numpy as np
from ....utils.python_utils.loss_functions import meanSquaredError

class RegressionMixin:
    """
    Mixin class providing common methods for regression models

    This class includes loss (MSE), target validation, and various evaluation metrics such as R-Squared, adjusted R-Squared,
    RMSE, MAE, MAPE, and explained variance computation
    """
    
    def _validateTarget(self, Y: np.ndarray) -> np.ndarray:
        """
        Validates the target array by ensuring it is a 2D NumPy array of numerical values.
        
        Args:
            Y (np.ndarray): The target values.
        
        Returns:
            np.ndarray: A validated, C-contiguous, float32 NumPy array.
        
        Raises:
            ValueError: If the input is not a valid 2D numerical NumPy array.
        """
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
        """
        Computes the mean squared error (MSE) loss between the true and predicted values.
        
        Args:
            Y_true (np.ndarray): The ground truth target values.
            Y_pred (np.ndarray): The predicted target values.
            language (str, optional): The language implementation of the loss function. Defaults to 'python'.
        
        Returns:
            float: The mean squared error.
        """
        return meanSquaredError(Y_true, Y_pred, language)
    
    def RSquared(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        Computes the R-squared (coefficient of determination) score.
        
        Args:
            Y_true (np.ndarray): The ground truth target values.
            Y_pred (np.ndarray): The predicted target values.
        
        Returns:
            float: The R-squared value.
        """
        return 1 - self.loss(Y_true, Y_pred) / np.var(Y_true)
    
    def adjustedRSquared(self, Y_true: np.ndarray, Y_pred: np.ndarray, n_features: int) -> float:
        """
        Computes the adjusted R-squared score.
        
        Args:
            Y_true (np.ndarray): The ground truth target values.
            Y_pred (np.ndarray): The predicted target values.
            n_features (int): The number of features used in the model.
        
        Returns:
            float: The adjusted R-squared value.
        
        Raises:
            ValueError: If the number of features is greater than or equal to the number of samples - 1.
        """
        n = len(Y_true)
        return 1 - (1 - self.RSquared(Y_true, Y_pred)) * (n - 1) / (n - n_features - 1)
    
    def MSE(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        Computes the mean squared error (MSE).
        
        Args:
            Y_true (np.ndarray): The ground truth target values.
            Y_pred (np.ndarray): The predicted target values.
        
        Returns:
            float: The mean squared error.
        """
        return self.loss(Y_true, Y_pred)
    
    def RMSE(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        Computes the root mean squared error (RMSE).
        
        Args:
            Y_true (np.ndarray): The ground truth target values.
            Y_pred (np.ndarray): The predicted target values.
        
        Returns:
            float: The root mean squared error.
        """
        return np.sqrt(self.loss(Y_true, Y_pred))
    
    def MAE(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        Computes the mean absolute error (MAE).
        
        Args:
            Y_true (np.ndarray): The ground truth target values.
            Y_pred (np.ndarray): The predicted target values.
        
        Returns:
            float: The mean absolute error.
        """
        return np.mean(np.abs(Y_true - Y_pred))
    
    def MAPE(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        Computes the mean absolute percentage error (MAPE).
        
        Args:
            Y_true (np.ndarray): The ground truth target values.
            Y_pred (np.ndarray): The predicted target values.
        
        Returns:
            float: The mean absolute percentage error.
        
        Raises:
            ValueError: If any element in Y_true is zero to prevent division by zero.
        """
        return np.mean(np.abs((Y_true - Y_pred) / (Y_true + 1e-7))) * 100
    
    def explainedVariance(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        Computes the explained variance score.
        
        Args:
            Y_true (np.ndarray): The ground truth target values.
            Y_pred (np.ndarray): The predicted target values.
        
        Returns:
            float: The explained variance score.
        """
        return 1 - np.var(Y_true - Y_pred) / np.var(Y_true)
    
    def getStats(self, Y_true: np.ndarray, Y_pred: np.ndarray, n_features: int) -> dict:
        """
        Computes multiple regression metrics and returns them as a dictionary.
        
        Args:
            Y_true (np.ndarray): The ground truth target values.
            Y_pred (np.ndarray): The predicted target values.
            n_features (int): The number of features used in the model.
        
        Returns:
            dict: A dictionary containing various regression metrics.
        """
        return {
            'R-Squared': self.RSquared(Y_true, Y_pred),
            'Adjusted R-Squared': self.adjustedRSquared(Y_true, Y_pred, n_features),
            'MSE': self.MSE(Y_true, Y_pred),
            'RMSE': self.RMSE(Y_true, Y_pred),
            'MAE': self.MAE(Y_true, Y_pred),
            'MAPE': self.MAPE(Y_true, Y_pred),
            'Explained Variance': self.explainedVariance(Y_true, Y_pred)
        }