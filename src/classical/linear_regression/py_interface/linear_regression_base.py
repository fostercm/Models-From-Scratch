from classical.classical_model import ClassicalModel
import numpy as np

class LinearRegressionBase(ClassicalModel):
    
    def __init__(self) -> None:
        """Initialize the model parameters"""
        self.params = {'beta': None}

    def fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fit the Linear Regression model to the training data

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            Y (np.ndarray): Target matrix of shape (n_samples, n_targets)
        """
        # Validate the input arrays
        X, Y = super()._validate_input(X,Y)
        
        # Pad the feature matrix with ones for the bias term
        X = np.hstack((np.ones((X.shape[0], 1)), X), dtype=np.float32)
        
        return X, Y
        
        # To be implemented in derived classes

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the learned model

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        """
        # Validate the input array
        X, _ = super()._validate_input(X)
        
        # Check if the model is fitted
        if self.params['beta'] is None:
            raise ValueError("Model is not fitted")
        
        # Pad the feature matrix with ones for the bias term
        X = np.hstack((np.ones((X.shape[0], 1)), X), dtype=np.float32)
        
        # Check if the input dimensions match the model parameters
        if X.shape[1] != self.params['beta'].shape[0]:
            raise ValueError("The number of columns in X must be equal to the number of features in the model")
        
        return X
        
        # To be implemented in derived classes
    
    def cost(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the Mean Squared Error cost

        Args:
            Y_pred (np.ndarray): Predicted target values
            Y (np.ndarray): True target values

        Returns:
            float: Mean Squared Error
        """
        Y_pred, Y = super()._validate_input(Y_pred, Y)
        
        return Y_pred, Y
        
        # To be implemented in derived classes