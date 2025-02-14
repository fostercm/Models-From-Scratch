import numpy as np
from ...linear_regression.py_interface.linear_regression_base import LinearRegressionBase

class LogisticRegressionBase(LinearRegressionBase):
    """
    A base class for Logistic Regression model.

    This class provides methods for fitting the logistic regression model,
    predicting target values, and calculating the cost (Cross Entropy Loss).

    Attributes:
        params (dict): A dictionary storing the model parameters, including 'beta'.
    """
    
    def __init__(self, learning_rate: float=0.01, tolerance: float=0.01, max_iters: int=10000) -> None:
        """
        Initialize the Logistic Regression model parameters.

        Initializes the 'beta' parameter to None. This will be populated
        when the model is fitted to the training data.
        """
        super().__init__()
        self.params['num_classes'] = None
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iters = max_iters
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fit the Logistic Regression model to the training data.
        
        This method fits a logistic regression model to the provided training data
        (X and Y) using the gradient descent approach. It pads the feature matrix
        with a bias term (a column of ones) and validates the input data.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features), where
                            n_samples is the number of training samples and 
                            n_features is the number of features per sample.
            Y (np.ndarray): Target matrix of shape (n_samples, n_targets), where
                            n_samples is the number of training samples and 
                            n_targets is the number of targets per sample.

        Returns:
            np.ndarray: The modified feature matrix (with the bias term added) and the target values (Y).

        Raises:
            ValueError: If the output dimension of Y is not 1.
        """
        # Validate the input arrays
        X, Y = super().fit(X, Y)
        
        # Check if the output dimension of Y is 1
        if Y.shape[1] != 1:
            raise ValueError(f"Output dimension of Y is {Y.shape[1]}, expected 1.")
        
        # Set the number of classes
        self.params['num_classes'] = len(np.unique(Y))
        
        # Check if there is only one class
        if self.params['num_classes'] == 1:
            raise ValueError(f"Only one class found in Y. Please check the target values.")
        
        # Check if the output classes are from 0 to num_classes-1
        if not np.array_equal(np.unique(Y), np.arange(self.params['num_classes'])):
            raise ValueError(f"Output classes should be from 0 to num_classes-1.")
        
        # Transform the target values to one-hot encoding
        if self.params['num_classes'] > 2:
            Y_new = np.zeros((Y.shape[0], self.params['num_classes']), dtype=np.float32)
            Y = Y.astype(np.int32)
            Y_new[np.arange(Y.shape[0]), Y.flatten()] = 1
        
        return X, Y_new if self.params['num_classes'] > 2 else Y

    def predict(self, X: np.ndarray, pad: bool=True) -> np.ndarray:
        """
        Predict target values using the learned model.
        
        This method predicts the target values based on the input features X using
        the learned model parameters.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features), where
                            n_samples is the number of samples and n_features is the number of features per sample.
        
        Returns:
            np.ndarray: Predicted target values.
        """
        # Validate the input arrays and pad X with ones
        X = super().predict(X, pad)    
        
        return X
    
    def cost(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the Cross Entropy (CE) cost.

        This method calculates the Cross Entropy Loss between the predicted values
        (Y_pred) and the true target values (Y). This is a common cost function
        used in classification tasks.

        Args:
            Y_pred (np.ndarray): Predicted target values of shape (n_samples, n_classes).
            Y (np.ndarray): True target values of shape (n_samples, n_classes).

        Returns:
            float: The Mean Squared Error between Y_pred and Y.

        Raises:
            ValueError: If the dimensions of Y_pred and Y do not match.
        """
        # Validate the input arrays
        Y_pred, Y = super().cost(Y_pred, Y)
        
        return Y_pred, Y