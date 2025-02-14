import numpy as np
np.seterr(all='ignore')
from .logistic_regression_base import LogisticRegressionBase
from ....utils.python_utils.loss_functions import crossEntropy
from ....utils.python_utils.activation_functions import sigmoid, softmax

class LogisticRegressionPython(LogisticRegressionBase):
    """
    A simple implementation of Logistic Regression using gradient descent.

    This class implements logistic regression using gradient descent to compute the model parameters.
    It inherits from the LogisticRegressionBase class and provides the basic functionality for
    fitting the model, making predictions, and calculating the cost (Cross Entropy Loss).
    """
    
    def __init__(self) -> None:
        """
        Initialize the LogisticRegressionPython model.

        This method initializes the model by calling the parent class constructor
        and setting up the necessary model parameters.
        """
        super().__init__()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the Logistic Regression model to the training data using gradient descent.

        This method validates the input data and computes the model parameters (beta)
        using gradient descent: β = β - lr * ∇J(β), where J(β) is the cost function and lr is the learning rate
        It also checks for convergence by comparing the norm of the gradient with a tolerance value
        If the gradient never falls below the tolerance, the model stops after max_iters iterations

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            Y (np.ndarray): Target matrix of shape (n_samples, 1).

        Raises:
            ValueError: If the dimensions of X and Y are not compatible.
        """
        # Validate the input arrays and pad X with ones
        X, Y = super().fit(X, Y)
        
        # Initialize the model parameters
        if self.params['num_classes'] == 2:
            # Initialize beta for binary classification
            self.params['beta'] = np.zeros((X.shape[1], 1), dtype=np.float32)
        else:
            # Initialize beta for multi-class classification
            self.params['beta'] = np.zeros((X.shape[1], self.params['num_classes']), dtype=np.float32)
        
        # Gradient Descent
        for _ in range(self.iterations):
            # Compute the predicted values
            Y_pred = self.predict(X, pad=False)
            
            # Compute the gradient of the cost function
            gradient = X.T @ (Y_pred - Y) / X.shape[0]
            
            # Update the model parameters
            self.params['beta'] -= self.learning_rate * gradient

    def predict(self, X: np.ndarray, pad: bool=True) -> np.ndarray:
        """
        Predict target values using the learned model.

        This method makes predictions by computing the dot product of the feature matrix (X)
        and the learned model parameters (beta), it then passes these values through a sigmoid or softmax function.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values of shape (n_samples, n_targets).

        Raises:
            ValueError: If the model is not fitted before calling predict.
        """
        # Validate the input array and pad X with ones
        X = super().predict(X, pad)
        
        if self.params['num_classes'] == 2:
            # Return the predicted values for binary classification
            return sigmoid(X @ self.params['beta'])
        else:
            # Return the predicted values for multi-class classification
            return softmax(X @ self.params['beta'])

    def cost(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the Cross Entropy (CE) cost between the predicted and true target values.

        This method calculates the CE using the formula: 
        CE = -1/n_samples * ΣΣ(Y * log(Y_pred))

        Args:
            Y_pred (np.ndarray): Predicted target values of shape (n_samples, n_targets).
            Y (np.ndarray): True target values of shape (n_samples, n_targets).

        Returns:
            float: The Mean Squared Error (MSE) between the predicted and true values.

        Raises:
            ValueError: If the dimensions of Y_pred and Y do not match.
        """
        Y_pred, Y = super().cost(Y_pred, Y)
        return crossEntropy(Y_pred, Y)