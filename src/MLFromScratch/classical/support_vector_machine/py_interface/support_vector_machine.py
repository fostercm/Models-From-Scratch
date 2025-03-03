from...base.abstract.supervised_model import SupervisedModel
from ...base.mixin.classification_mixin import ClassificationMixin
import numpy as np
from typing import Literal

class SVM(SupervisedModel, ClassificationMixin):
    """
    Support Vector Machine (SVM) implementation supporting classification and regression tasks.

    Attributes:
        task (str): Specifies the type of task ('classification' or 'regression').
        language (str): Specifies the implementation language ('Python', 'C', or 'CUDA').
        kernel (str): Kernel type used in the model ('linear', 'poly', 'rbf').
        degree (int): Degree for polynomial kernel (only used if kernel='poly').
        offset (float): Offset for polynomial kernel.
        gamma (float): Kernel coefficient for 'rbf' kernel.
        C (float): Regularization parameter.
        iterations (int): Number of iterations for optimization.
        tolerance (float): Convergence tolerance.
        alpha_threshold (float): Threshold for support vectors.
    """
    
    def __init__(
        self, 
        task: Literal['classification', 'regression'] = 'classification', 
        language: Literal['Python','C','CUDA'] = 'Python',
        kernel: Literal['linear', 'poly', 'rbf'] = 'linear',
        degree: int = 3,
        offset: float = 0.0,
        gamma: float = 1.0,
        C: float = 1.0,
        iterations: int = 1000,
        tolerance: float = 1e-3,
        alpha_threshold: float = 0.05
        ) -> None: 
        """
        Initializes an SVM model with the specified hyperparameters.

        Args:
            task (str): The type of task ('classification' or 'regression').
            language (str): The implementation language ('Python', 'C', or 'CUDA').
            kernel (str): Kernel type ('linear', 'poly', or 'rbf').
            degree (int): Polynomial kernel degree (used when kernel='poly').
            offset (float): Offset term for the polynomial kernel.
            gamma (float): Kernel coefficient for 'rbf' kernel.
            C (float): Regularization parameter.
            iterations (int): Number of optimization iterations.
            tolerance (float): Convergence tolerance for stopping criteria.
            alpha_threshold (float): Threshold to determine support vectors.
        """
        # Check language is valid
        if language not in ['Python', 'C', 'CUDA']:
            raise ValueError("Invalid language, must be 'Python', 'C' or 'CUDA'")
        
        # Check kernel is valid
        if kernel not in ['linear', 'poly', 'rbf']:
            raise ValueError("Invalid kernel, must be 'linear', 'poly' or 'rbf'")
        
        if kernel == 'poly' and degree < 1:
            raise ValueError("Invalid degree, must be greater than 0")
        
        if kernel == 'rbf' and gamma < 0:
            raise ValueError("Invalid gamma, must be greater than 0")
        
        kwargs = {
            'language': language,
            'kernel': kernel,
            'degree': degree,
            'offset': offset,
            'gamma': gamma,
            'C': C,
            'iterations': iterations,
            'tolerance': tolerance,
            'alpha_threshold': alpha_threshold
        }
        
        super().__init__(**kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the SVM model to the provided training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Raises:
            ValueError: If target values are not binary for classification.
        """
        # Validate the input data
        X = self._validateInput(X)
        y = self._validateTarget(y)
        self._validateInputPair(X, y)
        
        # Turn the target values from 0 to -1
        y = np.where(y == 0, -1, 1)
        
        # Initialize parameters
        alphas = np.zeros(X.shape[0])
        bias = 0
        tolerance = self.params["tolerance"]
        indices = np.arange(X.shape[0])
        
        # Compute the kernel matrix
        K = self._compute_kernel_matrix(X)
        
        # Begin the optimization
        for _ in range(self.params["iterations"]):
            
            # Keep track of the number of alphas that have been changed in the current iteration
            changed = False
            
            # Iterate over all the samples
            for i in range(X.shape[0]):
                
                # Compute the error for the current sample
                prediction_i = (alphas * y) @ K[:,i] + bias
                error_i = prediction_i - y[i]
                
                # Check if the current sample violates the KKT conditions
                if (y[i] * error_i < -tolerance and alphas[i] < self.params["C"]) or (y[i] * error_i > tolerance and alphas[i] > 0):
                    
                    # Select a random sample j that is not equal to i
                    j = np.random.choice(np.delete(indices, i))
                    
                    # Compute the error for the sample j
                    prediction_j = (alphas * y) @ K[:,j] + bias
                    error_j = prediction_j - y[j]
                    
                    # Compute the bounds for alpha_i and alpha_j
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.params["C"], self.params["C"] + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.params["C"])
                        H = min(self.params["C"], alphas[i] + alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute the eta value
                    eta = 2 * K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0:
                        continue
                    
                    # Save the old alphas
                    alpha_i_old, alpha_j_old = alphas[i], alphas[j]
                    
                    # Update alpha_j
                    alphas[j] -= y[j] * (error_i - error_j) / eta
                    alphas[j] = np.clip(alphas[j], L, H)
                    
                    # Check if alpha_j has changed significantly
                    if np.abs(alphas[j] - alpha_j_old) < tolerance:
                        alphas[j] = alpha_j_old
                        continue
                    
                    # Update alpha_i
                    alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])
                    
                    # Compute the bias term
                    b1 = bias - error_i - y[i] * (alphas[i] - alpha_i_old) * K[i,i] - y[j] * (alphas[j] - alpha_j_old) * K[i,j]
                    b2 = bias - error_j - y[i] * (alphas[i] - alpha_i_old) * K[i,j] - y[j] * (alphas[j] - alpha_j_old) * K[j,j]
                    
                    # Update the bias term
                    if 0 < alphas[i] < self.params["C"]:
                        bias = b1
                    elif 0 < alphas[j] < self.params["C"]:
                        bias = b2
                    else:
                        bias = (b1 + b2) / 2
                    
                    # Note that alphas have changed
                    changed = True
            
            # Check if no alphas have changed
            if not changed:
                break
        
        # Save the support vectors and their corresponding labels
        support_vector_indices = alphas > self.params["alpha_threshold"]
        self.params["alphas"] = alphas[support_vector_indices]
        self.params["support_vectors"] = X[support_vector_indices]
        self.params["support_vector_labels"] = y[support_vector_indices]
        
        # Save the bias
        self.params["bias"] = bias
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for given input samples.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        # Validate the input data
        X = self._validateInput(X)
        
        # Initialize the predictions
        predictions = np.zeros(X.shape[0])
        
        # Compute the classes
        for i in range(X.shape[0]):
            prediction = 0
            for alpha, support_vector, support_vector_label in zip(self.params["alphas"], self.params["support_vectors"], self.params["support_vector_labels"]):
                if self.params["kernel"] == "linear":
                    prediction += alpha * support_vector_label * np.dot(X[i], support_vector)
                elif self.params["kernel"] == "poly":
                    prediction += alpha * support_vector_label * (np.dot(X[i], support_vector) + self.params["offset"]) ** self.params["degree"]
                elif self.params["kernel"] == "rbf":
                    prediction += alpha * support_vector_label * np.exp(-self.params["gamma"] * np.linalg.norm(X[i] - support_vector) ** 2)
            
            if prediction + self.params["bias"] > 0:
                prediction = 1
            else:
                prediction = 0
            
            predictions[i] = prediction
        
        return predictions
    
    def _validateTarget(self, y: np.ndarray):
        """
        Validates the target values for binary classification.

        Ensures that the target values are binary (i.e., only two unique values). 
        Raises a ValueError if the target is not binary.

        Args:
            y (np.ndarray): The target values.

        Returns:
            np.ndarray: The validated target values.
        """
        # Validate the target values
        y = super()._validateTarget(y)
        
        # Check if the target values are binary
        if len(np.unique(y)) != 2:
            raise ValueError("Support Vector Classification only supports binary classification")
        
        return y
    
    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the kernel matrix based on the specified kernel type.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Computed kernel matrix of shape (n_samples, n_samples).
        """
        # Compute the kernel matrix depending on the kernel type
        if self.params["kernel"] == "linear":
            return np.dot(X, X.T)
        elif self.params["kernel"] == "poly":
            return (np.dot(X, X.T) + self.params["offset"]) ** self.params["degree"]
        elif self.params["kernel"] == "rbf":
            return np.exp(-self.params["gamma"] * np.linalg.norm(X[:, None] - X[None, :], axis=-1) ** 2)
        else:
            raise ValueError("Invalid kernel type")