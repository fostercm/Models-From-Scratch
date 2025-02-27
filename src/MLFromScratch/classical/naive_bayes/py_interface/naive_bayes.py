import numpy as np
from ...base.abstract.supervised_model import SupervisedModel
from ...base.mixin.classification_mixin import ClassificationMixin
from ....utils.python_utils.activation_functions import softmax
from typing import Literal


class NaiveBayes(SupervisedModel, ClassificationMixin):

    def __init__(self, language: Literal['Python', 'C', 'CUDA'] = 'Python', smoothing: float=1e-9, alpha: float=1.0, variant: Literal["bernoulli", "multinomial", "gaussian"] = "gaussian") -> None:
        # Validate the language parameter
        self._validate_language(language)
        
        # Check smoothing is positive
        if not isinstance(smoothing, float) or smoothing <= 0:
            raise ValueError("Smoothing must be a positive float")
    
        # Check alpha is positive
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("Alpha must be a positive float")
        
        # Get the type of the model
        if variant not in ["bernoulli", "multinomial", "gaussian"]:
            raise ValueError("Invalid variant, must be 'bernoulli', 'multinomial' or 'gaussian'")
        
        # Initialize the model parameters
        super().__init__(language=language, smoothing=smoothing, alpha=alpha, variant=variant)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Validate the input arrays
        X = self._validateInput(X)
        y = self._validateTarget(y)
        self._validateInputPair(X, y)
        
        # Get the number of features
        n_features = X.shape[1]
        
        # Get the number of classes
        unique_y = np.unique(y)
        n_classes = len(unique_y)
        
        # Get the prior probabilities
        self.params["priors"] = np.bincount(y, minlength=n_classes) / len(y)

        # Calculate the parameters
        if self.params["variant"] == "bernoulli":
            self.params["likelihoods"] = np.zeros((n_features, n_classes))
            
            for i in range(n_classes):
                # Get the data for the current class
                X_class = X[y == i]

                # Calculate the likelihoods
                likelihoods = (X_class.sum(axis=0) + self.params["alpha"]) / (X_class.shape[0] + self.params["alpha"] * 2)

                # Store the likelihoods
                self.params["likelihoods"][:, i] = likelihoods

        elif self.params["variant"] == "multinomial":
            self.params["likelihoods"] = np.zeros((n_features, n_classes))
            
            for i in range(n_classes):
                # Get the data for the current class
                X_class = X[y == i]

                # Calculate the likelihoods
                likelihoods = (X_class.sum(axis=0) + self.params["alpha"]) / (X_class.sum() + self.params["alpha"] * n_features)

                # Store the likelihoods
                self.params["likelihoods"][:, i] = likelihoods

        elif self.params["variant"] == "gaussian":
            self.params["means"] = np.zeros((n_features, n_classes))
            self.params["variances"] = np.zeros((n_features, n_classes))
            
            for i in range(n_classes):
                # Get the data for the current class
                X_class = X[y == i]

                # Calculate the means and variances
                means = X_class.mean(axis=0)
                variances = X_class.var(axis=0) + self.params["smoothing"]

                # Store the means and variances
                self.params["means"][:, i] = means
                self.params["variances"][:, i] = variances
        
        else:
            raise ValueError("Invalid variant, must be 'bernoulli', 'multinomial' or 'gaussian'")
        
        # Set model as fitted
        self.params["fitted"] = True

    def predict(self, X: np.ndarray, output: Literal["class", "probability"] = "class") -> np.ndarray:
        
        # Check if the model is fitted
        if not self.params["fitted"]:
            raise ValueError("Model must be fitted before making predictions")
        
        # Validate the input array
        X = self._validateInput(X)

        # Check if the input dimensions match the model parameters
        if self.params["variant"] in ["bernoulli", "multinomial"]:
            if X.shape[1] != self.params["likelihoods"].shape[0]:
                raise ValueError("The number of columns in X must be equal to the number of features in the model")
        
        elif self.params["variant"] == "gaussian":
            if X.shape[1] != self.params["means"].shape[0]:
                raise ValueError("The number of columns in X must be equal to the number of features in the model")
        
        else:
            raise ValueError("Invalid variant, must be 'bernoulli', 'multinomial' or 'gaussian'")
        
        # Get log priors
        log_priors = np.log(self.params["priors"])

        if self.params["variant"] == "bernoulli":
            # Calculate the log likelihoods and complements
            log_likelihoods = np.log(self.params["likelihoods"])
            log_complements = np.log(1 - self.params["likelihoods"])
            
            # Calculate the class probabilities
            predictions = X @ log_likelihoods + (1 - X) @ log_complements + log_priors

        elif self.params["variant"] == "multinomial":
            # Calculate the log likelihoods
            log_likelihoods = np.log(self.params["likelihoods"])
            
            # Calculate the class probabilities
            predictions = X @ log_likelihoods + log_priors

        else:
            # Calculate the log probability densities
            log_norm_factors = -np.log(self.params["variances"] * np.sqrt(2 * np.pi))
            log_probability_densities = np.sum(
                log_norm_factors - 0.5 * ((X[:, :, None] - self.params["means"]) / self.params["variances"])** 2,
                axis=1
                )
            
            # Calculate the class probabilities
            predictions = log_probability_densities + log_priors

        # Return the class with the highest probability
        if output == "class":
            return np.argmax(predictions, axis=1)
        elif output == "probability":
            return softmax(predictions)
        else:
            raise ValueError("Invalid output type, must be 'class' or 'probability'")

    def _validateInput(self, X: np.ndarray, output: Literal["class", "probability"] = "class") -> np.ndarray:
        # Validate the input array
        super()._validateInput(X)
        
        if not isinstance(output, str) or output not in ["class", "probability"]:
            raise ValueError("Invalid output type, must be 'class' or 'probability'")
        
        # Perform additional checks based on the type of the model
        if self.params['variant'] == 'bernoulli':
            # Check if the input array is binary
            if not np.array_equal(np.unique(X), np.array([0, 1])):
                raise ValueError("Input array must be binary")
        
        elif self.params['variant'] == 'multinomial':
            # Check if the input array is non-negative
            if np.any(X < 0):
                raise ValueError("Input array must be non-negative")
            
            # Check if the input array is integer
            if not np.issubdtype(X.dtype, np.integer):
                raise ValueError("Input array must be integer")
        
        elif self.params['variant'] == 'gaussian':
            # Check if the input array is numeric
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError("Input array must be numeric")
        
        else:
            raise ValueError("Invalid type, must be 'bernoulli', 'multinomial' or 'gaussian")

        return X