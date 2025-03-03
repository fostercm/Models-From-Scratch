from abc import ABC
from typing import Literal

import numpy as np


class BaseModel(ABC):
    """Abstract base class for machine learning models.

    This class provides a foundation for implementing various machine learning models,
    including parameter management, input validation, and compatibility checks.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the model with given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments representing model parameters.
        """
        # Initialize the parameters
        self.params = kwargs
        
    def getParams(self) -> dict:
        """Returns the model parameters.

        Returns:
            dict: A dictionary containing the model parameters.
        """
        # Return the model parameters
        return self.params
    
    def loadParams(self, params: dict) -> None:
        """Loads model parameters from a dictionary.

        Args:
            params (dict): A dictionary containing parameter names and values.

        Raises:
            ValueError: If a parameter value is a non-numeric list or tuple that cannot
                be converted to a NumPy array.
        """
        # Load the model parameters
        for key, value in params.items():
            
            # Convert list or tuple to numpy array if possible
            if isinstance(value, (list, tuple)):
                try:
                    self.params[key] = np.array(value, dtype=np.float32)
                except ValueError:
                    raise ValueError(f"Parameter '{key}' contains non-numeric elements and cannot be converted")
                
            # Otherwise, set the parameter directly
            else:
                self.params[key] = value
    
    def _validateInput(self, X: np.ndarray) -> np.ndarray:
        """Validates input data and ensures it meets expected format.

        This method checks that the input data is a 2D numerical NumPy array
        with no NaN or infinite values and converts it to a C-contiguous float32 array.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: A validated and potentially converted NumPy array.

        Raises:
            ValueError: If the input is not a NumPy array.
            ValueError: If the input is not a 2D array.
            ValueError: If the input has no samples or features.
            ValueError: If the input contains non-numeric values.
            ValueError: If the input contains NaN or infinite values.
        """
        # Check if the input data is a numpy array
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be of type numpy.ndarray")
        
        # Check if the input data is 2D
        if len(X.shape) != 2:
            raise ValueError("Input data must be a 2D array")
        
        # Check for at least one sample in the input data
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Input data must have at least 1 sample and 1 feature")
        
        # Check if the input data is numerical
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("Input data must be numerical")
        
        # Check for NaN or infinite values in the input data
        if not np.isfinite(X).all():
            raise ValueError("Input data contains NaN or infinite values")
        
        # Ensure the input data is C-contiguous and float32
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X, dtype=np.float32)
        else:
            X = X.astype(np.float32, copy=False)
        
        return X
    
    def _validateInputPair(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Validates that input data and target labels are correctly formatted.

        Args:
            X (np.ndarray): The input feature matrix.
            Y (np.ndarray): The target labels.

        Raises:
            ValueError: If either input is not a NumPy array.
            ValueError: If the number of samples in `X` and `Y` do not match.
        """
        # Check both input data and targets are numpy arrays
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise ValueError("Input data and target data must be of type numpy.ndarray")
        
        # Validate the number of samples
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Input data and target data must have the same number of samples")
    
    def _validate_language(self, language: Literal['Python', 'C', 'CUDA']) -> None:
        """Validates that the selected language is one of the supported options.

        Args:
            language (Literal["Python", "C", "CUDA"]): The programming language.

        Raises:
            ValueError: If the provided language is not one of "Python", "C", or "CUDA".
        """
        # Validate the language parameter
        if language not in ['Python', 'C', 'CUDA']:
            raise ValueError("Language must be one of 'Python', 'C', or 'CUDA'")