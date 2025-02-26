import numpy as np
from abc import ABC

class BaseModel(ABC):
    """
    An abstract base class for machine learning models providing common utility functions

    This class is designed to be inherited by specific models that implement either:
    - `fit` and `predict` methods (e.g., supervised regression/classification models)
    - `transform` methods (e.g., unsupervised transformation models)

    Methods:
    --------
        `__init__() -> None`:
            Initializes the model parameter dictionary
            
        `_validateInput(X: np.ndarray) -> np.ndarray`:
            Validates and preprocesses input data, ensuring it meets numerical, dimensional, and memory layout requirements
    """
    def __init__(self, **kwargs) -> None:
        """
        Initializes the model parameter dictionary `params`
        """
        self.params = {}
        for key, value in kwargs.items():
            self.params[key] = value
        
    def getParams(self) -> dict:
        """
        Retrieve the model's parameters.

        Returns
        --------
            dict:
                A dictionary containing the model's parameters, where values can be
                numbers, strings, or NumPy arrays
        """
        return self.params
    
    def loadParams(self, params: dict) -> None:
        """
        Load parameters into the model, converting lists or sequences into NumPy arrays

        This method updates the model's parameters from a given dictionary.
        If a parameter is a list or a sequence (excluding strings), it is 
        converted to a NumPy array of `float32` dtype. Other data types remain unchanged

        Parameters
        -----------
        params : dict
            A dictionary where keys are parameter names (str) and values 
            can be numbers, strings, or NumPy arrays

        Raises
        -------
        ValueError
            If a list contains non-numeric elements that cannot be converted to `float32`.
        """
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
        """
        Validates and preprocesses input data for model training or inference.

        This method ensures that the input:
        - Is a NumPy array (`numpy.ndarray`)
        - Is a two-dimensional matrix
        - Has at least one sample and one feature
        - Is numerical
        - Does not contain NaN or infinite values
        - Has `float32` data type
        - Is C-contiguous

        Parameters
        ----------
        X : np.ndarray
            The input data to be validated, expected to be a 2D NumPy array

        Returns
        -------
        np.ndarray
            The validated and C-contiguous numpy array with `float32` dtype

        Raises
        ------
        ValueError:
            - If `X` is not a NumPy array
            - If `X` is not a 2D array
            - If `X` is empty
            - If `X` contains NaN or infinite values
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
        """
        Validates that the input data and targets are a compatible pair (same number of samples)
        
        This method should be called after independent validation of the input data and targets,
        but bare minimum checks are performed to ensure no unexpected errors
        
        Parameters
        ----------
            X : np.ndarray
                The input data to be validated, expected to be a 2D NumPy array
            Y : np.ndarray
                The target labels to be validated, expected to be a NumPy array

        Returns
        -------
            None

        Raises
        -------
            ValueError:
                - If `X` and `Y` do not have the same number of samples
        """
        # Check both input data and targets are numpy arrays
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise ValueError("Input data and target data must be of type numpy.ndarray")
        
        # Validate the number of samples
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Input data and target data must have the same number of samples")
        
        return X, Y