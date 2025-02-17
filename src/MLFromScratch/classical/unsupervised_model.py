import numpy as np


class UnsupervisedModel:
    """
    A base class for classical machine learning models.

    This class provides common functionality for validating input data,
    storing model parameters, and providing placeholder methods for model
    fitting, prediction, and cost calculation. Subclasses should implement
    the fit, predict, and cost methods to define the specific behavior of
    different classical models (e.g., Linear Regression, Logistic Regression).

    Attributes:
        params (dict): A dictionary that holds the parameters of the model,
                       such as coefficients or weights after fitting.

    Methods:
        fit(X, Y): Fit the model to the training data (to be implemented in subclass).
        predict(X): Predict the target variable for the given data (to be implemented in subclass).
        cost(Y_pred, Y): Calculate the cost (e.g., loss function) for the model's predictions.
        get_params(): Get the parameters of the model.
        load_params(params): Load the model parameters.
    """

    def _validate_input(self, array1: np.ndarray, array2: np.ndarray = None) -> None:
        """
        Validate the input numpy arrays.

        Args:
            array1 (np.ndarray): Input numpy array.
            array2 (np.ndarray, optional): Second input numpy array, if applicable.

        Returns:
            tuple: A tuple containing the validated arrays (array1, array2).

        Raises:
            TypeError: If input arrays are not numpy arrays.
            ValueError: If input arrays are not 2D or have mismatched shapes or empty arrays.
        """
        if not isinstance(array1, np.ndarray):
            raise TypeError("Array 1 must be a numpy array")

        if len(array1.shape) != 2:
            raise ValueError("Array 1 must be a 2D array")

        if array1.size == 0:
            raise ValueError("Array 1 must not be empty")

        # Make sure array1 is a float32 array
        array1 = array1.astype(np.float32)

        if array2 is not None:
            if not isinstance(array2, np.ndarray):
                raise TypeError("Array 2 must be a numpy array")

            if len(array2.shape) != 2:
                raise ValueError("Array 2 must be a 2D array")

            if array2.size == 0:
                raise ValueError("Array 2 must not be empty")

            # Check if the number of rows in X and Y are equal
            if array1.shape[0] != array2.shape[0]:
                raise ValueError("The number of rows in X and Y must be equal")

            # Make sure array2 is a float32 array
            array2 = array2.astype(np.float32)

        return array1, array2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target variable for the given data (to be implemented in subclass)."""
        pass