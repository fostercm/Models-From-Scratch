from .k_nearest_neighbors_classification import KNNClassification
from .k_nearest_neighbors_regression import KNNRegression
from typing import Literal

class KNN:
    """
    K-Nearest Neighbors (KNN) model selector for classification or regression tasks.

    This class provides a unified interface to either KNN for classification or KNN for regression. 
    It initializes the appropriate model based on the task specified and delegates method calls to the 
    corresponding model (either KNNClassification or KNNRegression). 

    Attributes:
        _model (Union[KNNClassification, KNNRegression]): The selected KNN model (classification or regression).
    """
    
    def __init__(self, task: str = 'classification', language: Literal['Python','C','CUDA'] = 'Python'):
        """
        Initialize the KNN model based on the specified task (classification or regression).

        Args:
            task (str, optional): The task type, either 'classification' or 'regression'. Default is 'classification'.
            language (Literal['Python', 'C', 'CUDA'], optional): The language for the model implementation. Default is 'Python'.

        Raises:
            ValueError: If the provided task is not 'classification' or 'regression'.
        """
        # Fetch the appropriate model based on the task
        if task == 'classification':
            self._model = KNNClassification(language=language)
        elif task == 'regression':
            self._model = KNNRegression(language=language)
        else:
            raise ValueError("Invalid task, must be 'classification' or 'regression'")

    def __getattr__(self, attr):
        """
        Delegate method calls to the underlying KNN model (either classification or regression).

        Args:
            attr (str): The name of the method being accessed.

        Returns:
            The corresponding method from the selected KNN model.

        Raises:
            AttributeError: If the requested method does not exist in the model.
        """
        return getattr(self._model, attr)