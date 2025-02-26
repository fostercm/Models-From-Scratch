from .k_nearest_neighbors_classification import KNNClassification
from .k_nearest_neighbors_regression import KNNRegression
from typing import Literal

class KNN:
    def __init__(self, task: str = 'classification', language: Literal['Python','C','CUDA'] = 'Python'):
        # Check language is valid
        if language not in ['Python', 'C', 'CUDA']:
            raise ValueError("Invalid language, must be 'Python', 'C' or 'CUDA'")
        
        # Fetch the appropriate model based on the task
        if task == 'classification':
            self._model = KNNClassification(language=language)
        elif task == 'regression':
            self._model = KNNRegression(language=language)
        else:
            raise ValueError("Invalid task, must be 'classification' or 'regression'")

    def __getattr__(self, attr):
        """Delegate method calls to the underlying model."""
        return getattr(self._model, attr)