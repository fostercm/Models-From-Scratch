from .decision_tree_classification import DecisionTreeClassification
from .decision_tree_regression import DecisionTreeRegression
from typing import Literal

class DecisionTree:
    def __init__(self, task: str = 'classification', language: Literal['Python','C','CUDA'] = 'Python', max_depth: int = 100, min_samples_split: int = 2, tolerance: float = 1e-4): 
        # Check language is valid
        if language not in ['Python', 'C', 'CUDA']:
            raise ValueError("Invalid language, must be 'Python', 'C' or 'CUDA'")
        
        # Fetch the appropriate model based on the task
        if task == 'classification':
            self._model = DecisionTreeClassification(language=language, max_depth=max_depth, min_samples_split=min_samples_split, tolerance=tolerance)
        elif task == 'regression':
            self._model = DecisionTreeRegression(language=language, max_depth=max_depth, min_samples_split=min_samples_split, tolerance=tolerance)
        else:
            raise ValueError("Invalid task, must be 'classification' or 'regression'")

    def __getattr__(self, attr):
        """Delegate method calls to the underlying model."""
        return getattr(self._model, attr)