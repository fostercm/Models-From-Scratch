from .random_forest_classification import RandomForestClassification
from .random_forest_regression import RandomForestRegression

from typing import Literal

class RandomForest:
    """
    A flexible Random Forest model that can perform either classification or regression 
    using an ensemble of decision trees.

    This class acts as a wrapper around `RandomForestClassification` and 
    `RandomForestRegression`, delegating method calls to the appropriate model 
    based on the specified task.

    Attributes:
        _model: An instance of either `RandomForestClassification` or `RandomForestRegression`, 
                depending on the selected task.
    """
    
    def __init__(self, task: Literal['classification', 'regression'] = 'classification', language: Literal['Python','C','CUDA'] = 'Python', n_trees = 5, max_depth: int = 100, min_samples_split: int = 2, tolerance: float = 1e-4,  feature_subset: Literal['all', 'sqrt', 'log2'] = 'sqrt'): 
        """
        Initializes the RandomForest model based on the selected task and parameters.

        Args:
            task (Literal['classification', 'regression'], optional): 
                The type of task to perform. Defaults to 'classification'.
            language (Literal['Python', 'C', 'CUDA'], optional): 
                The implementation language. Defaults to 'Python'.
            n_trees (int, optional): 
                The number of decision trees in the ensemble. Defaults to 5.
            max_depth (int, optional): 
                The maximum depth of each tree. Defaults to 100.
            min_samples_split (int, optional): 
                The minimum number of samples required to split a node. Defaults to 2.
            tolerance (float, optional): 
                The minimum improvement required to continue splitting. Defaults to 1e-4.
            feature_subset (Literal['all', 'sqrt', 'log2'], optional): 
                The strategy for selecting feature subsets at each split. Defaults to 'sqrt'.

        Raises:
            ValueError: If an invalid language or task is provided.
        """
        # Check language is valid
        if language not in ['Python', 'C', 'CUDA']:
            raise ValueError("Invalid language, must be 'Python', 'C' or 'CUDA'")
        
        # Fetch the appropriate model based on the task
        if task == 'classification':
            self._model = RandomForestClassification(task=task, language=language, n_trees=n_trees, max_depth=max_depth, min_samples_split=min_samples_split, tolerance=tolerance, feature_subset=feature_subset)
        elif task == 'regression':
            self._model = RandomForestRegression(task=task, language=language, n_trees=n_trees, max_depth=max_depth, min_samples_split=min_samples_split, tolerance=tolerance, feature_subset=feature_subset)
        else:
            raise ValueError("Invalid task, must be 'classification'")
    
    def __getattr__(self, attr):
        """
        Delegates method calls to the underlying model (`RandomForestClassification` or `RandomForestRegression`).

        Args:
            attr (str): The attribute or method name being accessed.

        Returns:
            The corresponding attribute or method from the underlying model.
        """
        return getattr(self._model, attr)