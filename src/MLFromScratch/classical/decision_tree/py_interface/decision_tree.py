from .decision_tree_classification import DecisionTreeClassification
from .decision_tree_regression import DecisionTreeRegression
from typing import Literal

class DecisionTree:
    """
    A wrapper class for decision tree models that can either perform classification or regression tasks.
    Depending on the specified task, this class initializes the appropriate model (DecisionTreeClassification or DecisionTreeRegression)
    and delegates the method calls to the selected model.

    Args:
        task (Literal['classification', 'regression']): The task type the decision tree will perform. 
                                                      Can be 'classification' or 'regression'. Default is 'classification'.
        language (Literal['Python', 'C', 'CUDA']): The programming language for the model's implementation. 
                                                   Can be 'Python', 'C', or 'CUDA'. Default is 'Python'.
        max_depth (int): The maximum depth of the decision tree. Default is 100.
        min_samples_split (int): The minimum number of samples required to split a node. Default is 2.
        tolerance (float): The tolerance for stopping the splitting of nodes based on mean squared error or impurity. Default is 1e-4.
        feature_subset (Literal['all', 'sqrt', 'log2']): The subset of features to consider for each split. 
                                                         Can be 'all', 'sqrt', or 'log2'. Default is 'all'.
    
    Attributes:
        _model (DecisionTreeModel): The decision tree model (either DecisionTreeClassification or DecisionTreeRegression) 
                                     based on the specified task.
    """
    
    def __init__(
        self, 
        task: Literal['classification', 'regression'] = 'classification', 
        language: Literal['Python','C','CUDA'] = 'Python', 
        max_depth: int = 100, 
        min_samples_split: int = 2, 
        tolerance: float = 1e-4, 
        feature_subset: Literal['all', 'sqrt', 'log2'] = 'all'
        ) -> None: 
        """
        Initializes a DecisionTree object that delegates model behavior to either a classification or regression decision tree.

        Args:
            task (Literal['classification', 'regression']): The task type, either 'classification' or 'regression'.
            language (Literal['Python', 'C', 'CUDA']): The implementation language for the model.
            max_depth (int): The maximum depth of the tree.
            min_samples_split (int): The minimum number of samples required to split a node.
            tolerance (float): The tolerance for stopping criterion.
            feature_subset (Literal['all', 'sqrt', 'log2']): The subset of features considered for splitting.

        Raises:
            ValueError: If an invalid language or task is provided.
        """
        # Check language is valid
        if language not in ['Python', 'C', 'CUDA']:
            raise ValueError("Invalid language, must be 'Python', 'C' or 'CUDA'")
        
        # Fetch the appropriate model based on the task
        if task == 'classification':
            self._model = DecisionTreeClassification(task=task, language=language, max_depth=max_depth, min_samples_split=min_samples_split, tolerance=tolerance, feature_subset=feature_subset)
        elif task == 'regression':
            self._model = DecisionTreeRegression(task=task, language=language, max_depth=max_depth, min_samples_split=min_samples_split, tolerance=tolerance, feature_subset=feature_subset)
        else:
            raise ValueError("Invalid task, must be 'classification' or 'regression'")

    def __getattr__(self, attr):
        """
        Delegates method calls to the underlying model (DecisionTreeClassification or DecisionTreeRegression).

        Args:
            attr (str): The name of the method being called.

        Returns:
            The corresponding method from the underlying model.

        Raises:
            AttributeError: If the attribute does not exist in the underlying model.
        """
        return getattr(self._model, attr)