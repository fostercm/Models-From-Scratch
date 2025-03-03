from ...decision_tree.py_interface.decision_tree import DecisionTree
from...base.abstract.supervised_model import SupervisedModel
from ...base.mixin.classification_mixin import ClassificationMixin
import numpy as np

class RandomForestClassification(SupervisedModel, ClassificationMixin):
    """
    A Random Forest classifier that builds an ensemble of decision trees for classification tasks.
    
    Attributes:
        _trees (list): A list of DecisionTree instances forming the ensemble.
    """
    
    def __init__(self, n_trees: int, **kwargs) -> None:
        """
        Initializes the Random Forest classifier with a specified number of decision trees.
        
        Args:
            n_trees (int): The number of decision trees in the forest.
            **kwargs: Additional keyword arguments passed to the DecisionTree instances.
        """
        super().__init__(**kwargs)
        
        # Initialize the decision trees
        self._trees = [DecisionTree(**kwargs) for _ in range(n_trees)]
    
    def fit(self, X, y):
        """
        Trains the Random Forest classifier using bootstrapped datasets for each decision tree.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
        """
        for tree in self._trees:
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
    
    def predict(self, X):
        """
        Predicts class labels for the given input data using majority voting.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        # Get the predictions from each tree
        predictions = np.array([tree.predict(X) for tree in self._trees])
        
        # Return the most common prediction for each sample
        predictions = np.array([np.argmax(np.bincount(prediction)) for prediction in predictions.T])
        
        return predictions