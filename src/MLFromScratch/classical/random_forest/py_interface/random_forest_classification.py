from ...decision_tree.py_interface.decision_tree import DecisionTree
from...base.abstract.supervised_model import SupervisedModel
from ...base.mixin.classification_mixin import ClassificationMixin
import numpy as np

class RandomForestClassification(SupervisedModel, ClassificationMixin):
    def __init__(self, n_trees: int, **kwargs) -> None:
        super().__init__(**kwargs)
        
        # Initialize the decision trees
        self._trees = [DecisionTree(**kwargs) for _ in range(n_trees)]
    
    def fit(self, X, y):
        for tree in self._trees:
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
    
    def predict(self, X):
        # Get the predictions from each tree
        predictions = np.array([tree.predict(X) for tree in self._trees], dtype=int)
        
        # Return the most common prediction for each sample
        predictions = np.array([np.argmax(np.bincount(prediction)) for prediction in predictions.T])
        
        return predictions