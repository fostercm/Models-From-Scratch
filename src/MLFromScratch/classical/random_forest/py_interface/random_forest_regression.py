from ...decision_tree.py_interface.decision_tree import DecisionTree
from...base.abstract.supervised_model import SupervisedModel
from ...base.mixin.regression_mixin import RegressionMixin
import numpy as np

class RandomForestRegression(SupervisedModel, RegressionMixin):
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
        
        # Return the mean prediction for each sample
        return np.mean(predictions, axis=0)