from ...decision_tree.py_interface.decision_tree import DecisionTree
from...base.abstract.supervised_model import SupervisedModel
from ...base.mixin.regression_mixin import RegressionMixin
import numpy as np

class RandomForestRegression(SupervisedModel, RegressionMixin):
    """
    A Random Forest Regressor that utilizes an ensemble of decision trees 
    to perform regression tasks by averaging predictions from multiple trees.

    This model is built using bootstrapped samples of the training data 
    and leverages DecisionTree-based regression.

    Attributes:
        _trees (list): A list of DecisionTree instances used for the ensemble.
    """
    
    def __init__(self, n_trees: int, **kwargs) -> None:
        """
        Initializes the RandomForestRegression model.

        Args:
            n_trees (int): The number of decision trees in the forest.
            **kwargs: Additional keyword arguments to configure the decision trees.
        """
        super().__init__(**kwargs)
        
        # Initialize the decision trees
        self._trees = [DecisionTree(**kwargs) for _ in range(n_trees)]
    
    def fit(self, X, y):
        """
        Trains the random forest on the provided dataset by fitting each decision tree
        using bootstrapped samples from the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        for tree in self._trees:
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
    
    def predict(self, X):
        """
        Predicts the target values for the given input data by averaging the predictions
        from all decision trees in the ensemble.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values of shape (n_samples,).
        """
        # Get the predictions from each tree
        predictions = np.array([tree.predict(X) for tree in self._trees], dtype=int)
        
        # Return the mean prediction for each sample
        return np.mean(predictions, axis=0)