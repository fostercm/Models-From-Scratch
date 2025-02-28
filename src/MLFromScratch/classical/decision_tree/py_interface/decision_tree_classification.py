from ...base.model.decision_tree_model_base import DecisionTreeModelBase
from ...base.mixin.classification_mixin import ClassificationMixin
from typing import Literal
import numpy as np


class DecisionTreeClassification(DecisionTreeModelBase, ClassificationMixin):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Validate the target vector
        X = self._validateInput(X)
        y = self._validateTarget(y)
        self._validateInputPair(X, y)
        
        # Set the number of features and outputs
        self.params["num_features"] = X.shape[1]
        self.params["num_outputs"] = 1
        self.params["num_classes"] = len(np.unique(y))
        
        # Fit the model to the data
        self._grow_tree(X, y, self.root, 0)
        
        # Set the model as fitted
        self.params["fitted"] = True
    
    def _split_node(self, X: np.ndarray, y: np.ndarray, node: DecisionTreeModelBase.TreeNode, depth: int) -> None:
        # Check if the node is a leaf node
        if depth >= self.params["max_depth"] or len(y) < self.params["min_samples_split"] or len(np.unique(y)) == 1:
            node.value = np.argmax(np.bincount(y))
            return
        
        # Find the best split
        best_gini = np.inf
        best_feature_index = None
        best_threshold = None
        
        sorted_indices = np.argsort(X, axis=0)
        
        for feature_index in range(X.shape[1]):
            # Get the sorted values of the feature
            sorted_idx = sorted_indices[:, feature_index]
            sorted_y = y[sorted_idx]
            sorted_X = X[sorted_idx, feature_index]
            
            # Track the class counts
            left_counts = np.zeros(self.params["num_classes"])
            left_counts[sorted_y[0]] = 1
            right_counts = np.bincount(y[1:], minlength=self.params["num_classes"])
            
            for i in range(1, len(y)):
                # Calculate the Gini impurity
                left_gini = 1 - np.sum((left_counts / i) ** 2)
                right_gini = 1 - np.sum((right_counts / (len(y) - i)) ** 2)
                gini = left_gini + right_gini
                
                # Update the best split
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = (sorted_X[i - 1] + sorted_X[i]) / 2
                
                # Update the class counts
                left_counts[sorted_y[i]] += 1
                right_counts[sorted_y[i]] -= 1
        
        # Set the node values
        node.feature_index = best_feature_index
        node.threshold = best_threshold
        
        # If the best split does not reduce the Gini impurity, set the node as a leaf node
        if self._gini_impurity(y) - best_gini < self.params["tolerance"]:
            node.value = np.argmax(np.bincount(y))
        
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Check if the model is fitted
        if self.params["fitted"] is False:
            raise ValueError("Model is not fitted")
        
        # Validate the input array
        X = super()._validateInput(X)
        
        # Check if the dimensions of the input array match the stored feature matrix
        if X.shape[1] != self.params["num_features"]:
            raise ValueError("The number of columns in X must be equal to the number of features in the model")
        
        return self._predict(X).flatten()
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        # Calculate the class probabilities
        class_probabilities = np.bincount(y) / len(y)
        
        # Calculate the Gini impurity
        gini = 1.0 - np.sum(class_probabilities ** 2)
        
        return gini