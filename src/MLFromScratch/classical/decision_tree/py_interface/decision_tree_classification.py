from ...base.model.decision_tree_model_base import DecisionTreeModelBase
from ...base.mixin.classification_mixin import ClassificationMixin
from typing import Literal
import numpy as np


class DecisionTreeClassification(DecisionTreeModelBase, ClassificationMixin):
    """
    A decision tree classifier that extends both the DecisionTreeModelBase and ClassificationMixin classes.
    This model performs classification tasks by learning decision rules from input data. It is based on 
    the decision tree algorithm, which splits the data at each node according to a feature threshold 
    to minimize impurity, and it supports configurable parameters such as maximum depth and minimum samples per split.

    Attributes:
        params (dict): A dictionary to store model parameters such as max depth, min samples split, and number of features.
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Initializes the DecisionTreeClassification model by calling the constructor of the parent classes.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the parent classes, typically includes hyperparameters 
                      like max depth, min samples split, etc.
        """
        super().__init__(**kwargs)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the decision tree classifier on the input feature matrix X and target labels y.

        This method validates the input, splits the data, and recursively grows the decision tree based on the input data. 
        It stores the model parameters and sets the model as fitted.

        Args:
            X (np.ndarray): The input feature matrix, where each row represents a sample and each column represents a feature.
            y (np.ndarray): The target vector, containing the class labels for each sample.

        Raises:
            ValueError: If the input data or target vector is invalid.
        """
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
        """
        Splits a node of the decision tree based on the best feature and threshold to minimize Gini impurity.

        This method determines the best feature and threshold to split the data at a given node in the decision tree. 
        It recursively grows the tree until the stopping criteria are met.

        Args:
            X (np.ndarray): The input feature matrix, where each row represents a sample and each column represents a feature.
            y (np.ndarray): The target vector, containing the class labels for each sample.
            node (TreeNode): The current node in the decision tree being split.
            depth (int): The current depth of the node in the tree.

        Raises:
            ValueError: If the node is invalid or there is an issue during the split process.
        """
        # Check if the node is a leaf node
        if depth >= self.params["max_depth"] or len(y) < self.params["min_samples_split"] or len(np.unique(y)) == 1:
            node.value = np.argmax(np.bincount(y))
            return
        
        # Find the best split
        best_gini = np.inf
        best_feature_index = None
        best_threshold = None
        
        # Get the feautres to consider
        if self.params["feature_subset"] == "sqrt":
            features = np.random.choice(X.shape[1], int(np.sqrt(X.shape[1])), replace=False)
        elif self.params["feature_subset"] == "log2":
            features = np.random.choice(X.shape[1], int(np.log2(X.shape[1])), replace=False)
        else:
            features = np.arange(X.shape[1])
        
        # Get the sorted indices of the feature matrix
        sorted_indices = np.argsort(X, axis=0)
        
        for feature_index in features:
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
                gini = (i / len(y)) * left_gini + ((len(y) - i) / len(y)) * right_gini
                
                # Update the best split if the Gini impurity is lower and the threshold exists
                if gini < best_gini and sorted_X[i - 1] != sorted_X[i]:
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
        """
        Predicts the class labels for the input feature matrix X using the fitted decision tree model.

        This method traverses the tree to predict the class label for each sample in the input matrix X.

        Args:
            X (np.ndarray): The input feature matrix, where each row represents a sample and each column represents a feature.

        Returns:
            np.ndarray: The predicted class labels for each sample in X.

        Raises:
            ValueError: If the model is not fitted or if the input data is invalid.
        """
        # Calculate the class probabilities
        class_probabilities = np.bincount(y) / len(y)
        
        # Calculate the Gini impurity
        gini = 1.0 - np.sum(class_probabilities ** 2)
        
        return gini