from ...base.model.decision_tree_model_base import DecisionTreeModelBase
from ...base.mixin.regression_mixin import RegressionMixin
from typing import Literal
import numpy as np


class DecisionTreeRegression(DecisionTreeModelBase, RegressionMixin):
    """
    A decision tree regressor that fits a model to the provided data and predicts continuous values.
    Inherits from DecisionTreeModelBase and RegressionMixin to implement regression functionality.
    The model splits nodes based on minimizing the mean squared error (MSE) criterion.

    Attributes:
        params (dict): A dictionary to store model parameters including feature counts, max depth, etc.
        root (DecisionTreeModelBase.TreeNode): The root node of the decision tree.
    """
    
    def __init__(self,**kwargs) -> None:
        """
        Initializes the DecisionTreeRegression model with specified parameters.

        Args:
            **kwargs: Additional parameters passed to the base model classes.
        """
        super().__init__(**kwargs)
        
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the decision tree model to the provided data by recursively splitting nodes
        based on the feature values and minimizing mean squared error.

        Args:
            X (np.ndarray): The input feature matrix (n_samples x n_features).
            Y (np.ndarray): The target value matrix (n_samples x n_outputs).
        
        Raises:
            ValueError: If the dimensions of X and Y are incompatible.
        """
        # Validate the target vector
        X = self._validateInput(X)
        Y = self._validateTarget(Y)
        self._validateInputPair(X, Y)
        
        # Set the number of features and outputs
        self.params["num_features"] = X.shape[1]
        self.params["num_outputs"] = Y.shape[1]
        
        # Fit the model to the data
        self._grow_tree(X, Y, self.root, 0)
        
        # Set the model as fitted
        self.params["fitted"] = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the input features based on the fitted decision tree model.

        Args:
            X (np.ndarray): The input feature matrix (n_samples x n_features).

        Returns:
            np.ndarray: The predicted target values (n_samples x n_outputs).

        Raises:
            ValueError: If the model is not fitted or if the number of features in X doesn't match the model.
        """
        # Check if the model is fitted
        if self.params["fitted"] is False:
            raise ValueError("Model is not fitted")
        
        # Validate the input array
        X = super()._validateInput(X)
        
        # Check if the dimensions of the input array match the stored feature matrix
        if X.shape[1] != self.params["num_features"]:
            raise ValueError("The number of columns in X must be equal to the number of features in the model")
        
        return self._predict(X)
    
    def _split_node(self, X: np.ndarray, Y: np.ndarray, node: DecisionTreeModelBase.TreeNode, depth: int) -> None:
        """
        Recursively splits a node in the decision tree based on the feature values and minimizes
        the mean squared error. If the stopping criteria are met, the node becomes a leaf node.

        Args:
            X (np.ndarray): The input feature matrix (n_samples x n_features).
            Y (np.ndarray): The target value matrix (n_samples x n_outputs).
            node (DecisionTreeModelBase.TreeNode): The current node to be split.
            depth (int): The current depth of the tree during recursion.
        
        Notes:
            - The function attempts to find the best feature and threshold to split on, 
              based on minimizing the mean squared error (MSE).
            - The function terminates if the max depth or minimum sample conditions are met, 
              or if further splitting does not reduce the MSE.
        """
        # Check if the node is a leaf node
        if depth >= self.params["max_depth"] or len(Y) < self.params["min_samples_split"]:
            node.value = np.mean(Y, axis=0)
            return
        
        # Find the best split
        best_mse = np.inf
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
            sorted_Y = Y[sorted_idx]
            sorted_X = X[sorted_idx, feature_index]
            
            # Track the left and right sums
            left_sum = np.zeros(self.params["num_outputs"])
            left_sum += sorted_Y[0]
            right_sum = np.sum(sorted_Y, axis=0) - left_sum
            
            for i in range(1, len(Y)):
                # Calculate the mean squared error
                left_mean = left_sum / i
                right_mean = right_sum / (len(Y) - i)
                
                left_mse = np.mean((sorted_Y[:i] - left_mean) ** 2)
                right_mse = np.mean((sorted_Y[i:] - right_mean) ** 2)
                
                mse = (i / len(Y)) * left_mse + ((len(Y) - i) / len(Y)) * right_mse
                            
                # Update the best split
                if mse < best_mse and sorted_X[i - 1] != sorted_X[i]:
                    best_mse = mse
                    best_feature_index = feature_index
                    best_threshold = (sorted_X[i - 1] + sorted_X[i]) / 2
                
                # Update the MSE
                left_sum += sorted_Y[i]
                right_sum -= sorted_Y[i]
        
        # Set the node values
        node.feature_index = best_feature_index
        node.threshold = best_threshold
        
        # If the best split does not reduce the Gini impurity, set the node as a leaf node
        if node.threshold == None or self.MSE(Y, np.mean(Y, axis=0)[None, :]) - best_mse < self.params["tolerance"]:
            node.value = np.mean(Y, axis=0)