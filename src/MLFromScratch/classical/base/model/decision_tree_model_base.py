from ..abstract.supervised_model import SupervisedModel
import numpy as np
from typing import Literal


class DecisionTreeModelBase(SupervisedModel):
    """
    Base class for a Decision Tree model. This class provides the basic 
    implementation for a decision tree, including tree node structure, 
    fitting, and prediction functionalities.
    """
    
    class TreeNode:
        """
        Represents a single node in the decision tree.

        Attributes:
            feature_index (int or None): Index of the feature used for splitting.
            threshold (float or None): Threshold value for splitting.
            left (TreeNode or None): Left child node.
            right (TreeNode or None): Right child node.
            value (float or None): Predicted value if the node is a leaf.
        """
        
        def __init__(self) -> None:
            self.feature_index = None
            self.threshold = None
            self.left = None
            self.right = None
            self.value = None

    def __init__(self, **kwargs) -> None:
        """
        Initializes the DecisionTreeModelBase instance.

        Args:
            **kwargs: Additional parameters to be passed to the superclass.
        """
        # Initialize the base class
        super().__init__(**kwargs)
        
        # Initialize the root node
        self.root = self._initialize_node()
        
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Trains the decision tree model using the provided input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            Y (np.ndarray): Target values of shape (n_samples,).
        """
        # Validate the input array
        X = super()._validateInput(X)
        super()._validateInputPair(X, Y)
        
        # Set the model as fitted
        self.params["fitted"] = True
    
    def _predict(self, X: np.ndarray, node: TreeNode = None) -> np.ndarray:
        """
        Predicts the target values for the given input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            node (TreeNode, optional): The current node in the tree for recursive traversal. 
                                       Defaults to the root node.

        Returns:
            np.ndarray: Predicted target values of shape (n_samples, num_outputs).
        """
        # Initialize the node as the root if not provided
        if node is None:
            node = self.root
        
        # Return the value if the node is a leaf node
        if node.value is not None:
            return np.full((X.shape[0], self.params["num_outputs"]), node.value)
        
        # Find rows that satisfy the condition of the node
        left_mask = X[:, node.feature_index] <= node.threshold
        right_mask = ~left_mask
        
        # Split the data and recursively call the predict function
        predictions = np.zeros((X.shape[0], self.params["num_outputs"]))
        predictions[left_mask] = self._predict(X[left_mask], node.left)
        predictions[right_mask] = self._predict(X[right_mask], node.right)
        
        return predictions.astype(int)
    
    def _initialize_node(self) -> TreeNode:
        """
        Creates and returns a new TreeNode.

        Returns:
            TreeNode: A new instance of a tree node.
        """
        return self.TreeNode()
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, node: TreeNode, depth: int) -> None:
        """
        Recursively grows the decision tree by splitting nodes.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
            node (TreeNode): The current node in the tree.
            depth (int): The current depth of the tree.
        """
        # Split the node
        self._split_node(X, y, node, depth)
        
        # Check if the node is a leaf node
        if node.value is not None:
            return
        
        # Find rows that satisfy the condition of the node
        left_mask = X[:, node.feature_index] <= node.threshold
        right_mask = ~left_mask
        
        # Recursively grow the tree
        node.left = self._initialize_node()
        self._grow_tree(X[left_mask], y[left_mask], node.left, depth + 1)
        
        node.right = self._initialize_node()
        self._grow_tree(X[right_mask], y[right_mask], node.right, depth + 1)