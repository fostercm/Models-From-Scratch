import numpy as np
from ....utils.python_utils.loss_functions import crossEntropy

class ClassificationMixin:
    """
    Mixin class providing common methods for classification models

    This class includes loss (MSE), target validation, and various evaluation metrics such as accuracy, 
    precision, recall, F1 score, and confusion matrix computation
    """
    
    def _validateTarget(self, y: np.ndarray) -> np.ndarray:
        """
        Validate the target array to ensure it meets classification requirements

        Parameters
        -----------
        y : np.ndarray
            A NumPy array representing the target labels

        Raises
        -------
        TypeError
            - If the target vector is not a NumPy array
        ValueError
            - If the target vector is not 1D
            - If the target vector doesn't have at least one sample
            - If the target vector is not numerical
            - If the target vector contains NaN or infinite values
            - If the target vector is not integer
            - If the target vector is not of the form 0, 1, 2, ..., n_classes - 1
            
        """
        # Check if the target vector is a numpy array
        if not isinstance(y, np.ndarray):
            raise TypeError("Target vector must be of type numpy.ndarray")

        # Check if the target vectory is 1D
        if len(y.shape) != 1:
            raise ValueError("Target vector must be 1D")
        
        # Check for at least one sample in the target vector
        if y.shape[0] == 0 :
            raise ValueError("Target vector must have at least 1 sample")
        
        # Check if the target vector is numerical
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("Target vector must be numerical")
        
        # Check for NaN or infinite values in the target vector
        if not np.isfinite(y).all():
            raise ValueError("Target vector contains NaN or infinite values")
        
        # Check if the target vector is integer
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Target vector must be integer")
        
        # Get the number of classes
        unique_y = np.unique(y)
        n_classes = len(unique_y)
        
        # Ensure there are at least two classes
        if n_classes < 2:
            raise ValueError("Target vector must have at least two classes")
        
        # Ensure the target vector is of the form 0, 1, 2, ..., n_classes - 1
        if not np.array_equal(unique_y, np.arange(np.max(y) + 1)):
            raise ValueError("Target vector must be of the form 0, 1, 2, ..., n_classes - 1")
        
        # Ensure the input data is C-contiguous and int32
        if not y.flags['C_CONTIGUOUS']:
            y = np.ascontiguousarray(y, dtype=np.int32)
        else:
            y = y.astype(np.int32, copy=False)
        
        return y

    def _oneHotEncode(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        """
        One-hot encode the target vector

        Parameters
        -----------
        y : np.ndarray
            A NumPy array representing the target labels

        Returns
        --------
        np.ndarray
            A one-hot encoded target matrix

        """
        # Initialize the one-hot encoded target matrix
        Y_new = np.zeros((y.shape[0], n_classes), dtype=np.int32)
        
        # Transform the target values to one-hot encoding
        y = y.astype(np.int32)
        Y_new[np.arange(y.shape[0]), y.flatten()] = 1
        
        return Y_new
    
    def loss(self, y_true, y_pred, language: str = 'python') -> float:
        """
        Compute the loss between the predicted and true labels.
        """
        return crossEntropy(y_true, y_pred, language) 
    
    def accuracy(self, y_true, y_pred):
        """Calculate the accuracy of the model."""
        return np.mean(y_true == y_pred)
    
    def confusionMatrix(self, y_true, y_pred):
        """Calculate the confusion matrix."""
        n_classes = len(np.unique(y_true))
        cm = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                cm[i, j] = np.sum((y_true == i) & (y_pred == j))
        return cm
    
    def precision(self, y_true, y_pred):
        """Calculate the precision of the model."""
        cm = self.confusionMatrix(y_true, y_pred)
        return np.diag(cm) / np.sum(cm, axis=0)
    
    def recall(self, y_true, y_pred):
        """Calculate the recall of the model."""
        cm = self.confusionMatrix(y_true, y_pred)
        return np.diag(cm) / np.sum(cm, axis=1)
    
    def f1(self, y_true, y_pred):
        """Calculate the F1 score of the model."""
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall)
    
    def getStats(self, y_true, y_pred):
        """Calculate the accuracy, precision, recall and F1 score of the model."""
        stats_dict = {}
        stats_dict['accuracy'] = self.accuracy(y_true, y_pred)
        stats_dict['precision'] = self.precision(y_true, y_pred)
        stats_dict['recall'] = self.recall(y_true, y_pred)
        stats_dict['f1'] = self.f1(y_true, y_pred)
        stats_dict['confusion'] = self.confusionMatrix(y_true, y_pred)
        return stats_dict