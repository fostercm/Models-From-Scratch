import numpy as np

def meanSquaredError(Y_pred, Y_true):
    """
    Compute the mean squared error between the predicted and true labels.
    
    Parameters
    ----------
    Y_pred : numpy array
        Predicted labels.
    Y_true : numpy array
        True labels.
    
    Returns
    -------
    float
        Mean squared error.
    """
    SCALE_FACTOR = 0.5 / Y_true.shape[0]
    return SCALE_FACTOR * np.linalg.norm(Y_pred - Y_true)**2