import numpy as np


def meanSquaredError(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
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
    return SCALE_FACTOR * np.linalg.norm(Y_pred - Y_true) ** 2


def crossEntropy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """
    Compute the cross-entropy loss between the predicted and true labels.

    Parameters
    ----------
    Y_pred : numpy array
        Predicted labels.
    Y_true : numpy array
        True labels.

    Returns
    -------
    float
        Cross-entropy loss.
    """
    SCALE_FACTOR = -1 / Y_true.shape[0]

    if Y_pred.shape[1] == 1:
        # Binary classification
        return SCALE_FACTOR * np.sum(
            Y_true * np.log(Y_pred + 1e-9) + (1 - Y_true) * np.log(1 - Y_pred)
        )
    else:
        # Multi-class classification
        return SCALE_FACTOR * np.sum(Y_true * np.log(Y_pred + 1e-9))
