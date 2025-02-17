import numpy as np


def sigmoid(X: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function for the input array X.

    Parameters
    ----------
    X : numpy array
        Input array.

    Returns
    -------
    numpy array
        Sigmoid of the input array.
    """
    return 1 / (1 + np.exp(-X))


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the softmax function for the input array X.

    Parameters
    ----------
    X : numpy array
        Input array.

    Returns
    -------
    numpy array
        Softmax of the input array.
    """
    exps = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
