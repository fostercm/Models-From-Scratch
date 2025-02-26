import numpy as np
import ctypes
import os

# Load the C and CUDA shared libraries
package_dir = os.path.dirname(os.path.abspath(__file__))

clib_path = os.path.join(package_dir, "../../lib/libc_utils_shared.so")
clib_path = os.path.normpath(clib_path)
clib = ctypes.CDLL(clib_path)

cudalib_path = os.path.join(package_dir, "../../lib/libcuda_utils_shared.so")
cudalib_path = os.path.normpath(cudalib_path)
cudalib = ctypes.CDLL(cudalib_path)


def meanSquaredError(Y_pred: np.ndarray, Y_true: np.ndarray, language: str = 'python') -> float:
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
    if language == 'python':
        return (0.5 / Y_true.shape[0]) * np.linalg.norm(Y_pred - Y_true) ** 2
    elif language == 'c':
        return clib.meanSquaredError(Y_pred, Y_true, Y_pred.shape[0], Y_pred.shape[1])
    elif language == 'cuda':
        return cudalib.meanSquaredError(Y_pred, Y_true, Y_pred.shape[0], Y_pred.shape[1])
        


def crossEntropy(Y_pred: np.ndarray, Y_true: np.ndarray, language: str = 'python') -> float:
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
    if language == 'python':
        SCALE_FACTOR = -1 / Y_true.shape[0]

        if Y_pred.shape[1] == 1:
            # Binary classification
            return SCALE_FACTOR * np.sum(Y_true * np.log(Y_pred + 1e-9) + (1 - Y_true) * np.log(1 - Y_pred))
        else:
            # Multi-class classification
            return SCALE_FACTOR * np.sum(Y_true * np.log(Y_pred + 1e-9))
    
    elif language == 'c':
        return clib.crossEntropy(Y_pred, Y_true, Y_pred.shape[0], Y_pred.shape[1])
    elif language == 'cuda':
        return cudalib.crossEntropy(Y_pred, Y_true, Y_pred.shape[0], Y_pred.shape[1])
