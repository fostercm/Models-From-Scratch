import numpy as np

def normalize(X):
    """Normalize the feature matrix X."""
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

def standardize(X):
    """Standardize the feature matrix X."""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)