import numpy as np

def normalize(X):
    """Normalize the feature matrix X."""
    return X - np.min(X, axis=1) / (np.max(X, axis=1) - np.min(X, axis=1))

def standardize(X):
    """Standardize the feature matrix X."""
    return (X - np.mean(X, axis=1)) / np.std(X, axis=1)