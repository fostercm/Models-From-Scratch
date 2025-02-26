from ...base.abstract.unsupervised_model import UnsupervisedModel
from typing import Literal, Tuple
import numpy as np

class KMeans(UnsupervisedModel):
    
    def __init__(self, language: Literal['Python', 'C', 'CUDA'] = 'Python', iterations: int=300, tol: float=1e-4) -> None:
        # Check language is valid
        if language not in ['Python', 'C', 'CUDA']:
            raise ValueError("Invalid language, must be 'Python', 'C' or 'CUDA'")
        
        # Check max_iter is positive
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("max_iter must be a positive integer")
        
        # Check tol is positive
        if not isinstance(tol, float) or tol <= 0:
            raise ValueError("tol must be a positive float")
        
        super().__init__(language=language, iterations=iterations, tol=tol)
    
    def transform(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
        # Validate the input array
        X = self._validateInput(X)
        
        # Get the number of samples and features
        n_samples, n_features = X.shape
        
        # Store the number of clusters
        self.n_clusters = n_clusters
        
        # Initialize the centroids
        centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]
        new_centroids = np.zeros((n_clusters, n_features))
        
        for i in range(self.params["iterations"]):
            # Assign each data point to the closest centroid
            centroid_distances = np.linalg.norm(X[:, None] - centroids, axis=2)
            labels = np.argmin(centroid_distances, axis=1)
            
            # Update the centroids
            for k in range(n_clusters):
                new_centroids[k] = X[labels == k].mean(axis=0)
            
            # Check if the centroids have converged
            if i % 100 == 0 and np.linalg.norm(new_centroids - centroids) < self.params["tol"]:
                break
            
            # Update the centroids
            centroids = new_centroids
        
        # Return the cluster labels and centroids
        return labels, centroids