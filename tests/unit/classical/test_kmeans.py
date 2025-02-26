import unittest
from MLFromScratch.classical import KMeans
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import silhouette_score

np.random.seed(42)


class TestKMeans(unittest.TestCase):

    def test_transform(self):

        # Generate synthetic data with 3 clusters
        n_samples = 30  # Total number of data points
        n_features = 2   # Dimensionality (e.g., 2D for visualization)
        centers = [(-5, 5), (5, 5), (0, -5)]  # Cluster centers
        cluster_std = [0.8, 0.7, 0.6]  # Spread of each cluster
        X, y = make_blobs(n_features=n_features, n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=42)

        # Fit KMeans to the data
        for model in [
            KMeans(language='Python'),
        ]:
            # Transform the data
            transformed_labels, centroids = model.transform(X, n_clusters=3)
            
            # Check the shape of the transformed data
            self.assertEqual(transformed_labels.shape, (n_samples,))
            self.assertEqual(centroids.shape, (3, n_features))
            
            # Check the silhouette score to ensure the clusters are well-separated
            score = silhouette_score(X, transformed_labels)
            self.assertGreater(score, 0.5)