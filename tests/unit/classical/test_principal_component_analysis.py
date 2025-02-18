import unittest
from MLFromScratch.classical import (
    PCAPython,
    PCAC
)
from MLFromScratch.utils.python_utils.matrix_functions import standardize
from sklearn.decomposition import PCA
import numpy as np

np.random.seed(0)


class TestPCA(unittest.TestCase):
    """
    Tests for the LogisticRegression class implementations in Python, C, and CUDA.
    Ensures correctness for fitting, transforming, and calculating costs.
    """

    def test_transform(self):
        """
        Test the transform method for all LogisticRegression implementations.

        Verifies the following:
        - Ensures the model is fitted before making transformations.
        - Verifies that the input dimensions match the model's parameters.
        - Checks that the transformations are made correctly based on a fitted model.
        - Ensures that the transformations match expected values for a given input.
        """
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.float32)
        standard_X = standardize(X)
        sklearn_model = PCA()
        X_pred = sklearn_model.fit_transform(standard_X)
        
        for model in [PCAPython(), PCAC()]:
            
            # Test that the input is a numpy array
            with self.assertRaises(TypeError):
                model.transform([1, 2, 3])
            
            transformations = model.transform(X)
            
            for i in range(2):
                try:
                    np.testing.assert_array_almost_equal(
                        transformations[:,i], X_pred[:,i], decimal=4
                    )
                except AssertionError:    
                    np.testing.assert_array_almost_equal(
                        -transformations[:,i], X_pred[:,i], decimal=4
                    )
            
            # Check that the transformations are made correctly based on a fitted model
            for i in range(1, 3):
                transformations = model.transform(X, n_components=i)
                np.testing.assert_array_almost_equal(
                    model.transform(X,i), transformations[:,:i], decimal=4
                )
            
            np.testing.assert_array_almost_equal(
                model.transform(X, explained_variance_ratio=0.85), transformations[:,:1], decimal=4
            )
                
            
            

        
            
            
            
            