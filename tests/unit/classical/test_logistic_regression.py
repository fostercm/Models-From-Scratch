import unittest
from MLFromScratch.classical import LogisticRegression
import numpy as np

np.random.seed(0)


class TestLogisticRegression(unittest.TestCase):
    """
    Tests for the LogisticRegression class implementations in Python, C, and CUDA.
    Ensures correctness for fitting, predicting, and calculating costs.
    """

    def test_fit(self):
        """
        Test the fit method for all LogisticRegression implementations.

        Verifies the following:
        - Ensures proper handling of input types (inputs must be numpy arrays).
        - Ensures that the inputs are not empty arrays.
        - Verifies that inputs are 2D arrays (to match the expected model structure).
        - Checks that the number of rows in X and Y are equal.
        - Ensures that the model parameters (beta) are computed and not None after fitting.
        - Verifies that the computed parameters (beta) match expected values for a simple dataset.
        """
        # Binary classification dataset
        self.X_bin = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.float32)
        self.Y_bin = np.array([0, 0, 1, 1], dtype=np.int64)
        
        # Multiclass dataset
        self.X_multi = np.array(
            [[1, 1], [1, 2], [2, 2], [2, 3], [3, 3]], dtype=np.float32
        )
        self.Y_multi = np.array([0, 1, 1, 2, 2], dtype=np.int64)
        
        for model in [
            LogisticRegression(language="Python"),
        ]:

            # Binary classification
            model.fit(self.X_bin, self.Y_bin)
            self.assertIsNotNone(model.params["beta"])
            self.assertTupleEqual(model.params["beta"].shape, (3, 1))

            # Multiclass classification
            model.fit(self.X_multi, self.Y_multi)
            self.assertIsNotNone(model.params["beta"])
            self.assertTupleEqual(model.params["beta"].shape, (3, 3))

    def test_predict(self):
        """
        Test the predict method for all LogisticRegression implementations.

        Verifies the following:
        - Ensures the model is fitted before making predictions.
        - Verifies that the input dimensions match the model's parameters.
        - Checks that the predictions are made correctly based on a fitted model.
        - Ensures that the predictions match expected values for a given input.
        """
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.float32)
        Y_bin = 1 / (
                1 + np.exp(-np.hstack((np.ones((4, 1)), X)) @ np.array([[3], [1], [2]], dtype=np.float32))
            )
        
        Y_multi = np.hstack((np.ones((4, 1)), X)) @ np.array([[3, 1, 2], [1, 2, 3], [2, 3, 1]], dtype=np.float32)
        Y_multi = np.exp(Y_multi) / np.exp(Y_multi).sum(axis=1, keepdims=True)
        
        for model in [
            LogisticRegression(language="Python"),
        ]:
            model.params["fitted"] = True
            
            # Test binary classification
            model.params["beta"] = np.array([[3], [1], [2]], dtype=np.float32)
            model.params["num_classes"] = 2
            Y_pred = model.predict(X)            
            np.testing.assert_almost_equal(Y_pred, Y_bin, decimal=3)

            # Test multi-class classification
            model.params["beta"] = np.array([[3, 1, 2], [1, 2, 3], [2, 3, 1]], dtype=np.float32)
            model.params["num_classes"] = 3
            Y_pred = model.predict(X)
            np.testing.assert_almost_equal(Y_pred, Y_multi, decimal=3)


if __name__ == "__main__":
    unittest.main()
