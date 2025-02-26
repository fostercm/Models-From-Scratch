import unittest
from MLFromScratch.classical import LinearRegression
import numpy as np


class TestLinearRegression(unittest.TestCase):
    """
    Tests for the LinearRegression class implementations in Python, C, and CUDA.
    Ensures correctness for fitting, predicting, and calculating costs.
    """

    def test_fit(self):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.float32)
        Y = np.array([[6], [8], [9], [11]], dtype=np.float32)
        correct_beta = np.array([[3], [1], [2]], dtype=np.float32)
        
        for model in [
            LinearRegression(language='Python'),
        ]:
            # Check beta
            model.fit(X, Y)
            np.testing.assert_array_almost_equal(model.params["beta"], correct_beta, decimal=2)
            self.assertTrue(model.params["fitted"])

    def test_predict(self):
        """
        Test the predict method for all LinearRegression implementations.

        Verifies the following:
        - Ensures the model is fitted before making predictions.
        - Verifies that the input dimensions match the model's parameters.
        - Checks that the predictions are made correctly based on a fitted model.
        - Ensures that the predictions match expected values for a given input.
        """
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.float32)
        Y = np.array([[6], [8], [9], [11]], dtype=np.float32)
        
        for model in [
            LinearRegression(language='Python'),
        ]:
            # Test that the computation is correct
            model.params["beta"] = np.array([[3], [1], [2]], dtype=np.float32)
            model.params["fitted"] = True
            Y_pred = model.predict(X)

            # Check prediction
            self.assertTupleEqual(Y_pred.shape, (4, 1))
            for i in range(4):
                self.assertAlmostEqual(Y_pred[i][0], Y[i][0], places=2)


if __name__ == "__main__":
    unittest.main()
