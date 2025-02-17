import unittest
from MLFromScratch.classical import (
    LinearRegressionPython,
    LinearRegressionC,
    LinearRegressionCUDA,
)
import numpy as np


class TestLinearRegression(unittest.TestCase):
    """
    Tests for the LinearRegression class implementations in Python, C, and CUDA.
    Ensures correctness for fitting, predicting, and calculating costs.
    """

    def test_fit(self):
        """
        Test the fit method for all LinearRegression implementations.

        Verifies the following:
        - Ensures proper handling of input types (inputs must be numpy arrays).
        - Ensures that the inputs are not empty arrays.
        - Verifies that inputs are 2D arrays (to match the expected model structure).
        - Checks that the number of rows in X and Y are equal.
        - Ensures that the model parameters (beta) are computed and not None after fitting.
        - Verifies that the computed parameters (beta) match expected values for a simple dataset.
        """
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.float32)
        Y = np.array([[6], [8], [9], [11]], dtype=np.float32)
        correct_beta = np.array([[3], [1], [2]], dtype=np.float32)
        
        for model in [
            LinearRegressionPython(),
            LinearRegressionC(),
            LinearRegressionCUDA(),
        ]:
            # Test that inputs are numpy arrays
            with self.assertRaises(TypeError):
                model.fit([1, 2, 3], [1, 2, 3])

            # Test that inputs aren't empty
            with self.assertRaises(ValueError):
                model.fit(np.empty(0), np.empty(0))

            # Test that inputs are 2D arrays
            with self.assertRaises(ValueError):
                model.fit(np.array([1]), np.array([1]))

            # Test that the number of rows in X and Y are equal
            with self.assertRaises(ValueError):
                model.fit(np.zeros((2, 3)), np.zeros((3, 3)))

            # Test that model parameters are evaluated
            model.fit(np.random.randn(10, 5), np.random.randn(10, 1))
            self.assertIsNotNone(model.params["beta"])

            # Test that the model is fitted correctly
            # Test that the computation is correct
            

            # Check beta
            model.fit(X, Y)
            self.assertTupleEqual(model.params["beta"].shape, (3, 1))
            for i in range(3):
                self.assertAlmostEqual(
                    model.params["beta"][i][0], correct_beta[i][0], places=2
                )

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
            LinearRegressionPython(),
            LinearRegressionC(),
            LinearRegressionCUDA(),
        ]:
            # Test that the model is fitted
            with self.assertRaises(ValueError):
                model.predict(np.random.randn(10, 5))

            # Test that the input dimensions match the model parameters
            with self.assertRaises(ValueError):
                model.params["beta"] = np.random.randn(5, 1).astype(np.float32)
                model.predict(np.random.randn(10, 3))

            # Test that the computation is correct
            model.params["beta"] = np.array([[3], [1], [2]], dtype=np.float32)
            Y_pred = model.predict(X)

            # Check prediction
            self.assertTupleEqual(Y_pred.shape, (4, 1))
            for i in range(4):
                self.assertAlmostEqual(Y_pred[i][0], Y[i][0], places=2)

    def test_cost(self):
        """
        Test the cost method for all LinearRegression implementations.

        Verifies the following:
        - Ensures proper handling of input types (inputs must be numpy arrays).
        - Ensures that inputs are not empty arrays.
        - Ensures that inputs are 2D arrays.
        - Verifies that the number of rows in Y_pred and Y are equal.
        - Validates that the cost computation correctly reflects the difference between predicted and actual values.
        - Compares the computed cost to a manually computed value for accuracy.
        """
        Y_pred = np.random.randn(10, 1)
        Y = np.random.randn(10, 1)
        cost = np.sum((Y_pred - Y) ** 2) / 20
        for model in [
            LinearRegressionPython(),
            LinearRegressionC(),
            LinearRegressionCUDA(),
        ]:
            # Test that inputs are numpy arrays
            with self.assertRaises(TypeError):
                model.cost([1, 2, 3], [1, 2, 3])

            # Test that inputs aren't empty
            with self.assertRaises(ValueError):
                model.cost(np.empty(0), np.empty(0))

            # Test that inputs are 2D arrays
            with self.assertRaises(ValueError):
                model.cost(np.array([1]), np.array([1]))

            # Test that the number of rows in Y_pred and Y are equal
            with self.assertRaises(ValueError):
                model.cost(np.zeros((2, 3)), np.zeros((3, 3)))

            # Test that the cost is evaluated correctly
            self.assertAlmostEqual(
                model.cost(Y_pred, Y), cost, places=3
            )

    def test_end_to_end(self):
        """
        End-to-end test for the LinearRegression models.

        Verifies the full workflow from fitting the model to making predictions and evaluating the cost.
        Specifically, this test ensures:
        - The model correctly fits the data and computes parameters (beta).
        - The predicted values align with the expected output.
        - The cost computation reflects the difference between predictions and actual values.

        This test uses a small synthetic dataset to validate the entire process of training and evaluating the model.
        """
        # Define the input and output arrays
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.float32)
        Y = np.array([[6, 6], [8, 9], [9, 11], [11, 14]], dtype=np.float32)
        correct_beta = np.array([[3, 1], [1, 2], [2, 3]], dtype=np.float32)
        
        for model in [
            LinearRegressionPython(),
            LinearRegressionC(),
            LinearRegressionCUDA(),
        ]:
            # Fit the model
            model.fit(X, Y)

            # Check beta
            self.assertTupleEqual(model.params["beta"].shape, (3, 2))
            for i in range(3):
                for j in range(2):
                    self.assertAlmostEqual(
                        model.params["beta"][i][j], correct_beta[i][j], places=2
                    )

            # Predict
            Y_pred = model.predict(X)

            # Check prediction
            self.assertTupleEqual(Y_pred.shape, (4, 2))
            for i in range(4):
                for j in range(2):
                    self.assertAlmostEqual(Y_pred[i][j], Y[i][j], places=2)

            # Cost
            cost = model.cost(Y_pred, Y)

            # Check cost
            self.assertAlmostEqual(cost, 0, places=3)


if __name__ == "__main__":
    unittest.main()
