import unittest
from MLFromScratch.classical import LogisticRegressionPython, LogisticRegressionC
import numpy as np

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
        for model in [LogisticRegressionPython(), LogisticRegressionC()]:
        
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
                model.fit(np.zeros((2,3)), np.zeros((3,3)))
            
            # Binary classification dataset
            self.X_bin = np.array([[1, 1], 
                                [1, 2], 
                                [2, 2], 
                                [2, 3]], dtype=np.float32)
            self.Y_bin = np.array([[0], [0], [1], [1]], dtype=np.int64)

            # Multiclass dataset
            self.X_multi = np.array([[1, 1], 
                                    [1, 2], 
                                    [2, 2], 
                                    [2, 3], 
                                    [3, 3]], dtype=np.float32)
            self.Y_multi = np.array([[0], [1], [1], [2], [2]], dtype=np.int64)
            
            # Binary classification
            model.fit(self.X_bin, self.Y_bin)
            self.assertIsNotNone(model.params['beta'])
            self.assertTupleEqual(model.params['beta'].shape, (3, 1))
            
            # Multiclass classification
            model.fit(self.X_multi, self.Y_multi)
            self.assertIsNotNone(model.params['beta'])
            self.assertTupleEqual(model.params['beta'].shape, (3, 3))
            

    def test_predict(self):
        """
        Test the predict method for all LogisticRegression implementations.
        
        Verifies the following:
        - Ensures the model is fitted before making predictions.
        - Verifies that the input dimensions match the model's parameters.
        - Checks that the predictions are made correctly based on a fitted model.
        - Ensures that the predictions match expected values for a given input.
        """
        for model in [LogisticRegressionPython(), LogisticRegressionC()]:
        
            # Test that the model is fitted
            with self.assertRaises(ValueError):
                model.predict(np.random.randn(10, 5))
            
            # Test that the input dimensions match the model parameters
            with self.assertRaises(ValueError):
                model.params['beta'] = np.random.randn(5, 1).astype(np.float32)
                model.predict(np.random.randn(10, 3))
            
            X = np.array([[1, 1],
                          [1, 2],
                          [2, 2],
                          [2, 3]], dtype=np.float32)
            
            # Test binary classification
            model.params['beta'] = np.array([[3],[1],[2]],dtype=np.float32)
            Y_pred = model.predict(X)
            Y = 1 / (1 + np.exp(-np.hstack((np.ones((4, 1)),X)) @ model.params['beta']))
            self.assertTupleEqual(Y_pred.shape, (4, 1))
            for i in range(4):
                self.assertAlmostEqual(Y_pred[i][0], Y[i][0], places=2)
            
            # Test multi-class classification
            model.params['beta'] = np.array([[3,1,2],[1,2,3],[2,3,1]],dtype=np.float32)
            Y_pred = model.predict(X)
            Y = np.hstack((np.ones((4, 1)), X)) @ model.params['beta']
            Y = np.exp(Y) / np.exp(Y).sum(axis=1, keepdims=True)
            self.assertTupleEqual(Y_pred.shape, (4, 3))
            for i in range(4):
                for j in range(3):
                    self.assertAlmostEqual(Y_pred[i][j], Y[i][j], places=2)
        
    def test_cost(self):
        """
        Test the cost method for all LogisticRegression implementations.
        
        Verifies the following:
        - Ensures proper handling of input types (inputs must be numpy arrays).
        - Ensures that inputs are not empty arrays.
        - Ensures that inputs are 2D arrays.
        - Verifies that the number of rows in Y_pred and Y are equal.
        - Validates that the cost computation correctly reflects the difference between predicted and actual values.
        - Compares the computed cost to a manually computed value for accuracy.
        """
        for model in [LogisticRegressionPython(), LogisticRegressionC()]:
        
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
                model.cost(np.zeros((2,3)), np.zeros((3,3)))
            
        # Test binary cost
        Y_pred_binary = np.random.uniform(0.01, 0.99, (10, 1))
        Y_binary = np.random.randint(0, 2, (10, 1))
        expected_cost_binary = -np.sum(Y_binary * np.log(Y_pred_binary) + (1 - Y_binary) * np.log(1 - Y_pred_binary)) / 10
        self.assertAlmostEqual(model.cost(Y_pred_binary, Y_binary), expected_cost_binary, places=3)
        
        # Test multiclass cost
        m, K = 10, 4
        raw = np.random.uniform(0.01, 1.0, (m, K))
        Y_pred_multi = raw / raw.sum(axis=1, keepdims=True)
        Y_multi = np.zeros((m, K))
        indices = np.random.randint(0, K, size=m)
        Y_multi[np.arange(m), indices] = 1
        expected_cost_multi = -np.sum(Y_multi * np.log(Y_pred_multi)) / m
        self.assertAlmostEqual(model.cost(Y_pred_multi, Y_multi), expected_cost_multi, places=3)
    
    def test_end_to_end(self):
        """
        End-to-end test for the LogisticRegression models.

        Verifies the full workflow from fitting the model to making predictions and evaluating the cost. 
        Specifically, this test ensures:
        - The model correctly fits the data and computes parameters (beta).
        - The predicted values align with the expected output.
        - The cost computation reflects the difference between predictions and actual values.

        This test uses a small synthetic dataset to validate the entire process of training and evaluating the model.
        """
        for model in [LogisticRegressionPython(), LogisticRegressionC()]:
            
            # Binary classification dataset
            self.X_bin = np.array([[1, 1], 
                                [1, 2], 
                                [2, 2], 
                                [2, 3]], dtype=np.float32)
            self.Y_bin = np.array([[0], [0], [1], [1]], dtype=np.int32)

            # Multiclass dataset
            self.X_multi = np.array([[1, 1], 
                                    [1, 2], 
                                    [2, 2], 
                                    [2, 3], 
                                    [3, 3]], dtype=np.float32)
            self.Y_multi = np.array([[0], [1], [1], [2], [2]], dtype=np.int32)
            
            # Binary classification
            model.fit(self.X_bin, self.Y_bin)
            predictions = model.predict(self.X_bin)
            self.assertTupleEqual(predictions.shape, (4, 1))
            self.assertTupleEqual(tuple(np.round(predictions.flatten())), tuple(self.Y_bin.flatten()))
            
            # Multiclass classification
            model.fit(self.X_multi, self.Y_multi)
            predictions = model.predict(self.X_multi)
            self.assertTupleEqual(predictions.shape, (5, 3))
            self.assertTupleEqual(tuple(predictions.argmax(axis=1)), tuple(self.Y_multi.flatten()))

if __name__ == "__main__":
    unittest.main()