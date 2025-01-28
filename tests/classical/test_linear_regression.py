import unittest
from classical import LinearRegressionPython, LinearRegressionC, LinearRegressionCUDA
import numpy as np

class TestLinearRegression(unittest.TestCase):
    
    def test_fit(self):
        for model in [LinearRegressionPython(), LinearRegressionC()]:
        
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
                
            # Test that model parameters are evaluated
            model.fit(np.random.randn(10, 5), np.random.randn(10, 1))
            self.assertIsNotNone(model.params['beta'])

    def test_predict(self):
        for model in [LinearRegressionPython(), LinearRegressionC()]:
        
            # Test that the model is fitted
            with self.assertRaises(ValueError):
                model.predict(np.random.randn(10, 5))
            
            # Test that the input dimensions match the model parameters
            with self.assertRaises(ValueError):
                model.fit(np.random.randn(10, 5), np.random.randn(10, 1))
                model.predict(np.random.randn(10, 3))
        
    def test_cost(self):
        for model in [LinearRegressionPython(), LinearRegressionC()]:
        
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
            
            # Test that the cost is evaluated correctly
            model.fit(np.random.randn(10, 5), np.random.randn(10, 1))
            self.assertIsNotNone(model.cost(np.random.randn(10, 1), np.random.randn(10, 1)))
    
    def test_end_to_end(self):
        for model in [LinearRegressionPython(), LinearRegressionC()]:
            
            # Define the input and output arrays
            X = np.array([[1, 1],
                          [1, 2],
                          [2, 2],
                          [2, 3]], dtype=np.float32)
            Y = np.array([[6],
                          [8],
                          [9],
                          [11]], dtype=np.float32)
            
            # Fit the model
            model.fit(X, Y)
            
            # Check beta
            correct_beta = np.array([[3],[1],[2]],dtype=np.float32)
            self.assertTupleEqual(model.params['beta'].shape, (3, 1))
            for i in range(3):
                self.assertAlmostEqual(model.params['beta'][i][0], correct_beta[i][0], places=2)
            
            # Predict
            Y_pred = model.predict(X)
            
            # Check prediction
            self.assertTupleEqual(Y_pred.shape, (4, 1))
            for i in range(4):
                self.assertAlmostEqual(Y_pred[i][0], Y[i][0], places=2)
            
            # Cost
            cost = model.cost(Y_pred, Y)
            
            # Check cost
            self.assertAlmostEqual(cost, 0, places=3)