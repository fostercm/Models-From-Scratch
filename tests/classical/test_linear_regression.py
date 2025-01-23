import unittest
from classical import LinearRegression
import numpy as np

class TestLinearRegression(unittest.TestCase):
    
    def test_fit(self):
        model = LinearRegression()
        
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
        self.assertIsNotNone(model.params)

    def test_predict(self):
        model = LinearRegression()
        
        # Test that the model is fitted
        with self.assertRaises(ValueError):
            model.predict(np.random.randn(10, 5))
        
        # Test that the input dimensions match the model parameters
        with self.assertRaises(ValueError):
            model.fit(np.random.randn(10, 5), np.random.randn(10, 1))
            model.predict(np.random.randn(10, 3))
        
    def test_cost(self):
        model = LinearRegression()
        
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
    
    def test_get_params(self):
        model = LinearRegression()
        
        # Test that model starts with no params
        self.assertIsNone(model.get_params())
        
        # Test that model has params after fitting
        model.fit(np.random.randn(10, 5), np.random.randn(10, 1))
        self.assertIsNotNone(model.get_params())