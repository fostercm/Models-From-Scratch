import unittest
import time
import tracemalloc
import os
from classical import LinearRegressionPython, LinearRegressionC, LinearRegressionCUDA
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

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
        for model in [LinearRegressionPython(), LinearRegressionC(), LinearRegressionCUDA()]:
        
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
            
            # Test that the model is fitted correctly
            # Test that the computation is correct
            X = np.array([[1, 1],
                          [1, 2],
                          [2, 2],
                          [2, 3]], dtype=np.float32)
            Y = np.array([[6],
                          [8],
                          [9],
                          [11]], dtype=np.float32)
            
            # Check beta
            model.fit(X, Y)
            correct_beta = np.array([[3],[1],[2]],dtype=np.float32)
            self.assertTupleEqual(model.params['beta'].shape, (3, 1))
            for i in range(3):
                self.assertAlmostEqual(model.params['beta'][i][0], correct_beta[i][0], places=2)

    def test_predict(self):
        """
        Test the predict method for all LinearRegression implementations.
        
        Verifies the following:
        - Ensures the model is fitted before making predictions.
        - Verifies that the input dimensions match the model's parameters.
        - Checks that the predictions are made correctly based on a fitted model.
        - Ensures that the predictions match expected values for a given input.
        """
        for model in [LinearRegressionPython(), LinearRegressionC(), LinearRegressionCUDA()]:
        
            # Test that the model is fitted
            with self.assertRaises(ValueError):
                model.predict(np.random.randn(10, 5))
            
            # Test that the input dimensions match the model parameters
            with self.assertRaises(ValueError):
                model.params['beta'] = np.random.randn(5, 1).astype(np.float32)
                model.predict(np.random.randn(10, 3))
            
            # Test that the computation is correct
            X = np.array([[1, 1],
                          [1, 2],
                          [2, 2],
                          [2, 3]], dtype=np.float32)
            Y = np.array([[6],
                          [8],
                          [9],
                          [11]], dtype=np.float32)
            model.params['beta'] = np.array([[3],[1],[2]],dtype=np.float32)
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
        for model in [LinearRegressionPython(), LinearRegressionC(), LinearRegressionCUDA()]:
        
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
            Y_pred = np.random.randn(10, 1)
            Y = np.random.randn(10, 1)
            self.assertAlmostEqual(model.cost(Y_pred, Y), np.sum((Y_pred - Y) ** 2) / 20, places=3)
    
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
        for model in [LinearRegressionPython(), LinearRegressionC(), LinearRegressionCUDA()]:
            
            # Define the input and output arrays
            X = np.array([[1, 1],
                          [1, 2],
                          [2, 2],
                          [2, 3]], dtype=np.float32)
            Y = np.array([[6,6],
                          [8,9],
                          [9,11],
                          [11,14]], dtype=np.float32)
            
            # Fit the model
            model.fit(X, Y)
            
            # Check beta
            correct_beta = np.array([[3,1],[1,2],[2,3]],dtype=np.float32)
            self.assertTupleEqual(model.params['beta'].shape, (3, 2))
            for i in range(3):
                for j in range(2):
                    self.assertAlmostEqual(model.params['beta'][i][j], correct_beta[i][j], places=2)
            
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

class TestBenchmarks(unittest.TestCase):
    """Benchmark different implementations and generate plots."""

    # Benchmark settings
    n_runs = 5
    plot_dir = "benchmarks/linear_regression"

    # Model classes to benchmark
    model_classes = {
        "Python": LinearRegressionPython,
        "C": LinearRegressionC,
        "CUDA": LinearRegressionCUDA,
        "Scikit-Learn": LinearRegression
    }

    # Dataset sizes to benchmark
    dataset_sizes = {
        "128x5": (128, 5),
        "256x50": (256, 50),
        "512x100": (512, 100),
        "1024x200": (1024, 200),
        "2048x400": (2048, 400)
    }

    @classmethod
    def setUpClass(cls):
        """Ensure the benchmark directory exists."""
        os.makedirs(cls.plot_dir, exist_ok=True)

    def benchmark_model(self, model_class, X, Y):
        """Runs multiple benchmark iterations and returns avg execution time & memory usage."""
        execution_times = []
        peak_memories = []

        for _ in range(self.n_runs):
            model = model_class()

            # Measure execution time
            start_time = time.perf_counter()
            model.fit(X, Y)
            execution_times.append(time.perf_counter() - start_time)

            # Measure memory usage
            tracemalloc.start()
            model.fit(X, Y)
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_memories.append(peak_mem)

        return np.mean(execution_times), np.mean(peak_memories) / 10**6  # Convert bytes to MB

    def test_linear_regression(self):
        """Benchmark all models across dataset sizes and generate plots."""
        execution_results = {model: [] for model in self.model_classes}
        memory_results = {model: [] for model in self.model_classes}
        dataset_labels = list(self.dataset_sizes.keys())

        for size_name, (rows, cols) in self.dataset_sizes.items():
            X, Y = np.random.randn(rows, cols), np.random.randn(rows, 1)

            for model_name, model_class in self.model_classes.items():
                avg_time, avg_mem = self.benchmark_model(model_class, X, Y)
                execution_results[model_name].append(avg_time)
                memory_results[model_name].append(avg_mem)
        
        # Distinguish C and CUDA
        memory_results["C"] = [memory + 0.1 for memory in memory_results["C"]]

        self.plot_results(execution_results, dataset_labels, "Execution Time (s)", "execution_time.png")
        self.plot_results(memory_results, dataset_labels, "Peak Memory Usage (MB)", "memory_usage.png")

    def plot_results(self, results, x_labels, ylabel, filename):
        """Generates and saves a plot for benchmark results."""
        plt.figure(figsize=(10, 6))

        for model_name, values in results.items():
            plt.plot(x_labels, values, marker='o', label=model_name)

        plt.xlabel("Dataset Size")
        plt.ylabel(ylabel)
        plt.title(f"Linear Regression {ylabel}")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.plot_dir, filename)
        plt.savefig(plot_path)
        plt.close()

if __name__ == "__main__":
    unittest.main()