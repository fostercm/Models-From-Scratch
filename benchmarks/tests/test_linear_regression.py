from MLFromScratch.classical import LinearRegressionPython, LinearRegressionC, LinearRegressionCUDA
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import unittest
import os
import time
import tracemalloc

class TestBenchmarks(unittest.TestCase):
    """Benchmark different implementations and generate plots."""

    # Benchmark settings
    n_runs = 3
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

            # Measure execution time and memory usage
            tracemalloc.start()
            start_time = time.perf_counter()
            model.fit(X, Y)
            execution_times.append(time.perf_counter() - start_time)
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