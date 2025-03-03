import unittest
from MLFromScratch.classical import SVM
import numpy as np
from sklearn.datasets import make_classification

class TestSVM(unittest.TestCase):

    def test(self):

        # Generate a synthetic classification dataset
        X_cla, y_cla = make_classification(n_samples=500, n_features=2, n_classes=2, 
                            n_clusters_per_class=1, n_redundant=0, 
                            n_informative=2, class_sep=6)
        X_cla = (X_cla - np.mean(X_cla, axis=0)) / np.std(X_cla, axis=0, ddof=1)
        
        for model in [
            SVM(language='Python', task='classification', kernel='rbf', iterations=100, tolerance=1e-3, alpha_threshold=0.1),
        ]:
            # Fit the model to the data
            model.fit(X_cla, y_cla)
            
            # Predict the target values and calculate the accuracy for the different distance types
            y_pred = model.predict(X_cla)
            self.assertGreater(model.accuracy(y_pred,y_cla), 0.9)