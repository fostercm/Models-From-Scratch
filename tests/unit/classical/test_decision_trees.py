import unittest
from MLFromScratch.classical import DecisionTree
import numpy as np
from sklearn.datasets import make_classification, make_regression
import matplotlib.pyplot as plt

np.random.seed(42)

class TestDecisionTrees(unittest.TestCase):

    def test(self):

        # Generate a synthetic classification dataset
        X_cla, y_cla = make_classification(n_samples=500, n_features=2, n_classes=3, 
                            n_clusters_per_class=1, n_redundant=0, 
                            n_informative=2, class_sep=6, random_state=42)
        
        X_reg, Y_reg = make_regression(n_samples=500, n_features=2, n_targets=1, 
                       noise=0.1, random_state=42)
        Y_reg = Y_reg.reshape(-1, 1)
        
        for model in [
            DecisionTree(language='Python', task='classification'),
        ]:
            # Fit the model to the data
            model.fit(X_cla, y_cla)
            
            # Predict the target values and calculate the accuracy for the different distance types
            y_pred = model.predict(X_cla)
            self.assertGreater(model.accuracy(y_pred,y_cla), 0.9)            
        
        for model in [
            DecisionTree(language='Python', task='regression', max_depth=5),
        ]:
            # Fit the model to the data
            model.fit(X_reg, Y_reg)
            
            # Predict the target values and calculate the accuracy for the different distance types
            Y_pred = model.predict(X_reg)
            self.assertGreater(model.RSquared(Y_pred,Y_reg), 0.8)