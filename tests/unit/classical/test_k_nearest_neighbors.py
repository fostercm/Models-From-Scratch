import unittest
from MLFromScratch.classical import KNN
import numpy as np
from sklearn.datasets import make_classification, make_regression

np.random.seed(42)

class TestKNN(unittest.TestCase):

    def test(self):

        # Generate a synthetic classification dataset
        X_cla, y_cla = make_classification(n_samples=500, n_features=2, n_classes=3, 
                            n_clusters_per_class=1, n_redundant=0, 
                            n_informative=2, class_sep=6, random_state=42)
        
        X_reg, Y_reg = make_regression(n_samples=500, n_features=2, n_targets=1, 
                       noise=0.1, random_state=42)
        Y_reg = Y_reg.reshape(-1, 1)
        
        for model in [
            KNN(language='Python', task='classification'),
        ]:
            # Fit the model to the data
            model.fit(X_cla, y_cla)
            
            # Predict the target values and calculate the accuracy for the different distance types
            y_pred = model.predict(X_cla, k=5)
            self.assertGreater(model.accuracy(y_pred,y_cla), 0.9)
            
            y_pred = model.predict(X_cla, k=5, distance_type='manhattan')
            self.assertGreater(model.accuracy(y_pred,y_cla), 0.9)
            
            y_pred = model.predict(X_cla, k=5, distance_type='cosine')
            self.assertGreater(model.accuracy(y_pred,y_cla), 0.9)
        
        for model in [
            KNN(language='Python', task='regression'),
        ]:
            # Fit the model to the data
            model.fit(X_reg, Y_reg)
            
            # Predict the target values and calculate the accuracy for the different distance types
            Y_pred = model.predict(X_reg, k=5)
            self.assertGreater(model.RSquared(Y_pred,Y_reg), 0.8)
            
            Y_pred = model.predict(X_reg, k=5, distance_type='manhattan')
            self.assertGreater(model.RSquared(Y_pred,Y_reg), 0.8)
            
            Y_pred = model.predict(X_reg, k=5, distance_type='cosine')
            self.assertGreater(model.RSquared(Y_pred,Y_reg), 0.8)