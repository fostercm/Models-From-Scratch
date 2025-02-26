import unittest
from MLFromScratch.classical import NaiveBayes
import numpy as np

np.random.seed(0)


class TestNaiveBayes(unittest.TestCase):

    def test_transform(self):
        bin_X = np.array([[1,0,0],[1,0,1],[0,1,1],[1,1,1],[1,1,0],[0,1,0]])
        multi_X = np.array([[1,2,0],[1,2,1],[0,2,1],[0,1,1],[0,0,1],[0,0,2]])
        gauss_X = np.array([[1.5,2.5,0.5],[1.5,2.5,1.5],[0.5,2.5,1.5],[0.5,1.5,1.5],[0.5,0.5,1.5],[0.5,0.5,2.5]])
        Y = np.array([0,0,1,1,2,2])
        
        for model in [
            NaiveBayes(language="Python", variant="bernoulli"),
        ]:
            model.fit(bin_X, Y)
            P = model.predict(bin_X)
            self.assertTrue(np.allclose(P, Y))
            
        for model in [
            NaiveBayes(language="Python", variant="multinomial"),
        ]:
            model.fit(multi_X, Y)
            P = model.predict(multi_X)
            self.assertTrue(np.allclose(P, Y))
        
        for model in [
            NaiveBayes(language="Python", variant="gaussian"),
        ]:
            model.fit(gauss_X, Y)
            P = model.predict(gauss_X)
            self.assertTrue(np.allclose(P, Y))