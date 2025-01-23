from abc import ABC, abstractmethod
import numpy as np

class ClassicalModel(ABC):
    
    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit the model to the data"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target variable for the given data"""
        pass

    @abstractmethod
    def cost(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """Calculate the cost of the model"""
        pass

    def get_params(self) -> dict:
        """Get the parameters of the model"""
        return self.params
    
    def load_params(self, params: dict) -> None:
        """Load the parameters of the model"""
        self.params = params