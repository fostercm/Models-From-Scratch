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

    @abstractmethod
    def get_params(self) -> dict:
        """Get the parameters of the model"""
        pass
    
    @abstractmethod
    def update_params(self) -> None:
        """Update the parameters of the model"""
        pass