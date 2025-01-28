import numpy as np

class ClassicalModel():
    
    def _validate_input(self, array1: np.ndarray, array2: np.ndarray = None) -> None:
        """
        Validate the input numpy arrays

        Args:
            array1 (np.ndarray): Input numpy array
            array2 (np.ndarray): Input numpy array
        """
        if not isinstance(array1, np.ndarray):
            raise TypeError("Array 1 must be a numpy array")
        
        if len(array1.shape) != 2:
            raise ValueError("Array 1 must be a 2D array")
        
        if array1.size == 0:
            raise ValueError("Array 1 must not be empty")
        
        # Make sure array1 is a float32 array
        array1 = array1.astype(np.float32)
        
        if array2 is not None:
            if not isinstance(array2, np.ndarray):
                raise TypeError("Array 2 must be a numpy array")
        
            if len(array2.shape) != 2:
                raise ValueError("Array 2 must be a 2D array")
            
            if array2.size == 0:
                raise ValueError("Array 2 must not be empty")
            
            # Check if the number of rows in X and Y are equal
            if array1.shape[0] != array2.shape[0]:
                raise ValueError("The number of rows in X and Y must be equal")
            
            # Make sure array2 is a float32 array
            array2 = array2.astype(np.float32)
        
        return array1, array2
            
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Predict the target variable for the given data"""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target variable for the given data"""
        pass

    def cost(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """Calculate the cost of the model"""
        pass

    def get_params(self) -> dict:
        """Get the parameters of the model"""
        return self.params
    
    def load_params(self, params: dict) -> None:
        """Load the parameters of the model"""
        self.params = params