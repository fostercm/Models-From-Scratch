from fastapi import HTTPException
from classical import LinearRegressionPython, LinearRegressionC
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np

def getModel(model_name, model_language):
    
    if model_language == "Python":
        if model_name == "Linear Regression":
            return LinearRegressionPython()
        else:
            raise HTTPException(status_code=400, detail="Model name not supported.")
    
    elif model_language == "C":
        if model_name == "Linear Regression":
            return LinearRegressionC()
        else:
            raise HTTPException(status_code=400, detail="Model name not supported.")
    
    else:
        raise HTTPException(status_code=400, detail="Model language not supported.")
    