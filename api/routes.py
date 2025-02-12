from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from .services import getModel
import pandas as pd
import json
import logging

router = APIRouter()

# Set up logging for the API
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.get("/")
async def root():
    """
    Root endpoint for the API
    
    Returns:
        dict: A dictionary confirming the API is running
    """
    return {"message": "ML API is running!"}


@router.get("/health")
async def health_check():
    """
    Checks the health of the API
    
    Returns:
        dict: A dictionary containing the status of the API
    """
    return {"status": "up"}


@router.post("/train")
async def train(
    model_name: str = Form(...), 
    language: str = Form(...), 
    indicator: UploadFile = File(...), 
    response: UploadFile = File(...)
):
    """
    Trains a machine learning model using provided indicator and response data

    Args:
        model_name (str): The name of the model to train.
        language (str): The programming language used to train the model
        indicator (UploadFile): The CSV file containing the indicator data
        response (UploadFile): The CSV file containing the response data

    Returns:
        dict: A dictionary containing model details and performance metrics
    """
    
    try:
        # Get the model object based on the provided model name and language
        model = getModel(model_name, language)
        
        # Read indicator and response CSV files into numpy arrays
        X = pd.read_csv(indicator.file).to_numpy()
        Y = pd.read_csv(response.file).to_numpy()
        
        # Train the model with the provided data
        model.fit(X, Y)
        
        # Store model parameters and loss
        json_params = {
            'model': f"{model_name} {language}",
            'loss': float(model.cost(model.predict(X), Y)),
            'params': {k: v.tolist() for k, v in model.get_params().items()}
        }
        
        logger.info(f"Model {model_name} trained successfully.")
        return json_params
    
    except Exception as e:
        logger.error(f"Error training model {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Model training failed")


@router.post("/predict")
async def predict(
    model_name: str = Form(...), 
    language: str = Form(...), 
    indicator: UploadFile = File(...), 
    model_file: UploadFile = File(...)
):
    """
    Predicts outputs using a trained machine learning model

    Args:
        model_name (str): The name of the model to use for predictions
        language (str): The programming language used for the model
        indicator (UploadFile): The CSV file containing indicator data for prediction
        model_file (UploadFile): The JSON file containing model parameters

    Returns:
        dict: A dictionary containing the predictions from the model
    """
    
    try:
        # Get the model object based on the provided model name and language
        model = getModel(model_name, language)
        
        # Load model parameters from the uploaded JSON file
        model_data = json.load(model_file.file)
        model.load_params(model_data['params'])
        
        # Read indicator CSV file into a dataframe
        indicator_df = pd.read_csv(indicator.file)
        
        # Convert dataframe into a numpy array for prediction
        X = indicator_df.to_numpy()
        
        # Predict outputs using the trained model
        Y = model.predict(X)
        
        logger.info(f"Prediction successful for model {model_name}.")
        return {"predictions": Y.tolist()}
    
    except Exception as e:
        logger.error(f"Error predicting with model {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")