from fastapi import APIRouter, Form, File, UploadFile
from .services import getModel
import pandas as pd
import json

router = APIRouter()

@router.post("/train")
async def train(
    model_name: str = Form(...),
    language: str = Form(...),
    indicator: UploadFile = File(...),
    response: UploadFile = File(...)
):
    
    # Get model
    model = getModel(model_name, language)
    
    # Read CSVs into dataframes
    indicator_df = pd.read_csv(indicator.file)
    response_df = pd.read_csv(response.file)
    
    # Modify dataframes into X and Y
    X = indicator_df.to_numpy()
    Y = response_df.to_numpy()
    
    # Train model
    model.fit(X, Y)
    
    # Store model data
    json_params = {}
    json_params['model'] = model_name + " " + language
    json_params['loss'] = float(model.cost(model.predict(X), Y))
    json_params['params'] = {k: v.tolist() for k, v in model.get_params().items()}
    
    return json_params

@router.post("/predict")
async def predict(
    model_name: str = Form(...),
    language: str = Form(...),
    indicator: UploadFile = File(...),
    model_file: UploadFile = File(...)
):
    
    # Get model
    model = getModel(model_name, language)
    
    # Load model data
    model_data = json.load(model_file.file)
    model.load_params(model_data['params'])
    
    # Read CSV into dataframe
    indicator_df = pd.read_csv(indicator.file)
    
    # Modify dataframe into X
    X = indicator_df.to_numpy()
    
    # Predict
    Y = model.predict(X)
    
    return {"predictions": Y.tolist()}