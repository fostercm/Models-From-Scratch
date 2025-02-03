from fastapi import APIRouter, HTTPException
from api.schemas.data_schemas import DataUploadRequest
from api.services.data_services import store_data

router = APIRouter()

@router.post("/upload-data")
async def upload_data(request: DataUploadRequest):
    """
    Uploads input data (and output data if training).
    """
    try:
        input_df, output_df = request.to_dataframes()

        if request.data_type == "training" and output_df is None:
            raise HTTPException(status_code=400, detail="Training data requires both input and output data.")
    
        if request.data_type == "prediction" and output_df is not None:
            raise HTTPException(status_code=400, detail="Prediction data should not include output data.")

        store_data(input_df, output_df, request.data_type)

        return {"message": f"{request.data_type} data uploaded successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
