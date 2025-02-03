import pandas as pd
import os

DATA_STORAGE_PATH = "data_storage"

def store_data(input_df: pd.DataFrame, output_df: pd.DataFrame, data_type: str):
    """
    Saves input and output data (if provided) as Parquet files.
    """
    os.makedirs(DATA_STORAGE_PATH, exist_ok=True)  # Ensure storage directory exists

    input_path = os.path.join(DATA_STORAGE_PATH, f"{data_type}_input.csv")
    input_df.to_csv(input_path, index=False)

    if data_type == "training" and output_df is not None:
        output_path = os.path.join(DATA_STORAGE_PATH, f"{data_type}_output.csv")
        output_df.to_csv(output_path, index=False)
