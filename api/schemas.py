from pydantic import BaseModel
from typing import Literal, Optional
import pandas as pd
from io import StringIO

class DataUploadRequest(BaseModel):
    """Schema for uploading data for training or prediction."""
    input_data: str  # CSV string for input data
    output_data: Optional[str] = None  # Optional CSV string for output (only required for training)
    data_type: Literal["training", "prediction"]  # Must be either training or prediction

    def to_dataframes(self):
        """Convert CSV strings to Pandas DataFrames."""
        input_df = pd.read_csv(StringIO(self.input_data))
        output_df = pd.read_csv(StringIO(self.output_data)) if self.output_data else None
        return input_df, output_df