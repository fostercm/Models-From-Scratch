from fastapi import HTTPException
import logging
from MLFromScratch.classical import LinearRegressionPython, LinearRegressionC, LinearRegressionCUDA

# Set up logging
logger = logging.getLogger(__name__)

def getModel(model_name: str, model_language: str):
    """
    Retrieves the model instance based on the provided model name and language.

    Args:
        model_name (str): The name of the model to retrieve.
        model_language (str): The programming language in which the model is implemented.

    Returns:
        object: The model instance.

    Raises:
        HTTPException: If the model name or language is not supported.
    """
    
    # Check for invalid language
    if model_language not in ["Python", "C", "CUDA"]:
        logger.error(f"Unsupported model language: {model_language}")
        raise HTTPException(status_code=400, detail="Model language not supported.")
    
    # Map model names to corresponding classes
    model_mapping = {
        "Linear Regression": {
            "Python": LinearRegressionPython,
            "C": LinearRegressionC,
            "CUDA": LinearRegressionCUDA
        }
    }

    # Check if the model exists for the specified language
    if model_name in model_mapping and model_language in model_mapping[model_name]:
        model_class = model_mapping[model_name][model_language]
        logger.info(f"Model '{model_name}' for language '{model_language}' selected.")
        return model_class()
    else:
        logger.error(f"Unsupported model '{model_name}' for language '{model_language}'")
        raise HTTPException(status_code=400, detail="Model name not supported.")