from fastapi import FastAPI
from api.routes import data_routes

# Create FastAPI app
app = FastAPI(
    title="Machine Learning API",
    description="API for uploading data, training models, and making predictions.",
    version="1.0.0"
)

# Include routes
app.include_router(data_routes.router, prefix="/data", tags=["Data"])

# Health check route
@app.get("/")
async def root():
    return {"message": "ML API is running!"}