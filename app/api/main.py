from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="Machine Learning API",
    description="API for uploading data, training models, and making predictions.",
    version="1.0.0"
)

# Allow requests from GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fostercm.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check route
@app.get("/")
async def root():
    return {"message": "ML API is running!"}