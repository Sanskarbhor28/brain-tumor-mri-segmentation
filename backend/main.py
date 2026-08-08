from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers.predict import router as predict_router


app = FastAPI(
    title="Brain Tumor Segmentation API",
    description=(
        "Brain Tumor MRI Segmentation using "
        "UNet, Residual UNet and UNet++"
    ),
    version="1.0.0"
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local development
        "http://localhost:5173",
        "http://127.0.0.1:5173",

        # Add your Vercel URL here after frontend deployment
        # Example:
        # "https://brain-tumor-research.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# SERVE OUTPUT IMAGES
# ==========================================================

app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)


# ==========================================================
# ROUTERS
# ==========================================================

app.include_router(
    predict_router
)


# ==========================================================
# HOME
# ==========================================================

@app.get("/")
async def home():
    return {
        "status": "success",
        "message": "Brain Tumor Segmentation API is Running",
        "models": [
            "UNet",
            "Residual UNet",
            "UNet++"
        ],
        "version": "1.0.0"
    }


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }