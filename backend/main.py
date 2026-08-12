from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers.predict import router as predict_router


# ==========================================================
# APP
# ==========================================================

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description=(
        "Brain Tumor MRI Segmentation using UNet++"
    ),
    version="1.0.0"
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        

        # Your deployed frontend
        # Add it here if you deploy frontend separately
        # "https://your-frontend.onrender.com",
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

        "message":
            "Brain Tumor Segmentation API is Running",

        "models": [
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