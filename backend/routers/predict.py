import os
import time
import torch

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from models.model_manager import MODELS, device
from utils.preprocess import preprocess_image
from utils.postprocess import save_mask
from utils.overlay import create_overlay


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


AVAILABLE_MODELS = [
    "unet",
    "residual_unet",
    "unetplusplus"
]


# ============================================================
# SINGLE MODEL PREDICTION
# ============================================================

@router.post("/")
async def predict(
    model_name: str = "unetplusplus",
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    if model_name not in AVAILABLE_MODELS:

        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Invalid model name.",
                "available_models": AVAILABLE_MODELS
            }
        )

    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    image_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(image_path, "wb") as f:
        f.write(await file.read())

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image = preprocess_image(image_path).to(device)

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    selected_model = MODELS.get(model_name)

    if selected_model is None:

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Model {model_name} is not loaded."
            }
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():

        output = selected_model(image)

        output = torch.sigmoid(output)

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = (
        time.time() - start_time
    ) * 1000

    # --------------------------------------------------------
    # Save mask
    # --------------------------------------------------------

    mask_name = (
        f"{model_name}_{file.filename}"
    )

    mask_path = os.path.join(
        OUTPUT_FOLDER,
        mask_name
    )

    save_mask(
        output,
        mask_path
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay_name = (
        f"overlay_{model_name}_{file.filename}"
    )

    overlay_path = os.path.join(
        OUTPUT_FOLDER,
        overlay_name
    )

    create_overlay(
        image_path,
        mask_path,
        overlay_path
    )

    # --------------------------------------------------------
    # Tumor percentage
    # --------------------------------------------------------

    mask = (
        output > 0.5
    ).float()

    tumor_pixels = mask.sum().item()

    total_pixels = mask.numel()

    tumor_percentage = (
        tumor_pixels /
        total_pixels
    ) * 100

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if tumor_pixels > 0:

        confidence = (
            output[mask.bool()]
            .mean()
            .item()
        )

    else:

        confidence = 0.0

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return JSONResponse(
        {
            "status": "success",
            "model": model_name,
            "filename": file.filename,

            "mask_file": mask_name,

            "overlay_file": overlay_name,

            "tumor_percentage": round(
                tumor_percentage,
                2
            ),

            "confidence": round(
                confidence,
                4
            ),

            "inference_time_ms": round(
                inference_time,
                2
            )
        }
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

@router.post("/compare")
async def compare_models(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    image_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(image_path, "wb") as f:
        f.write(await file.read())

    # --------------------------------------------------------
    # Preprocess ONCE
    # --------------------------------------------------------

    image = preprocess_image(
        image_path
    ).to(device)

    comparison_results = {}

    # ========================================================
    # RUN ALL MODELS
    # ========================================================

    for model_name in AVAILABLE_MODELS:

        selected_model = MODELS.get(
            model_name
        )

        if selected_model is None:

            comparison_results[model_name] = {
                "status": "error",
                "message": "Model not loaded"
            }

            continue

        # ----------------------------------------------------
        # Start timer
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            output = selected_model(image)

            output = torch.sigmoid(
                output
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_time = (
            time.time() - start_time
        ) * 1000

        # ----------------------------------------------------
        # Save mask
        # ----------------------------------------------------

        mask_name = (
            f"{model_name}_{file.filename}"
        )

        mask_path = os.path.join(
            OUTPUT_FOLDER,
            mask_name
        )

        save_mask(
            output,
            mask_path
        )

        # ----------------------------------------------------
        # Create overlay
        # ----------------------------------------------------

        overlay_name = (
            f"overlay_{model_name}_{file.filename}"
        )

        overlay_path = os.path.join(
            OUTPUT_FOLDER,
            overlay_name
        )

        create_overlay(
            image_path,
            mask_path,
            overlay_path
        )

        # ----------------------------------------------------
        # Calculate tumor area
        # ----------------------------------------------------

        mask = (
            output > 0.5
        ).float()

        tumor_pixels = (
            mask.sum().item()
        )

        total_pixels = (
            mask.numel()
        )

        tumor_percentage = (
            tumor_pixels /
            total_pixels
        ) * 100

        # ----------------------------------------------------
        # Calculate confidence
        # ----------------------------------------------------

        if tumor_pixels > 0:

            confidence = (
                output[mask.bool()]
                .mean()
                .item()
            )

        else:

            confidence = 0.0

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        comparison_results[model_name] = {

            "status": "success",

            "model": model_name,

            "mask_file": mask_name,

            "overlay_file": overlay_name,

            "tumor_percentage": round(
                tumor_percentage,
                2
            ),

            "confidence": round(
                confidence,
                4
            ),

            "inference_time_ms": round(
                inference_time,
                2
            )
        }

    # ========================================================
    # RETURN COMPARISON
    # ========================================================

    return JSONResponse(
        {
            "status": "success",

            "filename": file.filename,

            "models": comparison_results
        }
    )