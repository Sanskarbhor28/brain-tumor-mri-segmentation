import os
import time
import gc
import torch

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from models.model_manager import (
    load_model,
    unload_model,
    device,
)

from utils.preprocess import preprocess_image
from utils.postprocess import save_mask
from utils.overlay import create_overlay


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# AVAILABLE MODELS
# ============================================================

AVAILABLE_MODELS = [
    "unet",
    "residual_unet",
    "unetplusplus"
]


# ============================================================
# MEMORY CLEANUP
# ============================================================

def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


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

    with open(
        image_path,
        "wb"
    ) as f:

        f.write(
            await file.read()
        )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image = preprocess_image(
        image_path
    ).to(device)

    selected_model = None

    try:

        # ----------------------------------------------------
        # Load ONLY requested model
        # ----------------------------------------------------

        selected_model = load_model(
            model_name
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():

            output = selected_model(
                image
            )

            output = torch.sigmoid(
                output
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_time = (
            time.time() - start_time
        ) * 1000

        # ----------------------------------------------------
        # Move output to CPU
        # ----------------------------------------------------

        output = output.detach().cpu()

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
        # Tumor percentage
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
        # Confidence
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
        # Response
        # ----------------------------------------------------

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

    finally:

        # ----------------------------------------------------
        # VERY IMPORTANT
        # Release model after prediction
        # ----------------------------------------------------

        unload_model(
            selected_model
        )

        selected_model = None

        cleanup_memory()


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

    with open(
        image_path,
        "wb"
    ) as f:

        f.write(
            await file.read()
        )

    # --------------------------------------------------------
    # Preprocess ONCE
    # --------------------------------------------------------

    image = preprocess_image(
        image_path
    ).to(device)

    comparison_results = {}

    # ========================================================
    # RUN MODELS ONE AT A TIME
    # ========================================================

    for model_name in AVAILABLE_MODELS:

        selected_model = None

        try:

            # ------------------------------------------------
            # Load ONLY this model
            # ------------------------------------------------

            print()
            print(
                "=" * 60
            )

            print(
                f"Running model: {model_name}"
            )

            print(
                "=" * 60
            )

            selected_model = load_model(
                model_name
            )

            # ------------------------------------------------
            # Start timer
            # ------------------------------------------------

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            with torch.no_grad():

                output = selected_model(
                    image
                )

                output = torch.sigmoid(
                    output
                )

            if device.type == "cuda":
                torch.cuda.synchronize()

            inference_time = (
                time.time() - start_time
            ) * 1000

            # ------------------------------------------------
            # Move output to CPU
            #
            # This is important because the model will
            # immediately be unloaded.
            # ------------------------------------------------

            output = (
                output
                .detach()
                .cpu()
            )

            # ------------------------------------------------
            # Save mask
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Create overlay
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Calculate tumor area
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Calculate confidence
            # ------------------------------------------------

            if tumor_pixels > 0:

                confidence = (
                    output[mask.bool()]
                    .mean()
                    .item()
                )

            else:

                confidence = 0.0

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            comparison_results[
                model_name
            ] = {

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

            print(
                f"✓ {model_name} completed"
            )

        except Exception as e:

            # ------------------------------------------------
            # Don't stop comparison if one model fails
            # ------------------------------------------------

            comparison_results[
                model_name
            ] = {

                "status": "error",

                "model": model_name,

                "message": str(e)
            }

            print(
                f"✗ {model_name} failed: {e}"
            )

        finally:

            # ------------------------------------------------
            # CRITICAL:
            # Release model BEFORE loading next model
            # ------------------------------------------------

            unload_model(
                selected_model
            )

            selected_model = None

            cleanup_memory()

            print(
                f"✓ Memory cleaned after {model_name}"
            )

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