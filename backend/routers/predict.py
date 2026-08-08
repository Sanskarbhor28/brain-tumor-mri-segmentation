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
    AVAILABLE_MODELS,
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
    # Save image
    # --------------------------------------------------------

    safe_filename = os.path.basename(
        file.filename
    )

    image_path = os.path.join(
        UPLOAD_FOLDER,
        safe_filename
    )

    with open(
        image_path,
        "wb"
    ) as f:

        f.write(
            await file.read()
        )

    selected_model = None

    try:

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        image = preprocess_image(
            image_path
        ).to(device)

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

        output = (
            output
            .detach()
            .cpu()
        )

        # ----------------------------------------------------
        # Save mask
        # ----------------------------------------------------

        mask_name = (
            f"{model_name}_{safe_filename}"
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
            f"overlay_{model_name}_{safe_filename}"
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

                "filename": safe_filename,

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

    except Exception as e:

        print(
            f"Prediction error: {e}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

    finally:

        # ----------------------------------------------------
        # CRITICAL FOR RENDER FREE
        # ----------------------------------------------------

        if selected_model is not None:

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
    # On Render Free:
    # AVAILABLE_MODELS = ["unetplusplus"]
    #
    # Locally:
    # AVAILABLE_MODELS =
    # ["unet", "residual_unet", "unetplusplus"]
    # --------------------------------------------------------

    safe_filename = os.path.basename(
        file.filename
    )

    image_path = os.path.join(
        UPLOAD_FOLDER,
        safe_filename
    )

    with open(
        image_path,
        "wb"
    ) as f:

        f.write(
            await file.read()
        )

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

            print()
            print("=" * 60)
            print(
                f"Running model: {model_name}"
            )
            print("=" * 60)

            # ------------------------------------------------
            # Load ONLY current model
            # ------------------------------------------------

            selected_model = load_model(
                model_name
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Move output to CPU
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
                f"{model_name}_{safe_filename}"
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
            # Overlay
            # ------------------------------------------------

            overlay_name = (
                f"overlay_{model_name}_{safe_filename}"
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
            # Tumor area
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
            # Confidence
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
            # Free model before next one
            # ------------------------------------------------

            if selected_model is not None:

                unload_model(
                    selected_model
                )

            selected_model = None

            cleanup_memory()

            print(
                f"✓ Memory cleaned after "
                f"{model_name}"
            )

    # ========================================================
    # RETURN
    # ========================================================

    return JSONResponse(
        {
            "status": "success",

            "filename": safe_filename,

            "models": comparison_results
        }
    )