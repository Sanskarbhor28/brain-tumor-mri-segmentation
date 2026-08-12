import os
import time
import torch

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from models.model_manager import (
    MODELS,
    device,
    AVAILABLE_MODELS
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
# MODEL STATUS
# ============================================================

@router.get("/status")
async def model_status():

    return {
        "status": "success",

        "device":
            str(device),

        "available_models":
            AVAILABLE_MODELS,

        "loaded_models":
            list(
                MODELS.keys()
            )
    }


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

                "message":
                    "Invalid model name.",

                "available_models":
                    AVAILABLE_MODELS
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


    # --------------------------------------------------------
    # Get model
    # --------------------------------------------------------

    selected_model = MODELS.get(
        model_name
    )

    if selected_model is None:

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",

                "message":
                    f"Model {model_name} "
                    f"is not loaded."
            }
        )


    try:

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        image = preprocess_image(
            image_path
        ).to(device)


        # ----------------------------------------------------
        # Start timer
        # ----------------------------------------------------

        if device.type == "cuda":

            torch.cuda.synchronize()

        start_time = time.perf_counter()


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

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
            time.perf_counter()
            - start_time
        ) * 1000


        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        mask = (
            output > 0.5
        ).float()


        # ----------------------------------------------------
        # Tumor percentage
        # ----------------------------------------------------

        tumor_pixels = (
            mask.sum().item()
        )

        total_pixels = (
            mask.numel()
        )

        if total_pixels > 0:

            tumor_percentage = (
                tumor_pixels /
                total_pixels
            ) * 100

        else:

            tumor_percentage = 0.0


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        if tumor_pixels > 0:

            confidence = (
                output[
                    mask.bool()
                ]
                .mean()
                .item()
            )

        else:

            confidence = 0.0


        # ----------------------------------------------------
        # Save mask
        # ----------------------------------------------------

        mask_name = (
            f"{model_name}_"
            f"{safe_filename}"
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
            f"overlay_"
            f"{model_name}_"
            f"{safe_filename}"
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
        # Response
        # ----------------------------------------------------

        return JSONResponse(
            {
                "status":
                    "success",

                "model":
                    model_name,

                "filename":
                    safe_filename,

                "mask_file":
                    mask_name,

                "overlay_file":
                    overlay_name,

                "tumor_percentage":
                    round(
                        tumor_percentage,
                        2
                    ),

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "inference_time_ms":
                    round(
                        inference_time,
                        2
                    )
            }
        )


    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",

                "message":
                    str(e)
            }
        )


# ============================================================
# THREE MODEL COMPARISON
# ============================================================

@router.post("/compare")
async def compare_models(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Save uploaded image
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


    # --------------------------------------------------------
    # Preprocess ONCE
    # --------------------------------------------------------

    image = preprocess_image(
        image_path
    ).to(device)


    comparison_results = {}


    # ========================================================
    # RUN ALL THREE MODELS
    # ========================================================

    for model_name in AVAILABLE_MODELS:

        print()
        print(
            f"Running {model_name}..."
        )


        # ----------------------------------------------------
        # Get model
        # ----------------------------------------------------

        selected_model = MODELS.get(
            model_name
        )


        if selected_model is None:

            comparison_results[
                model_name
            ] = {

                "status":
                    "error",

                "message":
                    "Model not loaded"
            }

            continue


        try:

            # ------------------------------------------------
            # Synchronize GPU
            # ------------------------------------------------

            if device.type == "cuda":

                torch.cuda.synchronize()


            # ------------------------------------------------
            # Start timer
            # ------------------------------------------------

            start_time = (
                time.perf_counter()
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            with torch.no_grad():

                output = (
                    selected_model(
                        image
                    )
                )

                output = torch.sigmoid(
                    output
                )


            # ------------------------------------------------
            # Synchronize GPU
            # ------------------------------------------------

            if device.type == "cuda":

                torch.cuda.synchronize()


            # ------------------------------------------------
            # Inference time
            # ------------------------------------------------

            inference_time = (
                time.perf_counter()
                - start_time
            ) * 1000


            # ------------------------------------------------
            # Binary mask
            # ------------------------------------------------

            mask = (
                output > 0.5
            ).float()


            # ------------------------------------------------
            # Tumor area
            # ------------------------------------------------

            tumor_pixels = (
                mask.sum().item()
            )

            total_pixels = (
                mask.numel()
            )


            if total_pixels > 0:

                tumor_percentage = (
                    tumor_pixels /
                    total_pixels
                ) * 100

            else:

                tumor_percentage = 0.0


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            if tumor_pixels > 0:

                confidence = (
                    output[
                        mask.bool()
                    ]
                    .mean()
                    .item()
                )

            else:

                confidence = 0.0


            # ------------------------------------------------
            # Save mask
            # ------------------------------------------------

            mask_name = (
                f"{model_name}_"
                f"{safe_filename}"
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
                f"overlay_"
                f"{model_name}_"
                f"{safe_filename}"
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
            # Store result
            # ------------------------------------------------

            comparison_results[
                model_name
            ] = {

                "status":
                    "success",

                "model":
                    model_name,

                "mask_file":
                    mask_name,

                "overlay_file":
                    overlay_name,

                "tumor_percentage":
                    round(
                        tumor_percentage,
                        2
                    ),

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "inference_time_ms":
                    round(
                        inference_time,
                        2
                    )
            }


            print(
                f"✓ {model_name} completed"
            )


        except Exception as e:

            print(
                f"✗ {model_name} failed:",
                repr(e)
            )

            comparison_results[
                model_name
            ] = {

                "status":
                    "error",

                "model":
                    model_name,

                "message":
                    str(e)
            }


    # --------------------------------------------------------
    # Cleanup temporary image tensor
    # --------------------------------------------------------

    del image


    if device.type == "cuda":

        torch.cuda.synchronize()


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return JSONResponse(
        {
            "status":
                "success",

            "filename":
                safe_filename,

            "models":
                comparison_results
        }
    )