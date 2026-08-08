import os
import sys
import torch

from huggingface_hub import hf_hub_download


# ============================================================
# PROJECT ROOT
# ============================================================

# Current file:
#
# C:\Project\BrainTumorResearch\backend\models\model_manager.py
#
# Go:
#
# models -> backend -> BrainTumorResearch
#
# Therefore we need TWO ".."

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"Project root: {PROJECT_ROOT}")


# ============================================================
# IMPORT MODEL ARCHITECTURES
# ============================================================

from src.model import UNet
from src.residual_model import ResidualUNet
from src.unetplusplus_model import UNetPlusPlus


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# HUGGING FACE
# ============================================================

HF_REPO_ID = (
    "CursedShadow2x8/"
    "brain-tumor-segmentation-models"
)

HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# CHECKPOINT FILES
# ============================================================

CHECKPOINT_FILES = {

    "unet":
        "unet_epoch_5.pth",

    "residual_unet":
        "best_residual_unet.pth",

    "unetplusplus":
        "best_unetplusplus.pth",
}


# ============================================================
# MODEL STORAGE
# ============================================================

MODELS = {}


# ============================================================
# DOWNLOAD CHECKPOINT
# ============================================================

def download_checkpoint(model_name: str) -> str:

    filename = CHECKPOINT_FILES[model_name]

    print()
    print("=" * 60)
    print(f"Downloading checkpoint: {model_name}")
    print(f"Repository: {HF_REPO_ID}")
    print(f"File: {filename}")
    print("=" * 60)

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=HF_TOKEN if HF_TOKEN else None,
    )

    print(
        f"✓ Checkpoint ready: {path}"
    )

    return path


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path: str
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Your checkpoints were saved using:
    #
    # checkpoint["model_state_dict"]

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint["model_state_dict"]
        )

    else:

        # Fallback for raw state_dict
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# LOAD ALL MODELS
# ============================================================

def load_models():

    print()
    print("=" * 60)
    print("LOADING BRAIN TUMOR SEGMENTATION MODELS")
    print("=" * 60)


    # ========================================================
    # UNET
    # ========================================================

    print()
    print("Loading UNet...")

    unet_checkpoint = (
        download_checkpoint("unet")
    )

    unet = UNet().to(device)

    load_checkpoint(
        unet,
        unet_checkpoint
    )

    MODELS["unet"] = unet

    print("✓ UNet loaded")


    # ========================================================
    # RESIDUAL UNET
    # ========================================================

    print()
    print("Loading Residual UNet...")

    residual_checkpoint = (
        download_checkpoint(
            "residual_unet"
        )
    )

    residual = (
        ResidualUNet().to(device)
    )

    load_checkpoint(
        residual,
        residual_checkpoint
    )

    MODELS["residual_unet"] = (
        residual
    )

    print(
        "✓ Residual UNet loaded"
    )


    # ========================================================
    # UNET++
    # ========================================================

    print()
    print("Loading UNet++...")

    unetplusplus_checkpoint = (
        download_checkpoint(
            "unetplusplus"
        )
    )

    unetpp = (
        UNetPlusPlus().to(device)
    )

    load_checkpoint(
        unetpp,
        unetplusplus_checkpoint
    )

    MODELS["unetplusplus"] = (
        unetpp
    )

    print("✓ UNet++ loaded")


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 60)
    print("✓ ALL MODELS LOADED SUCCESSFULLY")
    print("=" * 60)

    print()
    print("Available models:")

    for model_name in MODELS:

        print(
            f"  ✓ {model_name}"
        )

    print()
    print(
        f"Device: {device}"
    )

    print()


# ============================================================
# START MODEL LOADING
# ============================================================

load_models()