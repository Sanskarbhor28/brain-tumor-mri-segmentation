import os
import sys
import gc
import torch

from huggingface_hub import hf_hub_download


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# MODEL IMPORTS
# ============================================================

from src.model import UNet
from src.residual_model import ResidualUNet
from src.unetplusplus_model import UNetPlusPlus


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# CHECKPOINTS
# ============================================================

CHECKPOINT_FILES = {
    "unet": "unet_epoch_5.pth",
    "residual_unet": "best_residual_unet.pth",
    "unetplusplus": "best_unetplusplus.pth",
}


# ============================================================
# MODEL CLASSES
# ============================================================

MODEL_CLASSES = {
    "unet": UNet,
    "residual_unet": ResidualUNet,
    "unetplusplus": UNetPlusPlus,
}


# ============================================================
# CACHE DOWNLOADED FILES
# ============================================================

CHECKPOINT_PATHS = {}


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def get_checkpoint(model_name):

    if model_name in CHECKPOINT_PATHS:
        return CHECKPOINT_PATHS[model_name]

    filename = CHECKPOINT_FILES[model_name]

    print()
    print("=" * 60)
    print(f"Downloading/loading checkpoint: {model_name}")
    print(f"File: {filename}")
    print("=" * 60)

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=HF_TOKEN if HF_TOKEN else None,
    )

    CHECKPOINT_PATHS[model_name] = path

    print(f"✓ Checkpoint ready: {path}")

    return path


# ============================================================
# LOAD ONE MODEL
# ============================================================

def load_model(model_name):

    if model_name not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    print()
    print("=" * 60)
    print(f"LOADING {model_name.upper()}")
    print("=" * 60)

    checkpoint_path = get_checkpoint(
        model_name
    )

    model_class = MODEL_CLASSES[
        model_name
    ]

    model = model_class().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        f"✓ {model_name} loaded"
    )

    return model


# ============================================================
# UNLOAD MODEL
# ============================================================

def unload_model(model):

    if model is None:
        return

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("✓ Model memory released")