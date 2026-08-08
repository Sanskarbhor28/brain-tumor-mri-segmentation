import os
import sys
import gc
import torch

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
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

# Render Free should use CPU
device = torch.device("cpu")

print(f"Using device: {device}")

# ============================================================
# HUGGING FACE
# ============================================================

HF_REPO = "CursedShadow2x8/brain-tumor-segmentation-models"

# ============================================================
# RENDER MODE
# ============================================================

# Set DEPLOYMENT=render in Render Environment Variables
IS_RENDER = os.getenv("DEPLOYMENT", "").lower() == "render"

if IS_RENDER:
    AVAILABLE_MODELS = [
        "unetplusplus"
    ]
else:
    AVAILABLE_MODELS = [
        "unet",
        "residual_unet",
        "unetplusplus"
    ]

print("Available models:")
for model in AVAILABLE_MODELS:
    print(f"  ✓ {model}")

# ============================================================
# MODEL CACHE
# ============================================================

MODELS = {}


# ============================================================
# CHECKPOINT PATH
# ============================================================

def get_checkpoint(model_name):

    local_paths = {
        "unet": os.path.join(
            PROJECT_ROOT,
            "checkpoints",
            "unet_epoch_5.pth"
        ),

        "residual_unet": os.path.join(
            PROJECT_ROOT,
            "checkpoints",
            "residual_unet",
            "best_residual_unet.pth"
        ),

        "unetplusplus": os.path.join(
            PROJECT_ROOT,
            "checkpoints",
            "unetplusplus",
            "best_unetplusplus.pth"
        )
    }

    local_path = local_paths[model_name]

    # --------------------------------------------------------
    # Local machine
    # --------------------------------------------------------

    if os.path.exists(local_path):
        print(f"Using local checkpoint: {local_path}")
        return local_path

    # --------------------------------------------------------
    # Render / Hugging Face
    # --------------------------------------------------------

    print(
        f"Local checkpoint not found for {model_name}."
    )

    try:

        from huggingface_hub import hf_hub_download

        hf_files = {
            "unet": "unet_epoch_5.pth",
            "residual_unet": "best_residual_unet.pth",
            "unetplusplus": "best_unetplusplus.pth"
        }

        print(
            f"Downloading {hf_files[model_name]} "
            f"from Hugging Face..."
        )

        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=hf_files[model_name]
        )

        print(
            f"✓ Downloaded checkpoint: {downloaded}"
        )

        return downloaded

    except Exception as e:

        raise RuntimeError(
            f"Could not obtain checkpoint for "
            f"{model_name}: {e}"
        )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(model_name):

    if model_name == "unet":

        return UNet().to(device)

    elif model_name == "residual_unet":

        return ResidualUNet().to(device)

    elif model_name == "unetplusplus":

        return UNetPlusPlus().to(device)

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_name):

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if model_name not in AVAILABLE_MODELS:

        raise ValueError(
            f"Model '{model_name}' is not available. "
            f"Available: {AVAILABLE_MODELS}"
        )

    # --------------------------------------------------------
    # Return cached model if already loaded
    # --------------------------------------------------------

    if model_name in MODELS:

        print(
            f"✓ {model_name} already loaded"
        )

        return MODELS[model_name]

    print()
    print("=" * 60)
    print(f"Loading {model_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # Get checkpoint
    # --------------------------------------------------------

    checkpoint_path = get_checkpoint(
        model_name
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model(
        model_name
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print("Loading weights...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # Support both checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    MODELS[model_name] = model

    # Delete checkpoint from RAM
    del checkpoint
    del state_dict

    gc.collect()

    print(
        f"✓ {model_name} loaded successfully"
    )

    print("=" * 60)

    return model


# ============================================================
# UNLOAD MODEL
# ============================================================

def unload_model(model):

    if model is None:
        return

    # Find model name
    model_name = None

    for name, loaded_model in list(
        MODELS.items()
    ):

        if loaded_model is model:

            model_name = name
            break

    # Remove from cache
    if model_name is not None:

        del MODELS[model_name]

        print(
            f"✓ Removed {model_name} from memory"
        )

    # Delete reference
    del model

    # Python cleanup
    gc.collect()

    # CUDA cleanup
    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    print(
        "✓ Memory cleanup complete"
    )