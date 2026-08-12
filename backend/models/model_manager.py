import os
import sys
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

MODELS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.dirname(
    MODELS_DIR
)

PROJECT_ROOT = os.path.dirname(
    BACKEND_DIR
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORT MODELS
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

print("=" * 60)
print("BRAIN TUMOR SEGMENTATION")
print("=" * 60)

print(
    f"Using device: {device}"
)


# ============================================================
# CHECKPOINT PATHS
# ============================================================

CHECKPOINTS = {

    # Your actual file is directly inside checkpoints/
    "unet":
        os.path.join(
            PROJECT_ROOT,
            "checkpoints",
            "unet_epoch_5.pth"
        ),

    "residual_unet":
        os.path.join(
            PROJECT_ROOT,
            "checkpoints",
            "residual_unet",
            "best_residual_unet.pth"
        ),

    "unetplusplus":
        os.path.join(
            PROJECT_ROOT,
            "checkpoints",
            "unetplusplus",
            "best_unetplusplus.pth"
        )
}


# ============================================================
# AVAILABLE MODELS
# ============================================================

AVAILABLE_MODELS = [
    "unet",
    "residual_unet",
    "unetplusplus"
]


# ============================================================
# MODEL STORAGE
# ============================================================

MODELS = {}


# ============================================================
# CHECK CHECKPOINTS
# ============================================================

def check_checkpoints():

    print()
    print("=" * 60)
    print("CHECKING CHECKPOINTS")
    print("=" * 60)

    for model_name in AVAILABLE_MODELS:

        path = CHECKPOINTS[
            model_name
        ]

        print(
            f"Checking {model_name}: {path}"
        )

        if os.path.exists(path):

            print(
                f"✓ {model_name} checkpoint found"
            )

        else:

            print(
                f"✗ {model_name} checkpoint NOT FOUND"
            )

            raise FileNotFoundError(
                f"Checkpoint not found for "
                f"{model_name}: {path}"
            )


# ============================================================
# LOAD ONE CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    model_name
):

    checkpoint_path = CHECKPOINTS[
        model_name
    ]

    print(
        f"Loading weights for {model_name}..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # --------------------------------------------------------
    # Support different checkpoint formats
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

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


    # --------------------------------------------------------
    # Remove DataParallel prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value


    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )


    # --------------------------------------------------------
    # Cleanup checkpoint memory
    # --------------------------------------------------------

    del checkpoint
    del state_dict
    del cleaned_state_dict


    return model


# ============================================================
# LOAD ALL MODELS
# ============================================================

def load_models():

    global MODELS

    # --------------------------------------------------------
    # Check files first
    # --------------------------------------------------------

    check_checkpoints()


    print()
    print("=" * 60)
    print("LOADING MODELS")
    print("=" * 60)


    # ========================================================
    # UNET
    # ========================================================

    print()
    print("Loading UNet...")

    unet = UNet().to(
        device
    )

    unet = load_checkpoint(
        unet,
        "unet"
    )

    unet.eval()

    MODELS[
        "unet"
    ] = unet

    print(
        "✓ UNet loaded"
    )


    # ========================================================
    # RESIDUAL UNET
    # ========================================================

    print()
    print("Loading Residual UNet...")

    residual_unet = (
        ResidualUNet().to(
            device
        )
    )

    residual_unet = load_checkpoint(
        residual_unet,
        "residual_unet"
    )

    residual_unet.eval()

    MODELS[
        "residual_unet"
    ] = residual_unet

    print(
        "✓ Residual UNet loaded"
    )


    # ========================================================
    # UNET++
    # ========================================================

    print()
    print("Loading UNet++...")

    unetplusplus = (
        UNetPlusPlus().to(
            device
        )
    )

    unetplusplus = load_checkpoint(
        unetplusplus,
        "unetplusplus"
    )

    unetplusplus.eval()

    MODELS[
        "unetplusplus"
    ] = unetplusplus

    print(
        "✓ UNet++ loaded"
    )


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

    print("=" * 60)


# ============================================================
# LOAD ALL MODELS ON STARTUP
# ============================================================

load_models()