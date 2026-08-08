import os
import sys
import torch

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.unetplusplus_model import UNetPlusPlus

# Path to trained model
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "unetplusplus",
    "best_unetplusplus.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetPlusPlus(
    num_classes=1,
    input_channels=1,
    deep_supervision=False
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("✅ UNet++ loaded successfully")