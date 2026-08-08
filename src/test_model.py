import torch

from model import UNet


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = UNet().to(device)

# Fake MRI batch
x = torch.randn(
    2, 1, 256, 256,
    device=device
)

with torch.no_grad():
    output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

print(
    "Model parameters:",
    sum(p.numel() for p in model.parameters())
)