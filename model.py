import torch
import torch.nn as nn
import torchvision


def build_model():
    """Build the same ResNet18 binary classifier used by the team plan."""
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_model(model_path, device):
    """Load Member B's trained baseline.pth file."""
    model = build_model()

    checkpoint = torch.load(model_path, map_location=device)

    # Support either a plain state_dict or a checkpoint dictionary.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Handle models saved using DataParallel.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_state_dict[key.replace("module.", "", 1) if key.startswith("module.") else key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
