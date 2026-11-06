import torch
from torchvision import transforms
from PIL import Image

IMAGE_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def prepare_image(image):
    """Convert an uploaded PIL image into a model-ready batch tensor."""
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    return tensor


def predict_image(model, image, device):
    """Return the predicted label and both class probabilities."""
    input_tensor = prepare_image(image).to(device)

    with torch.no_grad():
        logit = model(input_tensor).reshape(-1)[0]
        real_probability = torch.sigmoid(logit).item()

    fake_probability = 1.0 - real_probability
    predicted_label = "Real" if real_probability >= 0.5 else "Fake"

    return {
        "label": predicted_label,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "input_tensor": input_tensor,
    }
