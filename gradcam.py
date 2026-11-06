import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class BinaryLogitTarget:
    """Grad-CAM target for a one-logit binary classifier.

    The project mapping is fake=0 and real=1. The model produces a
    single logit whose sigmoid is the probability of Real.
    """
    def __init__(self, explain_real):
        self.sign = 1.0 if explain_real else -1.0

    def __call__(self, model_output):
        return model_output.reshape(-1)[0] * self.sign   


def create_gradcam(model, input_tensor, original_image, predicted_label):
    """Create a Grad-CAM overlay for the predicted class."""
    target_layers = [model.layer4[-1]]
    targets = [BinaryLogitTarget(explain_real=(predicted_label == "Real"))]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    rgb_image = original_image.convert("RGB").resize((128, 128))
    rgb_array = np.asarray(rgb_image, dtype=np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_array,
        grayscale_cam[0],
        use_rgb=True
    )
    return visualization
