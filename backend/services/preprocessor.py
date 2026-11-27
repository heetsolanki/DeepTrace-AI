import torch
from PIL import Image
from torchvision import transforms

# This is the exact pipeline EfficientNet-B0 expects
preprocess_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(pil_image):
    """
    Takes a PIL Image (e.g., a cropped face).
    Returns a PyTorch tensor ready to feed into a model.
    Shape will be (3, 224, 224) — 3 color channels.
    """
    return preprocess_pipeline(pil_image)