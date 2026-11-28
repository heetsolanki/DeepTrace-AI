
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["baseline", "finetuned", "large"],
    default="large"
)
args = parser.parse_args()

SEED = 42
BATCH_SIZE = 16
SAMPLES_PER_CLASS = 1000

DATASET_PATH = Path(
    "data/raw/real_vs_fake/real-vs-fake"
)

MODEL_PATHS = {
    "baseline": Path("models/efficientnet_b0_best.pth"),
    "finetuned": Path("models/efficientnet_b0_finetuned.pth"),
    "large": Path("models/efficientnet_b0_large_best.pth")
}

MODEL_PATH = MODEL_PATHS[args.model]

print("Evaluating:", args.model)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Evaluation device:", device)

# Same preprocessing used during validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    DATASET_PATH / "valid",
    transform=transform
)

# Recreate the same validation subset
rng = random.Random(SEED)
indices = []

for class_id in range(len(dataset.classes)):
    class_indices = [
        i for i, label in enumerate(dataset.targets)
        if label == class_id
    ]

    rng.shuffle(class_indices)
    indices.extend(class_indices[:SAMPLES_PER_CLASS])

rng.shuffle(indices)

subset = Subset(dataset, indices)

loader = DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Load saved checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

assert (
    checkpoint["class_to_idx"]
    == dataset.class_to_idx
), "Class mapping mismatch!"

# Recreate the architecture
model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

true_labels = []
predicted_labels = []

with torch.inference_mode():
    for images, labels in loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        true_labels.extend(labels.tolist())
        predicted_labels.extend(
            predictions.cpu().tolist()
        )

print("\nDEEPTRACE AI - MODEL EVALUATION")
print("=" * 45)

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print(f"Accuracy: {accuracy:.4%}")

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=dataset.classes,
        digits=4,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1]
    )
)

print("\nSaved checkpoint epoch:",
      checkpoint["epoch"])

print("Saved validation accuracy:",
      checkpoint["validation_accuracy"])
