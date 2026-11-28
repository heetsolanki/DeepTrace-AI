
from pathlib import Path
import random
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ----------------------------------
# Command-line arguments
# ----------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=["baseline", "finetuned", "large", "full"],
    default="full"
)

parser.add_argument(
    "--split",
    choices=["valid", "test"],
    default="valid"
)

parser.add_argument(
    "--samples-per-class",
    type=int,
    default=None,
    help="Optional subset size per class"
)

args = parser.parse_args()

SEED = 42
BATCH_SIZE = 16

DATASET_PATH = Path(
    "data/raw/real_vs_fake/real-vs-fake"
)

MODEL_PATHS = {
    "baseline": Path(
        "models/efficientnet_b0_best.pth"
    ),
    "finetuned": Path(
        "models/efficientnet_b0_finetuned.pth"
    ),
    "large": Path(
        "models/efficientnet_b0_large_best.pth"
    ),
    "full": Path(
        "models/efficientnet_b0_full_best.pth"
    )
}

MODEL_PATH = MODEL_PATHS[args.model]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Evaluating model:", args.model)
print("Dataset split:", args.split)
print("Evaluation device:", device)

# ----------------------------------
# Image preprocessing
# ----------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    DATASET_PATH / args.split,
    transform=transform
)

# ----------------------------------
# Optional balanced subset
# ----------------------------------

evaluation_dataset = dataset

if args.samples_per_class is not None:

    if args.samples_per_class <= 0:
        parser.error(
            "--samples-per-class must be positive"
        )

    rng = random.Random(SEED)
    indices = []

    for class_id in range(len(dataset.classes)):

        class_indices = [
            i for i, label in enumerate(dataset.targets)
            if label == class_id
        ]

        if len(class_indices) < args.samples_per_class:
            raise ValueError(
                f"Class {class_id} has fewer than "
                f"{args.samples_per_class} images"
            )

        rng.shuffle(class_indices)

        indices.extend(
            class_indices[:args.samples_per_class]
        )

    rng.shuffle(indices)

    evaluation_dataset = Subset(dataset, indices)

print("Total images:", len(evaluation_dataset))
print("Class mapping:", dataset.class_to_idx)

loader = DataLoader(
    evaluation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

# ----------------------------------
# Load checkpoint
# ----------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

assert (
    checkpoint["class_to_idx"]
    == dataset.class_to_idx
), "Class mapping mismatch!"

# ----------------------------------
# Recreate EfficientNet-B0
# ----------------------------------

model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    len(dataset.classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

# ----------------------------------
# Model evaluation
# ----------------------------------

true_labels = []
predicted_labels = []

with torch.inference_mode():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        true_labels.extend(
            labels.tolist()
        )

        predicted_labels.extend(
            predictions.cpu().tolist()
        )

# ----------------------------------
# Results
# ----------------------------------

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
        labels=list(range(len(dataset.classes))),
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
        labels=list(range(len(dataset.classes)))
    )
)

print("\nSaved checkpoint epoch:",
      checkpoint["epoch"])

print("Saved validation accuracy:",
      checkpoint["validation_accuracy"])
