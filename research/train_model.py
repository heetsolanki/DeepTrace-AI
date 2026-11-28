
from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Configuration
SEED = 42
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.001

TRAIN_SAMPLES_PER_CLASS = 5000
VALID_SAMPLES_PER_CLASS = 1000

DATASET_PATH = Path(
    "data/raw/real_vs_fake/real-vs-fake"
)

MODEL_PATH = Path("models/efficientnet_b0_best.pth")

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training device:", device)

# Image transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(
    DATASET_PATH / "train",
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    DATASET_PATH / "valid",
    transform=valid_transform
)

assert train_dataset.class_to_idx == valid_dataset.class_to_idx

print("Class mapping:", train_dataset.class_to_idx)

def create_subset(dataset, samples_per_class):
    rng = random.Random(SEED)
    indices = []

    for class_id in range(len(dataset.classes)):
        class_indices = [
            i for i, label in enumerate(dataset.targets)
            if label == class_id
        ]

        rng.shuffle(class_indices)
        indices.extend(class_indices[:samples_per_class])

    rng.shuffle(indices)
    return Subset(dataset, indices)

train_subset = create_subset(
    train_dataset, TRAIN_SAMPLES_PER_CLASS
)

valid_subset = create_subset(
    valid_dataset, VALID_SAMPLES_PER_CLASS
)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Pretrained EfficientNet-B0
weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(weights=weights)

# Freeze pretrained feature extractor
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace classification head
num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features, 2
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)

best_accuracy = 0.0
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    model.train()

    # Keep frozen BatchNorm statistics unchanged
    model.features.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    model.eval()

    valid_loss_sum = 0.0
    valid_correct = 0
    valid_total = 0

    with torch.inference_mode():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss_sum += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            valid_correct += (
                predictions == labels
            ).sum().item()

            valid_total += labels.size(0)

    valid_loss = valid_loss_sum / valid_total
    valid_accuracy = valid_correct / valid_total

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4%}")
    print(f"Valid Loss: {valid_loss:.4f}")
    print(f"Valid Accuracy: {valid_accuracy:.4%}")

    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_dataset.class_to_idx,
            "epoch": epoch + 1,
            "validation_accuracy": valid_accuracy
        }, MODEL_PATH)

        print("Best model saved!")

print("\nTraining completed.")
print(f"Best validation accuracy: {best_accuracy:.4%}")
print(f"Model saved at: {MODEL_PATH}")
