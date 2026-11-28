
from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Configuration
SEED = 42
BATCH_SIZE = 16
EPOCHS = 5

TRAIN_PER_CLASS = 5000
VALID_PER_CLASS = 1000

DATASET_PATH = Path(
    "data/raw/real_vs_fake/real-vs-fake"
)

BASELINE_PATH = Path(
    "models/efficientnet_b0_best.pth"
)

FINETUNED_PATH = Path(
    "models/efficientnet_b0_finetuned.pth"
)

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Image transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
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

# Reproduce baseline sampling
def create_subset(dataset, count):
    rng = random.Random(SEED)
    indices = []

    for class_id in range(len(dataset.classes)):
        class_indices = [
            i for i, label in enumerate(dataset.targets)
            if label == class_id
        ]
        rng.shuffle(class_indices)
        indices.extend(class_indices[:count])

    rng.shuffle(indices)
    return Subset(dataset, indices)

train_loader = DataLoader(
    create_subset(train_dataset, TRAIN_PER_CLASS),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

valid_loader = DataLoader(
    create_subset(valid_dataset, VALID_PER_CLASS),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Initialize architecture
model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

# Load previously trained baseline
checkpoint = torch.load(
    BASELINE_PATH,
    map_location="cpu",
    weights_only=True
)

assert checkpoint["class_to_idx"] == train_dataset.class_to_idx

model.load_state_dict(checkpoint["model_state_dict"])

# Freeze entire feature extractor first
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Unfreeze final two feature blocks
for block in list(model.features.children())[-2:]:
    for parameter in block.parameters():
        parameter.requires_grad = True

# Classification head remains trainable
for parameter in model.classifier.parameters():
    parameter.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW([
    {
        "params": [
            p for p in model.features.parameters()
            if p.requires_grad
        ],
        "lr": 0.00001
    },
    {
        "params": model.classifier.parameters(),
        "lr": 0.0001
    }
])

best_accuracy = 0.0

FINETUNED_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

for epoch in range(EPOCHS):
    model.train()

    # Keep BatchNorm running statistics fixed
    for module in model.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_accuracy = train_correct / train_total
    train_loss /= train_total

    # Validation
    model.eval()

    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0

    with torch.inference_mode():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            valid_correct += (
                predictions == labels
            ).sum().item()

            valid_total += labels.size(0)

    valid_accuracy = valid_correct / valid_total
    valid_loss /= valid_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2%}")
    print(f"Valid Loss: {valid_loss:.4f}")
    print(f"Valid Accuracy: {valid_accuracy:.2%}")

    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_dataset.class_to_idx,
            "epoch": epoch + 1,
            "validation_accuracy": valid_accuracy
        }, FINETUNED_PATH)

        print("Best fine-tuned model saved!")

print("\nFine-tuning completed!")
print(f"Best validation accuracy: {best_accuracy:.2%}")
