
from pathlib import Path
import csv
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, recall_score

SEED = 42
BATCH_SIZE = 16
EPOCHS = 5
PATIENCE = 3

TRAIN_PER_CLASS = 20000
VALID_PER_CLASS = 1000

DATASET_PATH = Path(
    "data/raw/real_vs_fake/real-vs-fake"
)

INPUT_MODEL = Path(
    "models/efficientnet_b0_finetuned.pth"
)

OUTPUT_MODEL = Path(
    "models/efficientnet_b0_large_best.pth"
)

HISTORY_PATH = Path(
    "research/large_training_history.csv"
)

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training device:", device)

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

def create_subset(dataset, count):
    rng = random.Random(SEED)
    indices = []

    for class_id in range(len(dataset.classes)):
        class_indices = [
            i for i, label in enumerate(dataset.targets)
            if label == class_id
        ]

        if len(class_indices) < count:
            raise ValueError(
                f"Not enough images for class {class_id}"
            )

        rng.shuffle(class_indices)
        indices.extend(class_indices[:count])

    rng.shuffle(indices)
    return Subset(dataset, indices)

train_loader = DataLoader(
    create_subset(train_dataset, TRAIN_PER_CLASS),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

valid_loader = DataLoader(
    create_subset(valid_dataset, VALID_PER_CLASS),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print("Training images:", len(train_loader.dataset))
print("Validation images:", len(valid_loader.dataset))

# Load previously fine-tuned model
checkpoint = torch.load(
    INPUT_MODEL,
    map_location="cpu",
    weights_only=True
)

assert checkpoint["class_to_idx"] == train_dataset.class_to_idx

model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)

model.load_state_dict(checkpoint["model_state_dict"])

# Freeze feature extractor
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Unfreeze final two feature blocks
for block in list(model.features.children())[-2:]:
    for parameter in block.parameters():
        parameter.requires_grad = True

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
        "lr": 0.000005
    },
    {
        "params": model.classifier.parameters(),
        "lr": 0.00005
    }
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)

best_f1 = -1.0
best_accuracy = 0.0
epochs_without_improvement = 0

OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

history = []

for epoch in range(EPOCHS):
    model.train()

    # Keep BatchNorm running statistics fixed
    for module in model.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    train_loss_sum = 0.0
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

        train_loss_sum += loss.item() * images.size(0)
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    # Validation
    model.eval()

    valid_loss_sum = 0.0
    valid_total = 0
    true_labels = []
    predicted_labels = []

    with torch.inference_mode():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            valid_loss_sum += loss.item() * images.size(0)
            valid_total += labels.size(0)

            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(
                predictions.cpu().tolist()
            )

    valid_loss = valid_loss_sum / valid_total

    valid_accuracy = sum(
        p == t
        for p, t in zip(predicted_labels, true_labels)
    ) / valid_total

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    fake_recall = recall_score(
        true_labels,
        predicted_labels,
        pos_label=train_dataset.class_to_idx["fake"],
        zero_division=0
    )

    # Scheduler monitors validation F1
    scheduler.step(macro_f1)

    current_lr = optimizer.param_groups[1]["lr"]

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2%}")
    print(f"Valid Loss: {valid_loss:.4f}")
    print(f"Valid Accuracy: {valid_accuracy:.2%}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Fake Recall: {fake_recall:.2%}")
    print(f"Classifier LR: {current_lr:.8f}")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
        "macro_f1": macro_f1,
        "fake_recall": fake_recall,
        "classifier_lr": current_lr
    })

    # Save history after every epoch
    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )
        writer.writeheader()
        writer.writerows(history)

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_accuracy = valid_accuracy
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_dataset.class_to_idx,
            "epoch": epoch + 1,
            "validation_accuracy": valid_accuracy,
            "macro_f1": macro_f1,
            "fake_recall": fake_recall
        }, OUTPUT_MODEL)

        print("Best large-dataset model saved!")

    else:
        epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

print("\nTraining completed!")
print(f"Best validation accuracy: {best_accuracy:.2%}")
print(f"Best macro F1: {best_f1:.4f}")
print("Model:", OUTPUT_MODEL)
print("History:", HISTORY_PATH)
