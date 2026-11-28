
from pathlib import Path
from collections import Counter

DATASET = Path(
    "data/raw/real_vs_fake/real-vs-fake"
)

extensions = {".jpg", ".jpeg", ".png", ".webp"}

print("DEEPTRACE AI - DATASET INSPECTION")
print("=" * 45)

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET}"
    )

for split in ["train", "valid", "test"]:
    split_path = DATASET / split

    print(f"\n{split.upper()} DATASET")

    if not split_path.exists():
        print("Directory not found")
        continue

    counts = Counter()

    for file in split_path.rglob("*"):
        if file.is_file() and file.suffix.lower() in extensions:
            relative = file.relative_to(split_path)
            label = relative.parts[0] if len(relative.parts) > 1 else "unclassified"
            counts[label] += 1

    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")

    print("Total:", sum(counts.values()))

print("\nCSV FILES")
for csv_file in DATASET.glob("*.csv"):
    print(csv_file.name)

print("\nInspection complete.")
