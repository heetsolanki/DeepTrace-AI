
from pathlib import Path
from collections import defaultdict
import hashlib

DATASET_PATH = Path(
    "data/raw/real_vs_fake/real-vs-fake"
)

SPLITS = ["train", "valid", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def calculate_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()

def main():
    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(DATASET_PATH)

    hash_locations = defaultdict(set)
    image_count = 0

    print("DEEPTRACE AI - DATASET DUPLICATE CHECK")
    print("=" * 45)

    for split in SPLITS:
        split_path = DATASET_PATH / split

        if not split_path.is_dir():
            raise FileNotFoundError(split_path)

        count = 0

        for path in split_path.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower() not in EXTENSIONS
            ):
                continue

            image_hash = calculate_hash(path)
            hash_locations[image_hash].add(split)

            count += 1
            image_count += 1

        print(f"{split}: {count} images checked")

    print("\nChecking cross-split duplicates...")

    for first, second in [
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test")
    ]:
        duplicates = sum(
            1
            for locations in hash_locations.values()
            if first in locations and second in locations
        )

        print(
            f"{first} vs {second}: "
            f"{duplicates} duplicate hashes"
        )

    print(f"\nTotal images checked: {image_count}")
    print(
        "Unique file hashes:",
        len(hash_locations)
    )

    print("\nNote: This checks exact file duplicates only.")
    print(
        "Visually similar images and shared "
        "source identities are not detected."
    )

if __name__ == "__main__":
    main()
