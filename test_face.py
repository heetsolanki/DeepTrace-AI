from backend.services.detector import extract_face
from backend.services.preprocessor import preprocess_image

image_path = "test_face.jpg"

# 1. Extract the face
print(f"Step 1: Extracting face from {image_path}...")
face = extract_face(image_path)
if not face:
    print("No face found. Stopping.")
    exit()
print(f"   Face size: {face.size[1]}x{face.size[0]} pixels")

# 2. Preprocess it
print("Step 2: Preprocessing (resize + normalize)...")
tensor = preprocess_image(face)
print(f"   Tensor shape: {tensor.shape}")
print(f"   Value range: {tensor.min().item():.3f} to {tensor.max().item():.3f}")
print(f"   Data type: {tensor.dtype}")