import torch
from torchvision import models
from backend.services.detector import extract_face
from backend.services.preprocessor import preprocess_image

print("=" * 50)
print("DeepTrace AI - Model Inference Test")
print("=" * 50)

# 1. Load EfficientNet-B0 with pretrained ImageNet weights
print("\n[1] Loading EfficientNet-B0 (pretrained on ImageNet)...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# model.eval() disables dropout and switches batch norm
# to evaluation mode. Without this, results are inconsistent.
model.eval()

# Move model to GPU
model.to("cuda")
print(f"   Model loaded on: {next(model.parameters()).device}")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# 2. Get a preprocessed face tensor
print("\n[2] Preprocessing test face...")
face = extract_face("test_face.jpg")
if not face:
    print("   Error: No face found.")
    exit()
tensor = preprocess_image(face)

# Models expect a BATCH. We add a dimension at the front:
# (3, 224, 224) becomes (1, 3, 224, 224)
# The "1" means "one image in this batch"
tensor = tensor.unsqueeze(0).to("cuda")
print(f"   Input tensor shape: {tensor.shape}")

# 3. Run inference
print("\n[3] Running forward pass on GPU...")
with torch.no_grad():
    # torch.no_grad() disables gradient tracking.
    # We do not need gradients during inference, so this
    # saves memory and runs faster.
    output = model(tensor)

print(f"   Output tensor shape: {output.shape}")

# 4. Show top predictions (these are ImageNet classes, NOT real/fake)
top5 = torch.topk(output[0], 5)
print(f"\n[4] Top 5 ImageNet predictions (this is NOT a deepfake result):")
for i in range(5):
    confidence = torch.softmax(output[0], dim=0)[top5.indices[i]].item()
    print(f"   {top5.indices[i].item():>5}  {confidence:.4f}")

print("\n" + "=" * 50)
print("NOTE: These 1,000 classes are ImageNet categories")
print("(cats, cars, guitars, etc.). This model is NOT yet a")
print("deepfake detector. We will retrain the top layer next.")
print("=" * 50)