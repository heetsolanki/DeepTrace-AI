import torch

print("--- DeepTrace AI Environment Check ---")
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name     : {torch.cuda.get_device_name(0)}")
    print("Status          : Ready for GPU-accelerated Deep Learning!")
else:
    print("Status          : WARNING - PyTorch cannot access GPU. Running on CPU.")