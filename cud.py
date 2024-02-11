import torch

# Set the default device to CUDA
torch.cuda.set_device(0)  # 0 represents the GPU device index

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
    print("CuDNN is enabled:", torch.backends.cudnn.enabled)
else:
    print("GPU is not available")
