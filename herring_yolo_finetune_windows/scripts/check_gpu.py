import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
