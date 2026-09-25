python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
Write-Host "If you have an NVIDIA GPU, install PyTorch CUDA first using the command from https://pytorch.org/get-started/locally/"
Write-Host "Example for CUDA 12.1: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
pip install -r requirements.txt
Write-Host "Setup complete."
