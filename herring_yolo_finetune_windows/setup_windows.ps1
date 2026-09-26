Write-Host "Creating Python virtual environment..."
python -m venv .venv

Write-Host "Activating environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing requirements..."
pip install -r requirements.txt

Write-Host "Checking GPU availability..."
python scripts\check_gpu.py

Write-Host "Setup complete."
