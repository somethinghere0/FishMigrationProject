Write-Host "Creating Python virtual environment..."
python -m venv .venv

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing requirements..."
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "Place Herring.mp4 inside the videos folder, then run:"
Write-Host "powershell -ExecutionPolicy Bypass -File run_windows.ps1"
