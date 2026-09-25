#!/bin/bash
set -e

echo "Creating Python virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Place Herring.mp4 inside the videos folder, then run:"
echo "bash run_mac.sh"
