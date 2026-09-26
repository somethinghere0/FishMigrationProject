from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Working directory:", Path.cwd())

for mod in ["yaml", "ultralytics", "cv2", "torch", "numpy", "pandas", "tqdm"]:
    spec = importlib.util.find_spec(mod)
    print(f"{mod:12s}:", "FOUND" if spec else "MISSING", spec.origin if spec else "")

try:
    import yaml
    print("PyYAML version:", yaml.__version__)
except Exception as e:
    print("PyYAML import error:", repr(e))

try:
    import ultralytics
    print("Ultralytics version:", ultralytics.__version__)
except Exception as e:
    print("Ultralytics import error:", repr(e))
