"""Check whether the active Python environment can see a CUDA-capable PyTorch GPU."""
from __future__ import annotations
import platform
import sys

print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
try:
    import torch
except Exception as exc:
    print(f"PyTorch import failed: {exc}")
    print("Install dependencies in this exact Python environment first.")
    raise SystemExit(2)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime bundled with PyTorch: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GiB")
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"CUDA matrix test: PASS | result={y[0, 0].item():.4f}")
    raise SystemExit(0)

print("CUDA is not available in this Python environment.")
print("Check that this is the same interpreter used by PyCharm/run.py, that the GPU is NVIDIA, and that its driver supports the selected PyTorch CUDA wheel.")
raise SystemExit(1)
