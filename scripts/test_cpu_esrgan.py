import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import spandrel
import numpy as np
from enhance import config as C

print("- Loading model on CPU")
m = spandrel.ModelLoader().load_from_file(C.ESRGAN_MODEL)
net = m.model.to("cpu").eval()

print("- Creating dummy array (2240x1260 to 4480x2520 via x2)")
dummy = torch.randn(1, 3, 315, 560, device="cpu", dtype=torch.float32)

print("- Warming up")
with torch.inference_mode():
    out = net(dummy)

print("- Testing speed")
t0 = time.time()
with torch.inference_mode():
    for _ in range(5):
        out = net(dummy)

dt = time.time() - t0
print(f"CPU PyTorch rate: {(5/dt):.2f} fps (1 batch)")
