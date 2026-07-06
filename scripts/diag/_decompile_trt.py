"""Decompile the .pyc file using decompyle3."""
import marshal
import struct

path = r"D:\W_LBM - Copy\wanphys\examples\__pycache__\fluid_grid_lbm_oneway_moving_rigid_visual_trt.cpython-311.pyc"

with open(path, "rb") as f:
    raw = f.read()

code = marshal.loads(raw[16:])

from decompyle3 import deparse_code
out_path = r"D:\W_LBM - Copy\wanphys\examples\fluid_grid_lbm_oneway_moving_rigid_visual_trt.py"
with open(out_path, "w", encoding="utf-8") as out:
    deparse_code(3.11, code, out)

print(f"Recovered source written to {out_path}")
import os
print(f"File size: {os.path.getsize(out_path)} bytes")