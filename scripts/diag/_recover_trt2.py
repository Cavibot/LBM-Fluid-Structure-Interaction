"""Recover source code from .pyc cache - try different header layouts."""
import marshal
import struct

path = r"D:\W_LBM - Copy\wanphys\examples\__pycache__\fluid_grid_lbm_oneway_moving_rigid_visual_trt.cpython-311.pyc"

with open(path, "rb") as f:
    raw = f.read()

print(f"File size: {len(raw)} bytes")
print(f"Magic bytes: {raw[:4].hex()}")
print(f"Bytes 4-8: {raw[4:8].hex()}")

# Try different offset positions for marshal data
offsets_to_try = [4, 8, 12, 16]

for offset in offsets_to_try:
    try:
        code = marshal.loads(raw[offset:])
        print(f"\nSUCCESS at offset={offset}!")
        print(f"Code object name: {code.co_name}")
        print(f"Code object filename: {code.co_filename}")
        break
    except Exception as e:
        print(f"  offset={offset}: {e}")