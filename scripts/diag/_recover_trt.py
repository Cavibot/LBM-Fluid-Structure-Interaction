"""Recover source code from .pyc cache and write a .py file."""
import marshal
import struct
import dis

path = r"D:\W_LBM - Copy\wanphys\examples\__pycache__\fluid_grid_lbm_oneway_moving_rigid_visual_trt.cpython-311.pyc"

with open(path, "rb") as f:
    magic = f.read(4)
    flags = struct.unpack("<I", f.read(4))[0]
    # Skip hash/optional fields
    if flags & 0x1:
        f.read(8)  # hash
    f.read(4)  # mtime
    code = marshal.load(f)

# Try to de-compile
try:
    import uncompyle6
    from uncompyle6 import deparse_code
    with open(r"D:\W_LBM - Copy\wanphys\examples\fluid_grid_lbm_oneway_moving_rigid_visual_trt_recovered.py", "w") as out:
        deparse_code(2, code, out)
    print("Recovered with uncompyle6")
except ImportError:
    try:
        import decompyle3
    except ImportError:
        print("No decompiler available. Printing bytecode info instead.")
        print(dis.code_info(code))
        # Try pycdc approach - just show consts and names
        print("\n--- Names ---")
        for n in code.co_names:
            print(n)
        print("\n--- Consts ---")
        for c in code.co_consts:
            if isinstance(c, str):
                print(repr(c)[:200])
            elif isinstance(c, tuple):
                print(f"tuple({len(c)})")
            elif hasattr(c, 'co_code'):
                print(f"<code object {c.co_name}>")
            else:
                print(repr(c)[:100])