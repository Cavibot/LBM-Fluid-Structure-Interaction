# Bubble golden data (Phase 3)

## Generate (reference CUDA)

Build and run the exporter in `docs/Home-FSLBM`:

```text
target: export_bubble_golden
source: docs/Home-FSLBM/export_bubble_golden.cpp
```

See `docs/wanphys/home_fslbm/phase3_bubble_golden_zh.md` for build/run/copy steps.

## Install here

Copy exported `bubble_*` directories into this folder:

```text
golden_data/
  bubble_ccl_3spheres_r3/
  bubble_init_two_bubbles/
  ...
```

Until they exist, `test_regression_bubble.py` skips the corresponding cases.
