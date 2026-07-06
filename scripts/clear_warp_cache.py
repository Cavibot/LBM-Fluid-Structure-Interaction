#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Utility script to clear Warp's kernel cache.

Run this script if you encounter errors like:
  "Failed to lookup kernel function _xxx_cuda_kernel_forward in module"

This forces Warp to recompile all CUDA kernels from scratch.
"""

import sys


def main():
    """Clear Warp kernel cache and print confirmation."""
    try:
        import warp as wp
    except ImportError:
        print("Error: Warp is not installed. Please install it first:")
        print("  pip install warp-lang")
        sys.exit(1)

    print(f"Warp version: {wp.__version__}")
    print(f"Cache directory: {wp.config.kernel_cache_dir}")
    
    try:
        wp.clear_kernel_cache()
        print("\n✓ Kernel cache cleared successfully!")
        print("\nAll CUDA kernels will be recompiled on next use.")
    except Exception as e:
        print(f"\n✗ Failed to clear cache: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
