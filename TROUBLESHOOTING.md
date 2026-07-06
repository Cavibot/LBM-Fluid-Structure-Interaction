# Troubleshooting Guide

This document contains solutions to common issues you might encounter when working with WanPhys.

## Warp CUDA Kernel Issues

### Error: "Failed to lookup kernel function _xxx_cuda_kernel_forward in module"

**Symptoms:**
```
Failed to lookup kernel function _apply_pressure_gradient_7b7d533b_cuda_kernel_forward in module
```

**Cause:**
This error occurs when Warp's kernel cache becomes corrupted or outdated. This can happen due to:
- Code changes to kernel functions
- Warp version updates
- System updates or driver changes
- Interrupted compilation processes

**Solutions:**

#### Solution 1: Use the provided utility script (Recommended)
```bash
python scripts/clear_warp_cache.py
```

#### Solution 2: Clear cache from Python
```python
import warp as wp
wp.clear_kernel_cache()
```

#### Solution 3: Manual cache clearing
Delete the Warp cache directory:
- **Windows**: `%LOCALAPPDATA%\NVIDIA\warp\Cache\<version>`
- **Linux**: `~/.cache/nvidia/warp/<version>`
- **macOS**: `~/Library/Caches/nvidia/warp/<version>`

#### Solution 4: Force recompilation in code
Add this at the start of your script:
```python
import warp as wp
wp.config.mode = "debug"  # Disable caching during development
```

**After clearing the cache:**
The kernels will be automatically recompiled on the next run. The first execution may take longer than usual.

---

## Python Environment Issues

### Error: "ModuleNotFoundError: No module named 'warp'"

**Solution:**
Ensure you're using the correct Python environment:
```bash
# Activate the virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Or use the full path
E:/github/WanPhys/.venv/Scripts/python.exe your_script.py
```

### Error: "CUDA not available"

**Solution:**
1. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Check Warp CUDA support:
   ```python
   import warp as wp
   wp.init()
   print(wp.get_devices())
   ```

3. If CUDA is not detected, reinstall Warp with CUDA support:
   ```bash
   pip install --upgrade warp-lang
   ```

---

## Fluid Simulation Issues

### Performance Issues

**Symptoms:**
- Simulation runs very slowly
- GPU utilization is low

**Solutions:**

1. **Check device placement:**
   ```python
   model = FluidModel(nx=128, ny=128, nz=128, device="cuda:0")
   ```

2. **Reduce grid resolution** for testing:
   ```python
   # Start with smaller grids
   model = FluidModel(nx=32, ny=32, nz=32)
   ```

3. **Profile your code:**
   ```python
   import warp as wp
   with wp.ScopedTimer("simulation"):
       solver.step(state_in, state_out, dt)
   ```

### Numerical Instabilities

**Symptoms:**
- NaN or Inf values in velocity/pressure
- Simulation explodes
- Unphysical behavior

**Solutions:**

1. **Reduce timestep:**
   ```python
   dt = 0.001  # Start with smaller timestep
   ```

2. **Check CFL condition:**
   ```python
   max_velocity = ...  # Calculate from state
   cfl = max_velocity * dt / dx
   print(f"CFL number: {cfl}")  # Should be < 1.0
   ```

3. **Increase pressure iterations:**
   ```python
   model = FluidModel(..., pressure_iterations=100)
   ```

---

## Build and Installation Issues

### Error during pip install

**Solution:**
1. Update pip and setuptools:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. Install in editable mode:
   ```bash
   pip install -e .
   ```

---

## Getting Help

If you encounter an issue not covered here:

1. Check the [GitHub Issues](https://github.com/your-repo/issues)
2. Review the [documentation](docs/index.rst)
3. Run with verbose logging:
   ```python
   import warp as wp
   wp.config.verbose = True
   ```

4. Create a minimal reproducible example and file an issue
