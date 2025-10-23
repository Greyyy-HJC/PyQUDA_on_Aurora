# PyQUDA on Aurora (Intel GPU)

This guide provides step-by-step instructions for setting up and running PyQUDA on Aurora supercomputer with Intel GPU support.

## Overview

PyQUDA is a Python interface for QUDA (QCD on CUDA), a library for lattice QCD computations. This setup enables PyQUDA to run on Intel GPUs using SYCL backend on Aurora.

## Prerequisites

- Access to Aurora supercomputer
- Intel oneAPI toolkit with SYCL support
- Python environment with PyTorch and opt-einsum

## Installation Steps

### 1. Install QUDA with SYCL Support

Clone the QUDA repository with SYCL support:

```bash
git clone -b feature/sycl https://github.com/lattice/quda.git
cd quda
```

Configure and build QUDA for Intel GPU:

```bash
./configure-quda
ninja
```

The `configure-quda` script sets up the following key configurations:
- **Target**: SYCL backend for Intel GPU
- **SYCL Targets**: `intel_gpu_pvc` (Intel PVC GPU)
- **Compilers**: Intel SYCL compilers (`icpx`, `mpicxx`)
- **Features**: Multi-grid, distance preconditioning, QDP-JIT interface

### 2. Configure opt-einsum for Intel GPU Support

The Intel GPU (XPU) torch backend does not support complex tensor operations. To handle this limitation, you have two options:

#### Option A: Use sitecustomize.py (Recommended)

Place the provided `sitecustomize.py` file in any directory that is in your `PYTHONPATH`. For example:

```bash
# Option 1: Add to your current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Option 2: Copy to a system-wide location
cp sitecustomize.py /path/to/your/python/site-packages/

# Option 3: Add to your project directory and include in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project"
```

**What the sitecustomize.py does:**
- Automatically patches opt-einsum's contract function
- Detects XPU complex tensors and falls back to CPU computation
- Transfers results back to XPU device when appropriate
- No source code modification required

#### Option B: Replace torch.py (Alternative)

If you prefer to modify the opt-einsum source code directly, replace the `opt-einsum/backends/torch.py` file with the provided modified version.

**Key modifications include:**
- XPU device detection and handling
- Fallback to CPU for complex tensor operations (due to oneDNN limitations on XPU)
- Proper device management for mixed CPU/GPU operations

### 3. Configure PyQUDA Backend

Initialize PyQUDA with the correct backend settings:

```python
import pyquda
pyquda.init(backend="torch", torch_backend="xpu")
```

## Usage Example

```python
import pyquda
import torch

# Initialize PyQUDA with Intel GPU backend
pyquda.init(backend="torch", torch_backend="xpu")

# Your PyQUDA computations will now run on Intel GPU
# PyQUDA will automatically handle tensor operations on XPU
```

## Important Notes

- **Complex Operations**: Some complex tensor operations may fallback to CPU due to oneDNN limitations on Intel GPU
- **Memory Management**: Ensure sufficient GPU memory for your computations
- **Performance**: The SYCL backend is optimized for Intel PVC GPUs on Aurora
- **Automatic Patching**: The sitecustomize.py file automatically handles XPU complex tensor fallbacks

## Troubleshooting

1. **Device Detection Issues**: Ensure Intel GPU drivers and oneAPI toolkit are properly installed
2. **Memory Errors**: Reduce problem size or use CPU fallback for large computations
3. **Compilation Issues**: Check that all SYCL environment variables are set correctly
4. **PYTHONPATH Issues**: Ensure sitecustomize.py is in a directory that's in your PYTHONPATH

## References

- [QUDA Documentation](https://github.com/lattice/quda)
- [Intel oneAPI SYCL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/sycl.html)
- [PyQUDA Documentation](https://github.com/claudiopica/PyQUDA) 