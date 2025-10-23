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

### 2. Modify opt-einsum for Intel GPU Support

The xpu torch backend does not support complex tensor operations. So we need to modify the opt-einsum library to properly handle Intel GPU (XPU) tensors.

Replace the `opt-einsum/backends/torch.py` file with the provided modified version.

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

Note that 