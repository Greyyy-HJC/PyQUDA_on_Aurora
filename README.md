# PyQUDA on Aurora (Intel GPU)

This guide provides step-by-step instructions for setting up and running PyQUDA on Aurora supercomputer with Intel GPU support.

## Overview

PyQUDA is a Python interface for QUDA (QCD on CUDA), a library for lattice QCD computations. This setup enables PyQUDA to run on Intel GPUs using SYCL backend on Aurora.

## Prerequisites

- Access to Aurora supercomputer
- Intel oneAPI toolkit with SYCL support
- Python environment: PyTorch (for Torch backend) or dpnp, and opt-einsum

## Python Environment Setup

It's recommended to use a virtual environment for this setup:

```bash
# Create a virtual environment
python3 -m venv pyquda_env
source pyquda_env/bin/activate

# Install XPU PyTorch packages
python3 -m pip install --no-deps torch==2.9.0+xpu torchvision==0.24.0+xpu torchaudio==2.9.0+xpu --index-url https://download.pytorch.org/whl/xpu
```

## Installation Steps

### 1. Install QUDA with SYCL Support

Clone the QUDA repository with SYCL support:

```bash
git clone -b feature/sycl https://github.com/lattice/quda.git
cd quda
```

Configure and build QUDA for Intel GPU:

**Important**: Before running `./configure-quda`, you need to modify the paths in the script:

1. **Update the prefix path** (around line 74, 78, 88, 94):
   ```bash
   # Change these lines to your desired installation directory
   prefix="/your/desired/installation/path"
   ```

2. **Update the QUDA source path** (around line 170-171):
   ```bash
   # Change this line to point to your QUDA source directory
   echo $CMAKE --fresh $o /path/to/your/quda/source
   $CMAKE --fresh $o /path/to/your/quda/source
   ```

Then run the configuration and build:

```bash
./configure-quda
ninja
```

The `configure-quda` script sets up the following key configurations:
- **Target**: SYCL backend for Intel GPU
- **SYCL Targets**: `intel_gpu_pvc` (Intel PVC GPU)
- **Compilers**: Intel SYCL compilers (`icpx`, `mpicxx`)
- **Features**: Multi-grid, distance preconditioning, QDP-JIT interface

### 2. Configure opt-einsum for Intel GPU Support (Torch backend)

If you use the Torch XPU backend, note that complex tensor operations are not supported natively. To handle this limitation, you have two options:

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

### 4. Alternative: Using dpnp for Complex Matrix Operations

For better support of complex matrix einsum operations on Intel GPU, you can use [dpnp](https://github.com/IntelPython/dpnp) as an alternative backend. dpnp is Intel's data parallel extension for NumPy that provides optimized operations on Intel GPUs.

#### Install dpnp

```bash
# Install dpnp via pip
python -m pip install --index-url https://software.repos.intel.com/python/pypi dpnp
```

#### Configure PyQUDA with dpnp

```python
import pyquda
import dpnp as np

# Initialize PyQUDA with dpnp backend
pyquda.init(backend="dpnp")

# dpnp provides native support for complex matrix operations on Intel GPU
# This eliminates the need for CPU fallbacks in complex einsum operations
```

#### Benefits of dpnp

- **Native Complex Support**: Full support for complex tensor operations on Intel GPU
- **Optimized Performance**: Intel-optimized implementations for Intel GPU hardware
- **NumPy Compatibility**: Drop-in replacement for NumPy with GPU acceleration
- **No Fallback Required**: Eliminates the need for CPU fallbacks in complex operations
- **No PyTorch Dependency**: dpnp works independently without requiring PyTorch installation

## Usage Examples

### Torch XPU backend
```python
import pyquda
import torch  # ensure XPU build installed with --no-deps

pyquda.init(backend="torch", torch_backend="xpu")
```

### dpnp backend
```python
import pyquda
import dpnp as np

pyquda.init(backend="dpnp")
```

## Log (Hacking PyQUDA on Aurora)

1. Initial validation (NumPy backend works):
   - QUDA and PyQUDA were built and installed successfully.
   - Because the `cupy` backend is not usable on Intel GPUs, we first validated functionality with the `numpy` backend.
   - Both pion 2pt and proton 2pt tests passed, but performance was slow.

2. Trying the Torch XPU backend (install with --no-deps):
   - To improve performance, we switched the backend to Torch (XPU build).
   - When installing the XPU build via pip, add `--no-deps` to avoid polluting the MPI environment: the XPU build of Torch brings Intel oneAPI/IMPI runtime shared libraries that can overshadow the system MPI and libfabric paths, causing `mpi4py` to link against the wrong objects during initialization.

3. Complex matmul limitation in XPU Torch and the temporary workaround:
   - We found XPU Torch lacks support for complex matrix multiplication. Both `opt_einsum` and `torch.einsum` fail on complex contractions with errors like: `RuntimeError: Complex data type matmul is not supported in oneDNN`.
   - Temporary workaround: use `sitecustomize.py` to force `opt_einsum` contractions to run on the CPU and then move the result back to XPU. This works but adds CPU round-trips and overhead.

4. Adopting the dpnp backend (native complex einsum):
   - We evaluated Intel-maintained `dpnp` (Data Parallel Extension for NumPy). Tests show `dpnp.einsum` natively supports complex matrix multiplication on Intel GPUs.
   - We added `dpnp` as a PyQUDA backend option and prefer `dpnp.einsum` on contraction paths to keep Aurora behavior as close as possible to the `cupy` backend.
   - Note: when choosing the `dpnp` path, do not install Torch (XPU) to avoid introducing oneAPI/IMPI runtimes that can pollute MPI. See the `dpnp` repository for details: https://github.com/IntelPython/dpnp

 