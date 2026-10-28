# scs-qmbboot

`scs-qmbboot` is a fork of [SCS](https://github.com/cvxgrp/scs)
specialized for the semidefinite programs in the quantum many-body bootstrap.
These problems typically involve a large number of positive semidefinite (PSD) cones
and a sparse constraint matrix `A`.
Exploiting this structure, we make the following optimizations.
The changes to the solver are each **mathematically equivalent** to
the corresponding step of upstream SCS and differ only in floating-point rounding,
so with the same settings the solver performs the same iteration
and **preserves the convergence behavior** of upstream SCS.

- **Parallel PSD projection.**
With many PSD blocks, the PSD projection dominates the iteration time
since upstream SCS projects the PSD blocks serially.
In `scs-qmbboot`, the PSD blocks are now projected in parallel with OpenMP.

- **Divide-and-conquer eigensolvers.**
The PSD projection needs the full eigendecomposition of every block.
It now uses `syevd`/`heevd` instead of `syevr`/`heevr`;
these drivers rely mostly on level-3 BLAS and are considerably faster,
at the cost of a larger workspace.

- **MKL Pardiso backend on the Gram reduction.**
Factorizing the full KKT matrix of bootstrap SDPs is expensive and hard to parallelize.
The dense backend of upstream SCS in [`linsys/cpu/dense`](linsys/cpu/dense)
avoids the full KKT matrix by eliminating `y` and solving the reduced system
with the symmetric positive definite Gram matrix `R_x + P + A' R_y^{-1} A`,
which is in general dense and is factorized by a dense Cholesky decomposition.
For bootstrap SDPs, however, every row of `A` has only a few nonzeros,
so the Gram matrix is sparse. We therefore introduce the new backend `mkl_gram`
in [`linsys/mkl/gram`](linsys/mkl/gram), which applies the same reduction
but keeps the Gram matrix sparse and factorizes it in parallel with MKL Pardiso.
Conversely, it is not suited to problems in which `A` has dense rows.

- **Parallel element-wise loops.**
Once the projection and the linear solve are parallel,
the remaining vector operations of each iteration become a noticeable serial cost;
they are now parallelized with OpenMP as well.

- **Anderson acceleration off by default.**
On bootstrap SDPs we observed no speedup from Anderson acceleration,
so it is disabled by default (`acceleration_lookback = 0`).
It can still be enabled through the settings.

- **Python bindings.**
The lightweight package `scs_qmbboot` in [`python/`](python/) wraps the C library
through ctypes and follows the interface of [scs-python](https://github.com/bodono/scs-python).
It supports the features needed for bootstrap SDPs
and uses `mkl_gram` as its default linear solver on Linux.

## Installation

Python package (on Linux, make sure MKL is installed):

```bash
export MKLROOT="..."
pip install git+https://github.com/JefferyWangSH/scs-qmbboot.git
```

```python
import scs_qmbboot as scs

data = {"A": ..., "b": ..., "c": ...}
cone = {"z": ..., "l": ..., "s": [...], "cs": [...]}
solver = scs.SCS(data, cone, linear_solver="mkl_gram", eps_abs=1e-6, eps_rel=1e-6)
sol = solver.solve()  # sol["x"], sol["y"], sol["s"], sol["info"]
```

C libraries with the `mkl_gram` backend and OpenMP:

```bash
export MKLROOT="..."

# with make
make mkl USE_OPENMP=1

# or with cmake: this will build all available backends
cmake -B build -DUSE_OPENMP=ON && cmake --build build
```

The upstream SCS README follows.

---

<h1 align="center" margin=0px>
<img src="https://github.com/cvxgrp/scs/blob/master/docs/src/_static/scs_logo.png" alt="Intersection of a cone and a polyhedron" width="450">
</h1>

[![Build Status](https://github.com/cvxgrp/scs/actions/workflows/build.yml/badge.svg)](https://github.com/cvxgrp/scs/actions/workflows/build.yml)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat)](https://www.cvxgrp.org/scs/)
[![PyPI Downloads](https://api.pepy.tech/personalized-badge/scs?period=month&units=international_system&left_color=grey&right_color=blue&left_text=PyPI%20downloads%2Fmonth)](https://pepy.tech/projects/scs)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/scs.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/scs)

SCS (`splitting conic solver`) is a numerical optimization package for solving
large-scale convex cone problems. The current version is `3.3.1`.

The full documentation is available [here](https://www.cvxgrp.org/scs/).

If you wish to cite SCS please cite the papers listed [here](https://www.cvxgrp.org/scs/citing).

## Overview

SCS solves convex cone programs of the form:

```text
minimize    (1/2) x'Px + c'x
subject to  Ax + s = b, s in K
```

where `K` is a product of convex cones. It uses an operator-splitting method
(ADMM) with Anderson acceleration to achieve fast convergence.

**Supported cones:** zero/linear (LP), box, second-order (SOCP), semidefinite
(SDP), complex semidefinite, exponential, power, and spectral cones
(log-determinant, nuclear norm, ell1-norm, sum-of-largest eigenvalues).

## Features

- Multiple linear solver backends: sparse direct (QDLDL), dense direct (LAPACK),
  iterative (CG), Intel MKL Pardiso, Apple Accelerate (macOS), NVIDIA cuDSS, GPU
- Anderson acceleration for faster convergence, including solve diagnostics
- Problem data normalization/equilibration for numerical stability
- Adaptive dual scaling, with per-row diagonal rescaling driven by residual
  profiles (`adaptive_diag_scale`, on by default)
- Warm-starting and incremental `b`/`c` updates via `scs_update`
- Ctrl-C signal handling for graceful interruption
- Configurable precision (`float`/`double`) and index type (`int`/`long long`)
- MIT licensed

## Building from Source

### Make (default)

```bash
make                    # Build direct + indirect solvers with BLAS/LAPACK
make test               # Build test binaries
./out/run_tests_direct  # Run direct solver tests
./out/run_tests_indirect
```

Note that the spectral-cone tests are skipped unless SCS is built with
`USE_SPECTRAL_CONES=1` (see the flag table below); a default `make test` run
reports them as `skipped`.

### CMake

```bash
mkdir build && cd build
cmake -DBUILD_TESTING=ON -DUSE_LAPACK=ON ..
cmake --build .
ctest
```

### Common Build Flags

| Flag | Default | Description |
|------|---------|-------------|
| `USE_LAPACK` | `1` | Enable BLAS/LAPACK (required for SDP support) |
| `DLONG` | `0` | Use 64-bit integer indexing |
| `SFLOAT` | `0` | Use single-precision floats |
| `USE_SPECTRAL_CONES` | `0` | Enable spectral cone support (requires LAPACK) |
| `USE_OPENMP` | `0` | Enable OpenMP parallelization |

Example: `make DLONG=1 USE_LAPACK=1`

## Language Interfaces

SCS has interfaces for several languages:

- **Python:** [scs-python](https://github.com/bodono/scs-python) (`pip install scs`, [PyPI](https://pypi.org/project/scs/))
- **Julia:** [SCS.jl](https://github.com/jump-dev/SCS.jl)
- **R:** [scs](https://cran.r-project.org/package=scs)
- **MATLAB:** [scs-matlab](https://github.com/bodono/scs-matlab) ([install guide](https://www.cvxgrp.org/scs/install/matlab.html))
- **Ruby:** [scs-ruby](https://github.com/ankane/scs-ruby)

In [CVXPY](https://www.cvxpy.org/), SCS is the default solver for semidefinite,
exponential-cone and power-cone problems (Clarabel handles LPs and SOCPs, OSQP
handles QPs).

## Project Structure

```text
include/        Public API and internal headers
src/            Core solver implementation
linsys/         Linear solver backends (pluggable architecture)
  cpu/direct/     Sparse LDLt (AMD ordering + QDLDL, default)
  cpu/indirect/   Conjugate gradient
  cpu/dense/      Dense Cholesky on the Gram reduction (LAPACK dpotrf)
  mkl/direct/     Intel MKL Pardiso
  accelerate/direct/  Apple Accelerate sparse LDLt (macOS)
  cudss/direct/   NVIDIA cuDSS
  gpu/indirect/   GPU iterative solver
  external/       Vendored dependencies (AMD, QDLDL)
test/           Test suite (minunit framework)
docs/src/       Sphinx documentation source
```

## License

MIT License. See [LICENSE.txt](LICENSE.txt).
