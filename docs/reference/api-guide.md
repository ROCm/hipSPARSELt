# hipSPARSELt API library

hipSPARSELt is a library that contains basic linear algebra subroutines (BLAS) for SPARSE matrices that
are written in the [HIP programming language](https://rocm.docs.amd.com/projects/HIP/en/latest/). It
sits between the application and a 'worker' SPARSE library, marshalling inputs into the backend library
and marshalling results back to the application. Supported backends are
[rocSPARSELt](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html) (included in
hipSPARSELt) and cuSPARSELt.

hipSPARSELt supports AMD's SPARSE matrix core technology on AMD RDNA/CDNA GPUs. It is
designed for use with C and C++ code.

```{note}
Code for hipSPARSELt is open source and hosted on
[GitHub](https://github.com/ROCmSoftwarePlatform/hipSPARSELt).
```

The hipSPARSELt library is organized as follows:

* @ref types_module
* @ref library_module
* @ref matrix_desc_module
* @ref matmul_module
* @ref matmul_desc_module
* @ref matmul_algo_module
* @ref helper_module
* @ref aux_module

```{note}
All hipSPARSELt library functions, unless otherwise stated, are non-blocking and are run asynchronously with respect to the host. They may return before the actual computation has finished. To force synchronization, use `hipDeviceSynchronize` or `hipStreamSynchronize`.
```
