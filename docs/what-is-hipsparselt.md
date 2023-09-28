  <meta name="description" content="hipSPARSELt is a SPARSE marshalling library with multiple
  supported backends">
  <meta name="keywords" content="hipSPARSELt, ROCm, SPARSE library, API">

# What is hipSPARSELt?

hipSPARSELt is a SPARSE marshalling library with multiple supported backends. It exposes a common
interface that provides Basic Linear Algebra Subroutines (BLAS) for sparse computation implemented
on top of AMD's ROCm runtime and toolchains.

The hipSPARSELt library is created using the [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/)
programming language and is optimized for AMD's latest discrete GPUs.

hipSPARSELt sits between the application and a 'worker' SPARSE library, marshalling inputs into the
backend library and marshalling results back to the application. It exports an interface that doesn't
require the client to change, regardless of the chosen backend. Current supported backends are:
[rocSPARSELt](https://github.com/ROCmSoftwarePlatform/hipSPARSELt/blob/develop/library/src/hcc_detail/rocsparselt) and [cuSPARSELt v0.3](https://docs.nvidia.com/cuda/cusparselt).
