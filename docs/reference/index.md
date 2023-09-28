<meta name="description" content="hipSPARSELt API reference guide">
<meta name="keywords" content="hipSPARSELt, ROCm, API library, API reference">

# hipSPARSELt API reference guide

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

The API reference guide is organized as follows:

* [Supported functions](./supported-functions.md)
* [Data type support](./data-type-support.md)
* [Device & stream management](./device-stream-manage.md)
* [Storage formats](./storage-format.md)
* [API library](../doxygen/docBin/html/index)
