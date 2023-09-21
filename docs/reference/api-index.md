# HipSPARSELt API reference guide

hipSPARSELt is a library that contains basic linear algebra subroutines for sparse matrices written in HIP for GPU devices. It is designed to be used from C and C++ code.

The functionality of hipSPARSELt is organized in the following categories:

* :ref:`hipsparselt_library_managment_functions_` describe functions that provide the library handle.
* :ref:`hipsparselt_matrix_descriptor_functions_` describe functions that used to define sparse and dense matrix.
* :ref:`hipsparselt_matmul_descriptor_functions_` describe functions that used to define how to do the matrix multiply.
* :ref:`hipsparselt_matmul_algorithm_functions_` describe functions that provide algorithms for doing the matrix multiply.
* :ref:`hipsparselt_matmul_functions_` describe operations that provide multiply of sparse matrices.
* :ref:`hipsparselt_helper_functions_` describe available helper functions that are required for subsequent library calls.

The code is open and hosted on GitHub:
[https://github.com/ROCmSoftwarePlatform/hipSPARSELt](https://github.com/ROCmSoftwarePlatform/hipSPARSELt)

hipSPARSELt SPARSE library supports AMD's sparse matrix core technique on AMD RDNA/CDNA GPUs that speeds up structured sparsity computation and improves AI framework and Inference engine specifically.
This is also a marshalling library, with multiple supported backends.

It sits between the application and a 'worker' SPARSE library, marshalling inputs into the backend library and marshalling results back to the application.

hipSPARSELt exports an interface that does not require the client to change, regardless of the chosen backend.
Currently, hipSPARSELt support rocSPARSELt (included in hipSPARSELt) and cuSPARSELt as backends.
