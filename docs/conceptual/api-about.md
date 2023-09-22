# About hipSPARSELt API

hipSPARSELt is a library that contains basic linear algebra subroutines for sparse matrices written in HIP
for GPU devices. It is designed to be used from C and C++ code.

The hipSPARSELt API is organized in the following functions:

* Library management: Provides the library handle
* Matrix descriptor: Used to define sparse and dense matrix
* Matmul descriptor: Used to define how to do the matrix multiply
* Matmul algorithm: Provides algorithms for doing the matrix multiply
* Matmul: Operations that provide multiply of sparse matrices
* Helper: Helper functions that are required for subsequent library calls

The code is open and hosted here: https://github.com/ROCmSoftwarePlatform/hipSPARSELt

hipSPARSELt SPARSE library supports AMD's sparse matrix core technique on AMD RDNA/CDNA GPUs
that speeds up structured sparsity computation and improves AI frameworks and inference engines.

This is also a marshalling library, with multiple supported backends. It sits between the application and
a 'worker' SPARSE library, marshalling inputs into the backend library and marshalling results back to
the application.

hipSPARSELt exports an interface that doesn't require the client to change, regardless of the chosen
backend. hipSPARSELt supports rocSPARSELt (included in hipSPARSELt) and cuSPARSELt as backends.

## Device and stream management

`hipSetDevice` and `hipGetDevice` are HIP device management APIs. They are *not* part of the
hipSPARSELt API.

## Asynchronous execution

All hipSPARSELt library functions, unless otherwise stated, are non-blocking and executed
asynchronously with respect to the host. They may return before the actual computation has finished.
To force synchronization, use `hipDeviceSynchronize` or `hipStreamSynchronize`. This will ensure that all
previously executed hipSPARSELt functions on the device/this particular stream have completed.

## HIP device management

Before a HIP kernel invocation, you need to call `hipSetDevice` to set a device (e.g., device 1). If you
don't explicitly call this device, the system sets it as device 0 by default. Unless you explicitly call
`hipSetDevice` to set to another device, HIP kernels are always launched on device 0.

The above is a HIP (and CUDA) device management approach and has nothing to do with hipSPARSELt.
hipSPARSELt honors the above approach and assumes you have already set the device before a
hipSPARSELt routine call.

Once you set the device, you can create a handle with `hipsparselt_init`.

Subsequent hipSPARSELt routines take this handle as an input parameter. hipSPARSELt only *queries*
(by `hipGetDevice`) the device; it *doesn't set* the device. If hipSPARSELt doesn't see a valid device, it
returns an error message. It is your responsibility to provide a valid device and ensure the device safety.

You *cannot* switch devices between `hipsparselt_init` and `hipsparselt_destroy`. If you want to change
devices, you must destroy the current handle and create another hipSPARSELt handle.

## HIP stream management

HIP kernels are always launched in a queue (also known as stream).

If you don't explicitly specify a stream, the system provides a default stream, maintained by the
system. You cannot create or destroy the default stream. However, you can freely create new streams
(with `hipStreamCreate`) and bind it to the hipSPARSELt operations, such as `hipsparselt_spmma_prune`
and `hipsparselt_matmul`. HIP kernels are invoked in hipSPARSELt routines.

If you create a stream, you are responsible for destroying it.

## Multiple streams and multiple devices

If the system under test has multiple HIP devices, you can run multiple concurrent hipSPARSELt
handles. However, you can't run a single hipSPARSELt handle on different discrete devices. Each handle
is associated with a particular singular device, and a new handle should be created for each additional
device.

## Storage formats

The structured sparsity storage format represents a ```{eval-rst} :math:`m \times n` ``` matrix by

|                |                                                 |
|-----------|----------------------------------|
| m            | Number of rows (integer)       |
| n             | Number of columns (integer)  |
| sparsity  | 50%, ratio of `nnz` elements in every 2:1 (int) or 4:2 (others) element along the row (4:2 means every 4 continuous elements will only have 2 `nnz` elements |
| compressed_matrix | Matrix of ``nnz`` elements containing the data) |
| metadata | Matrix of `nnz` elements containing the element indices in every 4:2 or 2:1 array along the row. The contents or structure of metadata is dependent on the chosen solution by backend implementation. |

Consider the following ```{eval-rst} :math:`4 \times 4` ``` matrix and the structured sparsity structures, with ```{eval-rst} :math:`m = 4, n = 4` ```:

```{eval-rst}
.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 0.0 \\
        0.0 & 0.0 & 3.0 & 4.0 \\
        0.0 & 6.0 & 7.0 & 0.0 \\
        0.0 & 6.0 & 0.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  Compressed A = \begin{pmatrix}
                  1.0 & 2.0 \\
                  3.0 & 4.0 \\
                  6.0 & 7.0 \\
                  6.0 & 8.0 \\
                \end{pmatrix}
  metadata =    \begin{pmatrix}
                  0 & 1 \\
                  2 & 3 \\
                  1 & 2 \\
                  1 & 3 \\
                \end{pmatrix}
```

## Data type support

| Input                           | Output                        | Compute Type                                 | Backend |
|-------------------------|-------------------------|----------------------------------------|-----------|
| HIPSPARSELT_R_16F   | HIPSPARSELT_R_16F    | HIPSPARSELT_COMPUTE_32F           | HIP  |
| HIPSPARSELT_R_16BF | HIPSPARSELT_R_16BF  | HIPSPARSELT_COMPUTE_32F           | HIP |
| HIPSPARSELT_R_8I      | HIPSPARSELT_R_8I       | HIPSPARSELT_COMPUTE_32I           | HIP / CUDA |
| HIPSPARSELT_R_8I      | HIPSPARSELT_R_16F    | HIPSPARSELT_COMPUTE_32I           | HIP / CUDA |
| HIPSPARSELT_R_16F   | HIPSPARSELT_R_16F    | HIPSPARSELT_COMPUTE_16F           | CUDA |
| HIPSPARSELT_R_16BF | HIPSPARSELT_R_16BF  | HIPSPARSELT_COMPUTE_16F           | CUDA |
| HIPSPARSELT_R_32F   | HIPSPARSELT_R_32F    | HIPSPARSELT_COMPUTE_TF32          | CUDA |
| HIPSPARSELT_R_32F   | HIPSPARSELT_R_32F    | HIPSPARSELT_COMPUTE_TF32_FAST | CUDA |
