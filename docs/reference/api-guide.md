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

Data types
Functions

* :ref:`hipsparselt_library_managment_functions_` describe functions that provide the library handle.
* :ref:`hipsparselt_matrix_descriptor_functions_` describe functions that used to define sparse and dense matrix.
* :ref:`hipsparselt_matmul_descriptor_functions_` describe functions that used to define how to do the matrix multiply.
* :ref:`hipsparselt_matmul_algorithm_functions_` describe functions that provide algorithms for doing the matrix multiply.
* :ref:`hipsparselt_matmul_functions_` describe operations that provide multiply of sparse matrices.
* :ref:`hipsparselt_helper_functions_` describe available helper functions that are required for subsequent library calls.

```{note}
All hipSPARSELt library functions, unless otherwise stated, are non-blocking and are run asynchronously with respect to the host. They may return before the actual computation has finished. To force synchronization, use `hipDeviceSynchronize` or `hipStreamSynchronize`.
```

## Device and stream management

`hipSetDevice` and `hipGetDevice` are HIP device management APIs. They are *not* part of the
hipSPARSELt API.



HIP Device Management
^^^^^^^^^^^^^^^^^^^^^
Before a HIP kernel invocation, users need to call :cpp:func:`hipSetDevice` to set a device, e.g. device 1. If users do not explicitly call it, the system by default sets it as device 0. Unless users explicitly call :cpp:func:`hipSetDevice` to set to another device, their HIP kernels are always launched on device 0.

The above is a HIP (and CUDA) device management approach and has nothing to do with hipSPARSELt. hipSPARSELt honors the approach above and assumes users have already set the device before a hipSPARSELt routine call.

Once users set the device, they create a handle with :ref:`hipsparselt_init`.

Subsequent hipSPARSELt routines take this handle as an input parameter. hipSPARSELt ONLY queries (by :cpp:func:`hipGetDevice`) the user's device; hipSPARSELt does NOT set the device for users. If hipSPARSELt does not see a valid device, it returns an error message. It is the users' responsibility to provide a valid device to hipSPARSELt and ensure the device safety.

Users CANNOT switch devices between :ref:`hipsparselt_init` and :ref:`hipsparselt_destroy`. If users want to change device, they must destroy the current handle and create another hipSPARSELt handle.

HIP Stream Management
^^^^^^^^^^^^^^^^^^^^^
HIP kernels are always launched in a queue (also known as stream).

If users do not explicitly specify a stream, the system provides a default stream, maintained by the system. Users cannot create or destroy the default stream. However, users can freely create new streams (with :cpp:func:`hipStreamCreate`) and bind it to the hipSPARSELt operations, such as :ref:`hipsparselt_spmma_prune` and :ref:`hipsparselt_matmul`. HIP kernels are invoked in hipSPARSELt routines. If users create a stream, they are responsible for destroying it.

Multiple Streams and Multiple Devices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If the system under test has multiple HIP devices, users can run multiple hipSPARSELt handles concurrently, but can NOT run a single hipSPARSELt handle on different discrete devices. Each handle is associated with a particular singular device, and a new handle should be created for each additional device.

Storage Formats
^^^^^^^^^^^^^^^

Structured sparsity storage format
''''''''''''''''''''''''''''''''''
The Structured sparsity storage format represents a :math:`m \times n` matrix by

================ =====================================================================================================
m                number of rows (integer).
n                number of columns (integer).
sparsity         50%, ratio of ``nnz`` elemetns in every 2:1 (int) or 4:2 (others) element along the row.
                 4:2 means every 4 continuous elements will only have 2 ``nnz`` elements.
compresed_matrix matrix of ``nnz`` elements containing the data
metadata         matrix of ``nnz`` elements containing the element indices in every 4:2 or 2:1 array along the row.
                 contents or structure of metadata is dependent on the chosen solution by backend implementation.
================ =====================================================================================================

Consider the following :math:`4 \times 4` matrix and the structured sparsity structures, with :math:`m = 4, n = 4`:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 0.0 \\
        0.0 & 0.0 & 3.0 & 4.0 \\
        0.0 & 6.0 & 7.0 & 0.0 \\
        0.0 & 6.0 & 0.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  Compresed A = \begin{pmatrix}
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

=====================
Using hipSPARSELt API
=====================


Types
-----

hipsparseLtHandle_t
^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: hipsparseLtHandle_t

hipsparseLtMatDescriptor_t
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: hipsparseLtMatDescriptor_t

hipsparseLtMatmulDescriptor_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: hipsparseLtMatmulDescriptor_t

hipsparseLtMatmulAlgSelection_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: hipsparseLtMatmulAlgSelection_t

hipsparseLtMatmulPlan_t
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: hipsparseLtMatmulPlan_t

hipsparseLtDatatype_t
^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: hipsparseLtDatatype_t

hipsparseLtSparsity_t
^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: hipsparseLtSparsity_t

hipsparseLtMatDescAttribute_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: hipsparseLtMatDescAttribute_t

hipsparseLtComputetype_t
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenenum:: hipsparseLtComputetype_t

hipsparseLtMatmulDescAttribute_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: hipsparseLtMatmulDescAttribute_t

hipsparseLtMatmulAlg_t
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: hipsparseLtMatmulAlg_t

hipsparseLtPruneAlg_t
^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: hipsparseLtMatmulAlgAttribute_t

hipsparseLtMatmulAlgAttribute_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: hipsparseLtPruneAlg_t

hipsparseLtSplitKMode_t
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: hipsparseLtSplitKMode_t

.. _api:

Exported Sparse Functions
-------------------------

Library Management Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+------------------------------------------+
|Function name                             |
+------------------------------------------+
|:cpp:func:`hipsparseLtInit`               |
+------------------------------------------+
|:cpp:func:`hipsparseLtDestroy`            |
+------------------------------------------+
|:cpp:func:`hipsparseLtGetVersion`         |
+------------------------------------------+
|:cpp:func:`hipsparseLtGetProperty`        |
+------------------------------------------+

Matrix Descriptor Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================================================================================= ====== ====== ==== ======== ====
Function name                                                                           int    single half bfloat16 int8
                                                                                        (CUDA)
======================================================================================= ====== ====== ==== ======== ====
:cpp:func:`hipsparseLtDenseDescriptorInit() <hipsparseLtDenseDescriptorInit>`             x      x      x     x      x
:cpp:func:`hipsparseLtStructuredDescriptorInit() <hipsparseLtStructuredDescriptorInit>`   x      x      x     x      x
======================================================================================= ====== ====== ==== ======== ====

+--------------------------------------------+
|Function name                               |
+--------------------------------------------+
|:cpp:func:`hipsparseLtMatDescriptorDestroy` |
+--------------------------------------------+
|:cpp:func:`hipsparseLtMatDescSetAttribute`  |
+--------------------------------------------+
|:cpp:func:`hipsparseLtMatDescGetAttribute`  |
+--------------------------------------------+

Matmul Descriptor Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

=============================================================================== ====== ========= ====== ====== ===
Function name                                                                   TF32   TF32 Fast single half   int
                                                                                (CUDA) (CUDA)    (HIP)  (CUDA)
=============================================================================== ====== ========= ====== ====== ===
:cpp:func:`hipsparseLtMatmulDescriptorInit() <hipsparseLtMatmulDescriptorInit>`    x       x        x      x    x
=============================================================================== ====== ========= ====== ====== ===

+---------------------------------------------+
|Function name                                |
+---------------------------------------------+
|:cpp:func:`hipsparseLtMatmulDescriptorInit`  |
+---------------------------------------------+
|:cpp:func:`hipsparseLtMatmulDescSetAttribute`|
+---------------------------------------------+
|:cpp:func:`hipsparseLtMatmulDescGetAttribute`|
+---------------------------------------------+

Matmul Algorithm Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------------------------+
|Function name                                |
+---------------------------------------------+
|:cpp:func:`hipsparseLtMatmulAlgSelectionInit`|
+---------------------------------------------+
|:cpp:func:`hipsparseLtMatmulAlgSetAttribute` |
+---------------------------------------------+
|:cpp:func:`hipsparseLtMatmulAlgGetAttribute` |
+---------------------------------------------+

Matmul Functions
^^^^^^^^^^^^^^^^

+-----------------------------------------+
|Function name                            |
+-----------------------------------------+
|:cpp:func:`hipsparseLtMatmulGetWorkspace`|
+-----------------------------------------+
|:cpp:func:`hipsparseLtMatmulPlanInit`    |
+-----------------------------------------+
|:cpp:func:`hipsparseLtMatmulPlanDestroy` |
+-----------------------------------------+
|:cpp:func:`hipsparseLtMatmul`            |
+-----------------------------------------+
|:cpp:func:`hipsparseLtMatmulSearch`      |
+-----------------------------------------+

Helper Functions
^^^^^^^^^^^^^^^^

+-------------------------------------------+
|Function name                              |
+-------------------------------------------+
|:cpp:func:`hipsparseLtSpMMAPrune`          |
+-------------------------------------------+
|:cpp:func:`hipsparseLtSpMMAPruneCheck`     |
+-------------------------------------------+
|:cpp:func:`hipsparseLtSpMMAPrune2`         |
+-------------------------------------------+
|:cpp:func:`hipsparseLtSpMMAPruneCheck2`    |
+-------------------------------------------+
|:cpp:func:`hipsparseLtSpMMACompressedSize` |
+-------------------------------------------+
|:cpp:func:`hipsparseLtSpMMACompress`       |
+-------------------------------------------+
|:cpp:func:`hipsparseLtSpMMACompressedSize2`|
+-------------------------------------------+
|:cpp:func:`hipsparseLtSpMMACompress2`      |
+-------------------------------------------+


Asynchronous API
----------------
Except a functions having memory allocation inside preventing asynchronicity, all hipSPARSELt functions are configured to operate in non-blocking fashion with respect to CPU, meaning these library functions return immediately.

.. _hipsparselt_library_managment_functions_:

Library Management Functions
----------------------------

This library management describe functions that provide the library handle.

.. _hipsparselt_init:

hipsparseLtInit()
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtInit

.. _hipsparselt_destroy:

hipsparseLtDestroy()
^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtDestroy

.. _hipsparselt_get_version:

hipsparseLtGetVersion()
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtGetVersion

.. _hipsparselt_get_property:

hipsparseLtGetProperty()
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtGetProperty

.. _hipsparselt_matrix_descriptor_functions_:

Matrix Descriptor Functions
---------------------------

The matrix descriptor describe fuctions that used to define sparse and dense matrix

hipsparseLtDenseDescriptorInit()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtDenseDescriptorInit

hipsparseLtStructuredDescriptorInit()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtStructuredDescriptorInit

hipsparseLtMatDescriptorDestroy()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatDescriptorDestroy

hipsparseLtMatDescSetAttribute()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatDescSetAttribute

hipsparseLtMatDescGetAttribute()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatDescGetAttribute

.. _hipsparselt_matmul_descriptor_functions_:

Matmul Descriptor Functions
---------------------------

This matmul descriptor describe fuctions that used to define how to do the matrix multiply.

hipsparseLtMatmulDescriptorInit()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmulDescriptorInit

hipsparseLtMatmulDescSetAttribute()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmulDescSetAttribute

hipsparseLtMatmulDescGetAttribute()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmulDescGetAttribute

.. _hipsparselt_matmul_algorithm_functions_:

Matmul Algorithm Functions
--------------------------

This matmul algorithm describe functions that provide algorithms for doing the matrix multiply.

hipsparseLtMatmulAlgSelectionInit()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmulAlgSelectionInit

hipsparseLtMatmulAlgSetAttribute()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmulAlgSetAttribute

hipsparseLtMatmulAlgGetAttribute()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmulAlgGetAttribute

.. _hipsparselt_matmul_functions_:

Matmul Functions
----------------

This matmul describe operations that provide multiply of sparse matrices.

hipsparseLtMatmulGetWorkspace()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmulGetWorkspace

hipsparseLtMatmulPlanInit()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmulPlanInit

hipsparseLtMatmulPlanDestroy()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmulPlanDestroy

.. _hipsparselt_matmul:

hipsparseLtMatmul()
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmul

**Data Type Support**

=================== ==================== =============================== ============
       Input               Output                 Compute Type              Backend
=================== ==================== =============================== ============
HIPSPARSELT_R_16F    HIPSPARSELT_R_16F    HIPSPARSELT_COMPUTE_32F             HIP
HIPSPARSELT_R_16BF   HIPSPARSELT_R_16BF   HIPSPARSELT_COMPUTE_32F             HIP
HIPSPARSELT_R_8I     HIPSPARSELT_R_8I     HIPSPARSELT_COMPUTE_32I         HIP / CUDA
HIPSPARSELT_R_8I     HIPSPARSELT_R_16F    HIPSPARSELT_COMPUTE_32I         HIP / CUDA
HIPSPARSELT_R_16F    HIPSPARSELT_R_16F    HIPSPARSELT_COMPUTE_16F            CUDA
HIPSPARSELT_R_16BF   HIPSPARSELT_R_16BF   HIPSPARSELT_COMPUTE_16F            CUDA
HIPSPARSELT_R_32F    HIPSPARSELT_R_32F    HIPSPARSELT_COMPUTE_TF32           CUDA
HIPSPARSELT_R_32F    HIPSPARSELT_R_32F    HIPSPARSELT_COMPUTE_TF32_FAST      CUDA
=================== ==================== =============================== ============

hipsparseLtMatmulSearch()
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmulSearch

.. _hipsparselt_helper_functions_:

Helper Functions
----------------

This module holds available helper functions that are required for subsequent library calls

.. _hipsparselt_spmma_prune:

hipsparseLtSpMMAPrune()
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtSpMMAPrune

hipsparseLtSpMMAPruneCheck()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtSpMMAPruneCheck

hipsparseLtSpMMAPrune2()
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtSpMMAPrune2

hipsparseLtSpMMAPruneCheck2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtSpMMAPruneCheck2

hipsparseLtSpMMACompressedSize()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtSpMMACompressedSize

hipsparseLtSpMMACompress()
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtSpMMACompress

hipsparseLtSpMMACompressedSize2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtSpMMACompressedSize2

hipsparseLtSpMMACompress2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtSpMMACompress2