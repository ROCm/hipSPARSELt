# Functions

## Library management

* `hipsparseLtInit`
* `hipsparseLtDestroy`
* `hipsparseLtGetVersion`
* `hipsparseLtGetProperty`

## Matrix descriptor

| Function name | int (CUDA) | single | half | bfloat16 | int8 |
|------------------|-------------|-------|------|-----------|-----|
| `hipsparseLtDenseDescriptorInit() <hipsparseLtDenseDescriptorInit>` | x | x | x | x | x |
| `hipsparseLtStructuredDescriptorInit() <hipsparseLtStructuredDescriptorInit>` | x | x | x | x | x |

* `hipsparseLtMatDescriptorDestroy`
  .. doxygenfunction:: hipsparseLtMatDescriptorDestroy

* `hipsparseLtMatDescSetAttribute`
* `hipsparseLtMatDescGetAttribute`

## Matrix multiply (matmul)

* `hipsparseLtMatmulGetWorkspace`
  .. doxygenfunction:: hipsparseLtMatmulGetWorkspace

* `hipsparseLtMatmulPlanInit`
* `hipsparseLtMatmulPlanDestroy`
  .. doxygenfunction:: hipsparseLtMatmulPlanDestroy
* `hipsparseLtMatmul`
* `hipsparseLtMatmulSearch`
  .. doxygenfunction:: hipsparseLtMatmulSearch

## Matmul descriptor

| Function name | TF32 (CUDA) | TF32 Fast (CUDA) | single (HIP) | half (CUDA) | int |
|------------------|---------------|--------------------|--------------|---------------|----|
| `hipsparseLtMatmulDescriptorInit() <hipsparseLtMatmulDescriptorInit>`  | x | x | x | x | x |

* `hipsparseLtMatmulDescriptorInit`
* `hipsparseLtMatmulDescSetAttribute`
* `hipsparseLtMatmulDescGetAttribute`

## Matmul algorithm

* `hipsparseLtMatmulAlgSelectionInit`
* `hipsparseLtMatmulAlgSetAttribute`
* `hipsparseLtMatmulAlgGetAttribute`

## Helper

* `hipsparseLtSpMMAPrune`
* `hipsparseLtSpMMAPruneCheck`
* `hipsparseLtSpMMAPrune2`
* `hipsparseLtSpMMAPruneCheck2`
* `hipsparseLtSpMMACompressedSize`
* `hipsparseLtSpMMACompress`
* `hipsparseLtSpMMACompressedSize2`
* `hipsparseLtSpMMACompress2`




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


hipsparseLtMatmulPlanInit()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmulPlanInit



.. _hipsparselt_matmul:

hipsparseLtMatmul()
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: hipsparseLtMatmul




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