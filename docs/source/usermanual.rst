.. _user_manual:

***********
User Manual
***********

.. toctree::
   :maxdepth: 3
   :caption: Contents:

Introduction
============

hipSPARSELt is a library that contains basic linear algebra subroutines for sparse matrices written in HIP for GPU devices.
It is designed to be used from C and C++ code.
The functionality of hipSPARSELt is organized in the following categories:

* :ref:`hipsparselt_library_managment_functions_` describe functions that provide the library handle.
* :ref:`hipsparselt_matrix_descriptor_functions_` describe fuctions that used to define sparse and dense matrix.
* :ref:`hipsparselt_matmul_descriptor_functions_` describe fuctions that used to define how to do the matrix multiply.
* :ref:`hipsparselt_matmul_algorithm_functions_` describe functions that provide algortithms for doing the matrix multiply.
* :ref:`hipsparselt_matmul_functions_` describe operations that provide multiply of sparse matrices.
* :ref:`hipsparselt_helper_functions_` describe available helper functions that are required for subsequent library calls.

The code is open and hosted here: https://github.com/ROCmSoftwarePlatform/hipSPARSELt

hipSPARSELt SPARSE library supports AMD's sparse matrix core technique on AMD RDNA/CDNA GPUs that speeds up structured sparsity computation and improves AI framework and Inference engine specifically.
This is also a marshalling library, with multiple supported backends.
It sits between the application and a 'worker' SPARSE library, marshalling inputs into the backend library and marshalling results back to the application.
hipSPARSELt exports an interface that does not require the client to change, regardless of the chosen backend.
Currently, hipSPARSELt support rocSPARSELt (included in hipSPARSELt) and cuSPARSELt as backends.

.. _hipsparse_building:

Building and Installing
=======================

Prerequisites
-------------
hipSPARSELt requires a ROCm enabled platform, more information `here <https://rocm.github.io/>`_.

Installing pre-built packages
-----------------------------
hipSPARSELt can be installed from `AMD ROCm repository <https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories>`_.
For detailed instructions on how to set up ROCm on different platforms, see the `AMD ROCm Platform Installation Guide for Linux <https://rocm.github.io/ROCmInstall.html>`_.

hipSPARSELt can be installed on e.g. Ubuntu using

::

    $ sudo apt-get update
    $ sudo apt-get install hipsparselt

Once installed, hipSPARSELt can be used just like any other library with a C API.
The header file will need to be included in the user code in order to make calls into hipSPARSELt, and the hipSPARSELt shared library will become link-time and run-time dependent for the user application.

Building hipSPARSELt from source
------------------------------
Building from source is not necessary, as hipSPARSELt can be used after installing the pre-built packages as described above.
If desired, the following instructions can be used to build hipSPARSELt from source.
Furthermore, the following compile-time dependencies must be met

- `hipSPARSE <https://github.com/ROCmSoftwarePlatform/hipSPARSE>`_
- `git <https://git-scm.com/>`_
- `CMake <https://cmake.org/>`_ 3.5 or later
- `AMD ROCm <https://github.com/RadeonOpenCompute/ROCm>`_
- `googletest <https://github.com/google/googletest>`_ (optional, for clients)

Download hipSPARSELt
``````````````````
The hipSPARSELt source code is available at the `hipSPARSELt GitHub page <https://github.com/ROCmSoftwarePlatform/hipSPARSELt>`_.
Download the develop branch using:

::

  $ git clone -b develop https://github.com/ROCmSoftwarePlatform/hipSPARSELt.git
  $ cd hipSPARSELt

Below are steps to build different packages of the library, including dependencies and clients.
It is recommended to install hipSPARSELt using the `install.sh` script.

Using `install.sh` to build hipSPARSELt with dependencies
```````````````````````````````````````````````````````
The following table lists common uses of `install.sh` to build dependencies + library.

.. tabularcolumns::
      |\X{1}{6}|\X{5}{6}|

================= ====
Command           Description
================= ====
`./install.sh -h` Print help information.
`./install.sh -d` Build dependencies and library in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh` it is not necessary to rebuild the dependencies.
`./install.sh`    Build library in your local directory. It is assumed dependencies are available.
`./install.sh -i` Build library, then build and install hipSPARSELt package in `/opt/rocm/hipsparselt`. You will be prompted for sudo access. This will install for all users.
================= ====

Using `install.sh` to build hipSPARSELt with dependencies and clients
```````````````````````````````````````````````````````````````````
The client contains example code and unit tests. Common uses of `install.sh` to build them are listed in the table below.

.. tabularcolumns::
      |\X{1}{6}|\X{5}{6}|

=================== ====
Command             Description
=================== ====
`./install.sh -h`   Print help information.
`./install.sh -dc`  Build dependencies, library and client in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh` it is not necessary to rebuild the dependencies.
`./install.sh -c`   Build library and client in your local directory. It is assumed dependencies are available.
`./install.sh -idc` Build library, dependencies and client, then build and install hipSPARSELt package in `/opt/rocm/hipsparselt`. You will be prompted for sudo access. This will install for all users.
`./install.sh -ic`  Build library and client, then build and install hipSPARSELt package in `opt/rocm/hipsparselt`. You will be prompted for sudo access. This will install for all users.
=================== ====

Using individual commands to build hipSPARSELt
````````````````````````````````````````````
CMake 3.16.8 or later is required in order to build hipSPARSELt.

hipSPARSELt can be built using the following commands:

::

  # Create and change to build directory
  $ mkdir -p build/release ; cd build/release

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  $ cmake ../..

  # Compile hipSPARSELt library
  $ make -j$(nproc)

  # Install hipSPARSELt to /opt/rocm
  $ make install

GoogleTest is required in order to build hipSPARSELt clients.

hipSPARSELt with dependencies and clients can be built using the following commands:

::

  # Install googletest
  $ mkdir -p build/release/deps ; cd build/release/deps
  $ cmake ../../../deps
  $ make -j$(nproc) install

  # Change to build directory
  $ cd ..

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  $ cmake ../.. -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_SAMPLES=ON

  # Compile hipSPARSELt library
  $ make -j$(nproc)

  # Install hipSPARSELt to /opt/rocm
  $ make install

Simple Test
```````````
You can test the installation by running one of the hipSPARSELt examples, after successfully compiling the library with clients.

::

   # Navigate to clients binary directory
   $ cd hipSPARSELt/build/release/clients/staging

   # Execute hipSPARSELt example
   $ ./example_spmm_strided_batched -m 32 -n 32 -k 32 --batch_count 1

Supported Targets
-----------------
Currently, hipSPARSELt is supported under the following operating systems

- `Ubuntu 18.04 <https://ubuntu.com/>`_
- `Ubuntu 20.04 <https://ubuntu.com/>`_
- `CentOS 7 <https://www.centos.org/>`_
- `CentOS 8 <https://www.centos.org/>`_
- `SLES 15 <https://www.suse.com/solutions/enterprise-linux/>`_

To compile and run hipSPARSELt, `AMD ROCm Platform <https://github.com/RadeonOpenCompute/ROCm>`_ is required.

Device and Stream Management
============================
:cpp:func:`hipSetDevice` and :cpp:func:`hipGetDevice` are HIP device management APIs.
They are NOT part of the hipSPARSELt API.

Asynchronous Execution
----------------------
All hipSPARSELt library functions, unless otherwise stated, are non blocking and executed asynchronously with respect to the host. They may return before the actual computation has finished. To force synchronization, :cpp:func:`hipDeviceSynchronize` or :cpp:func:`hipStreamSynchronize` can be used. This will ensure that all previously executed hipSPARSELt functions on the device / this particular stream have completed.

HIP Device Management
---------------------
Before a HIP kernel invocation, users need to call :cpp:func:`hipSetDevice` to set a device, e.g. device 1. If users do not explicitly call it, the system by default sets it as device 0. Unless users explicitly call :cpp:func:`hipSetDevice` to set to another device, their HIP kernels are always launched on device 0.

The above is a HIP (and CUDA) device management approach and has nothing to do with hipSPARSELt. hipSPARSELt honors the approach above and assumes users have already set the device before a hipSPARSELt routine call.

Once users set the device, they create a handle with :ref:`hipsparselt_init`.

Subsequent hipSPARSELt routines take this handle as an input parameter. hipSPARSELt ONLY queries (by :cpp:func:`hipGetDevice`) the user's device; hipSPARSELt does NOT set the device for users. If hipSPARSELt does not see a valid device, it returns an error message. It is the users' responsibility to provide a valid device to hipSPARSELt and ensure the device safety.

Users CANNOT switch devices between :ref:`hipsparselt_init` and :ref:`hipsparselt_destroy`. If users want to change device, they must destroy the current handle and create another hipSPARSELt handle.

HIP Stream Management
---------------------
HIP kernels are always launched in a queue (also known as stream).

If users do not explicitly specify a stream, the system provides a default stream, maintained by the system. Users cannot create or destroy the default stream. However, users can freely create new streams (with :cpp:func:`hipStreamCreate`) and bind it to the hipSPARSELt operations, such as :ref:`hipsparselt_spmma_prune` and :ref:`hipsparselt_matmul`. HIP kernels are invoked in hipSPARSELt routines. If users create a stream, they are responsible for destroying it.

Multiple Streams and Multiple Devices
-------------------------------------
If the system under test has multiple HIP devices, users can run multiple hipSPARSELt handles concurrently, but can NOT run a single hipSPARSELt handle on different discrete devices. Each handle is associated with a particular singular device, and a new handle should be created for each additional device.

Storage Formats
===============

Structured sparsity storage format
------------------
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
                  1.0 & 2.0 & 0.0 & 0.0 \\
                  0.0 & 0.0 & 3.0 & 4.0 \\
                  0.0 & 6.0 & 7.0 & 0.0 \\
                  0.0 & 6.0 & 0.0 & 8.0 \\
                \end{pmatrix}
  metadata =    \begin{pmatrix}
                  0 & 1 \\
                  2 & 3 \\
                  1 & 2 \\
                  1 & 3 \\
                \end{pmatrix}

Types
=====

hipsparseLtHandle_t
-----------------

.. doxygentypedef:: hipsparseLtHandle_t

hipsparseLtMatDescriptor_t
-------------------

.. doxygentypedef:: hipsparseLtMatDescriptor_t

hipsparseLtMatmulDescriptor_t
-----------------

.. doxygentypedef:: hipsparseLtMatmulDescriptor_t

hipsparseLtMatmulAlgSelection_t
------------

.. doxygentypedef:: hipsparseLtMatmulAlgSelection_t

hipsparseLtMatmulPlan_t
------------

.. doxygentypedef:: hipsparseLtMatmulPlan_t

hipsparseLtDatatype_t
--------------

.. doxygentypedef:: hipsparseLtDatatype_t

hipsparseLtSparsity_t
-------------

.. doxygentypedef:: hipsparseLtSparsity_t

hipsparseLtMatDescAttribute_t
------------

.. doxygentypedef:: hipsparseLtMatDescAttribute_t

hipsparseComputetype_t
------------

.. doxygentypedef:: hipsparseComputetype_t

hipsparseLtMatmulDescAttribute_t
--------------

.. doxygentypedef:: hipsparseLtMatmulDescAttribute_t

hipsparseLtMatmulAlg_t
-------------

.. doxygentypedef:: hipsparseLtMatmulAlg_t

hipsparseLtPruneAlg_t
--------------

.. doxygentypedef:: hipsparseLtPruneAlg_t

hipsparseLtSplitKMode_t
-----------

.. doxygentypedef:: hipsparseLtSplitKMode_t

.. _api:

Exported Sparse Functions
=========================

Library Management Functions
----------------------------

+------------------------------------------+
|Function name                             |
+------------------------------------------+
|:cpp:func:`hipsparseLtInit`               |
+------------------------------------------+
|:cpp:func:`hipsparseLtDestroy`            |
+------------------------------------------+

Matrix Descriptor Functions
---------------------------

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
---------------------------

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
--------------------------

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
----------------

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
----------------

+-------------------------------------------+
|Function name                              |
+-------------------------------------------+
|:cpp:func:`hipsparseLtSpMMAPrune`          |
+-------------------------------------------+
|:cpp:func:`hipsparseLtSpMMAPruneCheck`     |
+-------------------------------------------+
|:cpp:func:`hipsparseLtSpMMAPrune2`         |
+-------------------------------------------+
|:cpp:func:`hipsparseLtSpMMAPruneCheck2`    |        |
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
============================

This library management describe functions that provide the library handle.

.. _hipsparselt_init:

hipsparseLtInit()
-----------------

.. doxygenfunction:: hipsparseLtInit

.. _hipsparselt_destroy_:

hipsparseLtDestroy()
--------------------

.. doxygenfunction:: hipsparseLtDestroy


Matrix Descriptor Functions
===========================

The matrix descriptor describe fuctions that used to define sparse and dense matrix

hipsparseLtDenseDescriptorInit()
--------------------------------

.. doxygenfunction:: hipsparseLtDenseDescriptorInit

hipsparseLtMatDescriptorDestroy()
---------------------------------

.. doxygenfunction:: hipsparseLtMatDescriptorDestroy

hipsparseLtMatDescSetAttribute()
--------------------------------

.. doxygenfunction:: hipsparseLtMatDescSetAttribute

hipsparseLtMatDescGetAttribute()
--------------------------------

.. doxygenfunction:: hipsparseLtMatDescGetAttribute


Matmul Descriptor Functions
===========================

This matmul descriptor describe fuctions that used to define how to do the matrix multiply.

hipsparseLtMatmulDescriptorInit()
---------------------------------

.. doxygenfunction:: hipsparseLtMatmulDescriptorInit

hipsparseLtMatmulDescSetAttribute()
-----------------------------------

.. doxygenfunction:: hipsparseLtMatmulDescSetAttribute

hipsparseLtMatmulDescGetAttribute()
-----------------------------------

.. doxygenfunction:: hipsparseLtMatmulDescGetAttribute


Matmul Algorithm Functions
==========================

This matmul algorithm describe functions that provide algortithms for doing the matrix multiply.

hipsparseLtMatmulAlgSelectionInit()
-----------------------------------

.. doxygenfunction:: hipsparseLtMatmulAlgSelectionInit

hipsparseLtMatmulAlgSetAttribute()
----------------------------------

.. doxygenfunction:: hipsparseLtMatmulAlgSetAttribute

hipsparseLtMatmulAlgGetAttribute()
----------------------------------

.. doxygenfunction:: hipsparseLtMatmulAlgGetAttribute



Matmul Functions
================

This matmul describe operations that provide multiply of sparse matrices.

hipsparseLtMatmulGetWorkspace()
-------------------------------

.. doxygenfunction:: hipsparseLtMatmulGetWorkspace

hipsparseLtMatmulPlanInit()
---------------------------

.. doxygenfunction:: hipsparseLtMatmulPlanInit

hipsparseLtMatmulPlanDestroy()
------------------------------

.. doxygenfunction:: hipsparseLtMatmulPlanDestroy

.. _hipsparselt_matmul:

hipsparseLtMatmul()
-------------------

.. doxygenfunction:: hipsparseLtMatmul

hipsparseLtMatmulSearch()
-------------------------

.. doxygenfunction:: hipsparseLtMatmulSearch


Helper Functions
================

This module holds available helper functions that are required for subsequent library calls

.. _hipsparselt_spmma_prune:

hipsparseLtSpMMAPrune()
-----------------------

.. doxygenfunction:: hipsparseLtSpMMAPrune

hipsparseLtSpMMAPruneCheck()
----------------------------

.. doxygenfunction:: hipsparseLtSpMMAPruneCheck

hipsparseLtSpMMAPrune2()
------------------------

.. doxygenfunction:: hipsparseLtSpMMAPrune2

hipsparseLtSpMMAPruneCheck2()
-----------------------------

.. doxygenfunction:: hipsparseLtSpMMAPruneCheck2

hipsparseLtSpMMACompressedSize()
--------------------------------

.. doxygenfunction:: hipsparseLtSpMMACompressedSize

hipsparseLtSpMMACompress()
--------------------------

.. doxygenfunction:: hipsparseLtSpMMACompress

hipsparseLtSpMMACompressedSize2()
---------------------------------

.. doxygenfunction:: hipsparseLtSpMMACompressedSize2

hipsparseLtSpMMACompress2()
---------------------------

.. doxygenfunction:: hipsparseLtSpMMACompress2


