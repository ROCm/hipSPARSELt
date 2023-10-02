.. meta::
   :description: hipSPARSELt API reference guide
   :keywords: hipSPARSELt, ROCm, API library, API reference

.. _api-reference:
********************************
hipSPARSELt API reference guide
********************************

hipSPARSELt is a library that contains basic linear algebra subroutines (BLAS) for SPARSE matrices that
are written in the `HIP programming language <https://rocm.docs.amd.com/projects/HIP/en/latest/>`_. It
sits between the application and a 'worker' SPARSE library, marshalling inputs into the backend library
and marshalling results back to the application. Supported backends are
`rocSPARSELt <https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html>`_ (included in
hipSPARSELt) and cuSPARSELt.

hipSPARSELt supports AMD's SPARSE matrix core technology on AMD RDNA/CDNA GPUs. It is
designed for use with C and C++ code.

.. note::
    Code for hipSPARSELt is open source and hosted on
    `GitHub <https://github.com/ROCmSoftwarePlatform/hipSPARSELt>`_.

The API reference guide is organized into the following sections:

* :ref:`supported-functions`
* :ref:`data-type-support`
* :ref:`device-stream-manage`
* :ref:`storage-format`
* :ref:`porting`
* :doc:`API library <../doxygen/docBin/html/index>`
