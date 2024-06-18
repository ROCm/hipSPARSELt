.. meta::
   :description: hipSPARSELt API library data type support
   :keywords: hipSPARSELt, ROCm, API library, API reference, data type, support

.. _data-type-support:

******************************************
Data type support
******************************************

* Supported input and output types.

  .. list-table:: Supported Input/Output Types
    :header-rows: 1
    :name: supported-input-output-types

    *
      - Input/Output Types
      - Library Data Type
      - AMD Supports
      - CUDA Supports
    *
      - int8
      - HIP_R_8I
      - ✅
      - ✅
    *
      - float8
      - HIP_R_8F_E4M3_FNUZ
      - ❌
      - ❌
    *
      - bfloat8
      - HIP_R_8F_E5M2_FNUZ
      - ❌
      - ❌
    *
      - int16
      - Not Supported
      - ❌
      - ❌
    *
      - float16
      - HIP_R_16F
      - ✅
      - ✅
    *
      - bfloat16
      - HIP_R_16BF
      - ✅
      - ✅
    *
      - int32
      - Not Supported
      - ❌
      - ❌
    *
      - tensorfloat32
      - Not Supported
      - ❌
      - ❌
    *
      - float32
      - HIP_R_32F
      - ❌
      - ✅
    *
      - float64
      - Not Supported
      - ❌
      - ❌

* Supported accumulator types.

  .. list-table:: Supported Compute Types
    :header-rows: 1
    :name: supported-accumulator-types

    *
      - Accumulator Types
      - Library Data Type
      - AMD Supports
      - CUDA Supports
    *
      - int8
      - Not Supported
      - ❌
      - ❌
    *
      - float8
      - Not Supported
      - ❌
      - ❌
    *
      - bfloat8
      - Not Supported
      - ❌
      - ❌
    *
      - int16
      - Not Supported
      - ❌
      - ❌
    *
      - float16
      - HIPSPARSELT_COMPUTE_16F
      - ❌
      - ✅
    *
      - bfloat16
      - Not Supported
      - ❌
      - ❌
    *
      - int32
      - HIPSPARSELT_COMPUTE_32I
      - ✅
      - ✅
    *
      - tensorfloat32
      - Not Supported
      - ❌
      - ✅
    *
      - float32
      - HIPSPARSELT_COMPUTE_32F
      - ✅
      - ❌
    *
      - float64
      - Not Supported
      - ❌
      - ❌

* List of supported compute types at specific input and output types:

  .. csv-table::
     :header: "Input", "Output", "Compute type", "Backend"

     "HIP_R_16F", "HIP_R_16F", "HIPSPARSELT_COMPUTE_32F", "HIP"
     "HIP_R_16BF", "HIP_R_16BF", "HIPSPARSELT_COMPUTE_32F", "HIP"
     "HIP_R_8I", "HIP_R_8I", "HIPSPARSELT_COMPUTE_32I", "HIP / CUDA"
     "HIP_R_8I", "HIP_R_16F", "HIPSPARSELT_COMPUTE_32I", "HIP / CUDA"
     "HIP_R_8I", "HIP_R_16BF", "HIPSPARSELT_COMPUTE_32I", "HIP / CUDA"
     "HIP_R_16F", "HIP_R_16F", "HIPSPARSELT_COMPUTE_16F", "CUDA"
     "HIP_R_16BF", "HIP_R_16BF", "HIPSPARSELT_COMPUTE_16F", "CUDA"
     "HIP_R_32F", "HIP_R_32F", "HIPSPARSELT_COMPUTE_TF32", "CUDA"
     "HIP_R_32F", "HIP_R_32F", "HIPSPARSELT_COMPUTE_TF32_FAST", "CUDA"
