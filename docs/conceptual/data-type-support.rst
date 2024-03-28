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
      - HIPSPARSELT_R_8I
      - ✅
      - ✅
    *
      - float8
      - HIPSPARSELT_R_8F
      - ❌
      - ❌
    *
      - bfloat8
      - HIPSPARSELT_R_8BF
      - ❌
      - ❌
    *
      - int16
      - Not Supported
      - ❌
      - ❌
    *
      - float16
      - HIPSPARSELT_R_16F
      - ✅
      - ✅
    *
      - bfloat16      
      - HIPSPARSELT_R_16BF
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
      - HIPSPARSELT_R_32F
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

     "HIPSPARSELT_R_16F", "HIPSPARSELT_R_16F", "HIPSPARSELT_COMPUTE_32F", "HIP"
     "HIPSPARSELT_R_16BF", "HIPSPARSELT_R_16BF", "HIPSPARSELT_COMPUTE_32F", "HIP"
     "HIPSPARSELT_R_8I", "HIPSPARSELT_R_8I", "HIPSPARSELT_COMPUTE_32I", "HIP / CUDA"
     "HIPSPARSELT_R_8I", "HIPSPARSELT_R_16F", "HIPSPARSELT_COMPUTE_32I", "HIP / CUDA"
     "HIPSPARSELT_R_16F", "HIPSPARSELT_R_16F", "HIPSPARSELT_COMPUTE_16F", "CUDA"
     "HIPSPARSELT_R_16BF", "HIPSPARSELT_R_16BF", "HIPSPARSELT_COMPUTE_16F", "CUDA"
     "HIPSPARSELT_R_32F", "HIPSPARSELT_R_32F", "HIPSPARSELT_COMPUTE_TF32", "CUDA"
     "HIPSPARSELT_R_32F", "HIPSPARSELT_R_32F", "HIPSPARSELT_COMPUTE_TF32_FAST", "CUDA"
