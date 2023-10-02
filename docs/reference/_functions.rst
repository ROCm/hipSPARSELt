.. meta::
   :description: HIP device and stream management with hipSPARSELt
   :keywords: hipSPARSELt, ROCm, API library, API reference, stream
management, device management

*****************
Matrix descriptor
*****************

| Function name | int (CUDA) | single | half | bfloat16 | int8 |
|------------------|-------------|-------|------|-----------|-----|
| `hipsparseLtDenseDescriptorInit() <hipsparseLtDenseDescriptorInit>` | x | x | x | x | x |
| `hipsparseLtStructuredDescriptorInit() <hipsparseLtStructuredDescriptorInit>` | x | x | x | x | x |

## Matmul descriptor

| Function name | TF32 (CUDA) | TF32 Fast (CUDA) | single (HIP) | half (CUDA) | int |
|------------------|---------------|--------------------|--------------|---------------|----|
| `hipsparseLtMatmulDescriptorInit() <hipsparseLtMatmulDescriptorInit>`  | x | x | x | x | x |


Asynchronous API
----------------
Except a functions having memory allocation inside preventing asynchronicity, all hipSPARSELt functions are configured to operate in non-blocking fashion with respect to CPU, meaning these library functions return immediately.
