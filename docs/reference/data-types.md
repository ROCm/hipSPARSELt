# Data types

```{eval-rst}
hipsparseLtHandle_t
.. doxygenstruct:: hipsparseLtHandle_t
```

hipsparseLtMatDescriptor_t
.. doxygenstruct:: hipsparseLtMatDescriptor_t

hipsparseLtMatmulDescriptor_t
.. doxygenstruct:: hipsparseLtMatmulDescriptor_t

hipsparseLtMatmulAlgSelection_t
.. doxygenstruct:: hipsparseLtMatmulAlgSelection_t

hipsparseLtMatmulPlan_t
.. doxygenstruct:: hipsparseLtMatmulPlan_t

hipsparseLtDatatype_t
.. doxygenenum:: hipsparseLtDatatype_t

hipsparseLtSparsity_t
.. doxygenenum:: hipsparseLtSparsity_t

hipsparseLtMatDescAttribute_t
.. doxygenenum:: hipsparseLtMatDescAttribute_t

hipsparseLtComputetype_t
.. doxygenenum:: hipsparseLtComputetype_t

hipsparseLtMatmulDescAttribute_t
.. doxygenenum:: hipsparseLtMatmulDescAttribute_t

hipsparseLtMatmulAlg_t
.. doxygenenum:: hipsparseLtMatmulAlg_t

hipsparseLtPruneAlg_t
.. doxygenenum:: hipsparseLtMatmulAlgAttribute_t

hipsparseLtMatmulAlgAttribute_t
.. doxygenenum:: hipsparseLtPruneAlg_t

hipsparseLtSplitKMode_t
.. doxygenenum:: hipsparseLtSplitKMode_t

.. _api:

## Data type support

| Input | Output | Compute Type | Backend |
|-------|---------|------------------|----------|
| HIPSPARSELT_R_16F | HIPSPARSELT_R_16F | HIPSPARSELT_COMPUTE_32F | HIP |
| HIPSPARSELT_R_16BF | HIPSPARSELT_R_16BF | HIPSPARSELT_COMPUTE_32F | HIP |
| HIPSPARSELT_R_8I | HIPSPARSELT_R_8I | HIPSPARSELT_COMPUTE_32I | HIP / CUDA |
| HIPSPARSELT_R_8I | HIPSPARSELT_R_16F | HIPSPARSELT_COMPUTE_32I | HIP / CUDA |
| HIPSPARSELT_R_16F | HIPSPARSELT_R_16F | HIPSPARSELT_COMPUTE_16F | CUDA |
| HIPSPARSELT_R_16BF | HIPSPARSELT_R_16BF | HIPSPARSELT_COMPUTE_16F | CUDA |
| HIPSPARSELT_R_32F | HIPSPARSELT_R_32F | HIPSPARSELT_COMPUTE_TF32 | CUDA |
| HIPSPARSELT_R_32F | HIPSPARSELT_R_32F | HIPSPARSELT_COMPUTE_TF32_FAST | CUDA |
