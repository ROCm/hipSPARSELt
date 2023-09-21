# Supported functions

* ROCm
  * AMD sparse MFMA matrix core support
  * Mixed-precision computation support:
    * FP16 input/output, FP32 Matrix Core accumulate
    * BFLOAT16 input/output, FP32 Matrix Core accumulate
    * INT8 input/output, INT32 Matrix Core accumulate
  * Matrix pruning and compression functionalities
  * Auto-tuning functionality (see `hipsparseLtMatmulSearch()`)
  * Batched Sparse Gemm support:
    * Single sparse matrix/Multiple dense matrices (Broadcast)
    * Multiple sparse and dense matrices
  * Activation function fuse in spmm kernel support:
    * ReLU
    * ClippedReLU (ReLU with upper bound and threshold setting)
    * GeLU
    * Abs
    * LeakyReLU
    * Sigmoid
    * Tanh
  * On Going Feature Development
    * Add support for Mixed-precision computation
      * FP8 input/output, FP32 Matrix Core accumulate
      * BF8 input/output, FP32 Matrix Core accumulate
      * Add kernel selection and genroator, used to provide the appropriate solution for the specific problem.
* CUDA
  * Support cusparseLt v0.3
