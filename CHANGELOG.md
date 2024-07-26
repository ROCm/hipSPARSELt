# Change Log for hipSPARSELt

## (Unreleased) hipSPARSELt 0.2.1

### Changes

* Refine test cases.

## (Unreleased) hipSPARSELt 0.2.0

### Additions

* Support Matrix B is a Structured Sparsity Matrix.

## hipSPARSELt 0.1.0

### Additions

* Enabled hipSPARSELt APIs
* Support for:
  * gfx940, gfx941, and gfx942 platforms
  * FP16, BF16, and INT8 problem types
  * ReLU, GELU, abs, sigmod, and tanh activation
  * GELU scaling
  * Bias vectors
  * cuSPARSELt v0.4 backend
* Integrated with Tensile Lite kernel generator
* Support for batched computation (single sparse x multiple dense and multiple sparse x
single dense)
* GoogleTest: hipsparselt-test
* `hipsparselt-bench` benchmarking tool
* Sample apps: `example_spmm_strided_batched`, `example_prune`, `example_compress`
