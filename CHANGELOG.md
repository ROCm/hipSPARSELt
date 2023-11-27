# Change Log for hipSPARSELt

## (Unreleased) hipSPARSELt 0.2.0

### Added

- Support Matrix B is a Structured Sparsity Matrix.

## (Unreleased) hipSPARSELt 0.1.0

### Added

- Enable hipSPARSELt APIs
- Support platform: gfx940, gfx941, gfx942
- Support problem type: fp16, bf16, int8
- Support activation: relu, gelu, abs, sigmod, tanh
- Support gelu scaling
- Support bias vector
- Support batched computation (single sparse x multiple dense, multiple sparse x
single dense)
- Support cuSPARSELt v0.4 backend
- Integreate with tensilelite kernel generator
- Add Gtest: hipsparselt-test
- Add benchmarking tool: hipsparselt-bench
- Add sample app: example_spmm_strided_batched, example_prune, example_compress
