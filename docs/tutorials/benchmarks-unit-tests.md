<meta name="description" content="Running benchmarks & unit tests">
<meta name="keywords" content="hipSPARSELt, ROCm, benchmarks, unit tests">

# Running benchmarks & unit tests

## Running benchmarks

To run benchmarks, hipSPARSELt has to be built with option -DBUILD_CLIENTS_BENCHMARKS=ON (or using ./install.sh -c).

```bash
# Go to hipSPARSELt build directory
cd hipSPARSELt/build/release

# Run benchmark, e.g.
./clients/staging/hipsparselt-bench -f spmm -i 200 -m 256 -n 256 -k 256
```

## Running unit tests

To run unit tests, hipSPARSELt has to be built with option -DBUILD_CLIENTS_TESTS=ON (or using ./install.sh -c)

```bash
# Go to hipSPARSELt build directory
cd hipSPARSELt; cd build/release

# Run all tests
./clients/staging/hipsparselt-test
```
