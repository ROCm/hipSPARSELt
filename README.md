# rocSPARSELt
rocSPARSELt provides general matrix-matrix operations for sparse computation implemented on top of AMD's Radeon Open eCosystem Platform [ROCm][] runtime and toolchains. rocSPARSELt is created using the [HIP][] programming language and optimized for AMD's latest discrete GPUs.

## Documentation (TBD)
The latest rocSPARSELt documentation and API description can be found [here][] or downloaded as [pdf][].

## Requirements
* Git
* CMake (3.5 or later)
* AMD [ROCm] 3.5 platform or later

Optional:
* [GTest][]
  * Required for tests.
  * Use GTEST_ROOT to specify GTest location.
  * If [GTest][] is not found, it will be downloaded and built automatically.

## Quickstart rocSPARSELt build and install

#### Install script
You can build rocSPARSELt using the *install.sh* script
```
# Clone rocSPARSELt using git
TBD

# Go to rocSPARSELt directory
cd rocSPARSELt

# Run install.sh script
# Command line options:
#   -h|--help         - prints help message
#   -i|--install      - install after build
#   -d|--dependencies - install build dependencies
#   -c|--clients      - build library clients too (combines with -i & -d)
#   -g|--debug        - build with debug flag
./install.sh -dci
```

#### CMake
All compiler specifications are determined automatically. The compilation process can be performed by
```
# Clone rocSPARSELt using git
TBD

# Go to rocSPARSELt directory, create and go to the build directory
cd rocSPARSELt; mkdir -p build/release; cd build/release

# Configure rocSPARSELt
# Build options:
#   BUILD_CLIENTS_TESTS      - build tests (OFF)
#   BUILD_CLIENTS_BENCHMARKS - build benchmarks (OFF)
#   BUILD_CLIENTS_SAMPLES    - build examples (ON)
#   BUILD_VERBOSE            - verbose output (OFF)
#   BUILD_SHARED_LIBS        - build rocSPARSELt as a shared library (ON)
CXX=/opt/rocm/bin/hipcc cmake -DBUILD_CLIENTS_TESTS=ON ../..

# Build
make

# Install
[sudo] make install
```

## Unit tests
To run unit tests, rocSPARSELt has to be built with option -DBUILD_CLIENTS_TESTS=ON.
```
# Go to rocSPARSELt build directory
cd rocSPARSELt; cd build/release

# Run all tests
./clients/staging/rocsparselt-test
```

## Benchmarks
To run benchmarks, rocSPARSELt has to be built with option -DBUILD_CLIENTS_BENCHMARKS=ON.
```
# Go to rocSPARSELt build directory
cd rocSPARSELt/build/release

# Run benchmark, e.g.
./clients/staging/rocsparselt-bench -f spmm -i 200
###TBD
```

## Support
Please use [the issue tracker][] for bugs and feature requests.

## License
The [license file][] can be found in the main repository.

[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[HIP]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/
[GTest]: https://github.com/google/googletest
[the issue tracker]: TBD
[license file]: TBD
