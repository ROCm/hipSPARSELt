# Installing hipSPARSELt (Linux)

## Prerequisites

hipSPARSELt requires a [ROCm-enabled platform](https://rocm.github.io/) and the
[hipSPARSE ROCm library](https://github.com/ROCmSoftwarePlatform/hipSPARSE) (for the header file).

## Installing pre-built packages

Install hipSPARSELt from the
[ROCm repository](https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories).

For detailed instructions on how to set up ROCm on different platforms, refer to the
[ROCm Linux installation guide](https://rocm.docs.amd.com/en/develop/tutorials/install/linux/index.html).

Using Ubuntu as an example, hipSPARSELt can be installed using

```bash
    sudo apt-get update
    sudo apt-get install hipsparselt
```

Once installed, hipSPARSELt can be used just like any other library with a C API.

The header file must be included in the user code in order to make calls into hipSPARSELt. The
hipSPARSELt shared library will become link-time and run-time dependent for the user application.

## Building hipSPARSELt from source

Although you can build from source, it's not necessary--hipSPARSELt can be used after installing the
pre-built packages described above. If you still want to build from source, use the following
instructions.

```{note}
The following compile-time dependencies must be met:

* [hipSPARSE](https://github.com/ROCmSoftwarePlatform/hipSPARSE)
* [git](https://git-scm.com/)
* [CMake](https://cmake.org/) 3.5 or later
* [AMD ROCm](https://github.com/RadeonOpenCompute/ROCm)
* (Optional, for clients) [googletest](https://github.com/google/googletest)
```

### Download hipSPARSELt

The hipSPARSELt source code is available on our
[GitHub page](https://github.com/ROCmSoftwarePlatform/hipSPARSELt)

Download the develop branch using:

```bash
git clone -b develop https://github.com/ROCmSoftwarePlatform/hipSPARSELt.git
cd hipSPARSELt
```

### Build library packages, including dependencies and clients

We recommended installing hipSPARSELt using the `install.sh` script.

1. Using `install.sh` to build hipSPARSELt with dependencies:

    The following table lists common uses of `install.sh` to build dependencies + library.

    | Command | Description |
    |-------------|--------------|
    | `./install.sh -h` | Print help information |
    | `./install.sh -d` | Build dependencies and library in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh`, it's not necessary to rebuild the dependencies. |
    | `./install.sh` | Build library in your local directory. It is assumed dependencies are available. |
    | `./install.sh -i` | Build library, then build and install hipSPARSELt package in `/opt/rocm/hipsparselt`. You will be prompted for sudo access. This will install for all users. |

2. Using `install.sh` to build hipSPARSELt with dependencies and clients:

    The client contains example code and unit tests. Common uses of `install.sh` to build them are listed in the table below.

    | Command | Description |
    |-------------|--------------|
    | `./install.sh -h` | Print help information |
    | `./install.sh -dc` | Build dependencies, library, and client in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh`, it's not necessary to rebuild the dependencies. |
    | `./install.sh -c` | Build library and client in your local directory. It is assumed dependencies are available. |
    | `./install.sh -idc` | Build library, dependencies, and client; then build and install hipSPARSELt package in `/opt/rocm/hipsparselt`. You will be prompted for sudo access. This will install for all users. |
    | `./install.sh -ic` | Build library and client, then build and install hipSPARSELt package in `opt/rocm/hipsparselt`. You will be prompted for sudo access. This will install for all users. |

3. Using individual commands to build hipSPARSELt:

    ```{note}
    CMake 3.16.8 or later is required in order to build hipSPARSELt.
    ```

    ```bash
        # Create and change the build directory
        $ mkdir -p build/release ; cd build/release

        # Change default install path (/opt/rocm); use -DCMAKE_INSTALL_PREFIX=<path> to adjust the path
        $  cmake ../..

        # Compile the hipSPARSELt library
        $ make -j$(nproc)

        # Install hipSPARSELt to `/opt/rocm`
        $ make install
      ```

    GoogleTest is required in order to build hipSPARSELt clients.

    Build hipSPARSELt with dependencies and clients using the following commands:

    ```bash
        # Install googletest
        $ mkdir -p build/release/deps ; cd build/release/deps
        $ cmake ../../../deps
        $ make -j$(nproc) install

        # Change to build directory
        $ cd ..

        # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
        $ cmake ../.. -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_SAMPLES=ON

        # Compile hipSPARSELt library
        $ make -j$(nproc)

        # Install hipSPARSELt to /opt/rocm
        $ make install
     ```

### Testing the installation

After successfully compiling the library with clients, you can test the installation by running a hipSPARSELt example:

``` bash
   # Navigate to clients binary directory
   $ cd hipSPARSELt/build/release/clients/staging

   # Execute hipSPARSELt example
   $ ./example_spmm_strided_batched -m 32 -n 32 -k 32 --batch_count 1
```
