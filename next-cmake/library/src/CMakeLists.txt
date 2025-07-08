# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../library/src")

target_sources(hipsparselt PRIVATE
    "${_CMAKE_CURRENT_SOURCE_DIR}/hipsparselt_ostream.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/auxiliary.cpp"
)

if(HIPSPARSELT_ENABLE_CUDA)
    add_subdirectory(nvcc_detail)
else()
    add_subdirectory(hcc_detail)

endif()

# ============================================================================
# Platform-specific include directories
# ============================================================================

if(HIPSPARSELT_ENABLE_CUDA)
    target_include_directories(hipsparselt PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../library/src/nvcc_detail"
    )
else()
    target_include_directories(hipsparselt PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../library/src/hcc_detail"
    )
endif()

# ============================================================================
# Compile definitions for internal API
# ============================================================================

target_compile_definitions(hipsparselt PRIVATE
    HIPSPARSELT_INTERNAL_API
)

# ============================================================================
# Compiler-specific optimizations
# ============================================================================

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(hipsparselt PRIVATE
        -mf16c  # Enable F16C intrinsics
    )
endif()

# ============================================================================
# Debug information for development builds
# ============================================================================

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hipsparselt PRIVATE
        -gsplit-dwarf
        -ggdb
    )
endif()

# ============================================================================
# Generate kernel parameter files (if needed)
# ============================================================================

if(NOT HIPSPARSELT_ENABLE_CUDA AND NOT HIPSPARSELT_ENABLE_TENSILELITE)
    # For builds without TensileLite, we may need to generate kernel parameter files
    # This is a placeholder for potential future kernel generation
    message(STATUS "Building without TensileLite - using fallback implementation")
endif()

# ============================================================================
# Source groups for IDE organization
# ============================================================================

source_group("Source Files\\Common" FILES ${hipsparselt_source_common})
source_group("Source Files\\Backend" FILES ${hipsparselt_source_backend})

if(HIPSPARSELT_ENABLE_TENSILELITE AND NOT HIPSPARSELT_ENABLE_CUDA)
    source_group("Source Files\\TensileLite" FILES ${tensile_source_files})
endif() 