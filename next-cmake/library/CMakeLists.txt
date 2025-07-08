# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ============================================================================
# Create hipSPARSELt library target
# ============================================================================

# Use component-level shared/static switch (don't modify BUILD_SHARED_LIBS)
if(HIPSPARSELT_BUILD_SHARED_LIBS)
    add_library(hipsparselt SHARED)
else()
    add_library(hipsparselt STATIC)
endif()

# Set library properties
rocm_set_soversion(hipsparselt ${hipsparselt_SOVERSION})

# Create alias for consistent usage
add_library(roc::hipsparselt ALIAS hipsparselt)

# ============================================================================
# Configure version header
# ============================================================================

# Configure version header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../library/include/hipsparselt-version.h.in"
    "${PROJECT_BINARY_DIR}/include/hipsparselt/hipsparselt-version.h"
    @ONLY
)

# Copy public header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../library/include/hipsparselt.h"
    "${PROJECT_BINARY_DIR}/include/hipsparselt/hipsparselt.h"
    COPYONLY
)

# ============================================================================
# Set target properties first
# ============================================================================

# Set C++ standard and features at target level
target_compile_features(hipsparselt PRIVATE cxx_std_17)

# Set target properties
set_target_properties(hipsparselt PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

# Set debug postfix
set_target_properties(hipsparselt PROPERTIES DEBUG_POSTFIX "-d")

# Set library naming for static builds
if(NOT HIPSPARSELT_BUILD_SHARED_LIBS)
    set_target_properties(hipsparselt PROPERTIES PREFIX "lib")
endif()

# ============================================================================
# Platform-specific compile definitions
# ============================================================================

if(HIPSPARSELT_ENABLE_CUDA)
    target_compile_definitions(hipsparselt 
        PRIVATE 
            __HIP_PLATFORM_NVIDIA__
    )
else()
    target_compile_definitions(hipsparselt 
        PRIVATE 
            __HIP_PLATFORM_AMD__
            ROCM_USE_FLOAT16
    )
endif()

# ============================================================================
# Conditional compile definitions
# ============================================================================

# Address sanitizer
if(HIPSPARSELT_ENABLE_ASAN)
    target_compile_options(hipsparselt PRIVATE 
        -fsanitize=address 
        -shared-libasan
    )
    target_link_options(hipsparselt PRIVATE 
        -fsanitize=address 
        -shared-libasan
        -fuse-ld=lld
    )
endif()

# Code coverage
if(HIPSPARSELT_BUILD_COVERAGE)
    target_compile_options(hipsparselt PRIVATE 
        -fprofile-arcs 
        -ftest-coverage
    )
    target_link_libraries(hipsparselt PRIVATE 
        --coverage 
        -lgcov
    )
endif()

# Marker support
if(HIPSPARSELT_ENABLE_MARKER)
    target_compile_definitions(hipsparselt PRIVATE HIPSPARSELT_ENABLE_MARKER)
    target_link_libraries(hipsparselt PRIVATE PkgConfig::ROCTX)
endif()

# Static library flag
if(NOT HIPSPARSELT_BUILD_SHARED_LIBS)
    target_compile_definitions(hipsparselt PRIVATE HIPSPARSELT_STATIC_LIB)
endif()

# ============================================================================
# Compiler-specific options
# ============================================================================

if(NOT HIPSPARSELT_ENABLE_CUDA)
    target_compile_options(hipsparselt PRIVATE 
        -Wall
        -Wno-unused-command-line-argument
    )
    
    # Build-type specific flags
    target_compile_options(hipsparselt PRIVATE
        $<$<CONFIG:Debug>:-O1>
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:-O3>
    )
endif()

# ============================================================================
# Include directories
# ============================================================================

target_include_directories(hipsparselt
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../library/src/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../library/src/hcc_detail/rocsparselt/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../library/src/hcc_detail/rocsparselt/src/include>
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../library/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# ============================================================================
# Add source directory
# ============================================================================

add_subdirectory(src)

# ============================================================================
# Link libraries
# ============================================================================

# Always link with HIP
target_link_libraries(hipsparselt PRIVATE hip::device)

# Platform-specific linking
if(HIPSPARSELT_ENABLE_CUDA)
    target_link_libraries(hipsparselt PRIVATE 
        ${CUDA_CUSPARSELT_LIBRARY}
        CUDA::cusparse
    )
else()
    target_link_libraries(hipsparselt PRIVATE ${CMAKE_DL_LIBS})
endif()

# ============================================================================
# Generate export header
# ============================================================================

generate_export_header(hipsparselt 
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/hipsparselt/hipsparselt-export.h"
)

# ============================================================================
# Installation
# ============================================================================

# Install library and headers
rocm_install_targets(
    TARGETS hipsparselt
    INCLUDE "${PROJECT_BINARY_DIR}/include"
)

# Export targets for find_package
if(NOT HIPSPARSELT_ENABLE_CUDA)
    rocm_export_targets(
        TARGETS roc::hipsparselt
        DEPENDS PACKAGE hip
        NAMESPACE roc::
    )
else()
    rocm_export_targets(
        TARGETS roc::hipsparselt
        DEPENDS PACKAGE HIP
        NAMESPACE roc::
    )
endif()

# ============================================================================
# Package kernel libraries
# ============================================================================

if(NOT HIPSPARSELT_ENABLE_CUDA)
    if(HIPSPARSELT_ENABLE_TENSILELITE)
        # Install TensileLite library files
        if(WIN32)
            set(HIPSPARSELT_TENSILE_LIBRARY_DIR "${CMAKE_INSTALL_BINDIR}/hipsparselt")
        else()
            set(HIPSPARSELT_TENSILE_LIBRARY_DIR "${CMAKE_INSTALL_LIBDIR}/hipsparselt")
        endif()
        
        # Install Tensile library directory
        install(
            DIRECTORY "${PROJECT_BINARY_DIR}/Tensile/library/"
            DESTINATION "${HIPSPARSELT_TENSILE_LIBRARY_DIR}"
            COMPONENT runtime
            FILES_MATCHING PATTERN "*"
        )
    endif()
endif()