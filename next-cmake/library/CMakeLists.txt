# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ============================================================================
# Create hipSPARSELt library target
# ============================================================================

set(_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../library")

if(HIPSPARSELT_BUILD_SHARED_LIBS)
    add_library(hipsparselt SHARED)
else()
    add_library(hipsparselt STATIC)
    target_compile_definitions(hipsparselt PRIVATE HIPSPARSELT_STATIC_LIB)
endif()

rocm_set_soversion(hipsparselt ${hipsparselt_SOVERSION})
add_library(roc::hipsparselt ALIAS hipsparselt)

target_compile_features(hipsparselt PRIVATE cxx_std_17)
if(HIPSPARSELT_ENABLE_CUDA)
    target_compile_definitions(hipsparselt PRIVATE __HIP_PLATFORM_NVIDIA__)
else()
    target_compile_definitions(hipsparselt PRIVATE __HIP_PLATFORM_AMD__ ROCM_USE_FLOAT16)
    # target_compile_options(hipsparselt PRIVATE 
    #     -Wall
    #     -Wno-unused-command-line-argument
    # )
    # target_compile_options(hipsparselt PRIVATE
    #     $<$<CONFIG:Debug>:-O1>
    #     $<$<CONFIG:Debug>:-g>
    #     $<$<CONFIG:Release>:-O3>
    # )
endif()

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

# TODO: Move this all to the end of the top-level CMakeLists.txt
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

if(HIPSPARSELT_ENABLE_MARKER)
    target_compile_definitions(hipsparselt PRIVATE HIPSPARSELT_ENABLE_MARKER)
    target_link_libraries(hipsparselt PRIVATE ${ROCTX64_LIBRARY})
endif()

set_target_properties(hipsparselt PROPERTIES
    CXX_EXTENSIONS OFF
    VISIBILITY_INLINES_HIDDEN ON
    CXX_VISIBILITY_PRESET "hidden"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

# # Set debug postfix
# set_target_properties(hipsparselt PROPERTIES DEBUG_POSTFIX "-d")

# # Set library naming for static builds
# if(NOT HIPSPARSELT_BUILD_SHARED_LIBS)
#     set_target_properties(hipsparselt PROPERTIES PREFIX "lib")
# endif()

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

target_link_libraries(hipsparselt PRIVATE hip::device)

# TODO: Review this section for correctness
if(HIPSPARSELT_ENABLE_CUDA)
    target_link_libraries(hipsparselt PRIVATE 
        ${CUDA_CUSPARSELT_LIBRARY}
        CUDA::cusparse
    )
else()
    target_link_libraries(hipsparselt PRIVATE ${CMAKE_DL_LIBS})
endif()

add_subdirectory(include)
add_subdirectory(src)

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