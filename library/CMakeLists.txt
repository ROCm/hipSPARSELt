# ########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print verbose compiler flags
if(BUILD_VERBOSE)
  include(../cmake/Verbose.cmake)
endif()

# Configure a header file to pass the hipSPARSE version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipsparselt-version.h.in"
               "${PROJECT_BINARY_DIR}/include/hipsparselt-version.h"
)

# Public hipSPARSE headers
set(hipsparselt_headers_public
  include/hipsparselt.h
  ${PROJECT_BINARY_DIR}/include/hipsparselt-version.h
)

if(NOT BUILD_CUDA)
    set(hipsparselt_headers_public ${hipsparselt_headers_public})
endif()
source_group("Header Files\\Public" FILES ${hipsparselt_headers_public})

include(GNUInstallDirs)

set(BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# Include sources
include(src/CMakeLists.txt)

# Create hipSPARSELt library
add_library(hipsparselt ${hipsparselt_source} ${hipsparselt_headers_public})
add_library(roc::hipsparselt ALIAS hipsparselt)

# Target compile definitions
if(NOT BUILD_CUDA)
    target_compile_options(hipsparselt PRIVATE -Wno-unused-command-line-argument -Wall)
    target_compile_definitions(hipsparselt PRIVATE ROCM_USE_FLOAT16 __HIP_PLATFORM_HCC__)
else()
    target_compile_definitions(hipsparselt PRIVATE __HIP_PLATFORM_NVCC__)
endif()

# Target compile features
target_compile_features(hipsparselt PRIVATE cxx_nullptr)

# Target include directories
target_include_directories(hipsparselt
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
                           PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
                                   $<INSTALL_INTERFACE:include>
)

if(BUILD_CUDA)
target_include_directories(hipsparselt
                           PUBLIC $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
)
else()
target_include_directories(hipsparselt
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/hcc_detail/rocsparselt/include>
                                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/hcc_detail/rocsparselt/src/include>
)
endif()

# Target link libraries
if(NOT BUILD_CUDA)
# Target link libraries
  target_link_libraries(hipsparselt PRIVATE hip::device roc::hipsparse ${DL_LIB})
else()
  target_link_libraries(hipsparselt PRIVATE /usr/lib/x86_64-linux-gnu/libcusparseLt.so ${CUDA_CUSPARSE_LIBRARY} roc::hipsparse )
endif()

# Target properties
rocm_set_soversion(hipsparselt ${hipsparselt_SOVERSION})
set_target_properties(hipsparselt PROPERTIES CXX_EXTENSIONS NO)
set_target_properties(hipsparselt PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(hipsparselt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_propertieS(hipsparselt PROPERTIES DEBUG_POSTFIX "-d")

# TODO ??
# Following boost conventions of prefixing 'lib' on static built libraries
if(NOT BUILD_SHARED_LIBS)
  set_target_properties(hipsparselt PROPERTIES PREFIX "lib")
endif()

# Generate export header
include(GenerateExportHeader)
generate_export_header(hipsparselt EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipsparselt-export.h)

# Install targets
rocm_install_targets(TARGETS hipsparselt ${SPMM_KERNEL_LIBS}
                     INCLUDE
                       ${CMAKE_SOURCE_DIR}/library/include
                       ${CMAKE_BINARY_DIR}/include
                     PREFIX hipsparselt
)

# Export targets
if(NOT BUILD_CUDA)
  rocm_export_targets(TARGETS roc::hipsparselt
                      PREFIX hipsparselt
                      DEPENDS PACKAGE hip
                      NAMESPACE roc::)
else()
  rocm_export_targets(TARGETS roc::hipsparselt
                      PREFIX hipsparselt
                      NAMESPACE roc::)
endif()

# Symbolic links
rocm_install_symlink_subdir(hipsparselt)

if(NOT BUILD_CUDA)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip-rocclr (>= 3.5.0)")
  set(CPACK_RPM_PACKAGE_REQUIRES "hip-rocclr >= 3.5.0")
endif()

# Package specific CPACK vars
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

if(NOT BUILD_CUDA)
  set(package_name hipsparselt)
else()
  set(package_name hipsparselt-alt)
endif()

set(HIPSPARSELT_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "Radeon Open Compute SPARSE library with xDL supported"
  MAINTAINER "hipSPARSELt Maintainer"
  LDCONFIG
  LDCONFIG_DIR ${HIPSPARSELT_CONFIG_DIR}
)
