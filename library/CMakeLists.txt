# ########################################################################
# Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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

# Configure a header file to pass the rocSPARSELt version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocsparselt-version.h.in"
               "${PROJECT_BINARY_DIR}/include/rocsparselt-version.h"
)

# Public rocSPARSELt headers
set(rocsparselt_headers_public
  include/rocsparselt-auxiliary.h
  include/rocsparselt-functions.h
  include/rocsparselt-types.h
  include/rocsparselt-bfloat16.h
  include/rocsparselt.h
   ${PROJECT_BINARY_DIR}/include/rocsparselt-version.h
)

source_group("Header Files\\Public" FILES ${rocsparselt_headers_public})

include(GNUInstallDirs)

set(BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# Include sources
include(src/CMakeLists.txt)

# Create rocSPARSELt library
add_library(rocsparselt ${rocsparselt_source} ${rocsparselt_headers_public})
add_library(roc::rocsparselt ALIAS rocsparselt)

# Target compile options
target_compile_options(rocsparselt PRIVATE -Wno-unused-command-line-argument -Wall)
target_compile_definitions(rocsparselt PRIVATE ROCM_USE_FLOAT16)

# Target include directories
target_include_directories(rocsparselt
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
                                   $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                                   $<BUILD_INTERFACE:${KERNEL_LAUNCHER_INC}>
                           PUBLIC  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   $<INSTALL_INTERFACE:include>
)

# Target link libraries
target_link_libraries(rocsparselt PRIVATE hip::device ${DL_LIB})
# Target properties
rocm_set_soversion(rocsparselt ${rocsparselt_SOVERSION})
set_target_properties(rocsparselt PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(rocsparselt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_properties(rocsparselt PROPERTIES DEBUG_POSTFIX "-d")

# Generate export header
include(GenerateExportHeader)
generate_export_header(rocsparselt EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocsparselt-export.h)

# Install targets
rocm_install_targets(TARGETS rocsparselt
                     INCLUDE
                       ${CMAKE_SOURCE_DIR}/library/include
                       ${CMAKE_BINARY_DIR}/include
                     PREFIX rocsparselt
)

# Export targets
rocm_export_targets(TARGETS roc::rocsparselt
                    PREFIX rocsparselt
                    DEPENDS PACKAGE hip
                    NAMESPACE roc::
)

# Symbolic links
rocm_install_symlink_subdir(rocsparselt)

# Package specific CPACK vars
set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip-rocclr (>= 3.5.0)")
set(CPACK_RPM_PACKAGE_REQUIRES "hip-rocclr >= 3.5.0")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

# Package name
set(package_name rocsparselt)

set(ROCSPARSELT_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "Radeon Open Compute SPARSE library with xDL supported"
  MAINTAINER "rocSPARSElt Maintainer <vihuan@amd.com>"
  LDCONFIG
  LDCONFIG_DIR ${ROCSPARSELT_CONFIG_DIR}
)
