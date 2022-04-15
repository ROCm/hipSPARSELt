# ########################################################################
# Copyright (c) 2016-2022 Advanced Micro Devices, Inc.
# ########################################################################

cmake_minimum_required( VERSION 3.16.8 )

add_definitions(-D_ROCSPARSELT_INTERNAL_BFLOAT16_)

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project( rocsparselt-clients LANGUAGES CXX C  )


if( SKIP_LIBRARY )
  include_directories(${ROCSPARSELT_LIBRARY_DIR}/include)
  #include_directories(${ROCSPARSELT_LIBRARY_DIR}/include/internal)
else()
  include_directories(${CMAKE_BINARY_DIR}/include)
  #include_directories(${CMAKE_BINARY_DIR}/include/internal)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# if it fails to find OpenMP compile and link flags in strange configurations it can just use non-parallel reference computation
# if there is no omp.h to find the client compilation will fail and this should be obvious, used to be REQUIRED
find_package(OpenMP)

if (TARGET OpenMP::OpenMP_CXX)
  set( COMMON_LINK_LIBS "OpenMP::OpenMP_CXX")
  list( APPEND COMMON_LINK_LIBS "-L${HIP_CLANG_ROOT}/lib;-Wl,-rpath=${HIP_CLANG_ROOT}/lib")
endif()

if (TARGET Threads::Threads)
  list( APPEND COMMON_LINK_LIBS "Threads::Threads")
endif()

message(STATUS "CLIENT COMMON CXX_OPTIONS: ${COMMON_CXX_OPTIONS}")
message(STATUS "CLIENT COMMON LINK: ${COMMON_LINK_LIBS}")

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

#include( build-options )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if( NOT TARGET rocsparselt )
  find_package( rocsparselt REQUIRED CONFIG PATHS ${ROCM_PATH}/rocsparselt /opt/rocm/rocsparselt ${ROCSPARSELT_LIBRARY_DIR})
endif( )

# Hip headers required of all clients; clients use hip to allocate device memory
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} /opt/rocm )
if ( NOT hip_FOUND )
  find_package( hip REQUIRED CONFIG PATHS ${ROCM_PATH} )
endif( )


if( BUILD_CLIENTS_SAMPLES )
  add_subdirectory( samples )
endif( )

if( BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)

  # Linking lapack library requires fortran flags
  find_package( cblas REQUIRED CONFIG )
  if (LINK_BLIS)
    set( BLIS_INCLUDE_DIR ${BUILD_DIR}/deps/blis/include/blis )
    set( BLIS_CPP ../common/blis_interface.cpp )
    set( BLAS_LIBRARY ${BUILD_DIR}/deps/blis/lib/libblis.so )
  else()
    set( BLAS_LIBRARY "blas" )
  endif()

  # common source files used in subdirectories benchmarks and gtest thus ../common
  set( rocsparselt_test_bench_common
      ../common/singletons.cpp
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ../common/argument_model.cpp
      ../common/rocsparselt_parse_data.cpp
      ../common/rocsparselt_arguments.cpp
      ../common/rocsparselt_random.cpp
      ${BLIS_CPP}
    )

  if( BUILD_CLIENTS_BENCHMARKS )
    add_subdirectory( benchmarks )
  endif( )

  if( BUILD_CLIENTS_TESTS )
    add_subdirectory( gtest )
  endif( )

endif()


set( ROCSPARSELT_COMMON "${PROJECT_BINARY_DIR}/staging/rocsparselt_common.yaml")
add_custom_command( OUTPUT "${ROCSPARSELT_COMMON}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocsparselt_common.yaml "${ROCSPARSELT_COMMON}"
                    DEPENDS include/rocsparselt_common.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( ROCSPARSELT_TEMPLATE "${PROJECT_BINARY_DIR}/staging/rocsparselt_template.yaml")
add_custom_command( OUTPUT "${ROCSPARSELT_TEMPLATE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocsparselt_template.yaml "${ROCSPARSELT_TEMPLATE}"
                    DEPENDS include/rocsparselt_template.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( ROCSPARSELT_GENTEST "${PROJECT_BINARY_DIR}/staging/rocsparselt_gentest.py")
add_custom_command( OUTPUT "${ROCSPARSELT_GENTEST}"
                    COMMAND ${CMAKE_COMMAND} -E copy common/rocsparselt_gentest.py "${ROCSPARSELT_GENTEST}"
                    DEPENDS common/rocsparselt_gentest.py
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

add_custom_target( rocsparselt-common DEPENDS "${ROCSPARSELT_COMMON}" "${ROCSPARSELT_TEMPLATE}" "${ROCSPARSELT_GENTEST}" )
