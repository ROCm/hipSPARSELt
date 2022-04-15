# ########################################################################
# Copyright (c) 2016-2022 Advanced Micro Devices, Inc.
# ########################################################################

set(rocsparselt_bench_source
  client.cpp
  )

add_executable( rocsparselt-bench ${rocsparselt_bench_source} ${rocsparselt_test_bench_common} )

# Internal header includes
target_include_directories( rocsparselt-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/spmm>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
)

# External header includes included as system files
target_include_directories( rocsparselt-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
)

target_link_libraries( rocsparselt-bench PRIVATE ${BLAS_LIBRARY} roc::rocsparselt )

target_link_libraries( rocsparselt-bench PRIVATE hip::host hip::device )

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # GCC or hip-clang needs specific flags to turn on f16c intrinsics
  target_compile_options( rocsparselt-bench PRIVATE -mf16c )
endif( )

target_compile_definitions( rocsparselt-bench PRIVATE ROCSPARSELT_BENCH ROCM_USE_FLOAT16 ROCSPARSELT_INTERNAL_API ${TENSILE_DEFINES} )
if ( NOT BUILD_FORTRAN_CLIENTS )
  target_compile_definitions( rocsparselt-bench PRIVATE CLIENTS_NO_FORTRAN )
endif()

target_compile_options(rocsparselt-bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)
# target_compile_options does not go to linker like CMAKE_CXX_FLAGS does, so manually add
if (NOT WIN32)
  target_link_libraries( rocsparselt-bench PRIVATE lapack cblas )
  list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs")
  if (NOT BUILD_FORTRAN_CLIENTS)
    list( APPEND COMMON_LINK_LIBS "-lgfortran") # for lapack
  endif()
else()
  list( APPEND COMMON_LINK_LIBS "libomp")
endif()
target_link_libraries( rocsparselt-bench PRIVATE ${COMMON_LINK_LIBS} )

set_target_properties( rocsparselt-bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

add_dependencies( rocsparselt-bench rocsparselt-common )
