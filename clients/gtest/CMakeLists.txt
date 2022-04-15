# ########################################################################
# Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
# ########################################################################

#enable_testing()
find_package( GTest REQUIRED )


set(rocsparselt_test_source
    rocsparselt_gtest_main.cpp
    rocsparselt_test.cpp
    prune_gtest.cpp
    compress_gtest.cpp
    spmm_gtest.cpp
  )

add_executable( rocsparselt-test ${rocsparselt_test_source} ${rocsparselt_test_bench_common} )

target_compile_definitions( rocsparselt-test PRIVATE GOOGLE_TEST )

# Internal header includes
target_include_directories( rocsparselt-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
)

# External header includes included as system files
target_include_directories( rocsparselt-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
)
message("BLIS_INCLUDE_DIR=" ${BLIS_INCLUDE_DIR})
target_link_libraries( rocsparselt-test PRIVATE ${BLAS_LIBRARY} ${GTEST_BOTH_LIBRARIES} roc::rocsparselt )

target_link_libraries( rocsparselt-test PRIVATE hip::host hip::device )

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # GCC or hip-clang needs specific flag to turn on f16c intrinsics
  target_compile_options( rocsparselt-test PRIVATE -mf16c )
endif( )

target_compile_definitions( rocsparselt-test PRIVATE ROCM_USE_FLOAT16 ROCSPARSELT_INTERNAL_API )

target_compile_options(rocsparselt-test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)
# target_compile_options does not go to linker like CMAKE_CXX_FLAGS does, so manually add

target_link_libraries( rocsparselt-test PRIVATE lapack cblas )
list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs")

target_link_libraries( rocsparselt-test PRIVATE ${COMMON_LINK_LIBS} )

set_target_properties( rocsparselt-test PROPERTIES
  IMPORT_PREFIX ""
  IMPORT_SUFFIX ".lib"
  LINKER_LANGUAGE CXX
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

set( ROCSPARSELT_TEST_DATA "${PROJECT_BINARY_DIR}/staging/rocsparselt_gtest.data")
add_custom_command( OUTPUT "${ROCSPARSELT_TEST_DATA}"
                    COMMAND ${python} ../common/rocsparselt_gentest.py -I ../include rocsparselt_gtest.yaml -o "${ROCSPARSELT_TEST_DATA}"
                    DEPENDS ../common/rocsparselt_gentest.py ../include/rocsparselt_common.yaml prune_gtest.yaml compress_gtest.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
add_custom_target( rocsparselt-test-data
                   DEPENDS "${ROCSPARSELT_TEST_DATA}" )
add_dependencies( rocsparselt-test rocsparselt-test-data rocsparselt-common )

