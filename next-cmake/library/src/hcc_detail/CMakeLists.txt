# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../library/src/hcc_detail")

target_sources(hipsparselt PRIVATE
    "${_CMAKE_CURRENT_SOURCE_DIR}/hipsparselt.cpp"
)

if(HIPSPARSELT_ENABLE_TENSILELITE)
    # TensileLite configuration
    set(TENSILE_LOGIC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../library/src/hcc_detail/rocsparselt/src/spmm/Tensile/Logic/asm_full")
    set(TENSILE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/Tensile")
    
    # Tensile build options
    set(TENSILE_OPTIONS)
    
    # Optional Tensile features
    if(HIPSPARSELT_ENABLE_MARKER)
        list(APPEND TENSILE_OPTIONS "ENABLE_MARKER")
    endif()
    
    # Verbose output
    if(HIPSPARSELT_ENABLE_VERBOSE)
        list(APPEND TENSILE_OPTIONS "PRINT_DEBUG")
    endif()
    
    # Use modern TensileCreateLibrary instead of legacy TensileCreateLibraryFiles
    TensileCreateLibrary(
        TENSILE_LOGIC_PATH    "${TENSILE_LOGIC_PATH}"
        TENSILE_OUTPUT_PATH   "${TENSILE_OUTPUT_PATH}"
        TENSILE_LIBRARY_NAME  "TensileLibrary"
        ARCHITECTURE          ${AMDGPU_TARGETS}
        CODE_OBJECT_VERSION   "5"
        COMPILER              "amdclang++"
        LIBRARY_FORMAT        "msgpack"
        PYTHON_EXECUTABLE     ${Python3_EXECUTABLE}
        ${TENSILE_OPTIONS}
    )
    
    # Add TensileLite source files
    set(tensile_source_files
        "${CMAKE_CURRENT_SOURCE_DIR}/../library/src/hcc_detail/rocsparselt/src/tensile_host.cpp"
    )
    
    list(APPEND hipsparselt_source_backend ${tensile_source_files})
    
    # TensileLite include directories
    target_include_directories(hipsparselt PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../library/src/hcc_detail/rocsparselt/src/Tensile"
    )
    
    # Link with TensileLite
    if(HIPSPARSELT_BUILD_SHARED_LIBS)
        target_link_libraries(hipsparselt PRIVATE TensileHost)
    else()
        # For static builds, get TensileHost properties and apply them
        get_target_property(TensileHost_INCLUDES TensileHost INCLUDE_DIRECTORIES)
        if(TensileHost_INCLUDES)
            target_include_directories(hipsparselt PRIVATE ${TensileHost_INCLUDES})
        endif()
        
        get_target_property(TensileHost_DEFINES TensileHost COMPILE_DEFINITIONS)
        if(TensileHost_DEFINES)
            target_compile_definitions(hipsparselt PRIVATE ${TensileHost_DEFINES})
        endif()
        
        # Add dependency to ensure TensileHost is built first
        add_dependencies(hipsparselt TensileHost)
    endif()
    
    # TensileLite compile definitions
    target_compile_definitions(hipsparselt PRIVATE 
        BUILD_WITH_TENSILE=1
        ROCSPARSELT_TENSILE_LAZY_LOAD=1
    )
    
    # Add dependency on TensileLibrary
    add_dependencies(hipsparselt TensileLibrary)
else()
    # No TensileLite support
    target_compile_definitions(hipsparselt PRIVATE BUILD_WITH_TENSILE=0)
endif()