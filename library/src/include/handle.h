/*! \file */
/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef HANDLE_H
#define HANDLE_H

#include "rocsparselt.h"
//#include "rocsparselt_ostream.hpp"
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>

struct _rocsparselt_attribute
{
    _rocsparselt_attribute(){};

    ~_rocsparselt_attribute();

    void clear();

    const void* data();

    size_t length();

    size_t get(void* out, size_t size);

    template <typename T>
    size_t get(T* out)
    {
        return get(out, sizeof(T));
    }

    void set(const void* in, size_t size);

    template <typename T>
    void set(const T* in)
    {
        set(in, sizeof(T));
    }

private:
    void*  _data      = nullptr;
    size_t _data_size = 0;
};

/********************************************************************************
 * \brief rocsparse_handle is a structure holding the rocsparselt library context.
 * It must be initialized using rocsparse_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocsparse_destroy_handle().
 *******************************************************************************/
struct _rocsparselt_handle
{
    // constructor
    _rocsparselt_handle();
    // destructor
    ~_rocsparselt_handle();

    // device id
    int device;
    // device properties
    hipDeviceProp_t properties;
    // device wavefront size
    int wavefront_size;
    // asic revision
    int asic_rev;

    // pointer mode ; default mode is host
    rocsparse_pointer_mode pointer_mode = rocsparse_pointer_mode_host;
    // logging mode
    rocsparse_layer_mode layer_mode;
    // device buffer
    size_t buffer_size;
    void*  buffer;

    // logging streams
    std::ofstream log_trace_ofs;
    std::ofstream log_bench_ofs;
    std::ostream* log_trace_os = nullptr;
    std::ostream* log_bench_os = nullptr;
};

/********************************************************************************
 * \brief rocsparse_mat_descr is a structure holding the rocsparselt matrix
 * content. It must be initialized using rocsparselt_dense_descr_init() or
 * rocsparselt_structured_descr_init()  and the retured handle must be passed
 * to all subsequent library function calls that involve the matrix.
 * It should be destroyed at the end using rocsparselt_mat_descr_destroy().
 *******************************************************************************/
struct _rocsparselt_mat_descr
{
    // constructor
    _rocsparselt_mat_descr(){};
    // destructor
    ~_rocsparselt_mat_descr(){};

    // matrix type
    rocsparselt_matrix_type m_type = rocsparselt_matrix_type_unknown;
    // num rows
    int64_t m = 0;
    // num cols
    int64_t n = 0;
    // leading dimension
    int64_t ld = 0;
    // memory alignment in bytes
    uint32_t alignment;
    // data type of the matrix
    rocsparselt_datatype type;
    // memory layout
    rocsparse_order order;
    // matrix sparsity ratio
    rocsparselt_sparsity sparsity;
    // matrix attributes
    _rocsparselt_attribute attributes[2];

    // info of compressed matrix, will auto fill at rocsparselt_matmul_descr_init().
    // num k after compressed.
    int64_t c_k;
    // memory alignment in bytes after compressed.
    int64_t c_ld;
};

/********************************************************************************
 * \brief rocsparse_matmul_descr holds the description of the matrix multiplication operation.
 * It is initialized and destroyed with rocsparselt_matmul_descr_init()
 * and rocsparselt_matmul_descr_destroy() functions respectively.
 *******************************************************************************/
struct _rocsparselt_matmul_descr
{
    // constructor
    _rocsparselt_matmul_descr(){};
    // destructor
    ~_rocsparselt_matmul_descr(){};

    // operation applied to the matrix A
    rocsparse_operation op_A;
    // operation applied to the matrix B
    rocsparse_operation op_B;
    // matrix description of the matrix A
    rocsparselt_mat_descr matrix_A;
    // matrix description of the matrix B
    rocsparselt_mat_descr matrix_B;
    // matrix description of the matrix C
    rocsparselt_mat_descr matrix_C;
    // matrix description of the matrix D
    rocsparselt_mat_descr matrix_D;
    //
    rocsparselt_compute_type compute_type;
    //data of rocsparselt_matmul_descr_attribute
    int                    activation_relu            = 0;
    float                  activation_relu_upperbound = 0.0f;
    float                  activation_relu_threshold  = 0.0f;
    int                    activation_gelu            = 0;
    _rocsparselt_attribute bias_pointer;
    int64_t                bias_stride = (int64_t)0;
};

/********************************************************************************
 * \brief rocsparselt_matmul_alg_selection holds the description of the matrix
 * multiplication algorithm.
 * It is initialized and destroyed with rocsparselt_matmul_alg_selection_init()
 * and rocsparselt_matmul_alg_selection_destroy() functions respectively.
 *******************************************************************************/
struct _rocsparselt_matmul_alg_selection
{
    // constructor
    _rocsparselt_matmul_alg_selection(){};
    // destructor
    ~_rocsparselt_matmul_alg_selection(){};
    //
    rocsparselt_matmul_alg alg;
    //data of rocsparselt_matmul_alg_attribute
    _rocsparselt_attribute attributes[3];
};

/********************************************************************************
 * \brief rocsparselt_matmul_plan holds the matrix multiplication execution plan,
 * namely all the information necessary to execute the rocsparselt_matmul() operation.
 * It is initialized and destroyed with rocsparselt_matmul_plan_init() and
 * rocsparselt_matmul_plan_destroy() functions respectively.
 *******************************************************************************/
struct _rocsparselt_matmul_plan
{
    // constructor
    _rocsparselt_matmul_plan(){};
    // destructor
    ~_rocsparselt_matmul_plan(){};
    //
    rocsparselt_matmul_descr matmul_descr;
    //
    rocsparselt_matmul_alg_selection alg_selection;
    //
    size_t workspace_size;
};

#endif // HANDLE_H
