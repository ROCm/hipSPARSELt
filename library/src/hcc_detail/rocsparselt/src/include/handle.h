/*! \file */
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <fstream>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>
#include <memory>

struct _rocsparselt_attribute
{
    _rocsparselt_attribute(){};
    _rocsparselt_attribute& operator=(const _rocsparselt_attribute& rhs);

    ~_rocsparselt_attribute();

    void clear();

    const void* data();

    size_t length();

    size_t get(void* out, size_t size) const;

    template <typename T>
    size_t get(T* out) const
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
    _rocsparselt_handle(){};
    // destructor
    ~_rocsparselt_handle(){};

    void init();
    void destroy();

    bool isInit() const
    {
        return is_init;
    };

    // device id
    int device;
    // device properties
    hipDeviceProp_t properties;
    // device wavefront size
    int wavefront_size;
    // asic revision
    int asic_rev;

    // pointer mode ; default mode is host
    rocsparselt_pointer_mode pointer_mode = rocsparselt_pointer_mode_host;
    // logging mode
    int  layer_mode;
    bool log_bench = false;

    // device buffer
    size_t buffer_size;
    void*  buffer;
    bool   is_init = false;

    // logging streams
    std::ofstream* log_trace_ofs = nullptr;
    std::ofstream* log_bench_ofs = nullptr;
    std::ostream*  log_trace_os  = nullptr;
    std::ostream*  log_bench_os  = nullptr;
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
    _rocsparselt_mat_descr(const _rocsparselt_handle* handle)
        : handle(handle){};
    _rocsparselt_mat_descr(const _rocsparselt_mat_descr& rhs)
        : handle(rhs.handle)
        , m_type(rhs.m_type)
        , m(rhs.m)
        , n(rhs.n)
        , ld(rhs.ld)
        , alignment(rhs.alignment)
        , type(rhs.type)
        , order(rhs.order)
        , sparsity(rhs.sparsity)
        , c_k(rhs.c_k)
        , c_ld(rhs.c_ld)
    {
        for(int i = 0; i < 2; i++)
        {
            attributes[i] = rhs.attributes[i];
        }
    };

    // destructor
    ~_rocsparselt_mat_descr()
    {
        clear();
    };

    _rocsparselt_mat_descr* clone()
    {
        return new _rocsparselt_mat_descr(*this);
    };

    void clear()
    {
        m_type = rocsparselt_matrix_type_unknown;
        for(int i = 0; i < 2; i++)
        {
            attributes[i].clear();
        }
    }

    bool isInit() const
    {
        return m_type == rocsparselt_matrix_type_unknown ? false : true;
    };

    friend std::ostream& operator<<(std::ostream& stream, const _rocsparselt_mat_descr& t);

    const _rocsparselt_handle* handle = nullptr;
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
    rocsparselt_order order;
    // matrix sparsity ratio
    rocsparselt_sparsity sparsity;
    // matrix attributes
    _rocsparselt_attribute attributes[2];

    // info of compressed matrix, will be auto filled at rocsparselt_matmul_descr_init().
    // numbeer of k after compressed.
    int64_t c_k = -1;
    // leading dimension of compressed matrix.
    int64_t c_ld = -1;
};

/********************************************************************************
 * \brief rocsparse_matmul_descr holds the description of the matrix multiplication operation.
 * It is initialized and destroyed with rocsparselt_matmul_descr_init()
 * and rocsparselt_matmul_descr_destroy() functions respectively.
 *******************************************************************************/
struct _rocsparselt_matmul_descr
{
    // constructor
    _rocsparselt_matmul_descr(const _rocsparselt_handle* handle)
        : handle(handle)
    {
        is_init = true;
    };

    _rocsparselt_matmul_descr(const _rocsparselt_matmul_descr& rhs)
        : handle(rhs.handle)
        , op_A(rhs.op_A)
        , op_B(rhs.op_B)
        , compute_type(rhs.compute_type)
        , activation_relu(rhs.activation_relu)
        , activation_relu_upperbound(rhs.activation_relu_upperbound)
        , activation_relu_threshold(rhs.activation_relu_threshold)
        , activation_gelu(rhs.activation_gelu)
        , activation_abs(rhs.activation_abs)
        , activation_leakyrelu(rhs.activation_leakyrelu)
        , activation_leakyrelu_alpha(rhs.activation_leakyrelu_alpha)
        , activation_sigmoid(rhs.activation_sigmoid)
        , activation_tanh(rhs.activation_tanh)
        , activation_tanh_alpha(rhs.activation_tanh_alpha)
        , activation_tanh_beta(rhs.activation_tanh_beta)
        , bias_pointer(rhs.bias_pointer)
        , bias_stride(rhs.bias_stride)
    {
        matrix_A  = rhs.matrix_A->clone();
        matrix_B  = rhs.matrix_B->clone();
        matrix_C  = rhs.matrix_C->clone();
        matrix_D  = rhs.matrix_D->clone();
        own_by_us = true;
        is_init   = true;
    };

    // destructor
    ~_rocsparselt_matmul_descr()
    {
        if(own_by_us)
        {
            delete matrix_A;
            delete matrix_B;
            delete matrix_C;
            delete matrix_D;
        }
        is_init = false;
    };

    bool isInit() const
    {
        return is_init;
    }

    friend std::ostream& operator<<(std::ostream& stream, const _rocsparselt_matmul_descr& t);

    const _rocsparselt_handle* handle = nullptr;

    // operation applied to the matrix A
    rocsparselt_operation op_A;
    // operation applied to the matrix B
    rocsparselt_operation op_B;
    // matrix description of the matrix A
    _rocsparselt_mat_descr* matrix_A;
    // matrix description of the matrix B
    _rocsparselt_mat_descr* matrix_B;
    // matrix description of the matrix C
    _rocsparselt_mat_descr* matrix_C;
    // matrix description of the matrix D
    _rocsparselt_mat_descr* matrix_D;
    //
    rocsparselt_compute_type compute_type;
    //data of rocsparselt_matmul_descr_attribute
    int     activation_relu            = 0;
    float   activation_relu_upperbound = std::numeric_limits<float>::infinity();
    float   activation_relu_threshold  = 0.0f;
    int     activation_gelu            = 0;
    int     activation_abs             = 0;
    int     activation_leakyrelu       = 0;
    float   activation_leakyrelu_alpha = 1.0f;
    int     activation_sigmoid         = 0;
    int     activation_tanh            = 0;
    float   activation_tanh_alpha      = 1.0f;
    float   activation_tanh_beta       = 1.0f;
    float*  bias_pointer               = nullptr;
    int64_t bias_stride                = 0;

private:
    bool own_by_us = false;
    bool is_init   = false;
};

struct _rocsparselt_matmul_config
{
    _rocsparselt_matmul_config(){}
    ~_rocsparselt_matmul_config(){}

    _rocsparselt_matmul_config(const _rocsparselt_matmul_config& rhs)
    {
        this->data = rhs.data;
        this->max_workspace_bytes = rhs.max_workspace_bytes;
    }

    union u {

        u(): ptr(nullptr){}
        ~u(){}

        u& operator=(const u& rhs)
        {
            if(this != &rhs)
               ptr = std::static_pointer_cast<void>(rhs.ptr);
            return *this;
        }

        std::shared_ptr<void> ptr;
        uint8_t data[48];
    } data;

    size_t max_workspace_bytes = 0;
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
    _rocsparselt_matmul_alg_selection(const _rocsparselt_handle* handle)
        : handle(handle)
    {
        is_init = true;
        configs = std::make_shared<std::vector<_rocsparselt_matmul_config>>();
    };
    // destructor
    ~_rocsparselt_matmul_alg_selection()
    {
        is_init = false;
    };

    bool isInit() const
    {
        return is_init;
    }

    friend std::ostream& operator<<(std::ostream&                            stream,
                                    const _rocsparselt_matmul_alg_selection& t);

    const _rocsparselt_handle* handle = nullptr;
    //

    std::shared_ptr<std::vector<_rocsparselt_matmul_config>> configs;

    rocsparselt_matmul_alg alg;
    //data of rocsparselt_matmul_alg_attribute
    int  config_id         = 0;
    int  config_max_id     = 0;
    int  search_iterations = 10;
    bool is_init           = false;
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
    _rocsparselt_matmul_plan(const _rocsparselt_handle* handle)
        : handle(handle){};
    // destructor
    ~_rocsparselt_matmul_plan()
    {
        clear();
    };

    bool isInit() const
    {
        return (matmul_descr == nullptr || alg_selection == nullptr) ? false : true;
    }

    void clear()
    {
        delete matmul_descr;
        matmul_descr  = nullptr;
        alg_selection = nullptr;
    }

    friend std::ostream& operator<<(std::ostream& stream, const _rocsparselt_matmul_plan& t);

    const _rocsparselt_handle* handle = nullptr;
    //
    _rocsparselt_matmul_descr* matmul_descr = nullptr;
    //
    _rocsparselt_matmul_alg_selection* alg_selection = nullptr;

    //
    size_t workspace_size;
};

#endif // HANDLE_H
