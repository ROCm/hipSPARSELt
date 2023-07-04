/*******************************************************************************
 *
 * MIT License
 *
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "handle.h"
#include "definitions.h"
#include "logging.h"
#include "status.h"
#include "utility.hpp"

#include <hip/hip_runtime.h>

ROCSPARSELT_KERNEL void init_kernel(){};

void _rocsparselt_handle::init()
{
    // Layer mode
    log_bench = false;
    char* str_layer_mode;
    if((str_layer_mode = getenv("HIPSPARSELT_LOG_LEVEL")) == NULL)
    {
        layer_mode = rocsparselt_layer_mode_none;
        if((str_layer_mode = getenv("HIPSPARSELT_LOG_MASK")) != NULL)
        {
            layer_mode = strtol(str_layer_mode, nullptr, 0);
        }
    }
    else
    {
        layer_mode = rocsparselt_layer_mode_none;
        switch(atoi(str_layer_mode))
        {
        case rocsparselt_layer_level_log_api:
            layer_mode |= rocsparselt_layer_mode_log_api;
        case rocsparselt_layer_level_log_info:
            layer_mode |= rocsparselt_layer_mode_log_info;
        case rocsparselt_layer_level_log_hints:
            layer_mode |= rocsparselt_layer_mode_log_hints;
        case rocsparselt_layer_level_log_trace:
            layer_mode |= rocsparselt_layer_mode_log_trace;
        case rocsparselt_layer_level_log_error:
            layer_mode |= rocsparselt_layer_mode_log_error;
            break;
        default:
            layer_mode = rocsparselt_layer_mode_none;
            break;
        }
    }

    if((str_layer_mode = getenv("HIPSPARSELT_LOG_BENCH")) != NULL)
    {
        log_bench = (atoi(str_layer_mode) > 0);
    }

    // Open log file
    if(layer_mode & 0xff)
    {
        log_trace_ofs = new std::ofstream();
        open_log_stream(&log_trace_os, log_trace_ofs, "HIPSPARSELT_LOG_FILE");
    }

    // Open log_bench file
    if(log_bench)
    {
        log_bench_ofs = new std::ofstream();
        open_log_stream(&log_bench_os, log_bench_ofs, "HIPSPARSELT_LOG_BENCH_FILE");
    }

    // Default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    log_trace(this, "handle::init", "hipGetDevice");

    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&properties, device));
    log_trace(this, "handle::init", "hipGetDeviceProperties", device);

    // Device wavefront size
    wavefront_size = properties.warpSize;

#if HIP_VERSION >= 307
    // ASIC revision
    asic_rev = properties.asicRevision;
#else
    asic_rev = 0;
#endif

    is_init = true;
}

void _rocsparselt_handle::destroy()
{
    is_init = false;
    // Close log files
    if(log_trace_ofs)
    {
        if(log_trace_ofs->is_open())
            log_trace_ofs->close();
        delete log_trace_ofs;
        log_trace_ofs = nullptr;
    }
    if(log_bench_ofs)
    {
        if(log_bench_ofs->is_open())
            log_bench_ofs->close();
        delete log_bench_ofs;
        log_bench_ofs = nullptr;
    }
}

_rocsparselt_attribute::~_rocsparselt_attribute()
{
    clear();
}

_rocsparselt_attribute& _rocsparselt_attribute::operator=(const _rocsparselt_attribute& rhs)
{
    clear();
    set(rhs._data, rhs._data_size);
    return *this;
};

void _rocsparselt_attribute::clear()
{
    set(nullptr, 0);
}

const void* _rocsparselt_attribute::data()
{
    return _data;
}
size_t _rocsparselt_attribute::length()
{
    return _data_size;
}

size_t _rocsparselt_attribute::get(void* out, size_t size) const
{
    if(out != nullptr && _data != nullptr && _data_size >= size)
    {
        memcpy(out, _data, size);
        return size;
    }
    return 0;
}

void _rocsparselt_attribute::set(const void* in, size_t size)
{
    if(in == nullptr || (_data != nullptr && _data_size != size))
    {
        free(_data);
        _data      = nullptr;
        _data_size = 0;
    }
    if(in != nullptr)
    {
        if(_data == nullptr)
            _data = malloc(size);
        memcpy(_data, in, size);
        _data_size = size;
    }
}

std::ostream& operator<<(std::ostream& stream, const _rocsparselt_mat_descr& t)
{
    stream << "{"
           << "ptr=" << (&t) << ", format=" << rocsparselt_matrix_type_to_string(t.m_type)
           << ", row=" << t.m << ", col=" << t.n << ", ld=" << t.ld << ", alignment=" << t.alignment
           << ", datatype=" << rocsparselt_datatype_to_string(t.type)
           << ", order=" << rocsparselt_order_to_string(t.order);
    if(t.m_type == rocsparselt_matrix_type_structured)
        stream << ", sparsity=" << rocsparselt_sparsity_to_string(t.sparsity);

    int     num_batches  = 1;
    int64_t batch_stride = t.n * t.ld;
    t.attributes[rocsparselt_mat_batch_stride].get(&batch_stride);
    t.attributes[rocsparselt_mat_num_batches].get(&num_batches);
    stream << ", batchSize=" << num_batches << ", batchStride=" << batch_stride << "}";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const _rocsparselt_matmul_descr& t)
{
    stream << "{"
           << "ptr=" << (&t) << ", opA=" << rocsparselt_operation_to_string(t.op_A)
           << ", opB=" << rocsparselt_operation_to_string(t.op_B) << ", matA=" << *(t.matrix_A)
           << ", matB=" << *(t.matrix_B) << ", matC=" << *(t.matrix_C);
    if(t.matrix_C != t.matrix_D)
        stream << ", matD=" << *(t.matrix_D);
    stream << ", computeType=" << rocsparselt_compute_type_to_string(t.compute_type)
           << ", activation_relu=" << t.activation_relu
           << ", activation_relu_upperbound=" << t.activation_relu_upperbound
           << ", activation_relu_threshold=" << t.activation_relu_threshold
           << ", activation_gelu=" << t.activation_gelu << ", activation_abs=" << t.activation_abs
           << ", activation_leakyrelu=" << t.activation_leakyrelu
           << ", activation_leakyrelu_alpha=" << t.activation_leakyrelu_alpha
           << ", activation_sigmoid=" << t.activation_sigmoid
           << ", activation_tanh=" << t.activation_tanh
           << ", activation_tanh_alpha=" << t.activation_tanh_alpha
           << ", alignactivation_tanh_betament=" << t.activation_tanh_beta
           << ", bias_pointer=" << t.bias_pointer << ", bias_stride=" << t.bias_stride << "}";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const _rocsparselt_matmul_alg_selection& t)
{
    stream << "{"
           << "ptr=" << (&t) << ", alg=" << t.alg << ", config_id=" << t.config_id
           << ", config_max_id=" << t.config_max_id <<  ", search_iterations=" << t.search_iterations << "}";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const _rocsparselt_matmul_plan& t)
{
    stream << "{"
           << "ptr=" << (&t) << ", matmul=" << *(t.matmul_descr)
           << ", alg_selection=" << *(t.alg_selection) << "}";
    return stream;
}
