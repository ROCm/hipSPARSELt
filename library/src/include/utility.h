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
#ifndef UTILITY_H
#define UTILITY_H

#include "handle.h"
#include "logging.h"
#include <algorithm>
#include <exception>

// Return the leftmost significant bit position
#if defined(rocsparse_ILP64)
static inline rocsparse_int rocsparse_clz(rocsparse_int n)
{
    return 64 - __builtin_clzll(n);
}
#else
static inline rocsparse_int rocsparse_clz(rocsparse_int n)
{
    return 32 - __builtin_clz(n);
}
#endif

// Return one on the device
static inline void rocsparselt_one(const rocsparselt_handle handle, float** one)
{
    *one = handle->sone;
}

static inline void rocsparselt_one(const rocsparselt_handle handle, double** one)
{
    *one = handle->done;
}

static inline void rocsparselt_one(const rocsparselt_handle handle, rocsparse_float_complex** one)
{
    *one = handle->cone;
}

static inline void rocsparselt_one(const rocsparselt_handle handle, rocsparse_double_complex** one)
{
    *one = handle->zone;
}

// if trace logging is turned on with
// (handle->layer_mode & rocsparse_layer_mode_log_trace) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_trace(rocsparselt_handle handle, H head, Ts&&... xs)
{
    if(nullptr != handle)
    {
        if(handle->layer_mode & rocsparse_layer_mode_log_trace)
        {
            std::string comma_separator = ",";

            std::ostream* os = handle->log_trace_os;
            log_arguments(*os, comma_separator, head, std::forward<Ts>(xs)...);
        }
    }
}

// if bench logging is turned on with
// (handle->layer_mode & rocsparse_layer_mode_log_bench) == true
// then
// log_bench will call log_arguments to log a string that
// can be input to the executable rocsparselt-bench.
template <typename H, typename... Ts>
void log_bench(rocsparselt_handle handle, H head, std::string precision, Ts&&... xs)
{
    if(nullptr != handle)
    {
        if(handle->layer_mode & rocsparse_layer_mode_log_bench)
        {
            std::string space_separator = " ";

            std::ostream* os = handle->log_bench_os;
            log_arguments(*os, space_separator, head, precision, std::forward<Ts>(xs)...);
        }
    }
}

// Trace log scalar values pointed to by pointer
template <typename T>
T log_trace_scalar_value(const T* value)
{
    return value ? *value : std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
T log_trace_scalar_value(rocsparselt_handle handle, const T* value)
{
    T host;
    if(value && handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipMemcpy(&host, value, sizeof(host), hipMemcpyDeviceToHost);
        value = &host;
    }
    return log_trace_scalar_value(value);
}

#define LOG_TRACE_SCALAR_VALUE(handle, value) log_trace_scalar_value(handle, value)

// Bench log scalar values pointed to by pointer
template <typename T>
T log_bench_scalar_value(const T* value)
{
    return (value ? *value : std::numeric_limits<T>::quiet_NaN());
}

template <typename T>
T log_bench_scalar_value(rocsparselt_handle handle, const T* value)
{
    T host;
    if(value && handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipMemcpy(&host, value, sizeof(host), hipMemcpyDeviceToHost);
        value = &host;
    }
    return log_bench_scalar_value(value);
}

#define LOG_BENCH_SCALAR_VALUE(handle, name) log_bench_scalar_value(handle, name)

// replaces X in string with s, d, c, z or h depending on typename T
template <typename T>
std::string replaceX(std::string input_string)
{
    if(std::is_same<T, float>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 's');
    }
    else if(std::is_same<T, double>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'd');
    }
    /*
    else if(std::is_same<T, rocsparse_float_complex>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'c');
    }
    else if(std::is_same<T, rocsparse_double_complex>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'z');
    }
    else if(std::is_same<T, rocsparse_half>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'h');
    }
    */
    return input_string;
}

//
// These macros can be redefined if the developer includes src/include/debug.h
//
#define ROCSPARSE_DEBUG_VERBOSE(msg__) (void)0
#define ROCSPARSE_RETURN_STATUS(token__) return rocsparse_status_##token__

// Convert the current C++ exception to rocsparse_status
// This allows extern "C" functions to return this function in a catch(...) block
// while converting all C++ exceptions to an equivalent rocsparse_status here
inline rocsparse_status exception_to_rocsparse_status(std::exception_ptr e
                                                      = std::current_exception())
try
{
    if(e)
        std::rethrow_exception(e);
    return rocsparse_status_success;
}
catch(const rocsparse_status& status)
{
    return status;
}
catch(const std::bad_alloc&)
{
    return rocsparse_status_memory_error;
}
catch(...)
{
    return rocsparse_status_internal_error;
}

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar_device_host(T x)
{
    return x;
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar_device_host(const T* xp)
{
    return *xp;
}

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T zero_scalar_device_host(T x)
{
    return static_cast<T>(0);
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T zero_scalar_device_host(const T* xp)
{
    return static_cast<T>(0);
}

//
// Provide some utility methods for enums.
//
struct rocsparselt_enum_utils
{
    template <typename U>
    static inline bool is_invalid(U value_);
};

template <>
inline bool rocsparselt_enum_utils::is_invalid(rocsparselt_sparsity value)
{
    switch(value)
    {
    case rocsparselt_sparsity_50_percent:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparselt_enum_utils::is_invalid(rocsparselt_matrix_type value_)
{
    switch(value_)
    {
    case rocsparselt_matrix_type_dense:
    case rocsparselt_matrix_type_structured:
    case rocsparselt_matrix_type_unknown:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparselt_enum_utils::is_invalid(rocsparselt_compute_type value_)
{
    switch(value_)
    {
    case rocsparselt_compute_f32:
    case rocsparselt_compute_i32:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparselt_enum_utils::is_invalid(rocsparselt_matmul_alg value_)
{
    switch(value_)
    {
    case rocsparselt_matmul_alg_default:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparselt_enum_utils::is_invalid(rocsparselt_matmul_alg_attribute value_)
{
    switch(value_)
    {
    case rocsparselt_matmul_alg_config_id:
    case rocsparselt_matmul_alg_config_max_id:
    case rocsparselt_matmul_search_iterations:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparselt_enum_utils::is_invalid(rocsparselt_prune_alg value_)
{
    switch(value_)
    {
    case rocsparselt_prune_smfmac_tile:
    case rocsparselt_prune_smfmac_strip:
    {
        return false;
    }
    }
    return true;
};

template <typename T>
struct floating_traits
{
    using data_t = T;
};

template <>
struct floating_traits<rocsparse_float_complex>
{
    using data_t = float;
};

template <>
struct floating_traits<rocsparse_double_complex>
{
    using data_t = double;
};

template <typename T>
using floating_data_t = typename floating_traits<T>::data_t;

#endif // UTILITY_H
