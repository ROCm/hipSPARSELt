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
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "handle.h"
#include "logging.h"
#include <algorithm>
#include <exception>

#pragma STDC CX_LIMITED_RANGE ON

inline bool isAligned(const void* pointer, size_t byte_count)
{
    return reinterpret_cast<uintptr_t>(pointer) % byte_count == 0;
}

constexpr const size_t rocsparselt_datatype_bytes(rocsparselt_datatype type)
{
    switch(type)
    {
    case rocsparselt_datatype_f16_r:
    case rocsparselt_datatype_bf16_r:
        return 2;
    case rocsparselt_datatype_f8_r:
    case rocsparselt_datatype_bf8_r:
    case rocsparselt_datatype_i8_r:
        return 1;
    }
    return 0;
}

// return precision string for rocsparselt_datatype
constexpr const char* rocsparselt_datatype_string(rocsparselt_datatype type)
{
    switch(type)
    {
    case rocsparselt_datatype_f16_r:
        return "f16_r";
    case rocsparselt_datatype_i8_r:
        return "i8_r";
    case rocsparselt_datatype_bf16_r:
        return "bf16_r";
    case rocsparselt_datatype_f8_r:
        return "f8_r";
    case rocsparselt_datatype_bf8_r:
        return "bf8_r";
    }
    return "invalid";
}

// return precision string for rocsparselt_compute_type
constexpr const char* rocsparselt_compute_type_string(rocsparselt_compute_type type)
{
    switch(type)
    {
    case rocsparselt_compute_i32:
        return "i32";
    case rocsparselt_compute_f32:
        return "f32";
    }
    return "invalid";
}

constexpr const char* rocsparselt_transpose_letter(rocsparse_operation op)
{
    switch(op)
    {
    case rocsparse_operation_none:
        return "N";
    case rocsparse_operation_transpose:
        return "T";
    case rocsparse_operation_conjugate_transpose:
        return "C";
    }
    return "invalid";
}
// Convert rocsparse_status to string
static const char* rocsparse_status_to_string(rocsparse_status status)
{
#define CASE(x) \
    case x:     \
        return #x
    switch(status)
    {
        CASE(rocsparse_status_success);
        CASE(rocsparse_status_invalid_handle);
        CASE(rocsparse_status_not_implemented);
        CASE(rocsparse_status_invalid_pointer);
        CASE(rocsparse_status_invalid_size);
        CASE(rocsparse_status_memory_error);
        CASE(rocsparse_status_internal_error);
        CASE(rocsparse_status_invalid_value);
        CASE(rocsparse_status_arch_mismatch);
        CASE(rocsparse_status_zero_pivot);
        CASE(rocsparse_status_not_initialized);
        CASE(rocsparse_status_type_mismatch);
        CASE(rocsparse_status_requires_sorted_storage);
        CASE(rocsparse_status_continue);
    }
#undef CASE
    // We don't use default: so that the compiler warns us if any valid enums are missing
    // from our switch. If the value is not a valid rocsparse_status, we return this string.
    return "<undefined rocsparse_status value>";
}
template <typename>
static constexpr char rocsparselt_precision_string[] = "invalid";
//template <> constexpr char rocsparselt_precision_string<rocsparselt_bfloat16      >[] = "bf16_r";
template <>
static constexpr char rocsparselt_precision_string<rocsparselt_half>[] = "f16_r";
template <>
static constexpr char rocsparselt_precision_string<float>[] = "f32_r";
template <>
static constexpr char rocsparselt_precision_string<double>[] = "f64_r";
template <>
static constexpr char rocsparselt_precision_string<int8_t>[] = "i8_r";
template <>
static constexpr char rocsparselt_precision_string<uint8_t>[] = "u8_r";
template <>
static constexpr char rocsparselt_precision_string<int32_t>[] = "i32_r";
template <>
static constexpr char rocsparselt_precision_string<uint32_t>[] = "u32_r";

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

template <typename... Ts>
inline std::string concatenate(Ts&&... vals)
{
    std::ostringstream msg;
    std::string        none_separator = "";
    each_args(log_arg{msg, none_separator}, std::forward<Ts>(vals)...);
    return msg.str();
}

template <bool T_Enable, typename... Ts>
inline std::string concatenate_if(Ts&&... vals)
{
    if(!T_Enable)
        return "";
    return concatenate(std::forward<Ts>(vals)...);
}

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

// for internal use during testing, fetch arch name
std::string rocsparselt_internal_get_arch_name();

#endif // UTILITY_H
