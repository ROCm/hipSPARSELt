/*! \file */
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

#pragma once
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "auxiliary.hpp"
#include "handle.h"
#include "hipsparselt-types.h"
#include "logging.h"
#include <algorithm>
#include <exception>

#pragma STDC CX_LIMITED_RANGE ON

hipsparseLtDatatype_t    RocSparseLtDatatypeToHIPDatatype(rocsparselt_datatype_ type);
hipsparseLtComputetype_t RocSparseLtComputetypeToHIPComputetype(rocsparselt_compute_type_ type);
hipsparseLtOperation_t   HCCOperationToHIPOperation(rocsparselt_operation_ op);

inline bool isAligned(const void* pointer, size_t byte_count)
{
    return reinterpret_cast<uintptr_t>(pointer) % byte_count == 0;
}

// return precision string for rocsparselt_datatype
constexpr const char* rocsparselt_datatype_string(rocsparselt_datatype type)
{
    return hipsparselt_datatype_to_string(RocSparseLtDatatypeToHIPDatatype(type));
}

// return precision string for rocsparselt_compute_type
constexpr const char* rocsparselt_compute_type_string(rocsparselt_compute_type type)
{
    return hipsparselt_computetype_to_string(RocSparseLtComputetypeToHIPComputetype(type));
}

constexpr const char* rocsparselt_transpose_letter(rocsparselt_operation op)
{
    return hipsparselt_operation_to_string(HCCOperationToHIPOperation(op));
}

template <typename>
static constexpr char rocsparselt_precision_string[] = "invalid";
template <>
static constexpr char rocsparselt_precision_string<hipsparseLtBfloat16>[] = "bf16_r";
template <>
static constexpr char rocsparselt_precision_string<hipsparseLtHalf>[] = "f16_r";
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

// if trace logging is turned on with
// (handle->layer_mode & rocsparselt_layer_mode_log_trace) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_trace(rocsparselt_handle handle, H head, Ts&&... xs)
{
    if(nullptr != handle)
    {
        if(handle->layer_mode & rocsparselt_layer_mode_log_trace)
        {
            std::string comma_separator = ",";

            std::ostream* os = handle->log_trace_os;
            log_arguments(*os, comma_separator, head, std::forward<Ts>(xs)...);
        }
    }
}

// if bench logging is turned on with
// (handle->layer_mode & rocsparselt_layer_mode_log_bench) == true
// then
// log_bench will call log_arguments to log a string that
// can be input to the executable rocsparselt-bench.
template <typename H, typename... Ts>
void log_bench(rocsparselt_handle handle, H head, std::string precision, Ts&&... xs)
{
    if(nullptr != handle)
    {
        if(handle->layer_mode & rocsparselt_layer_mode_log_bench)
        {
            std::string space_separator = " ";

            std::ostream* os = handle->log_bench_os;
            log_arguments(*os, space_separator, head, precision, std::forward<Ts>(xs)...);
        }
    }
}

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
#endif // UTILITY_H
