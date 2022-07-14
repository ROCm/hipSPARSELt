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

#include <hipsparselt.h>

HIPSPARSELT_EXPORT
constexpr const char* hipsparse_status_to_string(hipsparseStatus_t status)
{
#define CASE(x) \
    case x:     \
        return #x
    switch(status)
    {
        CASE(HIPSPARSE_STATUS_SUCCESS);
        CASE(HIPSPARSE_STATUS_NOT_INITIALIZED);
        CASE(HIPSPARSE_STATUS_ALLOC_FAILED);
        CASE(HIPSPARSE_STATUS_INVALID_VALUE);
        CASE(HIPSPARSE_STATUS_ARCH_MISMATCH);
        CASE(HIPSPARSE_STATUS_MAPPING_ERROR);
        CASE(HIPSPARSE_STATUS_EXECUTION_FAILED);
        CASE(HIPSPARSE_STATUS_INTERNAL_ERROR);
        CASE(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        CASE(HIPSPARSE_STATUS_ZERO_PIVOT);
        CASE(HIPSPARSE_STATUS_NOT_SUPPORTED);
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11003)
        CASE(HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES);
#endif
    }
#undef CASE
    // We don't use default: so that the compiler warns us if any valid enums are missing
    // from our switch. If the value is not a valid hipsparseStatus_t, we return this string.
    return "<undefined hipsparseStatus_t value>";
}

HIPSPARSELT_EXPORT
constexpr const char* hipsparselt_operation_to_string(hipsparseOperation_t value)
{
    switch(value)
    {
    case HIPSPARSE_OPERATION_NON_TRANSPOSE:
        return "N";
    case HIPSPARSE_OPERATION_TRANSPOSE:
        return "T";
    case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
        return "C";
    }
    return "invalid";
}

HIPSPARSELT_EXPORT
constexpr hipsparseOperation_t char_to_hipsparselt_operation(char value)
{
    switch(value)
    {
    case 'N':
    case 'n':
        return HIPSPARSE_OPERATION_NON_TRANSPOSE;
    case 'T':
    case 't':
        return HIPSPARSE_OPERATION_TRANSPOSE;
    case 'C':
    case 'c':
        return HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    default:
        return static_cast<hipsparseOperation_t>(-1);
    }
}

// return precision string for hipsparseLtDatatype_t
HIPSPARSELT_EXPORT
constexpr const char* hipsparselt_datatype_to_string(hipsparseLtDatatype_t type)
{
    switch(type)
    {
    case HIPSPARSELT_R_32F:
        return "f32_r";
    case HIPSPARSELT_R_16F:
        return "f16_r";
    case HIPSPARSELT_R_16BF:
        return "bf16_r";
    case HIPSPARSELT_R_8I:
        return "i8_r";
    case HIPSPARSELT_R_8F:
        return "f8_r";
    case HIPSPARSELT_R_8BF:
        return "bf8_r";
    }
    return "invalid";
}

// return precision string for hipsparseLtDatatype_t
HIPSPARSELT_EXPORT
constexpr const char* hipsparselt_computetype_to_string(hipsparseComputetype_t type)
{
    switch(type)
    {
    case HIPSPARSE_COMPUTE_16F:
        return "f16_r";
    case HIPSPARSE_COMPUTE_32I:
        return "i32_r";
    case HIPSPARSE_COMPUTE_32F:
        return "f32_r";
    case HIPSPARSE_COMPUTE_TF32:
        return "tf32_r";
    case HIPSPARSE_COMPUTE_TF32_FAST:
        return "tf32f_r";
    }
    return "invalid";
}

// clang-format off
HIPSPARSELT_EXPORT
constexpr hipsparseLtDatatype_t string_to_hipsparselt_datatype(const std::string& value)
{
    return
        value == "f32_r" || value == "s" ? HIPSPARSELT_R_32F  :
        value == "f16_r" || value == "h" ? HIPSPARSELT_R_16F  :
        value == "bf16_r"                ? HIPSPARSELT_R_16BF  :
        value == "i8_r"                  ? HIPSPARSELT_R_8I   :
        value == "f8_r"                  ? HIPSPARSELT_R_8F   :
        value == "bf8_r"                 ? HIPSPARSELT_R_8BF   :
        static_cast<hipsparseLtDatatype_t>(-1);
}

HIPSPARSELT_EXPORT
constexpr hipsparseComputetype_t string_to_hipsparselt_computetype(const std::string& value)
{
    return
        value == "f32_r" || value == "s" ? HIPSPARSE_COMPUTE_32F  :
        value == "i32_r"                 ? HIPSPARSE_COMPUTE_32I  :
        value == "f16_r" || value == "h" ? HIPSPARSE_COMPUTE_16F  :
        value == "tf32_r"                ? HIPSPARSE_COMPUTE_TF32  :
        value == "tf32f_r"               ? HIPSPARSE_COMPUTE_TF32_FAST  :
        static_cast<hipsparseComputetype_t>(-1);
}
// clang-format on

/*********************************************************************************************************
 * \brief The main structure for Numerical checking to detect numerical abnormalities such as NaN/zero/Inf
 *********************************************************************************************************/
typedef struct hipsparselt_check_numerics_s
{
    // Set to true if there is a NaN in the vector/matrix
    bool has_NaN = false;

    // Set to true if there is a zero in the vector/matrix
    bool has_zero = false;

    // Set to true if there is an Infinity in the vector/matrix
    bool has_Inf = false;
} hipsparselt_check_numerics_t;

/*******************************************************************************
* \brief  returns true if arg is NaN
********************************************************************************/
template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool hipsparselt_isnan(T)
{
    return false;
}

template <typename T, std::enable_if_t<!std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool hipsparselt_isnan(T arg)
{
    return std::isnan(arg);
}

__host__ __device__ inline bool hipsparselt_isnan(hipsparseLtHalf arg)
{
    union
    {
        hipsparseLtHalf fp;
        uint16_t        data;
    } x = {arg};
    return (~x.data & 0x7c00) == 0 && (x.data & 0x3ff) != 0;
}

/*******************************************************************************
* \brief  returns true if arg is Infinity
********************************************************************************/

template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool hipsparselt_isinf(T)
{
    return false;
}

template <typename T, std::enable_if_t<!std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool hipsparselt_isinf(T arg)
{
    return std::isinf(arg);
}

__host__ __device__ inline bool hipsparselt_isinf(hipsparseLtHalf arg)
{
    union
    {
        hipsparseLtHalf fp;
        uint16_t        data;
    } x = {arg};
    return (~x.data & 0x7c00) == 0 && (x.data & 0x3ff) == 0;
}

/*******************************************************************************
* \brief  returns true if arg is zero
********************************************************************************/

template <typename T>
__host__ __device__ inline bool hipsparselt_iszero(T arg)
{
    return arg == 0;
}
