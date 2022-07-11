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

#include "hipsparselt.h"
#include "hipsparselt_arguments.hpp"

template <typename T>
constexpr auto hipsparselt_type2datatype()
{
    if(std::is_same<T, hipsparseLtHalf>{})
        return HIPSPARSELT_R_16F;
    if(std::is_same<T, hipsparseLtBfloat16>{})
        return HIPSPARSELT_R_16BF;
    if(std::is_same<T, char>{})
        return HIPSPARSELT_R_8I;

    return HIPSPARSELT_R_16F; // testing purposes we default to f32 ex
}

// ----------------------------------------------------------------------------
// Calls TEST template based on the argument types. TEST<> is expected to
// return a functor which takes a const Arguments& argument. If the types do
// not match a recognized type combination, then TEST<void> is called.  This
// function returns the same type as TEST<...>{}(arg), usually bool or void.
// ----------------------------------------------------------------------------

// Simple functions which take only one datatype
//
// Even if the function can take mixed datatypes, this function can handle the
// cases where the types are uniform, in which case one template type argument
// is passed to TEST, and the rest are assumed to match the first.
template <template <typename...> class TEST>
auto hipsparselt_simple_dispatch(const Arguments& arg)
{
    switch(arg.a_type)
    {
    case HIPSPARSELT_R_16F:
        return TEST<hipsparseLtHalf>{}(arg);
    case HIPSPARSELT_R_16BF:
        return TEST<hipsparseLtBfloat16>{}(arg);
    default:
        return TEST<void>{}(arg);
    }
}

// gemm functions
template <template <typename...> class TEST>
auto hipsparselt_spmm_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, To = arg.c_type;
    auto       Tc = arg.compute_type;

    if(arg.b_type == Ti && arg.d_type == To)
    {
        if(Ti == To && To == HIPSPARSELT_R_16F && Tc == HIPSPARSE_COMPUTE_32F)
        {
            return TEST<hipsparseLtHalf, hipsparseLtHalf, float>{}(arg);
        }
        else if(Ti == To && To == HIPSPARSELT_R_16BF && Tc == HIPSPARSE_COMPUTE_32F)
        {
            return TEST<hipsparseLtBfloat16, hipsparseLtBfloat16, float>{}(arg);
        }
        else if(Ti == To && To == HIPSPARSELT_R_8I && Tc == HIPSPARSE_COMPUTE_32I)
        {
            return TEST<int8_t, int8_t, int32_t>{}(arg);
        }
    }
    return TEST<void>{}(arg);
}
