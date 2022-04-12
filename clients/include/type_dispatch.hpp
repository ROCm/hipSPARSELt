/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocsparselt.h"
#include "rocsparselt_arguments.hpp"

template <typename T>
constexpr auto rocsparselt_type2datatype()
{
    if(std::is_same<T, rocsparselt_half>{})
        return rocsparselt_datatype_f16_r;
    if(std::is_same<T, rocsparselt_bfloat16>{})
        return rocsparselt_datatype_bf16_r;
    if(std::is_same<T, char>{})
        return rocsparselt_datatype_i8_r;

    return rocsparselt_datatype_f16_r; // testing purposes we default to f32 ex
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
auto rocsparselt_simple_dispatch(const Arguments& arg)
{
    switch(arg.a_type)
    {
    case rocsparselt_datatype_f16_r:
        return TEST<rocsparselt_half>{}(arg);
    case rocsparselt_datatype_bf16_r:
        return TEST<rocsparselt_bfloat16>{}(arg);
    default:
        return TEST<void>{}(arg);
    }
}

// gemm functions
template <template <typename...> class TEST>
auto rocsparselt_spmm_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, To = arg.c_type;
    auto       Tc = arg.compute_type;

    if(arg.b_type == Ti && arg.d_type == To)
    {
        if(Ti == To && To == rocsparselt_datatype_f16_r && Tc == rocsparselt_compute_f32)
        {
            return TEST<rocsparselt_half, rocsparselt_half, float>{}(arg);
        }
        else if(Ti == To && To == rocsparselt_datatype_bf16_r && Tc == rocsparselt_compute_f32)
        {
            return TEST<rocsparselt_bfloat16, rocsparselt_bfloat16, float>{}(arg);
        }
        else if(Ti == To && To == rocsparselt_datatype_i8_r && Tc == rocsparselt_compute_i32)
        {
            return TEST<int8_t, int8_t, int32_t>{}(arg);
        }
    }
    return TEST<void>{}(arg);
}
