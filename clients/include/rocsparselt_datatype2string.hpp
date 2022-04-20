/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../../library/src/include/activation.hpp"
#include "rocsparselt.h"
#include <string>

enum class rocsparselt_initialization
{
    rand_int   = 111,
    trig_float = 222,
    hpl        = 333,
    special    = 444,
};

/* ============================================================================================ */
/*  Convert rocsparselt constants to lapack char. */

constexpr auto rocsparselt2char_operation(rocsparse_operation value)
{
    switch(value)
    {
    case rocsparse_operation_none:
        return 'N';
    case rocsparse_operation_transpose:
        return 'T';
    case rocsparse_operation_conjugate_transpose:
        return 'C';
    }
    return '\0';
}

// return precision string for rocsparselt_datatype
constexpr auto rocsparselt_datatype2string(rocsparselt_datatype type)
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

// return precision string for rocsparselt_datatype
constexpr auto rocsparselt_computetype2string(rocsparselt_compute_type type)
{
    switch(type)
    {
    case rocsparselt_compute_f32:
        return "f32_r";
    case rocsparselt_compute_i32:
        return "i32_r";
    }
    return "invalid";
}

constexpr auto rocsparselt_initialization2string(rocsparselt_initialization init)
{
    switch(init)
    {
    case rocsparselt_initialization::rand_int:
        return "rand_int";
    case rocsparselt_initialization::trig_float:
        return "trig_float";
    case rocsparselt_initialization::hpl:
        return "hpl";
    case rocsparselt_initialization::special:
        return "special";
    }
    return "invalid";
}

inline rocsparselt_internal_ostream& operator<<(rocsparselt_internal_ostream& os,
                                                rocsparselt_initialization    init)
{
    return os << rocsparselt_initialization2string(init);
}

/* ============================================================================================ */
/*  Convert lapack char constants to rocsparselt type. */

constexpr rocsparse_operation char2rocsparse_operation(char value)
{
    switch(value)
    {
    case 'N':
    case 'n':
        return rocsparse_operation_none;
    case 'T':
    case 't':
        return rocsparse_operation_transpose;
    case 'C':
    case 'c':
        return rocsparse_operation_conjugate_transpose;
    default:
        return static_cast<rocsparse_operation>(-1);
    }
}

// clang-format off
inline rocsparselt_initialization string2rocsparselt_initialization(const std::string& value)
{
    return
        value == "rand_int"   ? rocsparselt_initialization::rand_int   :
        value == "trig_float" ? rocsparselt_initialization::trig_float :
        value == "hpl"        ? rocsparselt_initialization::hpl        :
        value == "special"    ? rocsparselt_initialization::special        :
        static_cast<rocsparselt_initialization>(-1);
}

inline rocsparselt_datatype string2rocsparselt_datatype(const std::string& value)
{
    return
        value == "f16_r" || value == "h" ? rocsparselt_datatype_f16_r  :
        value == "bf16_r"                ? rocsparselt_datatype_bf16_r :
        value == "i8_r"                  ? rocsparselt_datatype_i8_r   :
        value == "f8_r"                  ? rocsparselt_datatype_f8_r   :
        value == "bf8_r"                  ? rocsparselt_datatype_bf8_r   :
        static_cast<rocsparselt_datatype>(-1);
}

inline rocsparselt_compute_type string2rocsparselt_compute_type(const std::string& value)
{
    return
        value == "f32_r" || value == "s" ? rocsparselt_compute_f32  :
        value == "i32_r"                 ? rocsparselt_compute_i32  :
        static_cast<rocsparselt_compute_type>(-1);
}
// clang-format on
