/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
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

#include <iostream>
#include <string>

enum class rocsparselt_activation_type
{
    none = 0,
    abs,
    clippedrelu,
    gelu,
    leakyrelu,
    relu,
    sigmoid,
    tanh,
    all,
    exp,
    count
};

constexpr const rocsparselt_activation_type
    string2rocsparselt_activation_type(const std::string& value)
{
    return value == "none"          ? rocsparselt_activation_type::none
           : value == "abs"         ? rocsparselt_activation_type::abs
           : value == "clippedrelu" ? rocsparselt_activation_type::clippedrelu
           : value == "gelu"        ? rocsparselt_activation_type::gelu
           : value == "leakyrelu"   ? rocsparselt_activation_type::leakyrelu
           : value == "relu"        ? rocsparselt_activation_type::relu
           : value == "sigmoid"     ? rocsparselt_activation_type::sigmoid
           : value == "tanh"        ? rocsparselt_activation_type::tanh
           : value == "all"         ? rocsparselt_activation_type::all
           : value == "exp"         ? rocsparselt_activation_type::exp
                                    : static_cast<rocsparselt_activation_type>(-1);
}

constexpr const char* rocsparselt_activation_type_string(rocsparselt_activation_type type)
{
    switch(type)
    {
    case rocsparselt_activation_type::abs:
        return "abs";
    case rocsparselt_activation_type::clippedrelu:
        return "clippedrelu";
    case rocsparselt_activation_type::exp:
        return "exp";
    case rocsparselt_activation_type::gelu:
        return "gelu";
    case rocsparselt_activation_type::leakyrelu:
        return "leakyrelu";
    case rocsparselt_activation_type::relu:
        return "relu";
    case rocsparselt_activation_type::sigmoid:
        return "sigmoid";
    case rocsparselt_activation_type::tanh:
        return "tanh";
    case rocsparselt_activation_type::all:
        return "all";
    case rocsparselt_activation_type::none:
        return "none";
    default:
        return "invalid";
    }
}
