/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#pragma once

/*!\file
 * \brief provides Floating point counts of Basic Linear Algebra Subprograms (BLAS) of Level 1, 2,
 * 3. Where possible we are using the values of NOP from the legacy BLAS files [sdcz]blas[23]time.f
 * for flop count.
 */

template <typename T>
constexpr double prune_strip_gflop_count(int64_t m, int64_t n)
{
    return (m * n) / 4.0 * 6 * 3 / 1e9;
}

/* \brief floating point counts of GEMM */
template <typename T>
constexpr double gemm_gflop_count(int64_t m, int64_t n, int64_t k)
{
    return (2.0 * m * n * k) / 1e9;
}

template <typename T>
constexpr double relu_gflop_count(int64_t m, int64_t n)
{
    return (m * n) / 1e9;
}

template <typename T>
constexpr double clippedrelu_gflop_count(int64_t m, int64_t n)
{
    return (m * n) / 1e9;
}

template <typename T>
constexpr double gelu_gflop_count(int64_t m, int64_t n)
{
    return (9.0 * m * n) / 1e9;
}
