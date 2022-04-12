/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../../library/src/include/rocsparselt_ostream.hpp"
#include "rocsparselt.h"
#include "rocsparselt_math.hpp"
#include "rocsparselt_random.hpp"
#include <cinttypes>
#include <iostream>
#include <omp.h>
#include <vector>

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize matrices with random values
template <typename T>
void rocsparselt_init(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
        size_t b_idx = i_batch * stride;
        for(size_t j = 0; j < N; ++j)
        {
            size_t col_idx = b_idx + j * lda;
            if(M > 4)
                random_run_generator<T>(A + col_idx, M);
            else
            {
                for(size_t i = 0; i < M; ++i)
                    A[col_idx + i] = random_generator<T>();
            }
        }
    }
}

// Initialize matrices with random values
template <typename T>
inline void rocsparselt_init(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocsparselt_init(A.data(), M, N, lda, stride, batch_count);
}

template <typename T>
void rocsparselt_init_sin(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
                A[i + offset] = static_cast<T>(sin(double(i + offset))); //force cast to double
        }
}

template <typename T>
inline void rocsparselt_init_sin(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocsparselt_init_sin(A.data(), M, N, lda, stride, batch_count);
}

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alernating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.
template <typename T>
void rocsparselt_init_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
            {
                auto value    = random_generator<T>();
                A[i + offset] = (i ^ j) & 1 ? value : negate(value);
            }
        }
}

template <typename T>
void rocsparselt_init_alternating_sign(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocsparselt_init_alternating_sign(A.data(), M, N, lda, stride, batch_count);
}

// Initialize matrix so adjacent entries have alternating sign.
template <typename T>
void rocsparselt_init_hpl_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
            {
                auto value    = random_hpl_generator<T>();
                A[i + offset] = (i ^ j) & 1 ? value : negate(value);
            }
        }
}

template <typename T>
void rocsparselt_init_hpl_alternating_sign(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocsparselt_init_hpl_alternating_sign(A.data(), M, N, lda, stride, batch_count);
}

template <typename T>
void rocsparselt_init_cos(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
                A[i + offset] = T(cos(double(i + offset))); //force cast to double
        }
}

template <typename T>
inline void rocsparselt_init_cos(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocsparselt_init_cos(A.data(), M, N, lda, stride, batch_count);
}

// Initialize vector with HPL-like random values
template <typename T>
void rocsparselt_init_hpl(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_hpl_generator<T>();
}

template <typename T>
void rocsparselt_init_hpl(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_hpl_generator<T>();
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
void rocsparselt_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(rocsparselt_nan_rng());
}

template <typename T>
void rocsparselt_init_nan(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(rocsparselt_nan_rng());
}

template <typename T>
void rocsparselt_init_nan_tri(
    bool upper, T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                T val                             = upper ? (j >= i ? T(rocsparselt_nan_rng()) : 0)
                                                          : (j <= i ? T(rocsparselt_nan_rng()) : 0);
                A[i + j * lda + i_batch * stride] = val;
            }
}

template <typename T>
void rocsparselt_init_nan(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocsparselt_nan_rng());
}

template <typename T>
void rocsparselt_init_nan(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocsparselt_init_nan(A.data(), M, N, lda, stride, batch_count);
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with Inf where appropriate */

template <typename T>
void rocsparselt_init_inf(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(rocsparselt_inf_rng());
}

template <typename T>
void rocsparselt_init_inf(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(rocsparselt_inf_rng());
}

template <typename T>
void rocsparselt_init_inf(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocsparselt_inf_rng());
}

template <typename T>
void rocsparselt_init_inf(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocsparselt_init_inf(A.data(), M, N, lda, stride, batch_count);
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with zero */

template <typename T>
void rocsparselt_init_zero(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocsparselt_zero_rng());
}

template <typename T>
void rocsparselt_init_zero(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(rocsparselt_zero_rng());
}

template <typename T>
void rocsparselt_init_alt_impl_big(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const rocsparselt_half ieee_half_max(65280.0);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_max);
}

template <typename T>
inline void rocsparselt_init_alt_impl_big(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const rocsparselt_half ieee_half_max(65280.0);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_max);
}

template <typename T>
void rocsparselt_init_alt_impl_small(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const rocsparselt_half ieee_half_small(0.0000607967376708984375);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_small);
}

template <typename T>
void rocsparselt_init_alt_impl_small(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const rocsparselt_half ieee_half_small(0.0000607967376708984375);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_small);
}

/* ============================================================================================ */
/*! \brief  matrix matrix initialization: copies from A into same position in B */
template <typename T>
void rocsparselt_copy_matrix(const T* A,
                             T*       B,
                             size_t   M,
                             size_t   N,
                             size_t   lda,
                             size_t   ldb,
                             size_t   stridea     = 0,
                             size_t   strideb     = 0,
                             size_t   batch_count = 1)
{

    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
        size_t stride_offset_a = i_batch * stridea;
        size_t stride_offset_b = i_batch * strideb;
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset_a = stride_offset_a + j * lda;
            size_t offset_b = stride_offset_b + j * ldb;
            memcpy(B + offset_b, A + offset_a, M * sizeof(T));
        }
    }
}
