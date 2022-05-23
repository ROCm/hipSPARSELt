/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief gtest unit compare two matrices float/double/complex */

#pragma once

#include "rocsparselt.h"
#include "rocsparselt_math.hpp"
#include "rocsparselt_test.hpp"
#include "rocsparselt_vector.hpp"

#ifndef GOOGLE_TEST
#define UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)
#define UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)
#else
#define UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)                  \
    do                                                                                           \
    {                                                                                            \
        for(size_t k = 0; k < batch_count; k++)                                                  \
            for(size_t j = 0; j < N; j++)                                                        \
                for(size_t i = 0; i < M; i++)                                                    \
                    if(rocsparselt_isnan(hCPU[i + j * size_t(lda) + k * strideA]))               \
                    {                                                                            \
                        ASSERT_TRUE(rocsparselt_isnan(hGPU[i + j * size_t(lda) + k * strideA])); \
                    }                                                                            \
                    else                                                                         \
                    {                                                                            \
                        UNIT_ASSERT_EQ(hCPU[i + j * size_t(lda) + k * strideA],                  \
                                       hGPU[i + j * size_t(lda) + k * strideA]);                 \
                    }                                                                            \
    } while(0)

#define UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)              \
    do                                                                                \
    {                                                                                 \
        for(size_t k = 0; k < batch_count; k++)                                       \
            for(size_t j = 0; j < N; j++)                                             \
                for(size_t i = 0; i < M; i++)                                         \
                    if(rocsparselt_isnan(hCPU[k][i + j * size_t(lda)]))               \
                    {                                                                 \
                        ASSERT_TRUE(rocsparselt_isnan(hGPU[k][i + j * size_t(lda)])); \
                    }                                                                 \
                    else                                                              \
                    {                                                                 \
                        UNIT_ASSERT_EQ(hCPU[k][i + j * size_t(lda)],                  \
                                       hGPU[k][i + j * size_t(lda)]);                 \
                    }                                                                 \
    } while(0)

//#define ASSERT_HALF_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))
//#define ASSERT_BF16_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))

#define ASSERT_HALF_EQ(a, b)                                    \
    do                                                          \
    {                                                           \
        rocsparselt_half absA    = (a > 0) ? a : -a;            \
        rocsparselt_half absB    = (b > 0) ? b : -b;            \
        rocsparselt_half absDiff = (a - b > 0) ? a - b : b - a; \
        ASSERT_TRUE(absDiff / (absA + absB + 1) < 0.01);        \
    } while(0)

#define ASSERT_BF16_EQ(a, b)                                                                       \
    do                                                                                             \
    {                                                                                              \
        const rocsparselt_bfloat16 bf16A    = static_cast<rocsparselt_bfloat16>(a);                \
        const rocsparselt_bfloat16 bf16B    = static_cast<rocsparselt_bfloat16>(b);                \
        const rocsparselt_bfloat16 bf16Zero = static_cast<rocsparselt_bfloat16>(0.0f);             \
        const rocsparselt_bfloat16 bf16One  = static_cast<rocsparselt_bfloat16>(1.0f);             \
        rocsparselt_bfloat16       absA     = (bf16A > bf16Zero) ? bf16A : -bf16A;                 \
        rocsparselt_bfloat16       absB     = (bf16B > bf16Zero) ? bf16B : -bf16B;                 \
        rocsparselt_bfloat16 absDiff = (bf16A - bf16B > bf16Zero) ? bf16A - bf16B : bf16B - bf16A; \
        ASSERT_TRUE(absDiff / (absA + absB + bf16One) < static_cast<rocsparselt_bfloat16>(0.1f));  \
    } while(0)

// Compare float to rocsparselt_bfloat16
// Allow the rocsparselt_bfloat16 to match the rounded or truncated value of float
// Only call ASSERT_FLOAT_EQ with the rounded value if the truncated value does not match
#include <gtest/internal/gtest-internal.h>
#define ASSERT_FLOAT_BF16_EQ(a, b)                                                             \
    do                                                                                         \
    {                                                                                          \
        using testing::internal::FloatingPoint;                                                \
        if(!FloatingPoint<float>(b).AlmostEquals(                                              \
               FloatingPoint<float>(rocsparselt_bfloat16(a, rocsparselt_bfloat16::truncate)))) \
            ASSERT_FLOAT_EQ(b, rocsparselt_bfloat16(a));                                       \
    } while(0)

#define ASSERT_FLOAT_COMPLEX_EQ(a, b)                  \
    do                                                 \
    {                                                  \
        auto ta = (a), tb = (b);                       \
        ASSERT_FLOAT_EQ(std::real(ta), std::real(tb)); \
        ASSERT_FLOAT_EQ(std::imag(ta), std::imag(tb)); \
    } while(0)

#define ASSERT_DOUBLE_COMPLEX_EQ(a, b)                  \
    do                                                  \
    {                                                   \
        auto ta = (a), tb = (b);                        \
        ASSERT_DOUBLE_EQ(std::real(ta), std::real(tb)); \
        ASSERT_DOUBLE_EQ(std::imag(ta), std::imag(tb)); \
    } while(0)

#endif // GOOGLE_TEST

// TODO: Replace std::remove_cv_t with std::type_identity_t in C++20
// It is only used to make T_hpa non-deduced
template <typename T, typename T_hpa = T>
void unit_check_general(
    int64_t M, int64_t N, int64_t lda, const std::remove_cv_t<T_hpa>* hCPU, const T* hGPU);

template <>
inline void unit_check_general(int64_t                     M,
                               int64_t                     N,
                               int64_t                     lda,
                               const rocsparselt_bfloat16* hCPU,
                               const rocsparselt_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<rocsparselt_bfloat16, float>(
    int64_t M, int64_t N, int64_t lda, const float* hCPU, const rocsparselt_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(
    int64_t M, int64_t N, int64_t lda, const rocsparselt_half* hCPU, const rocsparselt_half* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_HALF_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const float* hCPU, const float* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_FLOAT_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const double* hCPU, const double* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_DOUBLE_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const int64_t* hCPU, const int64_t* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const int8_t* hCPU, const int8_t* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_EQ);
}

template <typename T, typename T_hpa = T>
void unit_check_general(int64_t                        M,
                        int64_t                        N,
                        int64_t                        lda,
                        int64_t                        strideA,
                        const std::remove_cv_t<T_hpa>* hCPU,
                        const T*                       hGPU,
                        int64_t                        batch_count);

template <>
inline void unit_check_general(int64_t                     M,
                               int64_t                     N,
                               int64_t                     lda,
                               int64_t                     strideA,
                               const rocsparselt_bfloat16* hCPU,
                               const rocsparselt_bfloat16* hGPU,
                               int64_t                     batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<rocsparselt_bfloat16, float>(int64_t                     M,
                                                            int64_t                     N,
                                                            int64_t                     lda,
                                                            int64_t                     strideA,
                                                            const float*                hCPU,
                                                            const rocsparselt_bfloat16* hGPU,
                                                            int64_t                     batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(int64_t                 M,
                               int64_t                 N,
                               int64_t                 lda,
                               int64_t                 strideA,
                               const rocsparselt_half* hCPU,
                               const rocsparselt_half* hGPU,
                               int64_t                 batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(int64_t      M,
                               int64_t      N,
                               int64_t      lda,
                               int64_t      strideA,
                               const float* hCPU,
                               const float* hGPU,
                               int64_t      batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(int64_t       M,
                               int64_t       N,
                               int64_t       lda,
                               int64_t       strideA,
                               const double* hCPU,
                               const double* hGPU,
                               int64_t       batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(int64_t        M,
                               int64_t        N,
                               int64_t        lda,
                               int64_t        strideA,
                               const int64_t* hCPU,
                               const int64_t* hGPU,
                               int64_t        batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t       M,
                               int64_t       N,
                               int64_t       lda,
                               int64_t       strideA,
                               const int8_t* hCPU,
                               const int8_t* hGPU,
                               int64_t       batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <typename T, typename T_hpa = T>
void unit_check_general(int64_t                                    M,
                        int64_t                                    N,
                        int64_t                                    lda,
                        const host_vector<std::remove_cv_t<T_hpa>> hCPU[],
                        const host_vector<T>                       hGPU[],
                        int64_t                                    batch_count);

template <>
inline void unit_check_general(int64_t                                 M,
                               int64_t                                 N,
                               int64_t                                 lda,
                               const host_vector<rocsparselt_bfloat16> hCPU[],
                               const host_vector<rocsparselt_bfloat16> hGPU[],
                               int64_t                                 batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_BF16_EQ);
}

template <>
inline void
    unit_check_general<rocsparselt_bfloat16, float>(int64_t                                 M,
                                                    int64_t                                 N,
                                                    int64_t                                 lda,
                                                    const host_vector<float>                hCPU[],
                                                    const host_vector<rocsparselt_bfloat16> hGPU[],
                                                    int64_t batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(int64_t                             M,
                               int64_t                             N,
                               int64_t                             lda,
                               const host_vector<rocsparselt_half> hCPU[],
                               const host_vector<rocsparselt_half> hGPU[],
                               int64_t                             batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(int64_t                M,
                               int64_t                N,
                               int64_t                lda,
                               const host_vector<int> hCPU[],
                               const host_vector<int> hGPU[],
                               int64_t                batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const host_vector<int8_t> hCPU[],
                               const host_vector<int8_t> hGPU[],
                               int64_t                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t                  M,
                               int64_t                  N,
                               int64_t                  lda,
                               const host_vector<float> hCPU[],
                               const host_vector<float> hGPU[],
                               int64_t                  batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const host_vector<double> hCPU[],
                               const host_vector<double> hGPU[],
                               int64_t                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_DOUBLE_EQ);
}

template <typename T, typename T_hpa = T>
void unit_check_general(int64_t                              M,
                        int64_t                              N,
                        int64_t                              lda,
                        const std::remove_cv_t<T_hpa>* const hCPU[],
                        const T* const                       hGPU[],
                        int64_t                              batch_count);

template <>
inline void unit_check_general(int64_t                           M,
                               int64_t                           N,
                               int64_t                           lda,
                               const rocsparselt_bfloat16* const hCPU[],
                               const rocsparselt_bfloat16* const hGPU[],
                               int64_t                           batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_BF16_EQ);
}

template <>
inline void
    unit_check_general<rocsparselt_bfloat16, float>(int64_t                           M,
                                                    int64_t                           N,
                                                    int64_t                           lda,
                                                    const float* const                hCPU[],
                                                    const rocsparselt_bfloat16* const hGPU[],
                                                    int64_t                           batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(int64_t                       M,
                               int64_t                       N,
                               int64_t                       lda,
                               const rocsparselt_half* const hCPU[],
                               const rocsparselt_half* const hGPU[],
                               int64_t                       batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(int64_t          M,
                               int64_t          N,
                               int64_t          lda,
                               const int* const hCPU[],
                               const int* const hGPU[],
                               int64_t          batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t             M,
                               int64_t             N,
                               int64_t             lda,
                               const int8_t* const hCPU[],
                               const int8_t* const hGPU[],
                               int64_t             batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t            M,
                               int64_t            N,
                               int64_t            lda,
                               const float* const hCPU[],
                               const float* const hGPU[],
                               int64_t            batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(int64_t             M,
                               int64_t             N,
                               int64_t             lda,
                               const double* const hCPU[],
                               const double* const hGPU[],
                               int64_t             batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_DOUBLE_EQ);
}

template <typename T>
inline void trsm_err_res_check(T max_error, int64_t M, T forward_tolerance, T eps)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(max_error, forward_tolerance * eps * M);
#endif
}

template <typename T>
constexpr double get_epsilon()
{
    return std::numeric_limits<T>::epsilon();
}

template <typename T>
inline int64_t unit_check_diff(
    int64_t M, int64_t N, int64_t lda, int64_t stride, T* hCPU, T* hGPU, int64_t batch_count)
{
    int64_t error = 0;
    do
    {
        for(size_t k = 0; k < batch_count; k++)
            for(size_t j = 0; j < N; j++)
                for(size_t i = 0; i < M; i++)
                    if(rocsparselt_isnan(hCPU[i + j * size_t(lda) + k * stride]))
                    {
                        error += rocsparselt_isnan(hGPU[i + j * size_t(lda) + k * stride]) ? 0 : 1;
                    }
                    else
                    {
                        error += (hCPU[i + j * size_t(lda) + k * stride]
                                  == hGPU[i + j * size_t(lda) + k * stride])
                                     ? 0
                                     : 1;
                    }
    } while(0);
    return error;
}
