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
#include "cblas_interface.hpp"
#include "hipsparselt_vector.hpp"
#include "utility.hpp"
#include <bitset>
#include <omp.h>

// gemm
template <>
void cblas_gemm<hipsparseLtBfloat16, hipsparseLtBfloat16, float>(hipsparseOperation_t     transA,
                                                                 hipsparseOperation_t     transB,
                                                                 int64_t                    m,
                                                                 int64_t                    n,
                                                                 int64_t                    k,
                                                                 float                      alpha,
                                                                 const hipsparseLtBfloat16* A,
                                                                 int64_t                    lda,
                                                                 const hipsparseLtBfloat16* B,
                                                                 int64_t                    ldb,
                                                                 float                      beta,
                                                                 hipsparseLtBfloat16*       C,
                                                                 int64_t                    ldc,
                                                                 bool                       alt)
{
    // cblas does not support hipsparseLtBfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = static_cast<float>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipsparselt =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<hipsparseLtBfloat16>(C_float[i]);
}

template <>
void cblas_gemm<hipsparseLtBfloat16, float, float>(hipsparseOperation_t     transA,
                                                   hipsparseOperation_t     transB,
                                                   int64_t                    m,
                                                   int64_t                    n,
                                                   int64_t                    k,
                                                   float                      alpha,
                                                   const hipsparseLtBfloat16* A,
                                                   int64_t                    lda,
                                                   const hipsparseLtBfloat16* B,
                                                   int64_t                    ldb,
                                                   float                      beta,
                                                   float*                     C,
                                                   int64_t                    ldc,
                                                   bool                       alt)
{
    // cblas does not support hipsparseLtBfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipsparselt =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm<hipsparseLtHalf, hipsparseLtHalf, float>(hipsparseOperation_t transA,
                                                         hipsparseOperation_t transB,
                                                         int64_t                m,
                                                         int64_t                n,
                                                         int64_t                k,
                                                         float                  alpha,
                                                         const hipsparseLtHalf* A,
                                                         int64_t                lda,
                                                         const hipsparseLtHalf* B,
                                                         int64_t                ldb,
                                                         float                  beta,
                                                         hipsparseLtHalf*       C,
                                                         int64_t                ldc,
                                                         bool                   alt)
{
    // cblas does not support hipsparseLtHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    if(alt)
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i]
                = hipsparseLtBfloat16(float(A[i]), hipsparseLtBfloat16::truncate_t::truncate);
        for(size_t i = 0; i < sizeB; i++)
            B_float[i]
                = hipsparseLtBfloat16(float(B[i]), hipsparseLtBfloat16::truncate_t::truncate);
        for(size_t i = 0; i < sizeC; i++)
            C_float[i]
                = hipsparseLtBfloat16(float(C[i]), hipsparseLtBfloat16::truncate_t::truncate);
    }
    else
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = A[i];
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = B[i];
        for(size_t i = 0; i < sizeC; i++)
            C_float[i] = C[i];
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipsparselt =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = hipsparseLtHalf(C_float[i]);
}

template <>
void cblas_gemm<hipsparseLtHalf, float, float>(hipsparseOperation_t transA,
                                               hipsparseOperation_t transB,
                                               int64_t                m,
                                               int64_t                n,
                                               int64_t                k,
                                               float                  alpha,
                                               const hipsparseLtHalf* A,
                                               int64_t                lda,
                                               const hipsparseLtHalf* B,
                                               int64_t                ldb,
                                               float                  beta,
                                               float*                 C,
                                               int64_t                ldc,
                                               bool                   alt)
{
    // cblas does not support hipsparseLtHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? n : k) * size_t(ldb);

    host_vector<float> A_float(sizeA), B_float(sizeB);

    if(alt)
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i]
                = hipsparseLtBfloat16(float(A[i]), hipsparseLtBfloat16::truncate_t::truncate);
        for(size_t i = 0; i < sizeB; i++)
            B_float[i]
                = hipsparseLtBfloat16(float(B[i]), hipsparseLtBfloat16::truncate_t::truncate);
    }
    else
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = A[i];
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = B[i];
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipsparselt =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm<int8_t, int8_t, float>(hipsparseOperation_t transA,
                                       hipsparseOperation_t transB,
                                       int64_t              m,
                                       int64_t              n,
                                       int64_t              k,
                                       float                alpha,
                                       const int8_t*        A,
                                       int64_t              lda,
                                       const int8_t*        B,
                                       int64_t              ldb,
                                       float                beta,
                                       int8_t*              C,
                                       int64_t              ldc,
                                       bool                 alt)
{
    // cblas does not support int8_t input / int8_t output, however non-overflowing
    // 32-bit integer operations can be represented accurately with double-precision
    // floats, so convert to doubles and downcast result down to int32_t.
    // NOTE: This will not properly account for 32-bit integer overflow, however
    //       the result should be acceptable for testing.

    size_t const sizeA = ((transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : m) * size_t(lda);
    size_t const sizeB = ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : k) * size_t(ldb);
    size_t const sizeC = n * size_t(ldc);

    host_vector<double> A_double(sizeA);
    host_vector<double> B_double(sizeB);
    host_vector<double> C_double(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_double[i] = static_cast<double>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_double[i] = static_cast<double>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    cblas_dgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_double,
                lda,
                B_double,
                ldb,
                beta,
                C_double,
                ldc);

    auto saturate = [](double val) {
        val = std::nearbyint(val);
        val = val > 127.f ? 127.f : val < -128.f ? -128.f : val;
        return val;
    };

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<int8_t>(saturate(C_double[i]));
}

template <>
void cblas_gemm<int8_t, float, float>(hipsparseOperation_t transA,
                                      hipsparseOperation_t transB,
                                      int64_t              m,
                                      int64_t              n,
                                      int64_t              k,
                                      float                alpha,
                                      const int8_t*        A,
                                      int64_t              lda,
                                      const int8_t*        B,
                                      int64_t              ldb,
                                      float                beta,
                                      float*               C,
                                      int64_t              ldc,
                                      bool                 alt)
{
    // cblas does not support int8_t input / int8_t output, however non-overflowing
    // 32-bit integer operations can be represented accurately with double-precision
    // floats, so convert to doubles and downcast result down to int32_t.
    // NOTE: This will not properly account for 32-bit integer overflow, however
    //       the result should be acceptable for testing.

    size_t const sizeA = ((transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : m) * size_t(lda);
    size_t const sizeB = ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : k) * size_t(ldb);
    size_t const sizeC = n * size_t(ldc);

    host_vector<double> A_double(sizeA);
    host_vector<double> B_double(sizeB);
    host_vector<double> C_double(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_double[i] = static_cast<double>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_double[i] = static_cast<double>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    cblas_dgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_double,
                lda,
                B_double,
                ldb,
                beta,
                C_double,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<float>(C_double[i]);
}
