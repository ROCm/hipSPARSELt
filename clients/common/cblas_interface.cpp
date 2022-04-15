/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************/
#include "cblas_interface.hpp"
#include "rocsparselt_vector.hpp"
#include "utility.hpp"
#include <bitset>
#include <omp.h>

// gemm
template <>
void cblas_gemm<rocsparselt_bfloat16, rocsparselt_bfloat16, float>(rocsparse_operation transA,
                                                                   rocsparse_operation transB,
                                                                   int64_t             m,
                                                                   int64_t             n,
                                                                   int64_t             k,
                                                                   float               alpha,
                                                                   const rocsparselt_bfloat16* A,
                                                                   int64_t                     lda,
                                                                   const rocsparselt_bfloat16* B,
                                                                   int64_t                     ldb,
                                                                   float                       beta,
                                                                   rocsparselt_bfloat16*       C,
                                                                   int64_t                     ldc,
                                                                   bool                        alt)
{
    // cblas does not support rocsparselt_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocsparse_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocsparse_operation_none ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = static_cast<float>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocsparselt =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
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
        C[i] = static_cast<rocsparselt_bfloat16>(C_float[i]);
}

template <>
void cblas_gemm<rocsparselt_half, rocsparselt_half, float>(rocsparse_operation     transA,
                                                           rocsparse_operation     transB,
                                                           int64_t                 m,
                                                           int64_t                 n,
                                                           int64_t                 k,
                                                           float                   alpha,
                                                           const rocsparselt_half* A,
                                                           int64_t                 lda,
                                                           const rocsparselt_half* B,
                                                           int64_t                 ldb,
                                                           float                   beta,
                                                           rocsparselt_half*       C,
                                                           int64_t                 ldc,
                                                           bool                    alt)
{
    // cblas does not support rocsparselt_half, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocsparse_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocsparse_operation_none ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    if(alt)
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i]
                = rocsparselt_bfloat16(float(A[i]), rocsparselt_bfloat16::truncate_t::truncate);
        for(size_t i = 0; i < sizeB; i++)
            B_float[i]
                = rocsparselt_bfloat16(float(B[i]), rocsparselt_bfloat16::truncate_t::truncate);
        for(size_t i = 0; i < sizeC; i++)
            C_float[i]
                = rocsparselt_bfloat16(float(C[i]), rocsparselt_bfloat16::truncate_t::truncate);
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
    // printf("transA: rocsparselt =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
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
        C[i] = rocsparselt_half(C_float[i]);
}

template <>
void cblas_gemm<int8_t, int8_t, int32_t>(rocsparse_operation transA,
                                         rocsparse_operation transB,
                                         int64_t             m,
                                         int64_t             n,
                                         int64_t             k,
                                         float               alpha,
                                         const int8_t*       A,
                                         int64_t             lda,
                                         const int8_t*       B,
                                         int64_t             ldb,
                                         float               beta,
                                         int8_t*             C,
                                         int64_t             ldc,
                                         bool                alt)
{
    // cblas does not support int8_t input / int8_t output, however non-overflowing
    // 32-bit integer operations can be represented accurately with double-precision
    // floats, so convert to doubles and downcast result down to int32_t.
    // NOTE: This will not properly account for 32-bit integer overflow, however
    //       the result should be acceptable for testing.

    size_t const sizeA = ((transA == rocsparse_operation_none) ? k : m) * size_t(lda);
    size_t const sizeB = ((transB == rocsparse_operation_none) ? n : k) * size_t(ldb);
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
        C[i] = static_cast<int8_t>(C_double[i]);
}
