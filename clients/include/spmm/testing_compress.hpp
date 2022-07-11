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

#include "flops.hpp"
#include "hipsparselt.h"
#include "hipsparselt_datatype2string.hpp"
#include "hipsparselt_init.hpp"
#include "hipsparselt_math.hpp"
#include "hipsparselt_random.hpp"
#include "hipsparselt_test.hpp"
#include "hipsparselt_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

inline void extract_metadata(unsigned metadata, int& a, int& b, int& c, int& d)
{
    a = metadata & 0x03;
    b = (metadata >> 2) & 0x03;
    c = ((metadata >> 4) & 0x03);
    d = ((metadata >> 6) & 0x03);
}

template <typename T>
void self_validate(T*             A,
                   T*             C,
                   unsigned char* M,
                   int64_t        n1,
                   int64_t        n2,
                   int64_t        n3,
                   int64_t        s1,
                   int64_t        s2,
                   int64_t        s3,
                   int64_t        c_n1,
                   int64_t        c_n2,
                   int64_t        c_n3,
                   int64_t        c_s1,
                   int64_t        c_s2,
                   int64_t        c_s3,
                   int64_t        m_n1,
                   int64_t        m_n2,
                   int64_t        m_n3,
                   int64_t        m_s1,
                   int64_t        m_s2,
                   int64_t        m_s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    for(int i3 = 0; i3 < m_n3; i3++)
    {
        for(int i1 = 0; i1 < m_n1; i1++)
        {
            for(int i2 = 0; i2 < m_n2; i2++)
            {
                auto m_pos = (i1 * m_s1) + (i2 * m_s2) + (i3 * m_s3);
                int  idx[4];
                extract_metadata(M[m_pos], idx[0], idx[1], idx[2], idx[3]);
                idx[2] += 4;
                idx[3] += 4;
                int m_idx = 0;
                for(int i = 0; i < 8; i++)
                {
                    auto a_pos = (i1 * s1) + ((i2 * 8 + i) * s2) + (i3 * s3);
                    auto c_pos = (i1 * c_s1) + ((i2 * 4 + m_idx) * c_s2) + (i3 * c_s3);
                    T    a     = A[a_pos];
                    T    b     = static_cast<T>(0.0f);
                    if(i == idx[m_idx])
                    {

                        b = C[c_pos];
                        m_idx++;
                    }
                    CHECK_SUCCESS(a == b);
                }
            }
        }
    }
}

inline unsigned char generate_metadata(int a, int b, int c, int d)
{
    unsigned char metadata = (a)&0x03;
    metadata |= (b << 2) & 0x0C;
    metadata |= ((c - 4) << 4) & 0x30;
    metadata |= (((d - 4) << 6)) & 0xC0;
    return metadata;
}

template <typename Ti, typename Tc>
void compress(const Ti*      in,
              Ti*            out,
              unsigned char* metadata,
              int64_t        m,
              int64_t        n,
              int64_t        stride1,
              int64_t        stride2,
              int64_t        stride_b,
              int64_t        c_stride1,
              int64_t        c_stride2,
              int64_t        c_stride_b,
              int64_t        m_stride1,
              int64_t        m_stride2,
              int64_t        m_stride_b,
              int            num_batches)
{
    constexpr int tiles_y = 8;

    for(int b = 0; b < num_batches; b++)
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j += tiles_y)
            {
                Ti  values[4];
                int idx[4];
                int m_idx = 0;

                auto valid_data = [&](int midx, int index, Ti value) {
                    idx[midx]    = index;
                    values[midx] = value;
                };

                for(int k = 0; k < tiles_y; k++)
                {
                    auto offset = b * stride_b + i * stride1 + (j + k) * stride2;
                    Ti   value  = in[offset];

                    if(m_idx > 4)
                    {
                        hipsparselt_cerr << "Err - The given matrix is not a 2:4 sparse matrix"
                                         << std::endl;
                        CHECK_SUCCESS(false);
                    }

                    if((k == 3 && m_idx == 0) || (k == 7 && m_idx == 2))
                    {
                        offset = b * stride_b + i * stride1 + (j + k - 1) * stride2;
                        value  = in[offset];
                        valid_data(m_idx++, k - 1, value);
                    }
                    if((k == 3 && m_idx == 1) || (k == 7 && m_idx == 3)
                       || value != static_cast<Ti>(0))
                    {
                        offset = b * stride_b + i * stride1 + (j + k) * stride2;
                        value  = in[offset];
                        valid_data(m_idx++, k, value);
                    }
                }
                for(int k = 0; k < 4; k++)
                {
                    auto c_offset = b * c_stride_b + i * c_stride1 + (j / 2 + k) * c_stride2;
                    out[c_offset] = values[k];
                }

                unsigned char md = generate_metadata(idx[0], idx[1], idx[2], idx[3]);

                auto metadata_offset      = b * m_stride_b + i * m_stride1 + (j / 8) * m_stride2;
                metadata[metadata_offset] = md;
            }
        }
}

template <typename Ti, typename To, typename Tc>
void testing_compress_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const size_t safe_size = N * lda;

    const hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    const hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    // allocate memory on device
    device_vector<Ti> dA(safe_size);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());

    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, M, K, lda, arg.a_type, HIPSPARSE_ORDER_COLUMN);
    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COLUMN);
    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COLUMN);
    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COLUMN);
    hipsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);

    size_t workspace_size, compressed_size;
    hipsparseLtMatmulGetWorkspace(handle, alg_sel, &workspace_size);

    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel, workspace_size);

    // test version 1
    EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompressedSize(nullptr, plan, &compressed_size),
                            HIPSPARSE_STATUS_NOT_INITIALIZED);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompressedSize(handle, nullptr, &compressed_size),
                            HIPSPARSE_STATUS_NOT_INITIALIZED);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompressedSize(handle, plan, nullptr),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompressedSize(handle, plan, &compressed_size),
                            HIPSPARSE_STATUS_SUCCESS);

    device_vector<Ti> dA_1(compressed_size);
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());

    hipStream_t stream = nullptr;

    EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompress(nullptr, plan, dA, dA_1, stream),
                            HIPSPARSE_STATUS_NOT_INITIALIZED);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompress(handle, nullptr, dA, dA_1, stream),
                            HIPSPARSE_STATUS_NOT_INITIALIZED);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompress(handle, plan, nullptr, dA_1, stream),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompress(handle, plan, dA_1, nullptr, stream),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    // test version 2
    EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompressedSize2(nullptr, matA, &compressed_size),
                            HIPSPARSE_STATUS_NOT_INITIALIZED);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompressedSize2(handle, nullptr, &compressed_size),
                            HIPSPARSE_STATUS_NOT_INITIALIZED);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompressedSize2(handle, matA, nullptr),
                            HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompressedSize2(handle, matA, &compressed_size),
                            HIPSPARSE_STATUS_SUCCESS);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMACompress2(nullptr, matA, true, transA, dA, dA_1, stream),
        HIPSPARSE_STATUS_NOT_INITIALIZED);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMACompress2(handle, nullptr, true, transA, dA, dA_1, stream),
        HIPSPARSE_STATUS_NOT_INITIALIZED);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMACompress2(handle, matA, false, transA, dA, dA_1, stream),
        HIPSPARSE_STATUS_INTERNAL_ERROR);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMACompress2(handle, matA, true, transA, nullptr, dA_1, stream),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMACompress2(handle, matA, true, transA, dA_1, nullptr, stream),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

template <typename Ti,
          typename To,
          typename Tc,
          hipsparselt_batch_type btype = hipsparselt_batch_type::none>
void testing_compress(const Arguments& arg)
{
    int run_version = 1;
    if(strstr(arg.name, "compress2") != nullptr)
        run_version = 2;

    hipsparseOperation_t transA = char_to_hipsparselt_operation(arg.transA);
    hipsparseOperation_t transB = char_to_hipsparselt_operation(arg.transB);

    int64_t M = arg.M;
    int64_t N = arg.N;
    int64_t K = arg.K;

    int64_t lda = arg.lda;
    int64_t ldb = arg.ldb;
    int64_t ldc = arg.ldc;
    int64_t ldd = arg.ldd;

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used                = 0.0;
    double                   hipsparselt_error_c = 0.0;
    double                   hipsparselt_error_m = 0.0;
    bool                     HMM                 = arg.HMM;
    hipsparselt_local_handle handle{arg};
    hipStream_t              stream;
    hipStreamCreate(&stream);

    int64_t A_row = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? M : K;
    int64_t A_col = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K : M;
    int64_t B_row = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K : N;
    int64_t B_col = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? N : K;

    int64_t stride_1_a = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? 1 : lda;
    int64_t stride_2_a = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? lda : 1;

    constexpr bool do_batched         = (btype == hipsparselt_batch_type::batched);
    constexpr bool do_strided_batched = (btype == hipsparselt_batch_type::strided_batched);
    int            num_batches        = (do_batched || do_strided_batched ? arg.batch_count : 1);
    int64_t        stride_a           = do_strided_batched ? arg.stride_a : lda * A_col;
    int64_t        stride_b           = do_strided_batched ? arg.stride_b : ldb * B_col;
    int64_t        stride_c           = do_strided_batched ? arg.stride_c : ldc * M;
    int64_t        stride_d           = do_strided_batched ? arg.stride_c : ldd * M;

    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured,
                                     handle,
                                     A_row,
                                     A_col,
                                     lda,
                                     arg.a_type,
                                     HIPSPARSE_ORDER_COLUMN);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,
                                     handle,
                                     B_row,
                                     B_col,
                                     ldb,
                                     arg.b_type,
                                     HIPSPARSE_ORDER_COLUMN);
    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COLUMN);
    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COLUMN);

    bool invalid_size_a = M < 8 || K % 8 != 0 || lda < A_row;
    bool invalid_size_b = N < 8 || ldb < B_row;
    bool invalid_size_c = ldc < M;
    if(invalid_size_a)
    {
        EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_INVALID_VALUE);

        return;
    }
    if(invalid_size_b)
    {
        EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_INVALID_VALUE);

        return;
    }
    if(invalid_size_c)
    {
        EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_INVALID_VALUE);

        return;
    }

    if(do_batched || do_strided_batched)
    {
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matA, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matB, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matC, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matD, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
    }
    if(do_strided_batched)
    {
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matA, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_a, sizeof(int64_t)),
            HIPSPARSE_STATUS_SUCCESS);
    }

    hipsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);

    size_t workspace_size, compressed_size;
    hipsparseLtMatmulGetWorkspace(handle, alg_sel, &workspace_size);

    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel, workspace_size);

    if(run_version == 1)
    {
        EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompressedSize(handle, plan, &compressed_size),
                                HIPSPARSE_STATUS_SUCCESS);
    }
    else if(run_version == 2)
    {
        EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompressedSize2(handle, matA, &compressed_size),
                                HIPSPARSE_STATUS_SUCCESS);
    }
    const size_t size_A = stride_a == 0 ? lda * A_col * num_batches : stride_a * num_batches;
    const size_t size_A_pruned_copy     = arg.unit_check || arg.norm_check ? size_A : 0;
    const size_t size_A_compressed_copy = arg.unit_check || arg.norm_check ? compressed_size : 0;

    // allocate memory on device
    device_vector<Ti>            dA(size_A, 1, HMM);
    device_vector<unsigned char> dA_compressd(compressed_size, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_compressd.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ti>            hA(size_A);
    host_vector<Ti>            hA_pruned(size_A_pruned_copy);
    host_vector<unsigned char> hA_gold(size_A_compressed_copy);
    host_vector<unsigned char> hA_1(size_A_compressed_copy);

    hipsparselt_seedrand();

    // Initial Data on CPU
    if(arg.initialization == hipsparselt_initialization::rand_int)
    {
        hipsparselt_init<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
    }
    else if(arg.initialization == hipsparselt_initialization::trig_float)
    {
        hipsparselt_init_sin<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
    }
    else if(arg.initialization == hipsparselt_initialization::hpl)
    {
        hipsparselt_init_hpl<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
    }
    else if(arg.initialization == hipsparselt_initialization::special)
    {
        hipsparselt_init_alt_impl_big<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    if(run_version == 1)
    {
        EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompressedSize(handle, plan, &compressed_size),
                                HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune(
                handle, matmul, dA, dA, hipsparseLtPruneAlg_t(arg.prune_algo), stream),
            HIPSPARSE_STATUS_SUCCESS);
    }
    else if(run_version == 2)
    {
        EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompressedSize2(handle, matA, &compressed_size),
                                HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune2(
                handle, matA, true, transA, dA, dA, hipsparseLtPruneAlg_t(arg.prune_algo), stream),
            HIPSPARSE_STATUS_SUCCESS);
    }

    if(arg.unit_check || arg.norm_check)
    {
        //compressd matrix
        int64_t c_ld         = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? M : K / 2;
        int64_t c_stride_1_a = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? 1 : c_ld;
        int64_t c_stride_2_a = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? c_ld : 1;
        int64_t c_stride_a_r = K / 2 * M;
        int64_t c_stride_a   = stride_a == 0 ? 0 : c_stride_a_r;

        //metadata
        int64_t m_ld         = M;
        int64_t m_stride_1_a = K / 8;
        int64_t m_stride_2_a = 1;
        int64_t m_stride_a_r = M * m_stride_1_a;
        int64_t m_stride_a   = stride_a == 0 ? 0 : m_stride_a_r;

        auto metadata_offset = c_stride_a_r * sizeof(Ti) * (stride_a == 0 ? 1 : num_batches);

        hipStreamSynchronize(stream);
        hA_pruned.transfer_from(dA);

        if(run_version == 1)
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMACompress(handle, plan, dA, dA_compressd, stream),
                HIPSPARSE_STATUS_SUCCESS);
        else if(run_version == 2)
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMACompress2(handle, matA, true, transA, dA, dA_compressd, stream),
                HIPSPARSE_STATUS_SUCCESS);

        hipStreamSynchronize(stream);
        CHECK_HIP_ERROR(hA_1.transfer_from(dA_compressd));

        // now we can recycle gold matrix for reference purposes
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        compress<Ti, Tc>(hA_pruned,
                         reinterpret_cast<Ti*>(hA_gold.data()),
                         hA_gold.data() + metadata_offset,
                         M,
                         K,
                         stride_1_a,
                         stride_2_a,
                         stride_a,
                         c_stride_1_a,
                         c_stride_2_a,
                         c_stride_a,
                         m_stride_1_a,
                         m_stride_2_a,
                         m_stride_a,
                         num_batches);

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // check host error and norm
        if(arg.unit_check)
        {
            self_validate<Ti>(hA_pruned,
                              reinterpret_cast<Ti*>(hA_gold.data()),
                              hA_1 + metadata_offset,
                              M,
                              K,
                              num_batches,
                              stride_1_a,
                              stride_2_a,
                              stride_a,
                              M,
                              K / 2,
                              num_batches,
                              c_stride_1_a,
                              c_stride_2_a,
                              c_stride_a,
                              M,
                              K / 8,
                              num_batches,
                              m_stride_1_a,
                              m_stride_2_a,
                              m_stride_a);

            unit_check_general<Ti>(A_row,
                                   A_col / 2,
                                   c_ld,
                                   c_stride_a,
                                   reinterpret_cast<Ti*>(hA_gold.data()),
                                   reinterpret_cast<Ti*>(hA_1.data()),
                                   num_batches);
            unit_check_general<int8_t>(A_row,
                                       A_col / 8,
                                       A_row,
                                       m_stride_a,
                                       reinterpret_cast<int8_t*>(hA_gold + metadata_offset),
                                       reinterpret_cast<int8_t*>(hA_1 + metadata_offset),
                                       num_batches);
        }
        if(arg.norm_check)
        {
            hipsparselt_error_c = unit_check_diff<Ti>(A_row,
                                                      A_col / 2,
                                                      c_ld,
                                                      c_stride_a,
                                                      reinterpret_cast<Ti*>(hA_gold.data()),
                                                      reinterpret_cast<Ti*>(hA_1.data()),
                                                      num_batches);
            hipsparselt_error_m
                = unit_check_diff<int8_t>(A_row,
                                          A_col / 8,
                                          A_row,
                                          m_stride_a,
                                          reinterpret_cast<int8_t*>(hA_gold + metadata_offset),
                                          reinterpret_cast<int8_t*>(hA_1 + metadata_offset),
                                          num_batches);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
        {
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMACompress(handle, plan, dA, dA_compressd, stream),
                HIPSPARSE_STATUS_SUCCESS);
        }

        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMACompress(handle, plan, dA, dA_compressd, stream),
                HIPSPARSE_STATUS_SUCCESS);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA, e_transB, e_M, e_N, e_K, e_lda, e_stride_a, e_batch_count>{}
            .log_args<float>(hipsparselt_cout,
                             arg,
                             gpu_time_used,
                             ArgumentLogging::NA_value,
                             ArgumentLogging::NA_value,
                             cpu_time_used,
                             hipsparselt_error_c,
                             hipsparselt_error_m);
    }
}
