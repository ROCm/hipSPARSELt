/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "flops.hpp"
#include "rocsparselt.h"
#include "rocsparselt_datatype2string.hpp"
#include "rocsparselt_init.hpp"
#include "rocsparselt_math.hpp"
#include "rocsparselt_random.hpp"
#include "rocsparselt_test.hpp"
#include "rocsparselt_vector.hpp"
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
                    ASSERT_TRUE(a == b);
                }
            }
        }
    }
}

template <typename T>
void validate(T* A, T* B, int64_t n1, int64_t n2, int64_t n3, int64_t s1, int64_t s2, int64_t s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    for(int i3 = 0; i3 < n3; i3++)
    {
        for(int i1 = 0; i1 < n1; i1++)
        {
            for(int i2 = 0; i2 < n2; i2++)
            {
                auto a = A[(i1 * s1) + (i2 * s2) + (i3 * s3)];
                auto b = B[(i1 * s1) + (i2 * s2) + (i3 * s3)];
                ASSERT_TRUE(a == b);
            }
        }
    }
}

void validate_metadata(unsigned char* A,
                       unsigned char* B,
                       int64_t        n1,
                       int64_t        n2,
                       int64_t        n3,
                       int64_t        s1,
                       int64_t        s2,
                       int64_t        s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    for(int i3 = 0; i3 < n3; i3++)
    {
        for(int i1 = 0; i1 < n1; i1++)
        {
            for(int i2 = 0; i2 < n2; i2++)
            {
                auto a = A[(i1 * s1) + (i2 * s2) + (i3 * s3)];
                auto b = B[(i1 * s1) + (i2 * s2) + (i3 * s3)];
                ASSERT_TRUE(a == b);
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
                        rocsparselt_cerr << "Err - The given matrix is not a 2:4 sparse matrix"
                                         << std::endl;
                        ASSERT_TRUE(false);
                    }

                    if((k == 3 && m_idx == 0) || (k == 7 && m_idx == 2))
                    {
                        offset = b * stride_b + i * stride1 + (j + k - 1) * stride2;
                        value  = in[offset];
                        valid_data(m_idx++, k - 1, value);
                    }
                    if((k == 3 && m_idx == 1) || (k == 7 && m_idx == 3)
                       || static_cast<Tc>(value) != static_cast<Tc>(0))
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

    const rocsparse_operation transA = rocsparse_operation_none;
    const rocsparse_operation transB = rocsparse_operation_none;

    // allocate memory on device
    device_vector<Ti> dA(safe_size);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());

    rocsparselt_local_handle    handle{arg};
    rocsparselt_local_mat_descr matA(
        rocsparselt_matrix_type_structured, handle, M, K, lda, arg.a_type, rocsparse_order_column);
    rocsparselt_local_mat_descr matB(
        rocsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, rocsparse_order_column);
    rocsparselt_local_mat_descr matC(
        rocsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, rocsparse_order_column);
    rocsparselt_local_mat_descr matD(
        rocsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, rocsparse_order_column);
    rocsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);
    rocsparselt_local_matmul_alg_selection alg_sel(handle, matmul, rocsparselt_matmul_alg_default);

    size_t workspace_size, compressed_size;
    rocsparselt_matmul_get_workspace(handle, alg_sel, &workspace_size);

    rocsparselt_local_matmul_plan plan(handle, matmul, alg_sel, workspace_size);

    EXPECT_ROCSPARSELT_STATUS(rocsparselt_smfmac_compressed_size(nullptr, plan, &compressed_size),
                              rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSELT_STATUS(rocsparselt_smfmac_compressed_size(handle, nullptr, &compressed_size),
                              rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSELT_STATUS(rocsparselt_smfmac_compressed_size(handle, plan, nullptr),
                              rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSELT_STATUS(rocsparselt_smfmac_compressed_size(handle, plan, &compressed_size),
                              rocsparse_status_success);

    device_vector<Ti> dA_1(compressed_size);
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());

    hipStream_t stream = nullptr;

    EXPECT_ROCSPARSELT_STATUS(rocsparselt_smfmac_compress(nullptr, plan, dA, dA_1, stream),
                              rocsparse_status_invalid_handle);

    EXPECT_ROCSPARSELT_STATUS(rocsparselt_smfmac_compress(handle, nullptr, dA, dA_1, stream),
                              rocsparse_status_invalid_handle);

    EXPECT_ROCSPARSELT_STATUS(rocsparselt_smfmac_compress(handle, plan, nullptr, dA_1, stream),
                              rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSELT_STATUS(rocsparselt_smfmac_compress(handle, plan, dA_1, nullptr, stream),
                              rocsparse_status_invalid_pointer);
}

template <typename Ti, typename To, typename Tc>
void testing_compress(const Arguments& arg)
{
    rocsparse_operation transA = char2rocsparse_operation(arg.transA);
    rocsparse_operation transB = char2rocsparse_operation(arg.transB);

    int64_t M = arg.M;
    int64_t N = arg.N;
    int64_t K = arg.K;

    int64_t lda = arg.lda;
    int64_t ldb = arg.ldb;
    int64_t ldc = arg.ldc;

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used              = 0.0;
    double                   rocsparselt_error = 0.0;
    bool                     HMM               = arg.HMM;
    rocsparselt_local_handle handle{arg};
    hipStream_t              stream;
    hipStreamCreate(&stream);

    int64_t A_row = transA == rocsparse_operation_none ? M : K;
    int64_t A_col = transA == rocsparse_operation_none ? K : M;
    int64_t B_row = transB == rocsparse_operation_none ? K : N;
    int64_t B_col = transB == rocsparse_operation_none ? N : K;

    int64_t stride_1_a = transA == rocsparse_operation_none ? 1 : lda;
    int64_t stride_2_a = transA == rocsparse_operation_none ? lda : 1;

    int64_t c_stride_1_a = transA == rocsparse_operation_none ? 1 : K / 2;
    int64_t c_stride_2_a = transA == rocsparse_operation_none ? M : 1;
    int64_t c_stride_a
        = transA == rocsparse_operation_none ? K / 2 * c_stride_2_a : M * c_stride_1_a;

    int64_t m_stride_1_a = K / 8;
    int64_t m_stride_2_a = 1;
    int64_t m_stride_a   = M * m_stride_1_a;

    int     num_batches = arg.batch_count;
    int64_t stride_a    = arg.stride_a;
    int64_t stride_b    = arg.stride_b;
    int64_t stride_c    = arg.stride_c;
    int64_t stride_d    = stride_c;
    if(num_batches <= 0)
    {
        num_batches = 1;
        stride_a    = lda * A_col;
        stride_b    = ldb * B_col;
        stride_c    = ldc * N;
        stride_d    = stride_c;
    }

    rocsparselt_local_mat_descr matA(rocsparselt_matrix_type_structured,
                                     handle,
                                     A_row,
                                     A_col,
                                     lda,
                                     arg.a_type,
                                     rocsparse_order_column);
    rocsparselt_local_mat_descr matB(rocsparselt_matrix_type_dense,
                                     handle,
                                     B_row,
                                     B_col,
                                     ldb,
                                     arg.b_type,
                                     rocsparse_order_column);
    rocsparselt_local_mat_descr matC(
        rocsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, rocsparse_order_column);
    rocsparselt_local_mat_descr matD(
        rocsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, rocsparse_order_column);

    bool invalid_size_a = M < 8 || K % 8 != 0 || lda < A_row;
    bool invalid_size_b = N < 8 || ldb < B_row;
    bool invalid_size_c = ldc < M;
    if(invalid_size_a)
    {
        EXPECT_ROCSPARSELT_STATUS(matA.status(), rocsparse_status_invalid_size);

        return;
    }
    if(invalid_size_b)
    {
        EXPECT_ROCSPARSELT_STATUS(matB.status(), rocsparse_status_invalid_size);

        return;
    }
    if(invalid_size_c)
    {
        EXPECT_ROCSPARSELT_STATUS(matC.status(), rocsparse_status_invalid_size);

        return;
    }

    if(num_batches > 0)
    {
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matA, rocsparselt_mat_num_batches, &num_batches, sizeof(int)),
            rocsparse_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matB, rocsparselt_mat_num_batches, &num_batches, sizeof(int)),
            rocsparse_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matC, rocsparselt_mat_num_batches, &num_batches, sizeof(int)),
            rocsparse_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matD, rocsparselt_mat_num_batches, &num_batches, sizeof(int)),
            rocsparse_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matA, rocsparselt_mat_batch_stride, &stride_a, sizeof(int64_t)),
            rocsparse_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matB, rocsparselt_mat_batch_stride, &stride_b, sizeof(int64_t)),
            rocsparse_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matC, rocsparselt_mat_batch_stride, &stride_c, sizeof(int64_t)),
            rocsparse_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matD, rocsparselt_mat_batch_stride, &stride_d, sizeof(int64_t)),
            rocsparse_status_success);
    }

    rocsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);

    rocsparselt_local_matmul_alg_selection alg_sel(handle, matmul, rocsparselt_matmul_alg_default);

    size_t workspace_size, compressed_size;
    rocsparselt_matmul_get_workspace(handle, alg_sel, &workspace_size);

    rocsparselt_local_matmul_plan plan(handle, matmul, alg_sel, workspace_size);

    EXPECT_ROCSPARSELT_STATUS(rocsparselt_smfmac_compressed_size(handle, plan, &compressed_size),
                              rocsparse_status_success);

    const size_t size_A             = num_batches > 0 ? stride_a * num_batches : lda * A_col;
    const size_t size_A_pruned_copy = arg.unit_check || arg.norm_check || arg.timing ? size_A : 0;
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

    rocsparselt_seedrand();

    // Initial Data on CPU
    if(arg.initialization == rocsparselt_initialization::rand_int)
    {
        rocsparselt_init<Ti>(hA, A_row, A_col, lda);
    }
    else if(arg.initialization == rocsparselt_initialization::trig_float)
    {
        rocsparselt_init_sin<Ti>(hA, A_row, A_col, lda);
    }
    else if(arg.initialization == rocsparselt_initialization::hpl)
    {
        rocsparselt_init_hpl<Ti>(hA, A_row, A_col, lda);
    }
    else
    {
#ifdef GOOGLE_TEST
        FAIL() << "unknown initialization type";
        return;
#else
        rocsparselt_cerr << "unknown initialization type" << std::endl;
        rocsparselt_abort();
#endif
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    if(arg.unit_check || arg.norm_check || arg.timing)
    {
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_smfmac_prune(
                handle, matmul, dA, dA, rocsparselt_prune_smfmac_strip, stream),
            rocsparse_status_success);
        hipStreamSynchronize(stream);
        hA_pruned.transfer_from(dA);
    }

    if(arg.unit_check || arg.norm_check)
    {
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_smfmac_compress(handle, plan, dA, dA_compressd, stream),
            rocsparse_status_success);

        hipStreamSynchronize(stream);
        CHECK_HIP_ERROR(hA_1.transfer_from(dA_compressd));

        // now we can recycle gold matrix for reference purposes
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        compress<Ti, Tc>(hA_pruned,
                         reinterpret_cast<Ti*>(hA_gold.data()),
                         hA_gold.data() + c_stride_a * num_batches * sizeof(Ti),
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
            auto metadata_offset = c_stride_a * num_batches * sizeof(Ti);
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

            validate<Ti>(reinterpret_cast<Ti*>(hA_gold.data()),
                         reinterpret_cast<Ti*>(hA_1.data()),
                         M,
                         K / 2,
                         num_batches,
                         c_stride_1_a,
                         c_stride_2_a,
                         c_stride_a);
            validate_metadata(hA_gold + metadata_offset,
                              hA_1 + metadata_offset,
                              M,
                              K / 8,
                              num_batches,
                              m_stride_1_a,
                              m_stride_2_a,
                              m_stride_a);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
        {
            EXPECT_ROCSPARSELT_STATUS(
                rocsparselt_smfmac_compress(handle, plan, dA, dA_compressd, stream),
                rocsparse_status_success);
        }

        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            EXPECT_ROCSPARSELT_STATUS(
                rocsparselt_smfmac_compress(handle, plan, dA, dA_compressd, stream),
                rocsparse_status_success);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA, e_transB, e_M, e_N, e_K, e_alpha, e_lda, e_beta, e_ldb, e_ldc>{}
            .log_args<Ti>(rocsparselt_cout,
                          arg,
                          gpu_time_used,
                          ArgumentLogging::NA_value,
                          ArgumentLogging::NA_value,
                          cpu_time_used,
                          rocsparselt_error);
    }
}
