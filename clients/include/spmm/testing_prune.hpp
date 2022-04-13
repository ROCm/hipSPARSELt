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

template <typename Ti, typename Tc>
inline float norm2(Ti a, Ti b)
{
    auto ac = static_cast<Tc>(a);
    auto bc = static_cast<Tc>(b);

    return static_cast<Tc>(sqrt(ac * ac + bc * bc));
}

template <typename Ti, typename Tc>
void prune_strip(const Ti* in,
                 Ti*       out,
                 int64_t   m,
                 int64_t   n,
                 int64_t   stride1,
                 int64_t   stride2,
                 int       num_batches,
                 int64_t   stride_b)
{
    for(int b = 0; b < num_batches; b++)
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j += 4)
            {
                size_t pos[4];
                for(int k = 0; k < 4; k++)
                {
                    pos[k] = b * stride_b + i * stride1 + (j + k) * stride2;
                }

                auto max_norm2 = static_cast<Tc>(-1.0f);
                int  pos_a, pos_b;
                for(int a = 0; a < 4; a++)
                {
                    for(int b = a + 1; b < 4; b++)
                    {
                        auto norm2_v = norm2<Ti, double>(in[pos[a]], in[pos[b]]);
                        if(norm2_v > max_norm2)
                        {
                            pos_a     = a;
                            pos_b     = b;
                            max_norm2 = norm2_v;
                        }
                    }
                }

                for(int k = 0; k < 4; k++)
                {
                    if(k == pos_a || k == pos_b)
                    {
                        if(in != out)
                            out[pos[k]] = in[pos[k]];
                    }
                    else
                        out[pos[k]] = static_cast<Ti>(0.0f);
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
                auto pos     = (i1 * s1) + (i2 * s2) + (i3 * s3);
                auto value_a = A[pos];
                auto value_b = B[pos];
                ASSERT_TRUE(value_a == value_b);
            }
        }
    }
}

template <typename Ti, typename To, typename Tc>
void testing_prune_bad_arg(const Arguments& arg)
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

    hipStream_t stream = nullptr;

    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_smfmac_prune(nullptr, matmul, dA, dA, rocsparselt_prune_smfmac_strip, stream),
        rocsparse_status_invalid_handle);

    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_smfmac_prune(handle, nullptr, dA, dA, rocsparselt_prune_smfmac_strip, stream),
        rocsparse_status_invalid_handle);

    //TODO tile should be supported.
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_smfmac_prune(handle, matmul, dA, dA, rocsparselt_prune_smfmac_tile, stream),
        rocsparse_status_not_implemented);

    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_smfmac_prune(
            handle, matmul, dA, nullptr, rocsparselt_prune_smfmac_strip, stream),
        rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_smfmac_prune(
            handle, matmul, nullptr, dA, rocsparselt_prune_smfmac_strip, stream),
        rocsparse_status_invalid_pointer);
}

template <typename Ti, typename To, typename Tc>
void testing_prune(const Arguments& arg)
{
    rocsparselt_prune_alg prune_algo = arg.prune_algo;
    if(prune_algo == rocsparselt_prune_smfmac_tile)
        return;

    auto prune_cpu = prune_algo == rocsparselt_prune_smfmac_strip ? prune_strip<Ti, Tc> : nullptr;

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

    int     num_batches = arg.batch_count;
    int64_t stride_a    = arg.stride_a;
    int64_t stride_b    = arg.stride_b;
    int64_t stride_c    = arg.stride_c;
    int64_t stride_d    = stride_c;

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

    const size_t size_A      = stride_a == 0 ? A_col * lda : num_batches * stride_a;
    const size_t size_A_copy = arg.unit_check || arg.norm_check ? size_A : 0;

    // allocate memory on device
    device_vector<Ti> dA(size_A, 1, HMM);
    device_vector<Ti> dA_pruned(size_A, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_pruned.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ti> hA(size_A);
    host_vector<Ti> hA_gold(size_A_copy);
    host_vector<Ti> hA_1(size_A_copy);

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
    else if(arg.initialization == rocsparselt_initialization::special)
    {
        rocsparselt_init_alt_impl_big<Ti>(hA, A_row, A_col, lda);
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

    if(arg.unit_check || arg.norm_check)
    {
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_smfmac_prune(handle, matmul, dA, dA_pruned, prune_algo, stream),
            rocsparse_status_success);

        hipStreamSynchronize(stream);
        CHECK_HIP_ERROR(hA_1.transfer_from(dA_pruned));

        // now we can recycle gold matrix for reference purposes
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        prune_cpu(hA, hA_gold, M, K, stride_1_a, stride_2_a, num_batches, stride_a);

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        //releasing already used host memory
        hA = host_vector<Ti>();

        // check host error and norm
        if(arg.unit_check)
        {
            validate<Ti>(hA_gold, hA_1, M, K, num_batches, stride_1_a, stride_2_a, stride_a);
        }

        device_vector<int> d_valid(1, 1, HMM);
        int                h_valid = 0;
        //check the pruned matrix is sparisty 50 or not.
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_smfmac_prune_check(handle, matmul, dA_pruned, d_valid, stream),
            rocsparse_status_success);
        CHECK_HIP_ERROR(
            hipMemcpyAsync(&h_valid, d_valid, sizeof(int), hipMemcpyDeviceToHost, stream));
        hipStreamSynchronize(stream);
        ASSERT_TRUE(h_valid == 0);

        //check the original matrix is sparisty 50 or not.
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_smfmac_prune_check(handle, matmul, dA, d_valid, stream),
            rocsparse_status_success);
        CHECK_HIP_ERROR(
            hipMemcpyAsync(&h_valid, d_valid, sizeof(int), hipMemcpyDeviceToHost, stream));
        hipStreamSynchronize(stream);
        ASSERT_TRUE(h_valid != 0);
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
        {
            EXPECT_ROCSPARSELT_STATUS(
                rocsparselt_smfmac_prune(handle, matmul, dA, dA_pruned, prune_algo, stream),
                rocsparse_status_success);
        }

        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            EXPECT_ROCSPARSELT_STATUS(
                rocsparselt_smfmac_prune(handle, matmul, dA, dA_pruned, prune_algo, stream),
                rocsparse_status_success);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA, e_transB, e_M, e_N, e_K, e_alpha, e_lda, e_beta, e_ldb, e_ldc>{}
            .log_args<Ti>(rocsparselt_cout,
                          arg,
                          gpu_time_used,
                          prune_strip_gflop_count<Ti>(M, K),
                          ArgumentLogging::NA_value,
                          cpu_time_used,
                          rocsparselt_error);
    }
}
