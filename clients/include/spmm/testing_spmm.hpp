/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "near.hpp"
#include "norm.hpp"
#include "rocsparselt.h"
#include "rocsparselt_datatype2string.hpp"
#include "rocsparselt_init.hpp"
#include "rocsparselt_math.hpp"
#include "rocsparselt_random.hpp"
#include "rocsparselt_test.hpp"
#include "rocsparselt_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename Ti, typename To, typename Tc>
void testing_spmm_bad_arg(const Arguments& arg)
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
    device_vector<Ti> dA(safe_size / 2);
    device_vector<Ti> dB(safe_size);
    device_vector<Ti> dC(safe_size);
    device_vector<Ti> dD(safe_size);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());

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

    size_t                        workspace_size = 0;
    rocsparselt_local_matmul_plan plan(handle, matmul, alg_sel, workspace_size);

    void* workspace = nullptr;
    float alpha = 1.0, beta = 0.0;

    hipStream_t stream = nullptr;
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(nullptr, plan, &alpha, dA, dB, &beta, dC, dD, workspace, &stream, 1),
        rocsparse_status_invalid_handle);

    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, nullptr, &alpha, dA, dB, &beta, dC, dD, workspace, &stream, 1),
        rocsparse_status_invalid_handle);

    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, nullptr, dA, dB, &beta, dC, dD, workspace, &stream, 1),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, nullptr, dB, &beta, dC, dD, workspace, &stream, 1),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, dA, nullptr, &beta, dC, dD, workspace, &stream, 1),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, dA, dB, nullptr, dC, dD, workspace, &stream, 1),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, dA, dB, &beta, nullptr, dD, workspace, &stream, 1),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, dA, dB, &beta, dC, nullptr, workspace, &stream, 1),
        rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, dA, dB, &beta, dC, dD, workspace, &stream, -1),
        rocsparse_status_invalid_value);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, dA, dB, &beta, dC, dD, workspace, nullptr, 1),
        rocsparse_status_invalid_value);

    workspace_size = 1;
    rocsparselt_local_matmul_plan plan2(handle, matmul, alg_sel, workspace_size);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan2, &alpha, dA, dB, &beta, dC, dD, workspace, &stream, 0),
        rocsparse_status_invalid_value);
}

template <typename Ti, typename To, typename Tc>
void testing_spmm(const Arguments& arg)
{
    rocsparse_operation transA = char2rocsparse_operation(arg.transA);
    rocsparse_operation transB = char2rocsparse_operation(arg.transB);

    int64_t M          = arg.M;
    int64_t N          = arg.N;
    int64_t K          = arg.K;
    Tc      h_alpha_Tc = arg.get_alpha<Tc>();
    Tc      h_beta_Tc  = arg.get_beta<Tc>();
    int64_t lda        = arg.lda;
    int64_t ldb        = arg.ldb;
    int64_t ldc        = arg.ldc;
    int64_t ldd        = arg.ldd;

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used              = 0.0;
    double                   rocsparselt_error = 0.0;
    bool                     HMM               = arg.HMM;
    rocsparselt_local_handle handle{arg};
    hipStream_t              stream;
    hipStreamCreate(&stream);

    int     num_batches = arg.batch_count;
    int64_t stride_a    = arg.stride_a;
    int64_t stride_b    = arg.stride_b;
    int64_t stride_c    = arg.stride_c;
    int64_t stride_d    = stride_c;

    int64_t A_row = transA == rocsparse_operation_none ? M : K;
    int64_t A_col = transA == rocsparse_operation_none ? K : M;
    int64_t B_row = transB == rocsparse_operation_none ? K : N;
    int64_t B_col = transB == rocsparse_operation_none ? N : K;

    int64_t stride_1_a = transA == rocsparse_operation_none ? 1 : lda;
    int64_t stride_2_a = transA == rocsparse_operation_none ? lda : 1;

    int64_t c_stride_1_a = transA == rocsparse_operation_none ? 1 : K / 2;
    int64_t c_stride_2_a = transA == rocsparse_operation_none ? M : 1;
    int64_t c_stride_a_r
        = transA == rocsparse_operation_none ? K / 2 * c_stride_2_a : M * c_stride_1_a;
    int64_t c_stride_a = stride_a == 0 ? 0 : c_stride_a_r;

    int64_t m_stride_1_a = K / 8;
    int64_t m_stride_2_a = 1;
    int64_t m_stride_a_r = M * m_stride_1_a;
    int64_t m_stride_a   = stride_a == 0 ? 0 : m_stride_a_r;

    auto metadata_offset = c_stride_a_r * sizeof(Ti) * (stride_a == 0 ? 1 : num_batches);

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

    const size_t size_A = stride_a == 0 ? lda * A_col * num_batches : stride_a * num_batches;
    const size_t size_A_pruned_copy = arg.unit_check || arg.norm_check || arg.timing ? size_A : 0;

    const size_t size_B      = stride_b == 0 ? ldb * B_col * num_batches : stride_b * num_batches;
    const size_t size_C      = stride_c == 0 ? ldc * N * num_batches : stride_c * num_batches;
    const size_t size_D      = stride_d == 0 ? ldd * N * num_batches : stride_d * num_batches;
    const size_t size_D_copy = arg.unit_check || arg.norm_check ? size_D : 0;

    // allocate memory on device
    device_vector<Ti>            dA(size_A, 1, HMM);
    device_vector<Ti>            dB(size_B, 1, HMM);
    device_vector<To>            dC(size_C, 1, HMM);
    device_vector<To>            dD(size_D, 1, HMM);
    device_vector<unsigned char> dA_compressd(compressed_size, 1, HMM);
    device_vector<unsigned char> dWorkspace(workspace_size, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_compressd.memcheck());
    CHECK_DEVICE_ALLOCATION(dWorkspace.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ti> hA(size_A);
    host_vector<Ti> hA_pruned(size_A_pruned_copy);
    host_vector<Ti> hB(size_B);
    host_vector<To> hC(size_C);
    host_vector<To> hD_gold(size_D_copy);
    host_vector<To> hD_1(size_D_copy);

    rocsparselt_seedrand();

    // Initial Data on CPU
    if(arg.alpha_isnan<Tc>())
    {
        rocsparselt_init_nan<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
        rocsparselt_init_nan<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
    }
    else
    {
        if(arg.initialization == rocsparselt_initialization::rand_int)
        {
            rocsparselt_init<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
            rocsparselt_init_alternating_sign<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
        }
        else if(arg.initialization == rocsparselt_initialization::trig_float)
        {
            rocsparselt_init_sin<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
            rocsparselt_init_cos<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
        }
        else if(arg.initialization == rocsparselt_initialization::hpl)
        {
            rocsparselt_init_hpl<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
            rocsparselt_init_hpl<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
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
    }

    if(arg.beta_isnan<Tc>())
    {
        rocsparselt_init_nan<To>(hC, M, N, ldc, stride_c, num_batches);
    }
    else
    {
        if(arg.initialization == rocsparselt_initialization::rand_int)
            rocsparselt_init<To>(hC, M, N, ldc, stride_c, num_batches);
        else if(arg.initialization == rocsparselt_initialization::trig_float)
            rocsparselt_init_sin<To>(hC, M, N, ldc, stride_c, num_batches);
        else if(arg.initialization == rocsparselt_initialization::hpl)
            rocsparselt_init_hpl<To>(hC, M, N, ldc, stride_c, num_batches);
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_smfmac_prune(handle, matmul, dA, dA, rocsparselt_prune_smfmac_strip, stream),
        rocsparse_status_success);

    EXPECT_ROCSPARSELT_STATUS(rocsparselt_smfmac_compress(handle, plan, dA, dA_compressd, stream),
                              rocsparse_status_success);

    double rocsparselt_err = 0.0;
    if(arg.unit_check || arg.norm_check)
    {
        hipStreamSynchronize(stream);
        CHECK_HIP_ERROR(hA_pruned.transfer_from(dA));

        EXPECT_ROCSPARSELT_STATUS(rocsparselt_matmul(handle,
                                                     plan,
                                                     &h_alpha_Tc,
                                                     dA_compressd,
                                                     dB,
                                                     &h_beta_Tc,
                                                     dC,
                                                     dD,
                                                     dWorkspace,
                                                     &stream,
                                                     1),
                                  rocsparse_status_success);

        // now we can recycle gold matrix for reference purposes
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        for(int i = 0; i < num_batches; i++)
        {
            cblas_gemm<Ti, To, Tc>(transA,
                                   transB,
                                   M,
                                   N,
                                   K,
                                   h_alpha_Tc,
                                   hA_pruned + stride_a * i,
                                   lda,
                                   hB + stride_b * i,
                                   ldb,
                                   h_beta_Tc,
                                   hD_gold + stride_d * i,
                                   ldd,
                                   false);
        }

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // fetch GPU
        hipStreamSynchronize(stream);
        CHECK_HIP_ERROR(hD_1.transfer_from(dD));

        if(arg.unit_check)
        {
            if(std::is_same<To, rocsparselt_half>{} && K > 10000)
            {
                // For large K, rocsparse_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<To>;
                near_check_general<To>(M, N, ldd, stride_d, hD_gold, hD_1, num_batches, tol);
            }
            else
            {
                unit_check_general<To>(M, N, ldd, stride_d, hD_gold, hD_1, num_batches);
            }
        }

        if(arg.norm_check)
        {
            rocsparselt_err = std::abs(
                norm_check_general<To>('F', M, N, ldd, stride_d, hD_gold, hD_1, num_batches));
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
        {
            EXPECT_ROCSPARSELT_STATUS(rocsparselt_matmul(handle,
                                                         plan,
                                                         &h_alpha_Tc,
                                                         dA_compressd,
                                                         dB,
                                                         &h_beta_Tc,
                                                         dC,
                                                         dD,
                                                         dWorkspace,
                                                         &stream,
                                                         1),
                                      rocsparse_status_success);
        }

        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            EXPECT_ROCSPARSELT_STATUS(rocsparselt_matmul(handle,
                                                         plan,
                                                         &h_alpha_Tc,
                                                         dA_compressd,
                                                         dB,
                                                         &h_beta_Tc,
                                                         dC,
                                                         dD,
                                                         dWorkspace,
                                                         &stream,
                                                         1),
                                      rocsparse_status_success);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA,
                      e_transB,
                      e_M,
                      e_N,
                      e_K,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_beta,
                      e_ldb,
                      e_stride_b,
                      e_ldc,
                      e_stride_c,
                      e_ldd,
                      e_stride_d,
                      e_batch_count>{}
            .log_args<float>(rocsparselt_cout,
                             arg,
                             gpu_time_used,
                             gemm_gflop_count<float>(M, N, K),
                             ArgumentLogging::NA_value,
                             cpu_time_used,
                             rocsparselt_error);
    }
}
