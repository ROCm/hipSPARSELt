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
#include <cstddef>
#include <omp.h>

template <typename Ti, typename To, typename Tact, typename F>
void activation(int64_t m, int64_t n, int64_t ld, Ti* in, To* out, Tact arg1, Tact arg2, F& func)
{
    auto saturate_i8 = [](Tact val) {
        auto _val = std::nearbyint(static_cast<double>(val));
        _val      = _val > 127.f ? 127.f : _val < -128.f ? -128.f : _val;
        return static_cast<To>(_val);
    };

    auto saturate_o = [](Tact val) { return static_cast<To>(val); };

    To (*saturate)(Tact val);
    saturate = std::is_same<int8_t, To>() ? saturate_i8 : saturate_o;

    for(int i = 0; i < m; i++)
    {
#pragma omp parallel for
        for(int j = 0; j < n; j++)
        {
            auto pos     = j * ld + i;
            auto in_Tact = static_cast<Tact>(*(in + pos));
            *(out + pos) = saturate(func(in_Tact, arg1, arg2));
        }
    }
}

auto _relu = [](auto in, auto /*arg1*/, auto /*arg2*/) -> decltype(in) {
    return static_cast<decltype(in)>(std::max(static_cast<decltype(in)>(0), in));
};

auto _clippedrelu = [](auto in, auto arg1, auto arg2) -> decltype(in) {
    if(in > arg1)
        return static_cast<decltype(in)>(std::min(in, arg2));
    else
        return static_cast<decltype(in)>(0);
};

auto _gelu = [](auto in, auto /*arg1*/, auto /*arg2*/) -> decltype(in) {
    using Tc = float;

    constexpr auto k0    = static_cast<Tc>(0.7978845608028654);
    constexpr auto k1    = static_cast<Tc>(0.044715);
    Tc             in_Tc = static_cast<Tc>(in);

    return static_cast<decltype(in)>(
        0.5f * (in_Tc * (1.f + std::tanh(k0 * (in_Tc * (1.f + k1 * (in_Tc * in_Tc)))))));
};

auto _abs = [](auto in, auto /*arg1*/, auto /*arg2*/) -> decltype(in) {
    return static_cast<decltype(in)>(std::abs(static_cast<float>(in)));
};

auto _leakyrelu = [](auto in, auto arg1, auto /*arg2*/) -> decltype(in) {
    if(in > static_cast<decltype(in)>(0))
        return in;
    else
        return in * arg1;
};

auto _sigmoid = [](auto in, auto /*arg1*/, auto /*arg2*/) -> decltype(in) {
    using Tc = float;
    Tc in_Tc = static_cast<Tc>(in);
    return static_cast<decltype(in)>(1.f / (1.f + std::exp(-in_Tc)));
};

auto _tanh = [](auto in, auto arg1, auto arg2) -> decltype(in) {
    using Tc   = float;
    Tc in_Tc   = static_cast<Tc>(in);
    Tc arg1_Tc = static_cast<Tc>(arg1);
    Tc arg2_Tc = static_cast<Tc>(arg2);
    return static_cast<decltype(in)>(std::tanh(in_Tc * arg1_Tc) * arg2_Tc);
};

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

    const rocsparselt_operation transA = rocsparselt_operation_none;
    const rocsparselt_operation transB = rocsparselt_operation_none;

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
    rocsparselt_local_mat_descr matA(rocsparselt_matrix_type_structured,
                                     handle,
                                     M,
                                     K,
                                     lda,
                                     arg.a_type,
                                     rocsparselt_order_column);
    rocsparselt_local_mat_descr matB(
        rocsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, rocsparselt_order_column);
    rocsparselt_local_mat_descr matC(
        rocsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, rocsparselt_order_column);
    rocsparselt_local_mat_descr matD(
        rocsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, rocsparselt_order_column);
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
        rocsparselt_status_invalid_handle);

    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, nullptr, &alpha, dA, dB, &beta, dC, dD, workspace, &stream, 1),
        rocsparselt_status_invalid_handle);

    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, nullptr, dA, dB, &beta, dC, dD, workspace, &stream, 1),
        rocsparselt_status_invalid_pointer);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, nullptr, dB, &beta, dC, dD, workspace, &stream, 1),
        rocsparselt_status_invalid_pointer);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, dA, nullptr, &beta, dC, dD, workspace, &stream, 1),
        rocsparselt_status_invalid_pointer);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, dA, dB, nullptr, dC, dD, workspace, &stream, 1),
        rocsparselt_status_invalid_pointer);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, dA, dB, &beta, nullptr, dD, workspace, &stream, 1),
        rocsparselt_status_invalid_pointer);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, dA, dB, &beta, dC, nullptr, workspace, &stream, 1),
        rocsparselt_status_invalid_pointer);

    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, dA, dB, &beta, dC, dD, workspace, &stream, -1),
        rocsparselt_status_invalid_value);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan, &alpha, dA, dB, &beta, dC, dD, workspace, nullptr, 1),
        rocsparselt_status_invalid_value);

    workspace_size = 1;
    rocsparselt_local_matmul_plan plan2(handle, matmul, alg_sel, workspace_size);
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_matmul(handle, plan2, &alpha, dA, dB, &beta, dC, dD, workspace, &stream, 0),
        rocsparselt_status_invalid_value);
}

template <typename Ti,
          typename To,
          typename Tc,
          rocsparselt_batch_type btype = rocsparselt_batch_type::none>
void testing_spmm(const Arguments& arg)
{
    rocsparselt_operation transA = char2rocsparselt_operation(arg.transA);
    rocsparselt_operation transB = char2rocsparselt_operation(arg.transB);

    using Talpha = float;

    int64_t M       = arg.M;
    int64_t N       = arg.N;
    int64_t K       = arg.K;
    Talpha  h_alpha = arg.get_alpha<Talpha>();
    Talpha  h_beta  = arg.get_beta<Talpha>();
    int64_t lda     = arg.lda;
    int64_t ldb     = arg.ldb;
    int64_t ldc     = arg.ldc;
    int64_t ldd     = arg.ldd;

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used              = 0.0;
    double                   rocsparselt_error = 0.0;
    bool                     HMM               = arg.HMM;
    rocsparselt_local_handle handle{arg};
    hipStream_t              stream;
    hipStreamCreate(&stream);

    int64_t A_row = transA == rocsparselt_operation_none ? M : K;
    int64_t A_col = transA == rocsparselt_operation_none ? K : M;
    int64_t B_row = transB == rocsparselt_operation_none ? K : N;
    int64_t B_col = transB == rocsparselt_operation_none ? N : K;

    int64_t stride_1_a = transA == rocsparselt_operation_none ? 1 : lda;
    int64_t stride_2_a = transA == rocsparselt_operation_none ? lda : 1;

    constexpr bool do_batched         = (btype == rocsparselt_batch_type::batched);
    constexpr bool do_strided_batched = (btype == rocsparselt_batch_type::strided_batched);
    int            num_batches        = (do_batched || do_strided_batched ? arg.batch_count : 1);
    int64_t        stride_a           = do_strided_batched ? arg.stride_a : lda * A_col;
    int64_t        stride_b           = do_strided_batched ? arg.stride_b : ldb * B_col;
    int64_t        stride_c           = do_strided_batched ? arg.stride_c : ldc * M;
    int64_t        stride_d           = do_strided_batched ? arg.stride_c : ldd * M;

    rocsparselt_local_mat_descr matA(rocsparselt_matrix_type_structured,
                                     handle,
                                     A_row,
                                     A_col,
                                     lda,
                                     arg.a_type,
                                     rocsparselt_order_column);
    rocsparselt_local_mat_descr matB(rocsparselt_matrix_type_dense,
                                     handle,
                                     B_row,
                                     B_col,
                                     ldb,
                                     arg.b_type,
                                     rocsparselt_order_column);
    rocsparselt_local_mat_descr matC(
        rocsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, rocsparselt_order_column);
    rocsparselt_local_mat_descr matD(
        rocsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, rocsparselt_order_column);

    bool invalid_size_a = M < 8 || K % 8 != 0 || lda < A_row;
    bool invalid_size_b = N < 8 || ldb < B_row;
    bool invalid_size_c = ldc < M;
    if(invalid_size_a)
    {
        EXPECT_ROCSPARSELT_STATUS(matA.status(), rocsparselt_status_invalid_size);

        return;
    }
    if(invalid_size_b)
    {
        EXPECT_ROCSPARSELT_STATUS(matB.status(), rocsparselt_status_invalid_size);

        return;
    }
    if(invalid_size_c)
    {
        EXPECT_ROCSPARSELT_STATUS(matC.status(), rocsparselt_status_invalid_size);

        return;
    }

    if(do_batched || do_strided_batched)
    {
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matA, rocsparselt_mat_num_batches, &num_batches, sizeof(int)),
            rocsparselt_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matB, rocsparselt_mat_num_batches, &num_batches, sizeof(int)),
            rocsparselt_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matC, rocsparselt_mat_num_batches, &num_batches, sizeof(int)),
            rocsparselt_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matD, rocsparselt_mat_num_batches, &num_batches, sizeof(int)),
            rocsparselt_status_success);
    }

    if(do_strided_batched)
    {
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matA, rocsparselt_mat_batch_stride, &stride_a, sizeof(int64_t)),
            rocsparselt_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matB, rocsparselt_mat_batch_stride, &stride_b, sizeof(int64_t)),
            rocsparselt_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matC, rocsparselt_mat_batch_stride, &stride_c, sizeof(int64_t)),
            rocsparselt_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_mat_descr_set_attribute(
                handle, matD, rocsparselt_mat_batch_stride, &stride_d, sizeof(int64_t)),
            rocsparselt_status_success);
    }

    rocsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);

    int activation_on = 1;
    switch(arg.activation_type)
    {
    case rocsparselt_activation_type::clippedrelu:
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_matmul_descr_set_attribute(handle,
                                                   matmul,
                                                   rocsparselt_matmul_activation_relu_upperbound,
                                                   &arg.activation_arg2,
                                                   sizeof(float)),
            rocsparselt_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_matmul_descr_set_attribute(handle,
                                                   matmul,
                                                   rocsparselt_matmul_activation_relu_threshold,
                                                   &arg.activation_arg1,
                                                   sizeof(float)),
            rocsparselt_status_success);
    case rocsparselt_activation_type::relu:
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_matmul_descr_set_attribute(handle,
                                                   matmul,
                                                   rocsparselt_matmul_activation_relu,
                                                   &activation_on,
                                                   sizeof(activation_on)),
            rocsparselt_status_success);
        break;
    case rocsparselt_activation_type::gelu:
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_matmul_descr_set_attribute(handle,
                                                   matmul,
                                                   rocsparselt_matmul_activation_gelu,
                                                   &activation_on,
                                                   sizeof(activation_on)),
            rocsparselt_status_success);
        break;
    case rocsparselt_activation_type::abs:
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_matmul_descr_set_attribute(handle,
                                                   matmul,
                                                   rocsparselt_matmul_activation_abs,
                                                   &activation_on,
                                                   sizeof(activation_on)),
            rocsparselt_status_success);
        break;
    case rocsparselt_activation_type::leakyrelu:
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_matmul_descr_set_attribute(handle,
                                                   matmul,
                                                   rocsparselt_matmul_activation_leakyrelu,
                                                   &activation_on,
                                                   sizeof(activation_on)),
            rocsparselt_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_matmul_descr_set_attribute(handle,
                                                   matmul,
                                                   rocsparselt_matmul_activation_leakyrelu_alpha,
                                                   &arg.activation_arg1,
                                                   sizeof(float)),
            rocsparselt_status_success);
        break;
    case rocsparselt_activation_type::sigmoid:
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_matmul_descr_set_attribute(handle,
                                                   matmul,
                                                   rocsparselt_matmul_activation_sigmoid,
                                                   &activation_on,
                                                   sizeof(activation_on)),
            rocsparselt_status_success);
        break;
    case rocsparselt_activation_type::tanh:
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_matmul_descr_set_attribute(handle,
                                                   matmul,
                                                   rocsparselt_matmul_activation_tanh,
                                                   &activation_on,
                                                   sizeof(activation_on)),
            rocsparselt_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_matmul_descr_set_attribute(handle,
                                                   matmul,
                                                   rocsparselt_matmul_activation_tanh_alpha,
                                                   &arg.activation_arg1,
                                                   sizeof(float)),
            rocsparselt_status_success);
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_matmul_descr_set_attribute(handle,
                                                   matmul,
                                                   rocsparselt_matmul_activation_tanh_beta,
                                                   &arg.activation_arg2,
                                                   sizeof(float)),
            rocsparselt_status_success);
        break;
    default:
        activation_on = 0;
        break;
    }

    rocsparselt_local_matmul_alg_selection alg_sel(handle, matmul, rocsparselt_matmul_alg_default);

    size_t workspace_size, compressed_size;
    rocsparselt_matmul_get_workspace(handle, alg_sel, &workspace_size);

    rocsparselt_local_matmul_plan plan(handle, matmul, alg_sel, workspace_size);

    EXPECT_ROCSPARSELT_STATUS(rocsparselt_smfmac_compressed_size(handle, plan, &compressed_size),
                              rocsparselt_status_success);

    const size_t size_A = stride_a == 0 ? lda * A_col * num_batches : stride_a * num_batches;
    const size_t size_A_pruned_copy = arg.unit_check || arg.norm_check || arg.timing ? size_A : 0;

    const size_t size_B      = stride_b == 0 ? ldb * B_col * num_batches : stride_b * num_batches;
    const size_t size_C      = stride_c == 0 ? ldc * N * num_batches : stride_c * num_batches;
    const size_t size_D      = stride_d == 0 ? ldd * N * num_batches : stride_d * num_batches;
    const size_t size_D_copy = arg.unit_check || arg.norm_check ? size_D : 0;
    const size_t size_D_act_copy = activation_on ? size_D_copy : 0;

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
    host_vector<Ti>     hA(size_A);
    host_vector<Ti>     hA_pruned(size_A_pruned_copy);
    host_vector<Ti>     hB(size_B);
    host_vector<To>     hC(size_C);
    host_vector<To>     hD_gold(size_D_copy);
    host_vector<Talpha> hD_gold_act(size_D_copy);
    host_vector<To>     hD_1(size_D_copy);

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
        else if(arg.initialization == rocsparselt_initialization::special)
        {
            rocsparselt_init_alt_impl_big<Ti>(hA, A_row, A_col, lda, num_batches);
            rocsparselt_init_alt_impl_small<Ti>(hB, B_row, B_col, ldb, num_batches);
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
        else if(arg.initialization == rocsparselt_initialization::special)
            rocsparselt_init<To>(hC, M, N, ldc, stride_c, num_batches);
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(size_D_copy)
    {
        if(activation_on)
        {
            std::transform(hC.begin(), hC.end(), hD_gold_act.begin(), [](To c) -> Talpha {
                return static_cast<Talpha>(c);
            });
        }
        else
        {
            std::copy(hC.begin(), hC.end(), hD_gold.begin());
        }
    }
    EXPECT_ROCSPARSELT_STATUS(
        rocsparselt_smfmac_prune(handle, matmul, dA, dA, rocsparselt_prune_smfmac_strip, stream),
        rocsparselt_status_success);

    EXPECT_ROCSPARSELT_STATUS(rocsparselt_smfmac_compress(handle, plan, dA, dA_compressd, stream),
                              rocsparselt_status_success);

    if(arg.unit_check || arg.norm_check)
    {
        hipStreamSynchronize(stream);
        CHECK_HIP_ERROR(hA_pruned.transfer_from(dA));
        EXPECT_ROCSPARSELT_STATUS(
            rocsparselt_matmul(
                handle, plan, &h_alpha, dA_compressd, dB, &h_beta, dC, dD, dWorkspace, &stream, 1),
            rocsparselt_status_success);
        // now we can recycle gold matrix for reference purposes
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

#define activation_param \
    M, N, ldd, hD_gold_act + pos, hD_gold + pos, arg.activation_arg1, arg.activation_arg2
        for(int i = 0; i < num_batches; i++)
        {
            if(activation_on)
            {
                cblas_gemm<Ti, Talpha, Talpha>(transA,
                                               transB,
                                               M,
                                               N,
                                               K,
                                               h_alpha,
                                               hA_pruned + stride_a * i,
                                               lda,
                                               hB + stride_b * i,
                                               ldb,
                                               h_beta,
                                               hD_gold_act + stride_d * i,
                                               ldd,
                                               false);
                auto pos = stride_d * i;
                switch(arg.activation_type)
                {
                case rocsparselt_activation_type::clippedrelu:
                    activation(activation_param, ::_clippedrelu);
                    break;
                case rocsparselt_activation_type::gelu:
                    activation(activation_param, ::_gelu);
                    break;
                case rocsparselt_activation_type::relu:
                    activation(activation_param, ::_relu);
                    break;
                case rocsparselt_activation_type::abs:
                    activation(activation_param, ::_abs);
                    break;
                case rocsparselt_activation_type::leakyrelu:
                    activation(activation_param, ::_leakyrelu);
                    break;
                case rocsparselt_activation_type::sigmoid:
                    activation(activation_param, ::_sigmoid);
                    break;
                case rocsparselt_activation_type::tanh:
                    activation(activation_param, ::_tanh);
                    break;
                default:
                    continue;
                }
            }

            else
                cblas_gemm<Ti, To, Talpha>(transA,
                                           transB,
                                           M,
                                           N,
                                           K,
                                           h_alpha,
                                           hA_pruned + stride_a * i,
                                           lda,
                                           hB + stride_b * i,
                                           ldb,
                                           h_beta,
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
            rocsparselt_error = std::abs(
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
                                                         &h_alpha,
                                                         dA_compressd,
                                                         dB,
                                                         &h_beta,
                                                         dC,
                                                         dD,
                                                         dWorkspace,
                                                         &stream,
                                                         1),
                                      rocsparselt_status_success);
        }

        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            EXPECT_ROCSPARSELT_STATUS(rocsparselt_matmul(handle,
                                                         plan,
                                                         &h_alpha,
                                                         dA_compressd,
                                                         dB,
                                                         &h_beta,
                                                         dC,
                                                         dD,
                                                         dWorkspace,
                                                         &stream,
                                                         1),
                                      rocsparselt_status_success);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
        auto flops    = gemm_gflop_count<float>(M, N, K);
        switch(arg.activation_type)
        {
        case rocsparselt_activation_type::relu:
            flops += relu_gflop_count<float>(M, N);
            break;
        case rocsparselt_activation_type::clippedrelu:
            flops += clippedrelu_gflop_count<float>(M, N);
            break;
        case rocsparselt_activation_type::gelu:
            flops += gelu_gflop_count<float>(M, N);
            break;
        case rocsparselt_activation_type::abs:
            flops += abs_gflop_count<float>(M, N);
            break;
        case rocsparselt_activation_type::leakyrelu:
            flops += leakyrelu_gflop_count<float>(M, N);
            break;
        case rocsparselt_activation_type::sigmoid:
            flops += sigmoid_gflop_count<float>(M, N);
            break;
        case rocsparselt_activation_type::tanh:
            flops += tanh_gflop_count<float>(M, N);
            break;
        default:
            break;
        }
#define argument_param_nb                                                                     \
    e_transA, e_transB, e_M, e_N, e_K, e_alpha, e_lda, e_stride_a, e_beta, e_ldb, e_stride_b, \
        e_ldc, e_stride_c, e_ldd, e_stride_d
#define argument_param argument_param_nb, e_batch_count

        if(do_batched || do_strided_batched)
            ArgumentModel<argument_param>{}.log_args<float>(rocsparselt_cout,
                                                            arg,
                                                            gpu_time_used,
                                                            flops,
                                                            ArgumentLogging::NA_value,
                                                            cpu_time_used,
                                                            rocsparselt_error);
        else
            ArgumentModel<argument_param_nb>{}.log_args<float>(rocsparselt_cout,
                                                               arg,
                                                               gpu_time_used,
                                                               flops,
                                                               ArgumentLogging::NA_value,
                                                               cpu_time_used,
                                                               rocsparselt_error);
    }
}
