/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
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

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "hipsparselt_datatype2string.hpp"
#include "hipsparselt_init.hpp"
#include "hipsparselt_math.hpp"
#include "hipsparselt_random.hpp"
#include "hipsparselt_test.hpp"
#include "hipsparselt_vector.hpp"
#include "near.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include <cstddef>
#include <hipsparselt/hipsparselt.h>
#include <omp.h>

template <typename T, typename Tb = T, typename To = T>
void bias(int64_t m, int64_t n, int64_t ld, T* src, To* dest, Tb* bias)
{
    auto saturate_i8 = [](Tb val) {
        auto _val = std::nearbyint(static_cast<double>(val));
        _val      = _val > 127.f ? 127.f : _val < -128.f ? -128.f : _val;
        return static_cast<To>(_val);
    };

    auto saturate_o = [](Tb val) { return static_cast<To>(val); };

    To (*saturate)(Tb val);
    saturate = std::is_same<int8_t, To>() ? saturate_i8 : saturate_o;

    using TAccum = std::conditional_t<std::is_same<__half, Tb>::value, float, Tb>;

    for(int i = 0; i < m; i++)
    {
        Tb _bias = *(bias + i);
#pragma omp parallel for
        for(int j = 0; j < n; j++)
        {
            auto   pos = j * ld + i;
            TAccum src_Taccum;
            if constexpr(std::is_same<T, TAccum>())
                src_Taccum = *(src + pos);
            else
                src_Taccum = static_cast<TAccum>(*(src + pos));

            *(dest + pos) = saturate(src_Taccum + static_cast<TAccum>(_bias));
        }
    }
}

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

auto _gelu = [](auto in, auto arg1, auto /*arg2*/) -> decltype(in) {
    using Tc = float;

    constexpr auto k0    = static_cast<Tc>(0.7978845608028654);
    constexpr auto k1    = static_cast<Tc>(0.044715);
    Tc             in_Tc = static_cast<Tc>(in);

    auto out = 0.5f * (in_Tc * (1.f + std::tanh(k0 * (in_Tc * (1.f + k1 * (in_Tc * in_Tc))))));
    if(arg1 != 1) out *= arg1;

    return static_cast<decltype(in)>(out);
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

    const hipsparseOperation_t transA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    // allocate memory on device
    device_vector<Ti> dA(safe_size / 2);
    device_vector<Ti> dB(safe_size);
    device_vector<Ti> dC(safe_size);
    device_vector<Ti> dD(safe_size);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());

    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, M, K, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);

    size_t                        workspace_size = 0;
    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);

    void* workspace = nullptr;
    float alpha = 1.0, beta = 0.0;

    hipStream_t stream = nullptr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmul(nullptr, plan, &alpha, dA, dB, &beta, dC, dD, workspace, &stream, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmul(handle, nullptr, &alpha, dA, dB, &beta, dC, dD, workspace, &stream, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmul(handle, plan, nullptr, dA, dB, &beta, dC, dD, workspace, &stream, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmul(handle, plan, &alpha, nullptr, dB, &beta, dC, dD, workspace, &stream, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmul(handle, plan, &alpha, dA, nullptr, &beta, dC, dD, workspace, &stream, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmul(handle, plan, &alpha, dA, dB, nullptr, dC, dD, workspace, &stream, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmul(handle, plan, &alpha, dA, dB, &beta, nullptr, dD, workspace, &stream, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmul(handle, plan, &alpha, dA, dB, &beta, dC, nullptr, workspace, &stream, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmul(handle, plan, &alpha, dA, dB, &beta, dC, dD, workspace, &stream, -1),
        HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmul(handle, plan, &alpha, dA, dB, &beta, dC, dD, workspace, nullptr, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

template <typename Ti,
          typename To,
          typename Tc,
          typename TBias,
          hipsparselt_batch_type btype = hipsparselt_batch_type::none>
void testing_spmm(const Arguments& arg)
{
    hipsparseOperation_t transA = char_to_hipsparselt_operation(arg.transA);
    hipsparseOperation_t transB = char_to_hipsparselt_operation(arg.transB);

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
    double                   hipsparselt_error = 0.0;
    bool                     HMM               = arg.HMM;
    hipsparselt_local_handle handle{arg};
    hipStream_t              stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

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
    int64_t        stride_c           = do_strided_batched ? arg.stride_c : ldc * N;
    int64_t        stride_d           = do_strided_batched ? arg.stride_c : ldd * N;
    int64_t bias_stride = do_strided_batched ? arg.bias_stride == -1 ? M : arg.bias_stride : 0;
    hipsparseLtDatatype_t bias_type
        = arg.bias_type == 0 ? (arg.a_type == HIPSPARSELT_R_8I ? HIPSPARSELT_R_32F : arg.a_type)
                             : arg.bias_type;

    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured,
                                     handle,
                                     A_row,
                                     A_col,
                                     lda,
                                     arg.a_type,
                                     HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, B_row, B_col, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);

    hipsparseStatus_t eStatus
        = expected_hipsparse_status_of_matrix_size(arg.a_type, A_row, A_col, lda);
    EXPECT_HIPSPARSE_STATUS(matA.status(), eStatus);
    if(eStatus != HIPSPARSE_STATUS_SUCCESS)
        return;

    eStatus = expected_hipsparse_status_of_matrix_size(arg.b_type, B_row, B_col, ldb);
    EXPECT_HIPSPARSE_STATUS(matB.status(), eStatus);
    if(eStatus != HIPSPARSE_STATUS_SUCCESS)
        return;

    eStatus = expected_hipsparse_status_of_matrix_size(arg.c_type, M, N, ldc);
    EXPECT_HIPSPARSE_STATUS(matC.status(), eStatus);
    if(eStatus != HIPSPARSE_STATUS_SUCCESS)
        return;

    eStatus = expected_hipsparse_status_of_matrix_size(arg.d_type, M, N, ldd);
    EXPECT_HIPSPARSE_STATUS(matD.status(), eStatus);
    if(eStatus != HIPSPARSE_STATUS_SUCCESS)
        return;

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
        eStatus = expected_hipsparse_status_of_matrix_stride(stride_a, A_row, A_col, lda);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matA, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_a, sizeof(int64_t)),
            eStatus);
        if(eStatus != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatus = expected_hipsparse_status_of_matrix_stride(stride_b, B_row, B_col, ldb);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matB, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_b, sizeof(int64_t)),
            eStatus);
        if(eStatus != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatus = expected_hipsparse_status_of_matrix_stride(stride_c, M, N, ldc);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matC, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_c, sizeof(int64_t)),
            eStatus);
        if(eStatus != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatus = expected_hipsparse_status_of_matrix_stride(stride_d, M, N, ldd);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matD, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_d, sizeof(int64_t)),
            eStatus);
        if(eStatus != HIPSPARSE_STATUS_SUCCESS)
            return;
    }

    hipsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);

    int activation_on = 1;
    switch(arg.activation_type)
    {
    case hipsparselt_activation_type::clippedrelu:
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle,
                                              matmul,
                                              HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND,
                                              &arg.activation_arg2,
                                              sizeof(float)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle,
                                              matmul,
                                              HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD,
                                              &arg.activation_arg1,
                                              sizeof(float)),
            HIPSPARSE_STATUS_SUCCESS);
    case hipsparselt_activation_type::relu:
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle,
                                              matmul,
                                              HIPSPARSELT_MATMUL_ACTIVATION_RELU,
                                              &activation_on,
                                              sizeof(activation_on)),
            HIPSPARSE_STATUS_SUCCESS);
        break;
    case hipsparselt_activation_type::gelu:
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle,
                                              matmul,
                                              HIPSPARSELT_MATMUL_ACTIVATION_GELU,
                                              &activation_on,
                                              sizeof(activation_on)),
            HIPSPARSE_STATUS_SUCCESS);
            if(arg.activation_arg1 != 1)
                EXPECT_HIPSPARSE_STATUS(
                    hipsparseLtMatmulDescSetAttribute(handle,
                                                    matmul,
                                                    HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING,
                                                    &arg.activation_arg1,
                                                    sizeof(float)),
                    HIPSPARSE_STATUS_SUCCESS);
        break;
    case hipsparselt_activation_type::abs:
        EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulDescSetAttribute(handle,
                                                                  matmul,
                                                                  HIPSPARSELT_MATMUL_ACTIVATION_ABS,
                                                                  &activation_on,
                                                                  sizeof(activation_on)),
                                HIPSPARSE_STATUS_SUCCESS);
        break;
    case hipsparselt_activation_type::leakyrelu:
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle,
                                              matmul,
                                              HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU,
                                              &activation_on,
                                              sizeof(activation_on)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle,
                                              matmul,
                                              HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA,
                                              &arg.activation_arg1,
                                              sizeof(float)),
            HIPSPARSE_STATUS_SUCCESS);
        break;
    case hipsparselt_activation_type::sigmoid:
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle,
                                              matmul,
                                              HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID,
                                              &activation_on,
                                              sizeof(activation_on)),
            HIPSPARSE_STATUS_SUCCESS);
        break;
    case hipsparselt_activation_type::tanh:
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle,
                                              matmul,
                                              HIPSPARSELT_MATMUL_ACTIVATION_TANH,
                                              &activation_on,
                                              sizeof(activation_on)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle,
                                              matmul,
                                              HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA,
                                              &arg.activation_arg1,
                                              sizeof(float)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle,
                                              matmul,
                                              HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA,
                                              &arg.activation_arg2,
                                              sizeof(float)),
            HIPSPARSE_STATUS_SUCCESS);
        break;
    default:
        activation_on = 0;
        break;
    }

    hipsparselt_seedrand();

    const size_t size_bias
        = arg.bias_vector ? (bias_stride == 0 ? M : bias_stride * num_batches) : 0;

    device_vector<TBias> dBias(size_bias, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dBias.memcheck());
    host_vector<TBias> hBias(size_bias);
    if(arg.bias_vector)
    {
        hipsparselt_init<TBias>(hBias, M, 1, M, bias_stride, num_batches);
        CHECK_HIP_ERROR(dBias.transfer_from(hBias));
        void* _dBias = dBias;
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(
                handle, matmul, HIPSPARSELT_MATMUL_BIAS_POINTER, &_dBias, sizeof(void*)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(
                handle, matmul, HIPSPARSELT_MATMUL_BIAS_STRIDE, &bias_stride, sizeof(int64_t)),
            HIPSPARSE_STATUS_SUCCESS);
#ifdef __HIP_PLATFORM_HCC__
        EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulDescSetAttribute(handle,
                                                                  matmul,
                                                                  HIPSPARSELT_MATMUL_BIAS_TYPE,
                                                                  &bias_type,
                                                                  sizeof(hipsparseLtDatatype_t)),
                                HIPSPARSE_STATUS_SUCCESS);
#endif
    }

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);

    size_t workspace_size = 0, compressed_size = 0, compress_buffer_size = 0;

    {

        if(arg.search)
        {
            int config_max_id = 0;
            hipsparseLtMatmulAlgGetAttribute(
                handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &config_max_id, sizeof(int));
            for(int i = 0; i < config_max_id; i++)
            {
                hipsparseLtMatmulAlgSetAttribute(
                    handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &i, sizeof(int));
                hipsparselt_local_matmul_plan plan_tmp(handle, matmul, alg_sel);
                size_t                        ws = 0;
                EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(handle, plan_tmp, &ws),
                                        HIPSPARSE_STATUS_SUCCESS);
                workspace_size = std::max(workspace_size, ws);
            }
            hipsparseLtMatmulAlgSetAttribute(handle,
                                             alg_sel,
                                             HIPSPARSELT_MATMUL_SEARCH_ITERATIONS,
                                             &arg.search_iters,
                                             sizeof(int));
        }
        else
        {
            hipsparselt_local_matmul_plan plan_tmp(handle, matmul, alg_sel);
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtMatmulGetWorkspace(handle, plan_tmp, &workspace_size),
                HIPSPARSE_STATUS_SUCCESS);
        }
    }

    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMACompressedSize(handle, plan, &compressed_size, &compress_buffer_size),
        HIPSPARSE_STATUS_SUCCESS);

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
    device_vector<unsigned char> dA_compressBuffer(compress_buffer_size, 1, HMM);
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

    // Initial Data on CPU
    if(arg.alpha_isnan<Tc>())
    {
        hipsparselt_init_nan<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
        hipsparselt_init_nan<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
    }
    else
    {
        if(arg.initialization == hipsparselt_initialization::rand_int)
        {
            hipsparselt_init<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
            hipsparselt_init_alternating_sign<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
        }
        else if(arg.initialization == hipsparselt_initialization::trig_float)
        {
            hipsparselt_init_sin<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
            hipsparselt_init_cos<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
        }
        else if(arg.initialization == hipsparselt_initialization::hpl)
        {
            hipsparselt_init_hpl<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
            hipsparselt_init_hpl<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
        }
        else if(arg.initialization == hipsparselt_initialization::special)
        {
            hipsparselt_init_alt_impl_big<Ti>(hA, A_row, A_col, lda, num_batches);
            hipsparselt_init_alt_impl_small<Ti>(hB, B_row, B_col, ldb, num_batches);
        }
    }

    if(arg.beta_isnan<Tc>())
    {
        hipsparselt_init_nan<To>(hC, M, N, ldc, stride_c, num_batches);
    }
    else
    {
        if(arg.initialization == hipsparselt_initialization::rand_int)
            hipsparselt_init<To>(hC, M, N, ldc, stride_c, num_batches);
        else if(arg.initialization == hipsparselt_initialization::trig_float)
            hipsparselt_init_sin<To>(hC, M, N, ldc, stride_c, num_batches);
        else if(arg.initialization == hipsparselt_initialization::hpl)
            hipsparselt_init_hpl<To>(hC, M, N, ldc, stride_c, num_batches);
        else if(arg.initialization == hipsparselt_initialization::special)
            hipsparselt_init<To>(hC, M, N, ldc, stride_c, num_batches);
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(size_D_copy)
    {
        if(activation_on || arg.bias_vector)
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
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMAPrune(handle, matmul, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
        HIPSPARSE_STATUS_SUCCESS);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMACompress(handle, plan, dA, dA_compressd, dA_compressBuffer, stream),
        HIPSPARSE_STATUS_SUCCESS);

    if(arg.search)
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulSearch(
                handle, plan, &h_alpha, dA_compressd, dB, &h_beta, dC, dD, dWorkspace, &stream, 1),
            HIPSPARSE_STATUS_SUCCESS);
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        CHECK_HIP_ERROR(hA_pruned.transfer_from(dA));
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmul(
                handle, plan, &h_alpha, dA_compressd, dB, &h_beta, dC, dD, dWorkspace, &stream, 1),
            HIPSPARSE_STATUS_SUCCESS);
        // now we can recycle gold matrix for reference purposes
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

#define activation_param \
    M, N, ldd, hD_gold_act + pos, hD_gold + pos, arg.activation_arg1, arg.activation_arg2

        for(int i = 0; i < num_batches; i++)
        {
            if(activation_on || arg.bias_vector)
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
                if(arg.bias_vector)
                {
                    if(activation_on)
                        bias<Talpha, TBias>(M,
                                            N,
                                            ldd,
                                            hD_gold_act + pos,
                                            hD_gold_act + pos,
                                            hBias + bias_stride * i);
                    else
                        bias<Talpha, TBias, To>(
                            M, N, ldd, hD_gold_act + pos, hD_gold + pos, hBias + bias_stride * i);
                }

                if(activation_on)
                {
                    switch(arg.activation_type)
                    {
                    case hipsparselt_activation_type::clippedrelu:
                        activation(activation_param, ::_clippedrelu);
                        break;
                    case hipsparselt_activation_type::gelu:
                        activation(activation_param, ::_gelu);
                        break;
                    case hipsparselt_activation_type::relu:
                        activation(activation_param, ::_relu);
                        break;
                    case hipsparselt_activation_type::abs:
                        activation(activation_param, ::_abs);
                        break;
                    case hipsparselt_activation_type::leakyrelu:
                        activation(activation_param, ::_leakyrelu);
                        break;
                    case hipsparselt_activation_type::sigmoid:
                        activation(activation_param, ::_sigmoid);
                        break;
                    case hipsparselt_activation_type::tanh:
                        activation(activation_param, ::_tanh);
                        break;
                    default:
                        continue;
                    }
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
#undef activation_param

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // fetch GPU
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        CHECK_HIP_ERROR(hD_1.transfer_from(dD));

        if(arg.unit_check)
        {
            unit_check_general<To>(M, N, ldd, stride_d, hD_gold, hD_1, num_batches);
        }

        if(arg.norm_check)
        {
            hipsparselt_error = std::abs(
                norm_check_general<To>('F', M, N, ldd, stride_d, hD_gold, hD_1, num_batches));
        }

        // Debug
        //print_strided_batched("A", &hA[0], A_row, A_col, num_batches, 1, lda, stride_a);
        //print_strided_batched("B", &hB[0], B_row, B_col, num_batches, 1, ldb, stride_b);
        //print_strided_batched("C", &hC[0], M, N, num_batches, 1, ldc, stride_c);
        //if(arg.bias_vector)
        //    print_strided_batched("bias", &hBias[0], M, N, num_batches, 1, M, bias_stride);
        //print_strided_batched("hD_gold", &hD_gold[0], M, N, num_batches, 1, ldd, stride_d);
        //print_strided_batched("hD1", &hD_1[0], M, N, num_batches, 1, ldd, stride_d);
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        for(int i = 0; i < number_cold_calls; i++)
        {
            EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmul(handle,
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
                                    HIPSPARSE_STATUS_SUCCESS);
        }

        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmul(handle,
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
                                    HIPSPARSE_STATUS_SUCCESS);
        }
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
        auto flops    = gemm_gflop_count<float>(M, N, K);
        switch(arg.activation_type)
        {
        case hipsparselt_activation_type::relu:
            flops += relu_gflop_count<float>(M, N);
            break;
        case hipsparselt_activation_type::clippedrelu:
            flops += clippedrelu_gflop_count<float>(M, N);
            break;
        case hipsparselt_activation_type::gelu:
            flops += gelu_gflop_count<float>(M, N, arg.activation_arg1!=1);
            break;
        case hipsparselt_activation_type::abs:
            flops += abs_gflop_count<float>(M, N);
            break;
        case hipsparselt_activation_type::leakyrelu:
            flops += leakyrelu_gflop_count<float>(M, N);
            break;
        case hipsparselt_activation_type::sigmoid:
            flops += sigmoid_gflop_count<float>(M, N);
            break;
        case hipsparselt_activation_type::tanh:
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
            ArgumentModel<argument_param>{}.log_args<float>(hipsparselt_cout,
                                                            arg,
                                                            gpu_time_used,
                                                            flops,
                                                            ArgumentLogging::NA_value,
                                                            cpu_time_used,
                                                            hipsparselt_error);
        else
            ArgumentModel<argument_param_nb>{}.log_args<float>(hipsparselt_cout,
                                                               arg,
                                                               gpu_time_used,
                                                               flops,
                                                               ArgumentLogging::NA_value,
                                                               cpu_time_used,
                                                               hipsparselt_error);
    }
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}

template <typename Ti,
          typename To,
          typename Tc,
          hipsparselt_batch_type btype = hipsparselt_batch_type::none>
void testing_aux_plan_assign(const Arguments& arg)
{
    hipsparseOperation_t transA = char_to_hipsparselt_operation(arg.transA);
    hipsparseOperation_t transB = char_to_hipsparselt_operation(arg.transB);

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
    double                   hipsparselt_error = 0.0;
    bool                     HMM               = arg.HMM;
    hipsparselt_local_handle handle{arg};
    hipStream_t              stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    int64_t A_row = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? M : K;
    int64_t A_col = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K : M;
    int64_t B_row = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K : N;
    int64_t B_col = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? N : K;

    int64_t stride_1_a = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? 1 : lda;
    int64_t stride_2_a = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? lda : 1;

    constexpr bool do_batched         = (btype == hipsparselt_batch_type::batched);
    constexpr bool do_strided_batched = (btype == hipsparselt_batch_type::strided_batched);
    int            num_batches        = 5;
    int64_t        stride_a           = lda * A_col;
    int64_t        stride_b           = ldb * B_col;
    int64_t        stride_c           = ldc * N;
    int64_t        stride_d           = ldd * N;

    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured,
                                     handle,
                                     A_row,
                                     A_col,
                                     lda,
                                     arg.a_type,
                                     HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, B_row, B_col, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);

    hipsparseStatus_t eStatus
        = expected_hipsparse_status_of_matrix_size(arg.a_type, A_row, A_col, lda);
    EXPECT_HIPSPARSE_STATUS(matA.status(), eStatus);
    if(eStatus != HIPSPARSE_STATUS_SUCCESS)
        return;

    eStatus = expected_hipsparse_status_of_matrix_size(arg.b_type, B_row, B_col, ldb);
    EXPECT_HIPSPARSE_STATUS(matB.status(), eStatus);
    if(eStatus != HIPSPARSE_STATUS_SUCCESS)
        return;

    eStatus = expected_hipsparse_status_of_matrix_size(arg.c_type, M, N, ldc);
    EXPECT_HIPSPARSE_STATUS(matC.status(), eStatus);
    if(eStatus != HIPSPARSE_STATUS_SUCCESS)
        return;

    eStatus = expected_hipsparse_status_of_matrix_size(arg.d_type, M, N, ldd);
    EXPECT_HIPSPARSE_STATUS(matD.status(), eStatus);
    if(eStatus != HIPSPARSE_STATUS_SUCCESS)
        return;

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

    hipsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);

    // CHECK mat in matmul is a reference, hipsparseltMatmul() will use this new batch size.
    int new_num_batches = 2;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, matA, HIPSPARSELT_MAT_NUM_BATCHES, &new_num_batches, sizeof(int)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, matB, HIPSPARSELT_MAT_NUM_BATCHES, &new_num_batches, sizeof(int)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, matC, HIPSPARSELT_MAT_NUM_BATCHES, &new_num_batches, sizeof(int)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, matD, HIPSPARSELT_MAT_NUM_BATCHES, &new_num_batches, sizeof(int)),
        HIPSPARSE_STATUS_SUCCESS);

    int   activation_on   = 1;
    float activation_arg2 = 2.f;
    float activation_arg1 = 0.f;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle,
                                          matmul,
                                          HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND,
                                          &activation_arg2,
                                          sizeof(float)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle,
                                          matmul,
                                          HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD,
                                          &activation_arg1,
                                          sizeof(float)),
        HIPSPARSE_STATUS_SUCCESS);

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulDescSetAttribute(handle,
                                                              matmul,
                                                              HIPSPARSELT_MATMUL_ACTIVATION_RELU,
                                                              &activation_on,
                                                              sizeof(activation_on)),
                            HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);

    size_t workspace_size = 0, compressed_size = 0, compress_buffer_size = 0;
    int    search_iters  = 10;
    int    config_max_id = 0;
    hipsparseLtMatmulAlgGetAttribute(
        handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &config_max_id, sizeof(int));
    for(int i = 0; i < config_max_id; i++)
    {
        hipsparseLtMatmulAlgSetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &i, sizeof(int));
        hipsparselt_local_matmul_plan plan_tmp(handle, matmul, alg_sel);
        size_t                        ws = 0;
        EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(handle, plan_tmp, &ws),
                                HIPSPARSE_STATUS_SUCCESS);
        workspace_size = std::max(workspace_size, ws);
    }
    hipsparseLtMatmulAlgSetAttribute(
        handle, alg_sel, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &search_iters, sizeof(int));

    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMACompressedSize(handle, plan, &compressed_size, &compress_buffer_size),
        HIPSPARSE_STATUS_SUCCESS);

    const size_t size_A = stride_a == 0 ? lda * A_col * num_batches : stride_a * num_batches;
    const size_t size_A_pruned_copy = arg.unit_check || arg.norm_check || arg.timing ? size_A : 0;

    const size_t size_B      = stride_b == 0 ? ldb * B_col * num_batches : stride_b * num_batches;
    const size_t size_C      = stride_c == 0 ? ldc * N * num_batches : stride_c * num_batches;
    const size_t size_D      = stride_d == 0 ? ldd * N * num_batches : stride_d * num_batches;
    const size_t size_D_copy = size_D;
    const size_t size_D_act_copy = activation_on ? size_D_copy : 0;

    // allocate memory on device
    device_vector<Ti>            dA(size_A, 1, HMM);
    device_vector<Ti>            dB(size_B, 1, HMM);
    device_vector<To>            dC(size_C, 1, HMM);
    device_vector<To>            dD(size_D, 1, HMM);
    device_vector<unsigned char> dA_compressd(compressed_size, 1, HMM);
    device_vector<unsigned char> dA_compressBuffer(compress_buffer_size, 1, HMM);
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

    hipsparselt_seedrand();

    // Initial Data on CPU
    if(arg.alpha_isnan<Tc>())
    {
        hipsparselt_init_nan<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
        hipsparselt_init_nan<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
    }
    else
    {
        if(arg.initialization == hipsparselt_initialization::rand_int)
        {
            hipsparselt_init<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
            hipsparselt_init_alternating_sign<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
        }
        else if(arg.initialization == hipsparselt_initialization::trig_float)
        {
            hipsparselt_init_sin<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
            hipsparselt_init_cos<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
        }
        else if(arg.initialization == hipsparselt_initialization::hpl)
        {
            hipsparselt_init_hpl<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
            hipsparselt_init_hpl<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
        }
        else if(arg.initialization == hipsparselt_initialization::special)
        {
            hipsparselt_init_alt_impl_big<Ti>(hA, A_row, A_col, lda, num_batches);
            hipsparselt_init_alt_impl_small<Ti>(hB, B_row, B_col, ldb, num_batches);
        }
    }

    if(arg.beta_isnan<Tc>())
    {
        hipsparselt_init_nan<To>(hC, M, N, ldc, stride_c, num_batches);
    }
    else
    {
        if(arg.initialization == hipsparselt_initialization::rand_int)
            hipsparselt_init<To>(hC, M, N, ldc, stride_c, num_batches);
        else if(arg.initialization == hipsparselt_initialization::trig_float)
            hipsparselt_init_sin<To>(hC, M, N, ldc, stride_c, num_batches);
        else if(arg.initialization == hipsparselt_initialization::hpl)
            hipsparselt_init_hpl<To>(hC, M, N, ldc, stride_c, num_batches);
        else if(arg.initialization == hipsparselt_initialization::special)
            hipsparselt_init<To>(hC, M, N, ldc, stride_c, num_batches);
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
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMAPrune(handle, matmul, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
        HIPSPARSE_STATUS_SUCCESS);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMACompress(handle, plan, dA, dA_compressd, dA_compressBuffer, stream),
        HIPSPARSE_STATUS_SUCCESS);

    {
        auto check
            = [&](auto& plan, float activation_arg1, float activation_arg2, int num_batches) {
                  CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                  CHECK_HIP_ERROR(hA_pruned.transfer_from(dA));
                  EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmul(handle,
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
                                          HIPSPARSE_STATUS_SUCCESS);
                  // now we can recycle gold matrix for reference purposes
                  if(arg.timing)
                  {
                      cpu_time_used = get_time_us_no_sync();
                  }

#define activation_param \
    M, N, ldd, hD_gold_act + pos, hD_gold + pos, activation_arg1, activation_arg2
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
                          activation(activation_param, ::_clippedrelu);
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
#undef activation_param
                  // fetch GPU
                  CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                  CHECK_HIP_ERROR(hD_1.transfer_from(dD));
                  unit_check_general<To>(M, N, ldd, stride_d, hD_gold, hD_1, num_batches);
              };
        check(plan, activation_arg1, activation_arg2, new_num_batches);

        // CHECK mat in plsn is a copy, hipsparseltMatmul() will use old batch size.
        int new_num_batches2 = 5;
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matA, HIPSPARSELT_MAT_NUM_BATCHES, &new_num_batches2, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matB, HIPSPARSELT_MAT_NUM_BATCHES, &new_num_batches2, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matC, HIPSPARSELT_MAT_NUM_BATCHES, &new_num_batches2, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matD, HIPSPARSELT_MAT_NUM_BATCHES, &new_num_batches2, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        unit_check_general<To>(M, N, ldd, stride_d, hD_gold, hD_1, new_num_batches);

        // CHECK matmul in plan is a copy, modify the activation value outside will not impact the result.
        float new_activation_arg2 = 10.f;
        float new_activation_arg1 = 5.f;
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle,
                                              matmul,
                                              HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND,
                                              &new_activation_arg2,
                                              sizeof(float)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle,
                                              matmul,
                                              HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD,
                                              &new_activation_arg1,
                                              sizeof(float)),
            HIPSPARSE_STATUS_SUCCESS);
        check(plan, activation_arg1, activation_arg2, new_num_batches);

        //CHECK alg_sel in plan is a reference and plans will use same alg_sel.
        if(0)
        {
            EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulSearch(handle,
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
                                    HIPSPARSE_STATUS_SUCCESS);

            hipsparseLtMatmulPlan_t plan2 = plan;
            search_iters                  = 2;
            hipsparseLtMatmulAlgSetAttribute(
                handle, alg_sel, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &search_iters, sizeof(int));
            EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulSearch(handle,
                                                            &plan2,
                                                            &h_alpha,
                                                            dA_compressd,
                                                            dB,
                                                            &h_beta,
                                                            dC,
                                                            dD,
                                                            dWorkspace,
                                                            &stream,
                                                            1),
                                    HIPSPARSE_STATUS_SUCCESS);
        }
    }

    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}
