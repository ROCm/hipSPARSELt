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
#include <omp.h>

template <typename Ti, typename Tc>
inline Tc norm1(Ti a, Ti b)
{
    auto ac = static_cast<Tc>(a);
    auto bc = static_cast<Tc>(b);

    return static_cast<Tc>(abs(ac) + abs(bc));
}

template <typename Ti, typename Tc>
inline Tc norm1(Ti a, Ti b, Ti c, Ti d, Ti e, Ti f, Ti g, Ti h)
{
    auto ac = static_cast<Tc>(a);
    auto bc = static_cast<Tc>(b);
    auto cc = static_cast<Tc>(c);
    auto dc = static_cast<Tc>(d);
    auto ec = static_cast<Tc>(e);
    auto fc = static_cast<Tc>(f);
    auto gc = static_cast<Tc>(g);
    auto hc = static_cast<Tc>(h);
    return static_cast<Tc>(abs(ac) + abs(bc) + abs(cc) + abs(dc) + abs(ec) + abs(fc) + abs(gc)
                           + abs(hc));
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
#pragma omp parallel for
        for(int i = 0; i < m; i++)
        {
#pragma omp parallel for
            for(int j = 0; j < n; j += 4)
            {
                size_t pos[4];
                for(int k = 0; k < 4; k++)
                {
                    pos[k] = b * stride_b + i * stride1 + (j + k) * stride2;
                }

                auto max_norm1 = static_cast<Tc>(-1.0f);
                int  pos_a, pos_b;
                for(int a = 0; a < 4; a++)
                {
                    for(int b = a + 1; b < 4; b++)
                    {
                        auto norm1_v = norm1<Ti, double>(in[pos[a]], in[pos[b]]);
                        if(norm1_v > max_norm1)
                        {
                            pos_a     = a;
                            pos_b     = b;
                            max_norm1 = norm1_v;
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

typedef std::pair<int, int> pos_t;

// clang-format off
const static pos_t pos_patterns[90][4] = {
    {pos_t(0, 2), pos_t(0, 2), pos_t(1, 3), pos_t(1, 3)},  //ROW#(COL#,COL#), 0(0,2), 1(0,2), 2(1,3), 3(1,3)
    {pos_t(0, 2), pos_t(0, 3), pos_t(1, 3), pos_t(1, 2)},
    {pos_t(0, 2), pos_t(0, 3), pos_t(1, 2), pos_t(1, 3)},
    {pos_t(0, 2), pos_t(0, 1), pos_t(1, 3), pos_t(2, 3)},
    {pos_t(0, 2), pos_t(0, 1), pos_t(2, 3), pos_t(1, 3)},
    {pos_t(0, 2), pos_t(1, 3), pos_t(0, 2), pos_t(1, 3)},
    {pos_t(0, 2), pos_t(1, 3), pos_t(0, 3), pos_t(1, 2)},
    {pos_t(0, 2), pos_t(1, 3), pos_t(0, 1), pos_t(2, 3)},
    {pos_t(0, 2), pos_t(1, 3), pos_t(1, 3), pos_t(0, 2)},
    {pos_t(0, 2), pos_t(1, 3), pos_t(1, 2), pos_t(0, 3)},
    {pos_t(0, 2), pos_t(1, 3), pos_t(2, 3), pos_t(0, 1)},
    {pos_t(0, 2), pos_t(1, 2), pos_t(0, 3), pos_t(1, 3)},
    {pos_t(0, 2), pos_t(1, 2), pos_t(1, 3), pos_t(0, 3)},
    {pos_t(0, 2), pos_t(2, 3), pos_t(0, 1), pos_t(1, 3)},
    {pos_t(0, 2), pos_t(2, 3), pos_t(1, 3), pos_t(0, 1)},
    {pos_t(0, 3), pos_t(0, 2), pos_t(1, 3), pos_t(1, 2)},
    {pos_t(0, 3), pos_t(0, 2), pos_t(1, 2), pos_t(1, 3)},
    {pos_t(0, 3), pos_t(0, 3), pos_t(1, 2), pos_t(1, 2)},
    {pos_t(0, 3), pos_t(0, 1), pos_t(1, 2), pos_t(2, 3)},
    {pos_t(0, 3), pos_t(0, 1), pos_t(2, 3), pos_t(1, 2)},
    {pos_t(0, 3), pos_t(1, 3), pos_t(0, 2), pos_t(1, 2)},
    {pos_t(0, 3), pos_t(1, 3), pos_t(1, 2), pos_t(0, 2)},
    {pos_t(0, 3), pos_t(1, 2), pos_t(0, 2), pos_t(1, 3)},
    {pos_t(0, 3), pos_t(1, 2), pos_t(0, 3), pos_t(1, 2)},
    {pos_t(0, 3), pos_t(1, 2), pos_t(0, 1), pos_t(2, 3)},
    {pos_t(0, 3), pos_t(1, 2), pos_t(1, 3), pos_t(0, 2)},
    {pos_t(0, 3), pos_t(1, 2), pos_t(1, 2), pos_t(0, 3)},
    {pos_t(0, 3), pos_t(1, 2), pos_t(2, 3), pos_t(0, 1)},
    {pos_t(0, 3), pos_t(2, 3), pos_t(0, 1), pos_t(1, 2)},
    {pos_t(0, 3), pos_t(2, 3), pos_t(1, 2), pos_t(0, 1)},
    {pos_t(0, 1), pos_t(0, 2), pos_t(1, 3), pos_t(2, 3)},
    {pos_t(0, 1), pos_t(0, 2), pos_t(2, 3), pos_t(1, 3)},
    {pos_t(0, 1), pos_t(0, 3), pos_t(1, 2), pos_t(2, 3)},
    {pos_t(0, 1), pos_t(0, 3), pos_t(2, 3), pos_t(1, 2)},
    {pos_t(0, 1), pos_t(0, 1), pos_t(2, 3), pos_t(2, 3)},
    {pos_t(0, 1), pos_t(1, 3), pos_t(0, 2), pos_t(2, 3)},
    {pos_t(0, 1), pos_t(1, 3), pos_t(2, 3), pos_t(0, 2)},
    {pos_t(0, 1), pos_t(1, 2), pos_t(0, 3), pos_t(2, 3)},
    {pos_t(0, 1), pos_t(1, 2), pos_t(2, 3), pos_t(0, 3)},
    {pos_t(0, 1), pos_t(2, 3), pos_t(0, 2), pos_t(1, 3)},
    {pos_t(0, 1), pos_t(2, 3), pos_t(0, 3), pos_t(1, 2)},
    {pos_t(0, 1), pos_t(2, 3), pos_t(0, 1), pos_t(2, 3)},
    {pos_t(0, 1), pos_t(2, 3), pos_t(1, 3), pos_t(0, 2)},
    {pos_t(0, 1), pos_t(2, 3), pos_t(1, 2), pos_t(0, 3)},
    {pos_t(0, 1), pos_t(2, 3), pos_t(2, 3), pos_t(0, 1)},
    {pos_t(1, 3), pos_t(0, 2), pos_t(0, 2), pos_t(1, 3)},
    {pos_t(1, 3), pos_t(0, 2), pos_t(0, 3), pos_t(1, 2)},
    {pos_t(1, 3), pos_t(0, 2), pos_t(0, 1), pos_t(2, 3)},
    {pos_t(1, 3), pos_t(0, 2), pos_t(1, 3), pos_t(0, 2)},
    {pos_t(1, 3), pos_t(0, 2), pos_t(1, 2), pos_t(0, 3)},
    {pos_t(1, 3), pos_t(0, 2), pos_t(2, 3), pos_t(0, 1)},
    {pos_t(1, 3), pos_t(0, 3), pos_t(0, 2), pos_t(1, 2)},
    {pos_t(1, 3), pos_t(0, 3), pos_t(1, 2), pos_t(0, 2)},
    {pos_t(1, 3), pos_t(0, 1), pos_t(0, 2), pos_t(2, 3)},
    {pos_t(1, 3), pos_t(0, 1), pos_t(2, 3), pos_t(0, 2)},
    {pos_t(1, 3), pos_t(1, 3), pos_t(0, 2), pos_t(0, 2)},
    {pos_t(1, 3), pos_t(1, 2), pos_t(0, 2), pos_t(0, 3)},
    {pos_t(1, 3), pos_t(1, 2), pos_t(0, 3), pos_t(0, 2)},
    {pos_t(1, 3), pos_t(2, 3), pos_t(0, 2), pos_t(0, 1)},
    {pos_t(1, 3), pos_t(2, 3), pos_t(0, 1), pos_t(0, 2)},
    {pos_t(1, 2), pos_t(0, 2), pos_t(0, 3), pos_t(1, 3)},
    {pos_t(1, 2), pos_t(0, 2), pos_t(1, 3), pos_t(0, 3)},
    {pos_t(1, 2), pos_t(0, 3), pos_t(0, 2), pos_t(1, 3)},
    {pos_t(1, 2), pos_t(0, 3), pos_t(0, 3), pos_t(1, 2)},
    {pos_t(1, 2), pos_t(0, 3), pos_t(0, 1), pos_t(2, 3)},
    {pos_t(1, 2), pos_t(0, 3), pos_t(1, 3), pos_t(0, 2)},
    {pos_t(1, 2), pos_t(0, 3), pos_t(1, 2), pos_t(0, 3)},
    {pos_t(1, 2), pos_t(0, 3), pos_t(2, 3), pos_t(0, 1)},
    {pos_t(1, 2), pos_t(0, 1), pos_t(0, 3), pos_t(2, 3)},
    {pos_t(1, 2), pos_t(0, 1), pos_t(2, 3), pos_t(0, 3)},
    {pos_t(1, 2), pos_t(1, 3), pos_t(0, 2), pos_t(0, 3)},
    {pos_t(1, 2), pos_t(1, 3), pos_t(0, 3), pos_t(0, 2)},
    {pos_t(1, 2), pos_t(1, 2), pos_t(0, 3), pos_t(0, 3)},
    {pos_t(1, 2), pos_t(2, 3), pos_t(0, 3), pos_t(0, 1)},
    {pos_t(1, 2), pos_t(2, 3), pos_t(0, 1), pos_t(0, 3)},
    {pos_t(2, 3), pos_t(0, 2), pos_t(0, 1), pos_t(1, 3)},
    {pos_t(2, 3), pos_t(0, 2), pos_t(1, 3), pos_t(0, 1)},
    {pos_t(2, 3), pos_t(0, 3), pos_t(0, 1), pos_t(1, 2)},
    {pos_t(2, 3), pos_t(0, 3), pos_t(1, 2), pos_t(0, 1)},
    {pos_t(2, 3), pos_t(0, 1), pos_t(0, 2), pos_t(1, 3)},
    {pos_t(2, 3), pos_t(0, 1), pos_t(0, 3), pos_t(1, 2)},
    {pos_t(2, 3), pos_t(0, 1), pos_t(0, 1), pos_t(2, 3)},
    {pos_t(2, 3), pos_t(0, 1), pos_t(1, 3), pos_t(0, 2)},
    {pos_t(2, 3), pos_t(0, 1), pos_t(1, 2), pos_t(0, 3)},
    {pos_t(2, 3), pos_t(0, 1), pos_t(2, 3), pos_t(0, 1)},
    {pos_t(2, 3), pos_t(1, 3), pos_t(0, 2), pos_t(0, 1)},
    {pos_t(2, 3), pos_t(1, 3), pos_t(0, 1), pos_t(0, 2)},
    {pos_t(2, 3), pos_t(1, 2), pos_t(0, 3), pos_t(0, 1)},
    {pos_t(2, 3), pos_t(1, 2), pos_t(0, 1), pos_t(0, 3)},
    {pos_t(2, 3), pos_t(2, 3), pos_t(0, 1), pos_t(0, 1)},
};
// clang-format on

template <typename Ti, typename Tc>
void prune_tile(const Ti* in,
                Ti*       out,
                int64_t   m,
                int64_t   n,
                int64_t   stride1,
                int64_t   stride2,
                int       num_batches,
                int64_t   stride_b)
{
    for(int b = 0; b < num_batches; b++)
#pragma omp parallel for
        for(int i = 0; i < m; i += 4)
        {
#pragma omp parallel for
            for(int j = 0; j < n; j += 4)
            {

                Ti value[16];
                for(int x = 0; x < 4; x++)
                {
#pragma omp parallel for
                    for(int y = 0; y < 4; y++)
                    {
                        int64_t pos = b * stride_b + (i + x) * stride1 + (j + y) * stride2;

                        if((i + x) < m && (j + y) < n)
                        {
                            value[x * 4 + y] = in[pos];
                        }
                        else
                            value[x * 4 + y] = static_cast<Ti>(0.0f);
                    }
                }

                float norm_res[90];
                int   max_norm_idx = 0;
                float max_norm     = -1;

#pragma omp parallel for
                for(int pi = 0; pi < 90; pi++)
                {
                    auto pos_pattern = pos_patterns[pi];
                    norm_res[pi]     = norm1<Ti, double>(value[pos_pattern[0].first],
                                                     value[pos_pattern[0].second],
                                                     value[1 * 4 + pos_pattern[1].first],
                                                     value[1 * 4 + pos_pattern[1].second],
                                                     value[2 * 4 + pos_pattern[2].first],
                                                     value[2 * 4 + pos_pattern[2].second],
                                                     value[3 * 4 + pos_pattern[3].first],
                                                     value[3 * 4 + pos_pattern[3].second]);
                }
                for(int pi = 0; pi < 90; pi++)
                {
                    if(max_norm < norm_res[pi])
                    {
                        max_norm     = norm_res[pi];
                        max_norm_idx = pi;
                    }
                }

                auto pos_s = pos_patterns[max_norm_idx];
                for(int x = 0; x < 4; x++)
                {
#pragma omp parallel for
                    for(int y = 0; y < 4; y++)
                    {
                        if((i + x) < m && (j + y) < n)
                        {
                            int64_t pos = b * stride_b + (i + x) * stride1 + (j + y) * stride2;
                            if(pos_s[x].first == y || pos_s[x].second == y)
                            {
                                if(in != out)
                                    out[pos] = in[pos];
                            }
                            else
                                out[pos] = static_cast<Ti>(0.0f);
                        }
                    }
                }
            }
        }
}

template <typename Ti, typename Tc>
void tile_4x4_norm1(const Ti* in,
                    Tc*       out,
                    int64_t   m,
                    int64_t   n,
                    int64_t   stride1,
                    int64_t   stride2,
                    int       num_batches,
                    int64_t   stride_b)
{
    constexpr Tc TC_ZERO = static_cast<Tc>(0);
    for(int b = 0; b < num_batches; b++)
#pragma omp parallel for
        for(int i = 0; i < m; i += 4)
        {
#pragma omp parallel for
            for(int j = 0; j < n; j += 4)
            {

                Tc value = TC_ZERO;
                for(int x = 0; x < 4; x++)
                {
#pragma omp parallel for
                    for(int y = 0; y < 4; y++)
                    {
                        int64_t pos = b * stride_b + (i + x) * stride1 + (j + y) * stride2;

                        if((i + x) < m && (j + y) < n)
                        {
                            Tc tmp = static_cast<Tc>(in[pos]);
                            value += (tmp >= TC_ZERO ? tmp : -tmp);
                        }
                    }
                }
                for(int x = 0; x < 4; x++)
                {
#pragma omp parallel for
                    for(int y = 0; y < 4; y++)
                    {
                        int64_t pos = b * stride_b + (i + x) * stride1 + (j + y) * stride2;

                        if((i + x) < m && (j + y) < n)
                        {
                            if(x == 0 && y == 0)
                            {
                                out[pos] = value;
                            }
                            else
                            {
                                out[pos] = TC_ZERO;
                            }
                        }
                    }
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

    hipStream_t stream = nullptr;

    // test version 1
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMAPrune(nullptr, matmul, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
        HIPSPARSE_STATUS_NOT_INITIALIZED);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMAPrune(handle, nullptr, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
        HIPSPARSE_STATUS_NOT_INITIALIZED);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMAPrune(handle, matmul, dA, nullptr, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMAPrune(handle, matmul, nullptr, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
        HIPSPARSE_STATUS_INVALID_VALUE);

    // test version 2
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMAPrune2(
            nullptr, matA, true, transA, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
        HIPSPARSE_STATUS_NOT_INITIALIZED);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMAPrune2(
            handle, nullptr, true, transA, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
        HIPSPARSE_STATUS_NOT_INITIALIZED);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMAPrune2(
            handle, matA, false, transA, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
        HIPSPARSE_STATUS_INTERNAL_ERROR);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMAPrune2(
            handle, matA, true, transA, dA, nullptr, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
        HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtSpMMAPrune2(
            handle, matA, true, transA, nullptr, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

template <typename Ti,
          typename To,
          typename Tc,
          hipsparselt_batch_type btype = hipsparselt_batch_type::none>
void testing_prune(const Arguments& arg)
{
    int run_version = 1;

    if(strstr(arg.name, "prune2") != nullptr)
        run_version = 2;

    hipsparseLtPruneAlg_t prune_algo = hipsparseLtPruneAlg_t(arg.prune_algo);

    constexpr bool do_batched         = (btype == hipsparselt_batch_type::batched);
    constexpr bool do_strided_batched = (btype == hipsparselt_batch_type::strided_batched);

    void (*prune_cpu)(const Ti* in,
                      Ti*       out,
                      int64_t   m,
                      int64_t   n,
                      int64_t   stride1,
                      int64_t   stride2,
                      int       num_batches,
                      int64_t   stride_b);
    prune_cpu
        = prune_algo == HIPSPARSELT_PRUNE_SPMMA_STRIP ? prune_strip<Ti, Tc> : prune_tile<Ti, Tc>;

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
    gpu_time_used = cpu_time_used              = 0.0;
    double                   hipsparselt_error = 0.0;
    bool                     HMM               = arg.HMM;
    hipsparselt_local_handle handle{arg};
    hipStream_t              stream;
    hipStreamCreate(&stream);

    int64_t A_row = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? M : K;
    int64_t A_col = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K : M;
    int64_t B_row = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K : N;
    int64_t B_col = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? N : K;

    int64_t stride_1_a = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? 1 : lda;
    int64_t stride_2_a = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? lda : 1;

    int     num_batches = (do_batched || do_strided_batched ? arg.batch_count : 1);
    int64_t stride_a    = do_strided_batched ? arg.stride_a : lda * A_col;
    int64_t stride_b    = do_strided_batched ? arg.stride_b : ldb * B_col;
    int64_t stride_c    = do_strided_batched ? arg.stride_c : ldc * M;
    int64_t stride_d    = do_strided_batched ? arg.stride_c : ldd * M;

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
    bool invalid_size_d = ldd < M;
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
    if(invalid_size_d)
    {
        EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_INVALID_VALUE);

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

    const size_t size_A           = stride_a == 0 ? A_col * lda : num_batches * stride_a;
    const size_t size_A_copy      = arg.unit_check || arg.norm_check ? size_A : 0;
    const size_t size_A_norm_copy = prune_algo == HIPSPARSELT_PRUNE_SPMMA_TILE ? size_A_copy : 0;

    // allocate memory on device
    device_vector<Ti> dA(size_A, 1, HMM);
    device_vector<Ti> dA_pruned(size_A, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_pruned.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ti>    hA(size_A);
    host_vector<Ti>    hA_gold(size_A_copy);
    host_vector<Ti>    hA_1(size_A_copy);
    host_vector<float> hA_gold_norm(size_A_norm_copy);
    host_vector<float> hA_1_norm(size_A_norm_copy);

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

    if(arg.unit_check || arg.norm_check)
    {
        if(run_version == 1)
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMAPrune(handle, matmul, dA, dA_pruned, prune_algo, stream),
                HIPSPARSE_STATUS_SUCCESS);
        else if(run_version == 2)
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMAPrune2(
                    handle, matA, true, transA, dA, dA_pruned, prune_algo, stream),
                HIPSPARSE_STATUS_SUCCESS);

        hipStreamSynchronize(stream);
        CHECK_HIP_ERROR(hA_1.transfer_from(dA_pruned));

        //print_strided_batched("device", hA_1.data(), M, K, num_batches, stride_1_a, stride_2_a, stride_a);

        device_vector<int> d_valid(1, 1, HMM);
        int                h_valid = 0;
        //check the pruned matrix is sparisty 50 or not.
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck(handle, matmul, dA_pruned, d_valid, stream),
            HIPSPARSE_STATUS_SUCCESS);
        CHECK_HIP_ERROR(
            hipMemcpyAsync(&h_valid, d_valid, sizeof(int), hipMemcpyDeviceToHost, stream));
        hipStreamSynchronize(stream);
        CHECK_SUCCESS(h_valid == 0);

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

        if(prune_algo == HIPSPARSELT_PRUNE_SPMMA_TILE)
        {
            tile_4x4_norm1<Ti, float>(
                hA_1, hA_1_norm, M, K, stride_1_a, stride_2_a, num_batches, stride_a);
            tile_4x4_norm1<Ti, float>(
                hA_gold, hA_gold_norm, M, K, stride_1_a, stride_2_a, num_batches, stride_a);

            // check host error and norm
            if(arg.unit_check)
            {
                unit_check_general<float>(
                    A_row, A_col, lda, stride_a, hA_gold_norm, hA_1_norm, num_batches);
            }

            if(arg.norm_check)
            {
                hipsparselt_error = unit_check_diff<float>(
                    A_row, A_col, lda, stride_a, hA_gold_norm, hA_1_norm, num_batches);
            }
        }
        else if(prune_algo == HIPSPARSELT_PRUNE_SPMMA_STRIP)
        {
            // check host error and norm
            if(arg.unit_check)
            {
                unit_check_general<Ti>(A_row, A_col, lda, stride_a, hA_gold, hA_1, num_batches);
            }

            if(arg.norm_check)
            {
                hipsparselt_error
                    = unit_check_diff<Ti>(A_row, A_col, lda, stride_a, hA_gold, hA_1, num_batches);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
        {
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMAPrune(handle, matmul, dA, dA_pruned, prune_algo, stream),
                HIPSPARSE_STATUS_SUCCESS);
        }

        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMAPrune(handle, matmul, dA, dA_pruned, prune_algo, stream),
                HIPSPARSE_STATUS_SUCCESS);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        double (*gflop_count)(int64_t m, int64_t n);
        gflop_count = (prune_algo == HIPSPARSELT_PRUNE_SPMMA_STRIP) ? prune_strip_gflop_count<Ti>
                                                                    : prune_tile_gflop_count<Ti>;

        ArgumentModel<e_transA, e_transB, e_M, e_N, e_K, e_lda, e_stride_a, e_batch_count>{}
            .log_args<float>(hipsparselt_cout,
                             arg,
                             gpu_time_used,
                             gflop_count(M, K),
                             ArgumentLogging::NA_value,
                             cpu_time_used,
                             hipsparselt_error);
    }
}
