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

#include "definitions.h"
#include "handle.h"
#include "rocsparselt.h"
#include "utility.hpp"

#include "hipsparselt_ostream.hpp"
#include <hip/hip_runtime_api.h>

template <typename Ti, int SG0I, int SG1J, int TT0I, int TT1J>
__global__ void prune_check_kernel(const Ti* in,
                                   int*      out,
                                   int64_t   m,
                                   int64_t   n,
                                   int64_t   stride1,
                                   int64_t   stride2,
                                   int       num_batches,
                                   int64_t   batch_stride,
                                   int64_t   sizes)
{
    constexpr unsigned int MT0I = SG0I * TT0I;
    constexpr unsigned int MT1J = SG1J * TT1J;

    unsigned int serial = hc_get_workitem_id(0);
    unsigned int sg0I   = serial % SG0I;
    unsigned int sg1J   = serial / SG0I;

    int64_t stride = sg0I * stride1 + sg1J * TT1J * stride2;

    unsigned int wg0I    = hc_get_group_id(0);
    unsigned int wg1J    = hc_get_group_id(1);
    unsigned int batchId = hc_get_group_id(2);

    if((MT1J * wg1J + sg1J * TT1J) >= n || (MT0I * wg0I + sg0I * TT0I) >= m)
        return;

    int64_t wg_stride = MT1J * wg1J * stride2 + MT0I * wg0I * stride1;
    int64_t b_stride  = batchId * batch_stride;

    int64_t globalReadOffset = b_stride + wg_stride + stride;

    for(int i = 0; i < TT0I; i++)
    {
        for(int j = 0; j < TT1J; j += 4)
        {
            if(*out)
                return;

            int64_t offset = i * stride1 + j * stride2;
            int     nz     = 0;

#pragma unroll
            for(int k = 0; k < 4; k++)
            {
                int64_t pos = globalReadOffset + offset + k * stride2;
                if(pos < sizes)
                {
                    if(in[pos] != static_cast<Ti>(0.0))
                    {
                        nz++;
                    }
                }
            }
            if(nz > 2)
            {
                *out = 1;
                return;
            }
        }
    }
}

template <typename Ti, typename Tc>
__host__ __device__ inline Tc norm1(Ti a, Ti b)
{
    Tc ac = static_cast<Tc>(a);
    Tc bc = static_cast<Tc>(b);
    return static_cast<Tc>(abs(ac) + abs(bc));
}

template <typename T>
__host__ __device__ inline T sum(T a, T b, T c, T d, T e, T f, T g, T h)
{
    return a + b + c + d + e + f + g + h;
}

template <typename T>
__host__ __device__ inline T acc_sum8(T* v, uint8_t* p, int v_offset, int p_offset)
{
    return sum<T>(v[v_offset + p[p_offset]],
                  v[v_offset + p[p_offset + 1]],
                  v[v_offset + 1 * 4 + p[p_offset + 2]],
                  v[v_offset + 1 * 4 + p[p_offset + 3]],
                  v[v_offset + 2 * 4 + p[p_offset + 4]],
                  v[v_offset + 2 * 4 + p[p_offset + 5]],
                  v[v_offset + 3 * 4 + p[p_offset + 6]],
                  v[v_offset + 3 * 4 + p[p_offset + 7]]);
}

template <typename T, bool InPlace, typename = void>
__host__ __device__ inline void prune_if(bool prune, T* a, T b)
{
    if(prune)
        *a = static_cast<T>(0.0f);
    else
        *a = b;
}

template <typename T, bool InPlace, std::enable_if_t<InPlace>>
__host__ __device__ inline void prune_if(bool prune, T* a, T b)
{
    if(prune)
        *a = static_cast<T>(0.0f);
}

template <typename Ti, typename Tc, int SG0I, int SG1J, int TT0I, int TT1J, bool InPlace>
__global__ void prune_strip_kernel(const Ti* in,
                                   Ti*       out,
                                   int64_t   m,
                                   int64_t   n,
                                   int64_t   stride1,
                                   int64_t   stride2,
                                   int       num_batches,
                                   int64_t   batch_stride,
                                   int64_t   sizes)
{
    constexpr unsigned int MT0I = SG0I * TT0I;
    constexpr unsigned int MT1J = SG1J * TT1J;

    unsigned int serial = hc_get_workitem_id(0);
    unsigned int sg0I   = serial % SG0I;
    unsigned int sg1J   = serial / SG0I;
    int64_t      stride = sg0I * stride1 + sg1J * TT1J * stride2;

    unsigned int wg0I    = hc_get_group_id(0);
    unsigned int wg1J    = hc_get_group_id(1);
    unsigned int batchId = hc_get_group_id(2);

    if((MT1J * wg1J + sg1J * TT1J) >= n || (MT0I * wg0I + sg0I * TT0I) >= m)
        return;

    int64_t wg_stride = MT1J * wg1J * stride2 + MT0I * wg0I * stride1;
    int64_t b_stride  = batchId * batch_stride;

    int64_t globalReadOffset = b_stride + wg_stride + stride;

    for(int i = 0; i < TT0I; i++)
    {
        for(int j = 0; j < TT1J; j += 4)
        {
            int64_t offset = globalReadOffset + i * stride1 + j * stride2;
            Ti      values[4];
#pragma unroll
            for(int k = 0; k < 4; k++)
            {
                int64_t pos = offset + k * stride2;
                if(pos >= sizes)
                    values[k] = static_cast<Ti>(0.0f);
                else
                    values[k] = in[pos];
            }

            auto max_norm1 = static_cast<Tc>(-1.0);
            int  pos_a = 0, pos_b = 0;

#pragma unroll
            for(int a = 0; a < 4; a++)
            {
                for(int b = a + 1; b < 4; b++)
                {
                    auto norm1_v = norm1<Ti, Tc>(values[a], values[b]);
                    if(norm1_v > max_norm1)
                    {
                        pos_a     = a;
                        pos_b     = b;
                        max_norm1 = norm1_v;
                    }
                }
            }

#pragma unroll
            for(int k = 0; k < 4; k++)
            {
                int64_t pos = offset + k * stride2;
                prune_if<Ti, InPlace>(k != pos_a && k != pos_b, &out[pos], values[k]);
            }
        }
    }
}

template <typename Ti,
          typename Tc,
          int  SG0I,
          int  SG1J,
          int  TT0I,
          int  TT1J,
          int  PATTERNS_COUNT,
          int  THREADS_PER_SG,
          int  PATTERNS_PER_THREAD,
          bool InPlace>
__global__
    __launch_bounds__(SG0I* SG1J* THREADS_PER_SG) void prune_tile_kernel(const Ti* in,
                                                                         Ti*       out,
                                                                         int64_t   m,
                                                                         int64_t   n,
                                                                         int64_t   stride1,
                                                                         int64_t   stride2,
                                                                         int       num_batches,
                                                                         int64_t   batch_stride,
                                                                         int64_t   sizes)
{
    constexpr int  PAD = 0;
    __shared__ Tc  value_abs[(16 + PAD) * SG0I * SG1J];
    __shared__ Tc  norm_res[THREADS_PER_SG * SG0I * SG1J];
    __shared__ int norm_idx[THREADS_PER_SG * SG0I * SG1J];

    // 90 patterns, that pick 2 elements from each row and column from a 4x4 tile, total pick 8 elements.
    // the first pattern: 0, 2, 0, 2, 1, 3, 1, 3 => COL#(ROW#,ROW#) = 0(0,2), 1(0,2), 2(1,3), 3(1,3)
    __constant__ static uint8_t pos_patterns[90 * 4 * 2] = {
        0, 2, 0, 2, 1, 3, 1, 3, 0, 2, 0, 3, 1, 3, 1, 2, 0, 2, 0, 3, 1, 2, 1, 3, 0, 2, 0, 1, 1, 3,
        2, 3, 0, 2, 0, 1, 2, 3, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 3, 1, 2, 0, 2, 1, 3,
        0, 1, 2, 3, 0, 2, 1, 3, 1, 3, 0, 2, 0, 2, 1, 3, 1, 2, 0, 3, 0, 2, 1, 3, 2, 3, 0, 1, 0, 2,
        1, 2, 0, 3, 1, 3, 0, 2, 1, 2, 1, 3, 0, 3, 0, 2, 2, 3, 0, 1, 1, 3, 0, 2, 2, 3, 1, 3, 0, 1,
        0, 3, 0, 2, 1, 3, 1, 2, 0, 3, 0, 2, 1, 2, 1, 3, 0, 3, 0, 3, 1, 2, 1, 2, 0, 3, 0, 1, 1, 2,
        2, 3, 0, 3, 0, 1, 2, 3, 1, 2, 0, 3, 1, 3, 0, 2, 1, 2, 0, 3, 1, 3, 1, 2, 0, 2, 0, 3, 1, 2,
        0, 2, 1, 3, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 1, 2, 3, 0, 3, 1, 2, 1, 3, 0, 2, 0, 3,
        1, 2, 1, 2, 0, 3, 0, 3, 1, 2, 2, 3, 0, 1, 0, 3, 2, 3, 0, 1, 1, 2, 0, 3, 2, 3, 1, 2, 0, 1,
        0, 1, 0, 2, 1, 3, 2, 3, 0, 1, 0, 2, 2, 3, 1, 3, 0, 1, 0, 3, 1, 2, 2, 3, 0, 1, 0, 3, 2, 3,
        1, 2, 0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 1, 3, 0, 2, 2, 3, 0, 1, 1, 3, 2, 3, 0, 2, 0, 1, 1, 2,
        0, 3, 2, 3, 0, 1, 1, 2, 2, 3, 0, 3, 0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3, 0, 3, 1, 2, 0, 1,
        2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 3, 0, 2, 0, 1, 2, 3, 1, 2, 0, 3, 0, 1, 2, 3, 2, 3, 0, 1,
        1, 3, 0, 2, 0, 2, 1, 3, 1, 3, 0, 2, 0, 3, 1, 2, 1, 3, 0, 2, 0, 1, 2, 3, 1, 3, 0, 2, 1, 3,
        0, 2, 1, 3, 0, 2, 1, 2, 0, 3, 1, 3, 0, 2, 2, 3, 0, 1, 1, 3, 0, 3, 0, 2, 1, 2, 1, 3, 0, 3,
        1, 2, 0, 2, 1, 3, 0, 1, 0, 2, 2, 3, 1, 3, 0, 1, 2, 3, 0, 2, 1, 3, 1, 3, 0, 2, 0, 2, 1, 3,
        1, 2, 0, 2, 0, 3, 1, 3, 1, 2, 0, 3, 0, 2, 1, 3, 2, 3, 0, 2, 0, 1, 1, 3, 2, 3, 0, 1, 0, 2,
        1, 2, 0, 2, 0, 3, 1, 3, 1, 2, 0, 2, 1, 3, 0, 3, 1, 2, 0, 3, 0, 2, 1, 3, 1, 2, 0, 3, 0, 3,
        1, 2, 1, 2, 0, 3, 0, 1, 2, 3, 1, 2, 0, 3, 1, 3, 0, 2, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3,
        2, 3, 0, 1, 1, 2, 0, 1, 0, 3, 2, 3, 1, 2, 0, 1, 2, 3, 0, 3, 1, 2, 1, 3, 0, 2, 0, 3, 1, 2,
        1, 3, 0, 3, 0, 2, 1, 2, 1, 2, 0, 3, 0, 3, 1, 2, 2, 3, 0, 3, 0, 1, 1, 2, 2, 3, 0, 1, 0, 3,
        2, 3, 0, 2, 0, 1, 1, 3, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3, 0, 3, 0, 1, 1, 2, 2, 3, 0, 3, 1, 2,
        0, 1, 2, 3, 0, 1, 0, 2, 1, 3, 2, 3, 0, 1, 0, 3, 1, 2, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3, 0, 1,
        1, 3, 0, 2, 2, 3, 0, 1, 1, 2, 0, 3, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 3, 0, 2, 0, 1, 2, 3,
        1, 3, 0, 1, 0, 2, 2, 3, 1, 2, 0, 3, 0, 1, 2, 3, 1, 2, 0, 1, 0, 3, 2, 3, 2, 3, 0, 1, 0, 1,
    };

    constexpr unsigned int MT0I = SG0I * TT0I;
    constexpr unsigned int MT1J = SG1J * TT1J;

    const unsigned int serial
        = hc_get_workitem_id(0); //total has SG0I * SG1J * (THREADS_PER_SG) threads.
    const unsigned int serial_g  = serial / THREADS_PER_SG; //work idx of SG
    const unsigned int serial_gt = serial % THREADS_PER_SG; //thread idx in SG
    const unsigned int sg0I      = serial_g % SG0I;
    const unsigned int sg1J      = serial_g / SG0I;
    const int64_t      stride    = sg0I * TT0I * stride1 + sg1J * TT1J * stride2;

    const unsigned int wg0I    = hc_get_group_id(0);
    const unsigned int wg1J    = hc_get_group_id(1);
    const unsigned int batchId = hc_get_group_id(2);

    const int64_t wg_pos_x = MT0I * wg0I + sg0I * TT0I;
    const int64_t wg_pos_y = MT1J * wg1J + sg1J * TT1J;
    if((wg_pos_y) >= n || (wg_pos_x) >= m)
        return;

    const int64_t wg_stride = MT1J * wg1J * stride2 + MT0I * wg0I * stride1;
    const int64_t b_stride  = batchId * batch_stride;

    const int x              = serial_gt % 4;
    const int y              = serial_gt / 4;
    const int value_offset   = serial_g * (16 + PAD);
    const int c_value_offset = value_offset + serial_gt;

    const int norm_res_offset   = serial_g * THREADS_PER_SG;
    const int c_norm_res_offset = norm_res_offset + serial_gt;

    int64_t globalReadOffset = b_stride + wg_stride + stride + x * stride1 + y * stride2;

    int64_t pos;
    for(int j = 0; j < TT1J; j += 4)
    {
        pos = globalReadOffset;
        for(int i = 0; i < TT0I; i += 4)
        {
            Ti c_value = static_cast<Ti>(0.0f);
            // read 4x4 from the in matrix.
            {
                if((wg_pos_x + i + x) < m && (wg_pos_y + j + y) < n)
                {
                    c_value = in[pos];
                }
                value_abs[c_value_offset] = abs(static_cast<Tc>(c_value));
            }

            {
                int  offset             = serial_gt;
                auto pos_pattern_offset = offset << 3;
                Tc   max_norm           = static_cast<Tc>(-1.f);
                int  max_norm_idx_      = 0;

                __syncthreads(); //wait until value_abs[] ready

                // caculate norm1 result for each pattern
                Tc tmp_norm
                    = acc_sum8(&value_abs[0], &pos_patterns[0], value_offset, pos_pattern_offset);

#pragma unroll PATTERNS_PER_THREAD - 1
                for(int k = 0; k < PATTERNS_PER_THREAD - 1; k++)
                {
                    offset += THREADS_PER_SG;
                    if(offset < PATTERNS_COUNT)
                    {
                        if(max_norm < tmp_norm)
                        {
                            max_norm      = tmp_norm;
                            max_norm_idx_ = pos_pattern_offset;
                        }
                        pos_pattern_offset = offset << 3;
                        tmp_norm           = acc_sum8(
                            &value_abs[0], &pos_patterns[0], value_offset, pos_pattern_offset);
                        ;
                    }
                }
                if(max_norm < tmp_norm)
                {
                    max_norm      = tmp_norm;
                    max_norm_idx_ = pos_pattern_offset;
                }

                norm_res[c_norm_res_offset] = max_norm;
                norm_idx[c_norm_res_offset] = max_norm_idx_;
                __syncthreads(); //wait until norm_res[], norm_idx[] ready
            }

// find the pattern who has the largest norm1 value
#pragma unroll 4 //log2(THREADS_PER_SG)
            for(int tidxs = THREADS_PER_SG >> 1; tidxs > 0; tidxs >>= 1)
            {
                if(serial_gt < tidxs)
                {
                    Tc  a     = norm_res[c_norm_res_offset];
                    Tc  b     = norm_res[c_norm_res_offset + tidxs];
                    int b_idx = norm_idx[c_norm_res_offset + tidxs];
                    if(a < b)
                    {
                        norm_res[c_norm_res_offset] = b;
                        norm_idx[c_norm_res_offset] = b_idx;
                    }
                }
                __syncthreads(); //wait until norm_res[], norm_idx[] ready
            };
            //__syncthreads();

            // write 4x4 to the out matrix.
            {
                if((wg_pos_x + i + x) < m && (wg_pos_y + j + y) < n)
                {
                    prune_if<Ti, InPlace>(pos_patterns[norm_idx[norm_res_offset] + y * 2] != x
                                              && pos_patterns[norm_idx[norm_res_offset] + y * 2 + 1]
                                                     != x,
                                          &out[pos],
                                          c_value);
                }
            }
            pos += (stride1 << 2);
        }
        globalReadOffset += (stride2 << 2);
    }
}

template <typename Ti, typename Tc>
rocsparselt_status rocsparselt_smfmac_prune_template(const rocsparselt_handle handle,
                                                     int64_t                  m,
                                                     int64_t                  n,
                                                     int64_t                  stride0,
                                                     int64_t                  stride1,
                                                     int                      num_batches,
                                                     int64_t                  batch_stride,
                                                     rocsparselt_operation    op,
                                                     rocsparselt_order        order,
                                                     const Ti*                d_in,
                                                     Ti*                      d_out,
                                                     rocsparselt_prune_alg    pruneAlg,
                                                     hipStream_t              stream)
{
    if(pruneAlg == rocsparselt_prune_smfmac_strip)
    {
        constexpr int SG0I = 16;
        constexpr int SG1J = 4;
        constexpr int TT0I = 1;
        constexpr int TT1J = 4;
        constexpr int MT0I = SG0I * TT0I;
        constexpr int MT1J = SG1J * TT1J;

        int block_x = m / MT0I + (m % MT0I > 0 ? 1 : 0);
        int block_y = n / MT1J + (n % MT1J > 0 ? 1 : 0);

        void (*func)(const Ti* in,
                     Ti*       out,
                     int64_t   m,
                     int64_t   n,
                     int64_t   stride1,
                     int64_t   stride2,
                     int       num_batches,
                     int64_t   batch_stride,
                     int64_t   sizes);
        if(d_in == d_out)
            func = prune_strip_kernel<Ti, Tc, SG0I, SG1J, TT0I, TT1J, true>;
        else
            func = prune_strip_kernel<Ti, Tc, SG0I, SG1J, TT0I, TT1J, false>;
        hipLaunchKernelGGL(func, /* compute kernel*/
                           dim3(block_x, block_y, num_batches),
                           dim3(SG0I * SG1J),
                           0 /*dynamic shared*/,
                           stream,
                           d_in,
                           d_out,
                           m,
                           n,
                           stride0,
                           stride1,
                           num_batches,
                           batch_stride,
                           num_batches * batch_stride);
        return rocsparselt_status_success;
    }
    else if(pruneAlg == rocsparselt_prune_smfmac_tile)
    {
        constexpr int SG0I           = 4;
        constexpr int SG1J           = 4;
        constexpr int TT0I           = 4;
        constexpr int TT1J           = 4;
        constexpr int MT0I           = SG0I * TT0I;
        constexpr int MT1J           = SG1J * TT1J;
        constexpr int PATTERNS_COUNT = 90; // 90 pre-gernated pattens.
        constexpr int THREADS_PER_SG = 16; // fix at 16
        constexpr int PATTERNS_PER_THREAD
            = PATTERNS_COUNT / THREADS_PER_SG + (PATTERNS_COUNT % THREADS_PER_SG != 0 ? 1 : 0);

        int block_x = m / MT0I + (m % MT0I > 0 ? 1 : 0);
        int block_y = n / MT1J + (n % MT1J > 0 ? 1 : 0);

        void (*func)(const Ti* in,
                     Ti*       out,
                     int64_t   m,
                     int64_t   n,
                     int64_t   stride1,
                     int64_t   stride2,
                     int       num_batches,
                     int64_t   batch_stride,
                     int64_t   sizes);
        if(d_in == d_out)
            func = prune_tile_kernel<Ti,
                                     Tc,
                                     SG0I,
                                     SG1J,
                                     TT0I,
                                     TT1J,
                                     PATTERNS_COUNT,
                                     THREADS_PER_SG,
                                     PATTERNS_PER_THREAD,
                                     true>;
        else
            func = prune_tile_kernel<Ti,
                                     Tc,
                                     SG0I,
                                     SG1J,
                                     TT0I,
                                     TT1J,
                                     PATTERNS_COUNT,
                                     THREADS_PER_SG,
                                     PATTERNS_PER_THREAD,
                                     false>;
        hipLaunchKernelGGL(func, /* compute kernel*/
                           dim3(block_x, block_y, num_batches),
                           dim3(SG0I * SG1J * THREADS_PER_SG),
                           0 /*dynamic shared*/,
                           stream,
                           d_in,
                           d_out,
                           m,
                           n,
                           stride0,
                           stride1,
                           num_batches,
                           batch_stride,
                           num_batches * batch_stride);
        return rocsparselt_status_success;
    }
    return rocsparselt_status_not_implemented;
}

template <typename Ti>
rocsparselt_status rocsparselt_smfmac_prune_check_template(const rocsparselt_handle handle,
                                                           int64_t                  m,
                                                           int64_t                  n,
                                                           int64_t                  stride0,
                                                           int64_t                  stride1,
                                                           int                      num_batches,
                                                           int64_t                  batch_stride,
                                                           rocsparselt_operation    op,
                                                           rocsparselt_order        order,
                                                           const Ti*                d_in,
                                                           int*                     d_out,
                                                           hipStream_t              stream)
{
    constexpr int SG0I = 16;
    constexpr int SG1J = 4;
    constexpr int TT0I = 1;
    constexpr int TT1J = 4;
    constexpr int MT0I = SG0I * TT0I;
    constexpr int MT1J = SG1J * TT1J;

    int block_x = m / MT0I + (m % MT0I > 0 ? 1 : 0);
    int block_y = n / MT1J + (n % MT1J > 0 ? 1 : 0);

    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_out, 0, sizeof(int), stream));
    hipLaunchKernelGGL((prune_check_kernel<Ti, SG0I, SG1J, TT0I, TT1J>), /* compute kernel*/
                       dim3(block_x, block_y, num_batches),
                       dim3(SG0I * SG1J),
                       0 /*dynamic shared*/,
                       stream,
                       d_in,
                       d_out,
                       m,
                       n,
                       stride0,
                       stride1,
                       num_batches,
                       batch_stride,
                       num_batches * batch_stride);
    return rocsparselt_status_success;
}

#ifdef __cplusplus
extern "C" {
#endif

rocsparselt_status rocsparselt_smfmac_prune_impl(const rocsparselt_handle handle,
                                                 rocsparselt_mat_descr    matrix,
                                                 rocsparselt_operation    op,
                                                 const void*              d_in,
                                                 void*                    d_out,
                                                 rocsparselt_prune_alg    pruneAlg,
                                                 hipStream_t              stream)
{
    int64_t              m     = op == rocsparselt_operation_transpose ? matrix->n : matrix->m;
    int64_t              n     = op == rocsparselt_operation_transpose ? matrix->m : matrix->n;
    int64_t              ld    = matrix->ld;
    rocsparselt_order    order = matrix->order;
    rocsparselt_datatype type  = matrix->type;

    int     num_batches  = 1;
    int64_t batch_stride = 0;
    matrix->attributes[rocsparselt_mat_num_batches].get(&num_batches);
    matrix->attributes[rocsparselt_mat_batch_stride].get(&batch_stride);

    //set the number of batches to 1 since in the broadcast case, we only care about contents in first batch.
    if(batch_stride == 0) //boardcast case.
    {
        num_batches  = 1;
        batch_stride = matrix->n * ld;
    }

    int64_t stride0 = (op == rocsparselt_operation_transpose) ? ld : 1;
    int64_t stride1 = (op == rocsparselt_operation_transpose) ? 1 : ld;

#define PRUNE_PARAMS(T)                                                   \
    handle, m, n, stride0, stride1, num_batches, batch_stride, op, order, \
        reinterpret_cast<const T*>(d_in), reinterpret_cast<T*>(d_out), pruneAlg, stream

    switch(type)
    {
    case rocsparselt_datatype_f16_r:
        return rocsparselt_smfmac_prune_template<hipsparseLtHalf, float>(
            PRUNE_PARAMS(hipsparseLtHalf));
    case rocsparselt_datatype_bf16_r:
        return rocsparselt_smfmac_prune_template<hipsparseLtBfloat16, float>(
            PRUNE_PARAMS(hipsparseLtBfloat16));
    case rocsparselt_datatype_i8_r:
        return rocsparselt_smfmac_prune_template<int8_t, float>(PRUNE_PARAMS(int8_t));
    default:
        return rocsparselt_status_not_implemented;
    }
}

rocsparselt_status rocsparselt_smfmac_prune_check_impl(const rocsparselt_handle handle,
                                                       rocsparselt_mat_descr    matrix,
                                                       rocsparselt_operation    op,
                                                       const void*              d_in,
                                                       int*                     d_out,
                                                       hipStream_t              stream)
{

    int64_t              m     = op == rocsparselt_operation_transpose ? matrix->n : matrix->m;
    int64_t              n     = op == rocsparselt_operation_transpose ? matrix->m : matrix->n;
    int64_t              ld    = matrix->ld;
    rocsparselt_order    order = matrix->order;
    rocsparselt_datatype type  = matrix->type;

    int     num_batches  = 1;
    int64_t batch_stride = 0;
    matrix->attributes[rocsparselt_mat_num_batches].get(&num_batches);
    matrix->attributes[rocsparselt_mat_batch_stride].get(&batch_stride);

    //set the number of batches to 1 since in the broadcast case, we only care about contents in first batch.
    if(batch_stride == 0) //boardcast case.
    {
        num_batches  = 1;
        batch_stride = matrix->n * ld;
    }

    int64_t stride0 = (op == rocsparselt_operation_transpose) ? ld : 1;
    int64_t stride1 = (op == rocsparselt_operation_transpose) ? 1 : ld;

#define PRUNE_CHECK_PARAMS(T)                                             \
    handle, m, n, stride0, stride1, num_batches, batch_stride, op, order, \
        reinterpret_cast<const T*>(d_in), d_out, stream

    switch(type)
    {
    case rocsparselt_datatype_f16_r:
        return rocsparselt_smfmac_prune_check_template<hipsparseLtHalf>(
            PRUNE_CHECK_PARAMS(hipsparseLtHalf));
    case rocsparselt_datatype_bf16_r:
        return rocsparselt_smfmac_prune_check_template<hipsparseLtBfloat16>(
            PRUNE_CHECK_PARAMS(hipsparseLtBfloat16));
    case rocsparselt_datatype_i8_r:
        return rocsparselt_smfmac_prune_check_template<int8_t>(PRUNE_CHECK_PARAMS(int8_t));
    default:
        return rocsparselt_status_not_implemented;
    }
}

/********************************************************************************
 * \brief prunes a dense matrix according to the specified algorithm.
 *******************************************************************************/
rocsparselt_status rocsparselt_smfmac_prune(const rocsparselt_handle*       handle,
                                            const rocsparselt_matmul_descr* matmulDescr,
                                            const void*                     d_in,
                                            void*                           d_out,
                                            rocsparselt_prune_alg           pruneAlg,
                                            hipStream_t                     stream)

{
    // Check if handle is valid
    if(handle == nullptr || matmulDescr == nullptr || *handle == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(d_in == nullptr || d_out == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }

    // Check if prune alg is valid
    if(pruneAlg != rocsparselt_prune_smfmac_strip && pruneAlg != rocsparselt_prune_smfmac_tile)
    {
        return rocsparselt_status_not_implemented;
    }

    auto _matmulDescr = reinterpret_cast<const _rocsparselt_matmul_descr*>(matmulDescr);

    rocsparselt_mat_descr matrix;
    // Check if matrix A is a structured matrix
    if(_matmulDescr->matrix_A->m_type == rocsparselt_matrix_type_structured)
        matrix = _matmulDescr->matrix_A;
    else
        return rocsparselt_status_not_implemented;

    log_trace(*handle, "rocsparselt_smfmac_prune");
    return rocsparselt_smfmac_prune_impl(
        *handle, matrix, _matmulDescr->op_A, d_in, d_out, pruneAlg, stream);
}

/********************************************************************************
 * \brief prunes a dense matrix according to the specified algorithm.
 *******************************************************************************/
rocsparselt_status rocsparselt_smfmac_prune2(const rocsparselt_handle*    handle,
                                             const rocsparselt_mat_descr* sparseMatDescr,
                                             int                          isSparseA,
                                             rocsparselt_operation        op,
                                             const void*                  d_in,
                                             void*                        d_out,
                                             rocsparselt_prune_alg        pruneAlg,
                                             hipStream_t                  stream)
{
    // Check if handle is valid
    if(handle == nullptr || sparseMatDescr == nullptr || *handle == nullptr
       || *sparseMatDescr == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(d_in == nullptr || d_out == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }

    if(!isSparseA)
        return rocsparselt_status_not_implemented;

    if(op != rocsparselt_operation_none && op != rocsparselt_operation_transpose)
        return rocsparselt_status_invalid_value;

    // Check if prune alg is valid
    if(pruneAlg != rocsparselt_prune_smfmac_strip && pruneAlg != rocsparselt_prune_smfmac_tile)
    {
        return rocsparselt_status_not_implemented;
    }

    rocsparselt_mat_descr matrix = *sparseMatDescr;
    // Check if matrix A is a structured matrix
    if(matrix->m_type != rocsparselt_matrix_type_structured)
        return rocsparselt_status_not_implemented;

    log_trace(*handle, "rocsparselt_smfmac_prune2");
    return rocsparselt_smfmac_prune_impl(*handle, matrix, op, d_in, d_out, pruneAlg, stream);
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status rocsparselt_smfmac_prune_check(const rocsparselt_handle*       handle,
                                                  const rocsparselt_matmul_descr* matmulDescr,
                                                  const void*                     d_in,
                                                  int*                            d_out,
                                                  hipStream_t                     stream)
{
    // Check if handle is valid
    if(handle == nullptr || matmulDescr == nullptr || *handle == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(d_in == nullptr || d_out == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }

    auto _matmulDescr = reinterpret_cast<const _rocsparselt_matmul_descr*>(matmulDescr);

    rocsparselt_mat_descr matrix;
    // Check if matrix A is a structured matrix
    if(_matmulDescr->matrix_A->m_type == rocsparselt_matrix_type_structured)
        matrix = _matmulDescr->matrix_A;
    else
        return rocsparselt_status_not_implemented;

    log_trace(*handle, "rocsparselt_smfmac_prune_check");
    return rocsparselt_smfmac_prune_check_impl(
        *handle, matrix, _matmulDescr->op_A, d_in, d_out, stream);
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status rocsparselt_smfmac_prune_check2(const rocsparselt_handle*    handle,
                                                   const rocsparselt_mat_descr* sparseMatDescr,
                                                   int                          isSparseA,
                                                   rocsparselt_operation        op,
                                                   const void*                  d_in,
                                                   int*                         d_out,
                                                   hipStream_t                  stream)
{
    // Check if handle is valid
    if(handle == nullptr || sparseMatDescr == nullptr || *handle == nullptr
       || *sparseMatDescr == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }

    if(!isSparseA)
        return rocsparselt_status_not_implemented;

    if(op != rocsparselt_operation_none && op != rocsparselt_operation_transpose)
        return rocsparselt_status_invalid_value;

    // Check if pointer is valid
    if(d_in == nullptr || d_out == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }

    rocsparselt_mat_descr matrix = *sparseMatDescr;
    // Check if matrix A is a structured matrix
    if(matrix->m_type != rocsparselt_matrix_type_structured)
        return rocsparselt_status_not_implemented;

    log_trace(*handle, "rocsparselt_smfmac_prune_check2");
    return rocsparselt_smfmac_prune_check_impl(*handle, matrix, op, d_in, d_out, stream);
}

#ifdef __cplusplus
}
#endif
