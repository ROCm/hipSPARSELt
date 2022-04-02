/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSELT_SPMM_HPP
#define ROCSPARSELT_SPMM_HPP

//#include "gemm_tensile.hpp"

#include "handle.h"
#include "rocsparselt_ostream.hpp"
#include "utility.hpp"

#if BUILD_WITH_TENSILE
#include "tensile_host.hpp"
#else
#include "kernel_launcher.hpp"
#endif

template <typename Ti, typename To, typename Tc>
rocsparse_status spmm_batched_template(rocsparselt_handle   handle,
                                       rocsparse_operation  trans_a,
                                       rocsparse_operation  trans_b,
                                       int64_t              m,
                                       int64_t              n,
                                       int64_t              k,
                                       const Tc*            alpha,
                                       const Ti*            a,
                                       int64_t              ld_a,
                                       int64_t              batch_stride_a,
                                       int64_t              offset_a,
                                       const Ti*            b,
                                       int64_t              ld_b,
                                       int64_t              batch_stride_b,
                                       int64_t              offset_b,
                                       const Tc*            beta,
                                       const To*            c,
                                       int64_t              ld_c,
                                       int64_t              batch_stride_c,
                                       int64_t              offset_c,
                                       To*                  d,
                                       int64_t              ld_d,
                                       int64_t              batch_stride_d,
                                       int64_t              offset_d,
                                       int64_t              batch_count,
                                       bool                 strided_batch,
                                       bool                 sparseA,
                                       const unsigned char* metadata,
                                       int                  act_relu,
                                       float                act_relu_upperbound,
                                       float                act_relu_threshold,
                                       int                  act_gelu,
                                       const void*          bias_vector,
                                       int64_t              bias_stride,
                                       hipStream_t*         streams,
                                       int32_t              numStreams,
                                       int*                 config_id,
                                       const int            config_max_id,
                                       const int            search_iterations)
{

    RocsparseltContractionProblem<Ti, To, Tc> problem{handle,
                                                      trans_a,
                                                      trans_b,
                                                      m,
                                                      n,
                                                      k,
                                                      alpha,
                                                      a,
                                                      nullptr,
                                                      ld_a,
                                                      batch_stride_a,
                                                      offset_a,
                                                      b,
                                                      nullptr,
                                                      ld_b,
                                                      batch_stride_b,
                                                      offset_b,
                                                      beta,
                                                      c,
                                                      nullptr,
                                                      ld_c,
                                                      batch_stride_c,
                                                      offset_c,
                                                      d,
                                                      nullptr,
                                                      ld_d,
                                                      batch_stride_d,
                                                      offset_d,
                                                      batch_count,
                                                      strided_batch,
                                                      sparseA,
                                                      metadata,
                                                      streams,
                                                      numStreams};
#if BUILD_WITH_TENSILE
    return runContractionProblem(problem);
#else
    if(problem.trans_a == rocsparse_operation_none)
        if(problem.trans_b == rocsparse_operation_none)
            return runContractionProblem<Ti,
                                         To,
                                         Tc,
                                         rocsparse_operation_none,
                                         rocsparse_operation_none>(
                problem, config_id, config_max_id, search_iterations);
        else
            return runContractionProblem<Ti,
                                         To,
                                         Tc,
                                         rocsparse_operation_none,
                                         rocsparse_operation_transpose>(
                problem, config_id, config_max_id, search_iterations);
    else if(problem.trans_b == rocsparse_operation_none)
        return runContractionProblem<Ti,
                                     To,
                                     Tc,
                                     rocsparse_operation_transpose,
                                     rocsparse_operation_transpose>(
            problem, config_id, config_max_id, search_iterations);
    else
        return runContractionProblem<Ti,
                                     To,
                                     Tc,
                                     rocsparse_operation_transpose,
                                     rocsparse_operation_transpose>(
            problem, config_id, config_max_id, search_iterations);
#endif
}

template <typename Ti, typename To = Ti, typename Tc = To>
rocsparse_status spmm_typecasting(rocsparselt_handle   handle,
                                  rocsparse_operation  trans_a,
                                  rocsparse_operation  trans_b,
                                  int64_t              m,
                                  int64_t              n,
                                  int64_t              k,
                                  const void*          alpha,
                                  const void*          a,
                                  int64_t              ld_a,
                                  int64_t              batch_stride_a,
                                  int64_t              offset_a,
                                  const void*          b,
                                  int64_t              ld_b,
                                  int64_t              batch_stride_b,
                                  int64_t              offset_b,
                                  const void*          beta,
                                  const void*          c,
                                  int64_t              ld_c,
                                  int64_t              batch_stride_c,
                                  int64_t              offset_c,
                                  void*                d,
                                  int64_t              ld_d,
                                  int64_t              batch_stride_d,
                                  int64_t              offset_d,
                                  int64_t              batch_count,
                                  bool                 strided_batch,
                                  bool                 sparseA,
                                  const unsigned char* metadata,
                                  int                  act_relu,
                                  float                act_relu_upperbound,
                                  float                act_relu_threshold,
                                  int                  act_gelu,
                                  const void*          bias_vector,
                                  int64_t              bias_stride,
                                  hipStream_t*         streams,
                                  int32_t              numStreams,
                                  int*                 config_id,
                                  const int            config_max_id,
                                  const int            search_iterations)
{
    // check alignment of pointers before casting
    if(!isAligned(a, sizeof(Ti)) || !isAligned(b, sizeof(Ti)) || !isAligned(c, sizeof(Ti))
       || !isAligned(d, sizeof(To)))
    {
        rocsparselt_cerr << "memmory is not aligned" << std::endl;
        return rocsparse_status_invalid_size;
    }
    return spmm_batched_template(handle,
                                 trans_a,
                                 trans_b,
                                 m,
                                 n,
                                 k,
                                 reinterpret_cast<const Tc*>(alpha),
                                 reinterpret_cast<const Ti*>(a),
                                 ld_a,
                                 batch_stride_a,
                                 offset_a,
                                 reinterpret_cast<const Ti*>(b),
                                 ld_b,
                                 batch_stride_b,
                                 offset_b,
                                 reinterpret_cast<const Tc*>(beta),
                                 reinterpret_cast<const To*>(c),
                                 ld_c,
                                 batch_stride_c,
                                 offset_c,
                                 (To*)d,
                                 ld_d,
                                 batch_stride_d,
                                 offset_d,
                                 batch_count,
                                 strided_batch,
                                 sparseA,
                                 metadata,
                                 act_relu,
                                 act_relu_upperbound,
                                 act_relu_threshold,
                                 act_gelu,
                                 bias_vector,
                                 bias_stride,
                                 streams,
                                 numStreams,
                                 config_id,
                                 config_max_id,
                                 search_iterations);
}

inline rocsparse_status rocsparselt_spmm_template(rocsparselt_handle       handle,
                                                  rocsparse_operation      trans_a,
                                                  rocsparse_operation      trans_b,
                                                  int64_t                  m,
                                                  int64_t                  n,
                                                  int64_t                  k,
                                                  const void*              alpha,
                                                  const void*              a,
                                                  rocsparselt_datatype     a_type,
                                                  int64_t                  ld_a,
                                                  int64_t                  batch_stride_a,
                                                  int64_t                  offset_a,
                                                  const void*              b,
                                                  rocsparselt_datatype     b_type,
                                                  int64_t                  ld_b,
                                                  int64_t                  batch_stride_b,
                                                  int64_t                  offset_b,
                                                  const void*              beta,
                                                  const void*              c,
                                                  rocsparselt_datatype     c_type,
                                                  int64_t                  ld_c,
                                                  int64_t                  batch_stride_c,
                                                  int64_t                  offset_c,
                                                  void*                    d,
                                                  rocsparselt_datatype     d_type,
                                                  int64_t                  ld_d,
                                                  int64_t                  batch_stride_d,
                                                  int64_t                  offset_d,
                                                  int64_t                  batch_count,
                                                  bool                     strided_batch,
                                                  rocsparselt_compute_type compute_type,
                                                  bool                     sparseA,
                                                  const unsigned char*     metadata,
                                                  int                      act_relu,
                                                  float                    act_relu_upperbound,
                                                  float                    act_relu_threshold,
                                                  int                      act_gelu,
                                                  const void*              bias_vector,
                                                  int64_t                  bias_stride,
                                                  hipStream_t*             streams,
                                                  int32_t                  numStreams,
                                                  int*                     config_id,
                                                  const int                config_max_id,
                                                  const int                search_iterations)
{
    rocsparse_status rs_status = rocsparse_status_not_implemented;

#define EX_TYPECASTING_PARM                                                                   \
    handle, trans_a, trans_b, m, n, k, alpha, a, ld_a, batch_stride_a, offset_a, b, ld_b,     \
        batch_stride_b, offset_b, beta, c, ld_c, batch_stride_c, offset_c, d, ld_d,           \
        batch_stride_d, offset_d, batch_count, strided_batch, sparseA, metadata, act_relu,    \
        act_relu_upperbound, act_relu_threshold, act_gelu, bias_vector, bias_stride, streams, \
        numStreams, config_id, config_max_id, search_iterations

    if(a_type == rocsparselt_datatype_f16_r && b_type == rocsparselt_datatype_f16_r)
    {
        if(c_type == rocsparselt_datatype_f16_r && d_type == rocsparselt_datatype_f16_r)
        {
            if(compute_type == rocsparselt_compute_f32)
            {
                rs_status = spmm_typecasting<rocsparselt_half, rocsparselt_half, float>(
                    EX_TYPECASTING_PARM);
            }
        }
    }
    else
    {
        rs_status = rocsparse_status_not_implemented;
    }

    return rs_status;
}
#endif
