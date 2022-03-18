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

//#include "gemm_tensile.hpp"

#include "handle.h"
#include "utility.hpp"

inline rocsparse_status getOriginalSizes(rocsparse_operation opA,
                                         rocsparse_operation opB,
                                         int64_t             num_rows_a,
                                         int64_t             num_cols_a,
                                         int64_t             num_rows_b,
                                         int64_t             num_cols_b,
                                         int64_t&            m,
                                         int64_t&            n,
                                         int64_t&            k)
{
    // values of num_* are values after been transposed, redirect to before which been transposed.
    // initialized m,n,k by NN.
    m = num_rows_a, n = num_cols_b, k = num_cols_a;
    if(opA == rocsparse_operation_transpose)
    {
        m = num_cols_a;
        k = num_rows_a;
    }
    if(opB == rocsparse_operation_transpose)
    {
        n = num_rows_b;
        if(k != num_cols_b)
        {
            rocsparselt_cerr << "A, B matrix size are not matched" << std::endl;
            return rocsparse_status_invalid_size;
        }
    }
    else if(k != num_rows_b)
    {
        rocsparselt_cerr << "A, B matrix size are not matched" << std::endl;
        return rocsparse_status_invalid_size;
    }

    return rocsparse_status_success;
}

/*******************************************************************************
 * Get the offset of the metatdata (in bytes)
 ******************************************************************************/
inline int64_t rocsparselt_metadata_offset_in_compressed_matrix(int64_t              num_cols,
                                                                int64_t              ld,
                                                                int                  num_batches,
                                                                rocsparselt_datatype type)
{
    int64_t batch_stride = ld * num_cols;
    auto    bpe          = rocsparselt_datatype_bytes(type);
    int64_t offset       = num_batches * batch_stride * bpe;
    return offset;
}

/*******************************************************************************
 * Validate Arguments - matrix init.
 ******************************************************************************/
inline rocsparse_status validateArgs(rocsparselt_handle      handle,
                                     int64_t                 num_rows,
                                     int64_t                 num_cols,
                                     int64_t                 ld,
                                     uint32_t                alignment,
                                     rocsparselt_datatype    valueType,
                                     rocsparse_order         order,
                                     rocsparselt_matrix_type matrixType)
{
    if(num_rows < 8 || num_cols < 8)
    {
        rocsparselt_cerr << "row and col must larger than 8" << std::endl;
        return rocsparse_status_invalid_size;
    }

    if(matrixType == rocsparselt_matrix_type_structured)
        if(num_cols % 8 != 0 || num_rows % 8 != 0)
        {
            rocsparselt_cerr << "row and col must be the mutliplication of 8" << std::endl;
            return rocsparse_status_invalid_size;
        }

    if(order == rocsparse_order_row)
        return rocsparse_status_not_implemented;

    //TODO should support other datatype in the future.
    if(valueType != rocsparselt_datatype_f16_r)
        return rocsparse_status_not_implemented;

    return rocsparse_status_success;
}

/*******************************************************************************
 * Validate Arguments
 ******************************************************************************/
inline rocsparse_status validateArgs(rocsparselt_handle       handle,
                                     rocsparse_operation      opA,
                                     rocsparse_operation      opB,
                                     const void*              alpha,
                                     const void*              a,
                                     int64_t                  num_rows_a,
                                     int64_t                  num_cols_a,
                                     int64_t                  lda,
                                     const void*              b,
                                     int64_t                  num_rows_b,
                                     int64_t                  num_cols_b,
                                     int64_t                  ldb,
                                     const void*              beta,
                                     const void*              c,
                                     int64_t                  num_rows_c,
                                     int64_t                  num_cols_c,
                                     int64_t                  ldc,
                                     const void*              d,
                                     int64_t                  num_rows_d,
                                     int64_t                  num_cols_d,
                                     int64_t                  ldd,
                                     rocsparselt_datatype     type_a,
                                     rocsparselt_datatype     type_b,
                                     rocsparselt_datatype     type_c,
                                     rocsparselt_datatype     type_d,
                                     rocsparselt_compute_type compute_type,
                                     rocsparselt_matrix_type  matrix_type_a,
                                     rocsparselt_matrix_type  matrix_type_b,
                                     rocsparselt_matrix_type  matrix_type_c,
                                     rocsparselt_matrix_type  matrix_type_d,
                                     rocsparse_order          order_a = rocsparse_order_column,
                                     rocsparse_order          order_b = rocsparse_order_column,
                                     rocsparse_order          order_c = rocsparse_order_column,
                                     rocsparse_order          order_d = rocsparse_order_column,
                                     int                      num_batches_a       = 1,
                                     int                      num_batches_b       = 1,
                                     int                      num_batches_c       = 1,
                                     int                      num_batches_d       = 1,
                                     int64_t                  batch_stride_a      = 0,
                                     int64_t                  batch_stride_b      = 0,
                                     int64_t                  batch_stride_c      = 0,
                                     int64_t                  batch_stride_d      = 0,
                                     int                      act_relu            = 0,
                                     float                    act_relu_upperbound = 0.0f,
                                     float                    act_relu_threshold  = 0.0f,
                                     int                      act_gelu            = 0,
                                     const void*              bias_vector         = nullptr,
                                     int64_t                  bias_stride         = 0)
{
    // handle must be valid
    if(!handle)
        return rocsparse_status_invalid_handle;

    // sizes must not be negative
    if(num_rows_a < 0 || num_cols_a < 0 || num_rows_b < 0 || num_cols_b < 0 || num_rows_c < 0
       || num_cols_c < 0 || num_rows_d < 0 || num_cols_d < 0 || batch_stride_a < 0
       || batch_stride_b < 0 || batch_stride_c < 0 || batch_stride_d < 0)
    {
        rocsparselt_cerr << "matrix and stride size must be posstive" << std::endl;
        return rocsparse_status_invalid_size;
    }

    // number of batches of matrics A,B,C,D must be the same and negative
    if(num_batches_a != num_batches_b || num_batches_a != num_batches_c
       || num_batches_a != num_batches_d || num_batches_a < 1)
    {
        rocsparselt_cerr << " number of batches of matrics A,B,C,D must be the same and negative"
                         << std::endl;
        return rocsparse_status_invalid_size;
    }

    // sizes of matrics A,B,C,D must fulfill the matrix multiplication rule.
    // D = A x B + C
    // values of num_* are values after been transposed, redirect to before which been transposed.
    int64_t m, n, k;
    auto    status
        = getOriginalSizes(opA, opB, num_rows_a, num_cols_a, num_rows_b, num_cols_b, m, n, k);
    if(status != rocsparse_status_success)
        return status;

    if(m != num_rows_c || m != num_rows_d || n != num_cols_c || n != num_cols_d)
    {
        rocsparselt_cerr << " matrix size is not valid" << std::endl;
        return rocsparse_status_invalid_size;
    }

    // size of k must be a multiplication of 8
    if(k % 8 != 0)
    {
        rocsparselt_cerr << "k must be a multiplication of 8" << std::endl;
        return rocsparse_status_invalid_size;
    }

    // order must be column-major
    if((order_a & order_b & order_c & order_d) != rocsparse_order_column)
        return rocsparse_status_not_implemented;

    // leading dimensions must be valid
    if(num_rows_a > lda || num_rows_b > ldb || num_rows_c > ldc || num_rows_d > ldd)
    {
        rocsparselt_cerr << "num_rows" << std::endl;
        return rocsparse_status_invalid_size;
    }

    // data type of matrics must be the same
    if(type_a != type_b || type_a != type_c || type_a != type_c)
        return rocsparse_status_invalid_value;

    switch(type_a)
    {
    case rocsparselt_datatype_bf16_r:
    case rocsparselt_datatype_f16_r:
    case rocsparselt_datatype_f8_r:
    case rocsparselt_datatype_bf8_r:
        if(compute_type != rocsparselt_compute_f32)
            return rocsparse_status_invalid_value;
        break;
    case rocsparselt_datatype_i8_r:
        if(compute_type != rocsparselt_compute_i32)
            return rocsparse_status_invalid_value;
        break;
    }

    // quick return 0 is valid in BLAS
    // Note: k==0 is not a quick return, because C must still be multiplied by beta
    if(!m || !n || !num_batches_a)
        return rocsparse_status_success;

    if(!beta)
        return rocsparse_status_invalid_pointer;

    // pointers must be valid
    if((k && (!a || !b || !alpha)) || !c || !d)
        return rocsparse_status_invalid_pointer;

    if(act_relu < 0 || act_gelu < 0 || (bias_vector != nullptr && bias_stride < 0))
        return rocsparse_status_invalid_value;

    // Only matrix A can be structured matrix.
    if(matrix_type_a != rocsparselt_matrix_type_structured)
        return rocsparse_status_invalid_value;

    if(matrix_type_b != rocsparselt_matrix_type_dense
       || matrix_type_c != rocsparselt_matrix_type_dense
       || matrix_type_d != rocsparselt_matrix_type_dense)
        return rocsparse_status_invalid_value;

    return rocsparse_status_continue;
}

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
                                       int32_t              numStreams)
{
    return rocsparse_status_not_implemented;
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
                                  int32_t              numStreams)
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
                                 (const Tc*)alpha,
                                 (const Ti*)a,
                                 ld_a,
                                 batch_stride_a,
                                 offset_a,
                                 (const Ti*)b,
                                 ld_b,
                                 batch_stride_b,
                                 offset_b,
                                 (const Tc*)beta,
                                 (const To*)c,
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
                                 numStreams);
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
                                                  int32_t                  numStreams)
{
    rocsparse_status rs_status = rocsparse_status_not_implemented;

#define EX_TYPECASTING_PARM                                                                   \
    handle, trans_a, trans_b, m, n, k, alpha, a, ld_a, batch_stride_a, offset_a, b, ld_b,     \
        batch_stride_b, offset_b, beta, c, ld_c, batch_stride_c, offset_c, d, ld_d,           \
        batch_stride_d, offset_d, batch_count, strided_batch, sparseA, metadata, act_relu,    \
        act_relu_upperbound, act_relu_threshold, act_gelu, bias_vector, bias_stride, streams, \
        numStreams

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
