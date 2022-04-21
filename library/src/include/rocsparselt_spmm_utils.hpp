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
#ifndef ROCSPARSELT_SPMM_UTILS_HPP
#define ROCSPARSELT_SPMM_UTILS_HPP
#include "handle.h"
#include "rocsparselt_ostream.hpp"
#include <cxxabi.h>

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

template <typename T>
inline rocsparse_status validateSetAttributeDataSize(size_t dataSize,
                                                     size_t expectedSize = sizeof(T))
{
    if(expectedSize != dataSize)
    {
        int   status = -4;
        char* mname  = __cxxabiv1::__cxa_demangle(typeid(T).name(), NULL, NULL, &status);

        rocsparselt_cerr << "The parameter number 5 (dataSize) had an illegal value: "
                         << "expected " << expectedSize << " bytes(sizeof("
                         << (status == 0 ? mname : typeid(T).name()) << "))"
                         << ", current size " << dataSize << " bytes" << std::endl;

        if(status == 0)
            free(mname);
        return rocsparse_status_invalid_size;
    }
    return rocsparse_status_success;
}

template <>
inline rocsparse_status validateSetAttributeDataSize<void>(size_t dataSize, size_t expectedSize)
{
    if(expectedSize > dataSize)
    {
        rocsparselt_cerr << "The parameter number 5 (dataSize) had an illegal value: "
                         << "at least " << expectedSize << " bytes, current size " << dataSize
                         << " bytes" << std::endl;
        return rocsparse_status_invalid_size;
    }
    return rocsparse_status_success;
}

template <typename T>
inline rocsparse_status validateGetAttributeDataSize(size_t dataSize,
                                                     size_t expectedSize = sizeof(T))
{
    return validateGetAttributeDataSize<void>(dataSize, expectedSize);
}

template <>
inline rocsparse_status validateGetAttributeDataSize<void>(size_t dataSize, size_t expectedSize)
{
    if(expectedSize < dataSize)
    {
        rocsparselt_cerr << "The parameter number 5 (dataSize) had an illegal value: expected "
                         << expectedSize << " bytes, current size " << dataSize << " bytes"
                         << std::endl;
        return rocsparse_status_invalid_size;
    }
    return rocsparse_status_success;
}

/*******************************************************************************
 * Validate Matrix Arguments - matrix init.
 ******************************************************************************/
inline rocsparse_status validateMatrixArgs(rocsparselt_handle      handle,
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
        rocsparselt_cerr << "row and col must larger than 8, current are " << num_rows << " and "
                         << num_cols << std::endl;
        return rocsparse_status_invalid_size;
    }

    // leading dimensions must be valid
    if(num_rows > ld)
    {
        rocsparselt_cerr << "number of rows(" << num_rows << ") is larger than leading dimension("
                         << ld << ")" << std::endl;
        return rocsparse_status_invalid_size;
    }

    if(matrixType == rocsparselt_matrix_type_structured)
        if(num_cols % 8 != 0)
        {
            rocsparselt_cerr << "row and col must be the mutliplication of 8" << std::endl;
            return rocsparse_status_invalid_size;
        }

    if(order == rocsparse_order_row)
        return rocsparse_status_not_implemented;

    //TODO should support other datatype in the future.
    switch(valueType)
    {
    case rocsparselt_datatype_f16_r:
    case rocsparselt_datatype_bf16_r:
    case rocsparselt_datatype_i8_r:
        break;
    default:
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

/*******************************************************************************
 * Validate Matmul Descr. init Arguments - matrix init.
 ******************************************************************************/
inline rocsparse_status validateMatmulDescrArgs(rocsparselt_handle       handle,
                                                rocsparse_operation      opA,
                                                rocsparse_operation      opB,
                                                int64_t                  num_rows_a,
                                                int64_t                  num_cols_a,
                                                int64_t                  lda,
                                                int64_t                  num_rows_b,
                                                int64_t                  num_cols_b,
                                                int64_t                  ldb,
                                                int64_t                  num_rows_c,
                                                int64_t                  num_cols_c,
                                                int64_t                  ldc,
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
                                                rocsparselt_matrix_type  matrix_type_d)
{
    // handle must be valid
    if(!handle)
        return rocsparse_status_invalid_handle;

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

    // Only matrix A can be structured matrix.
    if(matrix_type_a != rocsparselt_matrix_type_structured)
    {
        rocsparselt_cerr << " Matrix A must be structrured matrix." << std::endl;
        return rocsparse_status_not_implemented;
    }

    if(matrix_type_b != rocsparselt_matrix_type_dense)
    {
        rocsparselt_cerr << " Matrix B cannot be structrured matrix." << std::endl;
        return rocsparse_status_not_implemented;
    }

    if(matrix_type_c != rocsparselt_matrix_type_dense
       || matrix_type_d != rocsparselt_matrix_type_dense)
        return rocsparse_status_invalid_value;

    return rocsparse_status_success;
}

/*******************************************************************************
 * Validate Matmul Arguments
 ******************************************************************************/
inline rocsparse_status validateMatmulArgs(rocsparselt_handle handle,
                                           int64_t            m,
                                           int64_t            n,
                                           int64_t            k,
                                           const void*        alpha,
                                           const void*        a,
                                           const void*        b,
                                           const void*        beta,
                                           const void*        c,
                                           const void*        d,
                                           int                num_batches_a  = 1,
                                           int                num_batches_b  = 1,
                                           int                num_batches_c  = 1,
                                           int                num_batches_d  = 1,
                                           int64_t            batch_stride_a = 0,
                                           int64_t            batch_stride_b = 0,
                                           int64_t            batch_stride_c = 0,
                                           int64_t            batch_stride_d = 0)
{
    // handle must be valid
    if(!handle)
        return rocsparse_status_invalid_handle;

    // sizes must not be negative
    if(batch_stride_a < 0 || batch_stride_b < 0 || batch_stride_c < 0 || batch_stride_d < 0)
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

    // quick return 0 is valid in BLAS
    // Note: k==0 is not a quick return, because C must still be multiplied by beta
    if(!m || !n || !num_batches_a)
        return rocsparse_status_success;

    if(!beta)
        return rocsparse_status_invalid_pointer;

    // pointers must be valid
    if((k && (!a || !b || !alpha)) || !c || !d)
        return rocsparse_status_invalid_pointer;

    return rocsparse_status_continue;
}

#endif
