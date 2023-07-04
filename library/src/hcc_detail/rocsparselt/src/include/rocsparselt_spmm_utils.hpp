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
#ifndef ROCSPARSELT_SPMM_UTILS_HPP
#define ROCSPARSELT_SPMM_UTILS_HPP
#include "handle.h"
#include "hipsparselt_ostream.hpp"
#include "utility.hpp"
#if BUILD_WITH_TENSILE
#include "tensile_host.hpp"
#else
#include "kernel_launcher.hpp"
#endif
#include <cxxabi.h>

inline rocsparselt_status getOriginalSizes(rocsparselt_operation opA,
                                           rocsparselt_operation opB,
                                           int64_t               num_rows_a,
                                           int64_t               num_cols_a,
                                           int64_t               num_rows_b,
                                           int64_t               num_cols_b,
                                           int64_t&              m,
                                           int64_t&              n,
                                           int64_t&              k)
{
    // values of num_* are values after been transposed, redirect to before which been transposed.
    // initialized m,n,k by NN.
    m = num_rows_a, n = num_cols_b, k = num_cols_a;
    if(opA == rocsparselt_operation_transpose)
    {
        m = num_cols_a;
        k = num_rows_a;
    }
    if(opB == rocsparselt_operation_transpose)
    {
        n = num_rows_b;
        if(k != num_cols_b)
        {
            hipsparselt_cerr << "A, B matrix size are not matched" << std::endl;
            return rocsparselt_status_invalid_size;
        }
    }
    else if(k != num_rows_b)
    {
        hipsparselt_cerr << "A, B matrix size are not matched" << std::endl;
        return rocsparselt_status_invalid_size;
    }

    return rocsparselt_status_success;
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

    auto datatype_bpe = [&] {
        switch(type)
        {
        case rocsparselt_datatype_f32_r:
            return 4;
        case rocsparselt_datatype_f16_r:
        case rocsparselt_datatype_bf16_r:
            return 2;
        case rocsparselt_datatype_f8_r:
        case rocsparselt_datatype_bf8_r:
        case rocsparselt_datatype_i8_r:
            return 1;
        default:
            return 0;
        }
    };

    auto    bpe    = datatype_bpe();
    int64_t offset = num_batches * batch_stride * bpe;
    return offset;
}

template <typename T>
inline rocsparselt_status validateSetAttributeDataSize(size_t dataSize,
                                                       size_t expectedSize = sizeof(T))
{
    if(expectedSize != dataSize)
    {
        int   status = -4;
        char* mname  = __cxxabiv1::__cxa_demangle(typeid(T).name(), NULL, NULL, &status);

        hipsparselt_cerr << "The parameter number 5 (dataSize) had an illegal value: "
                         << "expected " << expectedSize << " bytes(sizeof("
                         << (status == 0 ? mname : typeid(T).name()) << "))"
                         << ", current size " << dataSize << " bytes" << std::endl;

        if(status == 0)
            free(mname);
        return rocsparselt_status_invalid_size;
    }
    return rocsparselt_status_success;
}

template <>
inline rocsparselt_status validateSetAttributeDataSize<void>(size_t dataSize, size_t expectedSize)
{
    if(expectedSize > dataSize)
    {
        hipsparselt_cerr << "The parameter number 5 (dataSize) had an illegal value: "
                         << "at least " << expectedSize << " bytes, current size " << dataSize
                         << " bytes" << std::endl;
        return rocsparselt_status_invalid_size;
    }
    return rocsparselt_status_success;
}

template <typename T>
inline rocsparselt_status validateGetAttributeDataSize(size_t dataSize,
                                                       size_t expectedSize = sizeof(T))
{
    return validateGetAttributeDataSize<void>(dataSize, expectedSize);
}

template <>
inline rocsparselt_status validateGetAttributeDataSize<void>(size_t dataSize, size_t expectedSize)
{
    if(expectedSize > dataSize)
    {
        hipsparselt_cerr << "The parameter number 5 (dataSize) had an illegal value: expected "
                         << expectedSize << " bytes, current size " << dataSize << " bytes"
                         << std::endl;
        return rocsparselt_status_invalid_size;
    }
    return rocsparselt_status_success;
}

/*******************************************************************************
 * Validate Matrix Arguments - matrix init.
 ******************************************************************************/
inline rocsparselt_status validateMatrixArgs(const _rocsparselt_handle* handle,
                                             int64_t                    num_rows,
                                             int64_t                    num_cols,
                                             int64_t                    ld,
                                             uint32_t                   alignment,
                                             rocsparselt_datatype       valueType,
                                             rocsparselt_order          order,
                                             rocsparselt_matrix_type    matrixType)
{
    // handle must be valid
    if(handle == nullptr || !handle->isInit())
        return rocsparselt_status_invalid_handle;

    if(num_rows == 0 || num_cols == 0)
    {
        hipsparselt_cerr << "row and col cannot be zero, current are " << num_rows << " and "
                         << num_cols << std::endl;
        log_error(handle, __func__, "row and col cannot be 0");
        return rocsparselt_status_invalid_size;
    }

    int num_elements = 8;
    switch(valueType)
    {
    case rocsparselt_datatype_i8_r:
    case rocsparselt_datatype_f8_r:
    case rocsparselt_datatype_bf8_r:
        num_elements = 16;
        break;
    default:
        break;
    }

    if(num_rows < num_elements || num_cols < num_elements)
    {
        hipsparselt_cerr << "row and col must larger than " << num_elements << ", current are "
                         << num_rows << " and " << num_cols << std::endl;
        if(handle->layer_mode & rocsparselt_layer_mode_log_error)
        {
            std::ostringstream stream;
            stream << "row and col must >= " << num_elements;
            log_error(handle, __func__, stream.str());
        }
        return rocsparselt_status_not_implemented;
    }

    if(num_rows % num_elements != 0 || num_cols % num_elements)
    {
        hipsparselt_cerr << "row and col must be a multiple of " << num_elements << std::endl;
        if(handle->layer_mode & rocsparselt_layer_mode_log_error)
        {
            std::ostringstream stream;
            stream << "row and col must be a multiple of " << num_elements;
            log_error(handle, __func__, stream.str());
        }
        return rocsparselt_status_not_implemented;
    }

    // leading dimensions must be valid
    if(num_rows > ld)
    {
        hipsparselt_cerr << "number of rows(" << num_rows << ") is larger than leading dimension("
                         << ld << ")" << std::endl;
        log_error(handle, __func__, "row and col must >= ld");
        return rocsparselt_status_invalid_size;
    }

    if(order == rocsparselt_order_row)
    {
        hipsparselt_cerr << "rocsparselt_order_row is not supported" << std::endl;
        log_error(handle, __func__, "rocsparselt_order_row is not supported");
        return rocsparselt_status_not_implemented;
    }

    //TODO should support other datatype in the future.
    switch(valueType)
    {
    case rocsparselt_datatype_f16_r:
    case rocsparselt_datatype_bf16_r:
    case rocsparselt_datatype_i8_r:
        break;
    default:
        hipsparselt_cerr << "datatype (" << rocsparselt_datatype_to_string(valueType)
                         << ") is not supported" << std::endl;
        log_error(handle, __func__, "datatype is not supported");
        return rocsparselt_status_not_implemented;
    }
    return rocsparselt_status_success;
}

/*******************************************************************************
 * Validate Matmul Descr. init Arguments - matrix init.
 ******************************************************************************/
inline rocsparselt_status validateMatmulDescrArgs(const _rocsparselt_handle* handle,
                                                  rocsparselt_operation      opA,
                                                  rocsparselt_operation      opB,
                                                  int64_t                    num_rows_a,
                                                  int64_t                    num_cols_a,
                                                  int64_t                    lda,
                                                  int64_t                    num_rows_b,
                                                  int64_t                    num_cols_b,
                                                  int64_t                    ldb,
                                                  int64_t                    num_rows_c,
                                                  int64_t                    num_cols_c,
                                                  int64_t                    ldc,
                                                  int64_t                    num_rows_d,
                                                  int64_t                    num_cols_d,
                                                  int64_t                    ldd,
                                                  rocsparselt_datatype       type_a,
                                                  rocsparselt_datatype       type_b,
                                                  rocsparselt_datatype       type_c,
                                                  rocsparselt_datatype       type_d,
                                                  rocsparselt_compute_type   compute_type,
                                                  rocsparselt_matrix_type    matrix_type_a,
                                                  rocsparselt_matrix_type    matrix_type_b,
                                                  rocsparselt_matrix_type    matrix_type_c,
                                                  rocsparselt_matrix_type    matrix_type_d)
{
    // handle must be valid
    if(handle == nullptr || !handle->isInit())
        return rocsparselt_status_invalid_handle;

    auto is_op_valid = [](rocsparselt_operation op) {
        switch(op)
        {
        case rocsparselt_operation_none:
        case rocsparselt_operation_transpose:
            return true;
        default:
            return false;
        }
    };

    if(!is_op_valid(opA))
    {
        log_error(handle, __func__, "opA", opA, "is not valid");
        return rocsparselt_status_invalid_value;
    }
    if(!is_op_valid(opB))
    {
        log_error(handle, __func__, "opB", opB, "is not valid");
        return rocsparselt_status_invalid_value;
    }

    // sizes of matrics A,B,C,D must fulfill the matrix multiplication rule.
    // D = A x B + C
    // values of num_* are values after been transposed, redirect to before which been transposed.
    int64_t m, n, k;
    auto    status
        = getOriginalSizes(opA, opB, num_rows_a, num_cols_a, num_rows_b, num_cols_b, m, n, k);
    if(status != rocsparselt_status_success)
    {
        log_error(handle, __func__, "size of matrices are not matched");
        return status;
    }

    if(m != (num_rows_c | num_rows_d) || n != (num_cols_c | num_cols_d))
    {
        hipsparselt_cerr << "matrix size is not valid" << std::endl;
        log_error(handle, __func__, "matrix size is not valid");
        return rocsparselt_status_invalid_size;
    }

    // data type of matrics must be the same
    if(type_a != (type_b | type_c | type_d))
    {
        log_error(handle, __func__, "datatype of matrices are inconsistent");
        return rocsparselt_status_not_implemented;
    }

    switch(type_a)
    {
    case rocsparselt_datatype_bf16_r:
    case rocsparselt_datatype_f16_r:
    case rocsparselt_datatype_f8_r:
    case rocsparselt_datatype_bf8_r:
        if(compute_type != rocsparselt_compute_f32)
        {
            log_error(handle, __func__, "computType must be f32");
            return rocsparselt_status_not_implemented;
        }
        break;
    case rocsparselt_datatype_i8_r:
        if(compute_type != rocsparselt_compute_i32)
        {
            log_error(handle, __func__, "computType must be i32");
            return rocsparselt_status_not_implemented;
        }
        break;
    default:
        log_error(handle,
                  __func__,
                  "datatype",
                  rocsparselt_datatype_to_string(type_a),
                  "is not supported");
        return rocsparselt_status_not_implemented;
    }

    // Only matrix A can be structured matrix.
    if(matrix_type_a != rocsparselt_matrix_type_structured)
    {
        hipsparselt_cerr << "Matrix A must be structrured matrix." << std::endl;
        log_error(handle, __func__, "Matrix A must be structrured matrix");
        return rocsparselt_status_not_implemented;
    }

    if(matrix_type_b != rocsparselt_matrix_type_dense)
    {
        hipsparselt_cerr << "Matrix B cannot be structrured matrix." << std::endl;
        log_error(handle, __func__, "Matrix B cannot be structrured matrix");
        return rocsparselt_status_not_implemented;
    }

    if(matrix_type_c != rocsparselt_matrix_type_dense
       || matrix_type_d != rocsparselt_matrix_type_dense)
    {
        log_error(handle, __func__, "Matrix C and D must be dense matrix");
        return rocsparselt_status_invalid_value;
    }

    return rocsparselt_status_success;
}

template <typename Ti, typename To, typename Tc>
rocsparselt_status ConstructRocSparseLtProblem(const char*                                 caller,
                                               RocsparseltContractionProblem<Ti, To, Tc>** prob,
                                               const _rocsparselt_matmul_descr*            matDescr,
                                               const Tc*    alpha         = nullptr,
                                               const Tc*    beta          = nullptr,
                                               const Ti*    a             = nullptr,
                                               const Ti*    b             = nullptr,
                                               const To*    c             = nullptr,
                                               To*          d             = nullptr,
                                               bool         strided_batch = true,
                                               void*        workspace     = nullptr,
                                               size_t       workspaceSize = ~size_t{0},
                                               hipStream_t* streams       = nullptr,
                                               int32_t      numStreams    = 0);

template <typename Ti, typename To, typename Tc>
rocsparselt_status findTopConfigs(const _rocsparselt_matmul_descr*         matmulDescr,
                                  std::vector<_rocsparselt_matmul_config>* configs,
                                  int*                                     config_max_id,
                                  const int                                requestConfigs = 10)
{
    RocsparseltContractionProblem<Ti, To, Tc>* prob;
    Tc                                         alpha = static_cast<Tc>(1.0f);
    Tc                                         beta  = static_cast<Tc>(1.0f);
    auto                                       status
        = ConstructRocSparseLtProblem<Ti, To, Tc>(__func__, &prob, matmulDescr, &alpha, &beta);
    if(status != rocsparselt_status_success)
        return status;
    getBestSolutions<Ti, To, Tc>(*prob, requestConfigs, configs, config_max_id);
    delete prob;
    return status;
}
#endif
