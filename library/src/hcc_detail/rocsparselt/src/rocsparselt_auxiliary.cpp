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
#include "kernel_launcher.hpp"
#include "rocsparselt.h"
#include "rocsparselt_spmm_utils.hpp"
#include "utility.hpp"

#include <hip/hip_runtime_api.h>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief rocsparselt_handle is a structure holding the rocsparselt library context.
 * It must be initialized using rocsparselt_init()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocsparselt_destroy().
 *******************************************************************************/
rocsparselt_status rocsparselt_init(rocsparselt_handle* handle)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }
    else
    {

        // Allocate
        try
        {
            auto _handle = reinterpret_cast<_rocsparselt_handle*>(handle);
            _handle->init();
            log_trace(_handle, "rocsparselt_init");
        }
        catch(const rocsparselt_status& status)
        {
            return status;
        }
        return rocsparselt_status_success;
    }
}

/********************************************************************************
 * \brief destroy handle
 *******************************************************************************/
rocsparselt_status rocsparselt_destroy(const rocsparselt_handle* handle)
{
    if(handle == nullptr)
    {
        return rocsparselt_status_success;
    }

    auto _handle = reinterpret_cast<_rocsparselt_handle*>(const_cast<rocsparselt_handle*>(handle));
    if(!_handle->isInit())
    {
        return rocsparselt_status_invalid_handle;
    }

    log_trace(_handle, "rocsparse_destroy");
    // Destruct
    try
    {
        _handle->destroy();
    }
    catch(const rocsparselt_status& status)
    {
        return status;
    }
    return rocsparselt_status_success;
}

/********************************************************************************
 * \brief rocsparse_mat_descr is a structure holding the rocsparselt matrix
 * content. It must be initialized using rocsparselt_dense_descr_init() or
 * rocsparselt_structured_descr_init()  and the retured handle must be passed
 * to all subsequent library function calls that involve the matrix.
 * It should be destroyed at the end using rocsparselt_mat_descr_destroy().
 *******************************************************************************/
rocsparselt_status rocsparselt_dense_descr_init(const rocsparselt_handle* handle,
                                                rocsparselt_mat_descr*    matDescr,
                                                int64_t                   rows,
                                                int64_t                   cols,
                                                int64_t                   ld,
                                                uint32_t                  alignment,
                                                rocsparselt_datatype      valueType,
                                                rocsparselt_order         order)
{
    // Check if matDescr is valid
    if(handle == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }
    else if(matDescr == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
            if(!_handle->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }

            auto status = validateMatrixArgs(_handle,
                                             rows,
                                             cols,
                                             ld,
                                             alignment,
                                             valueType,
                                             order,
                                             rocsparselt_matrix_type_dense);
            if(status != rocsparselt_status_success)
                throw status;

            auto                   _matDescr = reinterpret_cast<_rocsparselt_mat_descr*>(matDescr);
            _rocsparselt_mat_descr tmpMatDescr;
            memcpy(_matDescr, &tmpMatDescr, sizeof(_rocsparselt_mat_descr));
            _matDescr->m_type    = rocsparselt_matrix_type_dense;
            _matDescr->m         = rows;
            _matDescr->n         = cols;
            _matDescr->ld        = ld;
            _matDescr->alignment = alignment;
            _matDescr->type      = valueType;
            _matDescr->order     = order;
            int     num_batches  = 1;
            int64_t batch_stride = cols * ld;
            _matDescr->attributes[rocsparselt_mat_batch_stride].set(&batch_stride);
            _matDescr->attributes[rocsparselt_mat_num_batches].set(&num_batches);
            log_trace(_handle, "rocsparselt_dense_descr_init");
        }
        catch(const rocsparselt_status& status)
        {
            return status;
        }
        return rocsparselt_status_success;
    }
}

/********************************************************************************
 * \brief rocsparse_mat_descr is a structure holding the rocsparselt matrix
 * content. It must be initialized using rocsparselt_dense_descr_init() or
 * rocsparselt_structured_descr_init()  and the retured handle must be passed
 * to all subsequent library function calls that involve the matrix.
 * It should be destroyed at the end using rocsparselt_mat_descr_destroy().
 *******************************************************************************/
rocsparselt_status rocsparselt_structured_descr_init(const rocsparselt_handle* handle,
                                                     rocsparselt_mat_descr*    matDescr,
                                                     int64_t                   rows,
                                                     int64_t                   cols,
                                                     int64_t                   ld,
                                                     uint32_t                  alignment,
                                                     rocsparselt_datatype      valueType,
                                                     rocsparselt_order         order,
                                                     rocsparselt_sparsity      sparsity)

{
    // Check if matDescr is valid
    if(handle == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }
    else if(matDescr == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
            if(!_handle->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }
            auto status = validateMatrixArgs(_handle,
                                             rows,
                                             cols,
                                             ld,
                                             alignment,
                                             valueType,
                                             order,
                                             rocsparselt_matrix_type_structured);
            if(status != rocsparselt_status_success)
                throw status;

            auto                   _matDescr = reinterpret_cast<_rocsparselt_mat_descr*>(matDescr);
            _rocsparselt_mat_descr tmpMatDescr;
            memcpy(_matDescr, &tmpMatDescr, sizeof(_rocsparselt_mat_descr));
            _matDescr->m_type    = rocsparselt_matrix_type_structured;
            _matDescr->m         = rows;
            _matDescr->n         = cols;
            _matDescr->ld        = ld;
            _matDescr->alignment = alignment;
            _matDescr->type      = valueType;
            _matDescr->order     = order;
            _matDescr->sparsity  = sparsity;
            int     num_batches  = 1;
            int64_t batch_stride = cols * ld;
            _matDescr->attributes[rocsparselt_mat_batch_stride].set(&batch_stride);
            _matDescr->attributes[rocsparselt_mat_num_batches].set(&num_batches);
            log_trace(_handle, "rocsparselt_structured_descr_init");
        }
        catch(const rocsparselt_status& status)
        {
            return status;
        }
        return rocsparselt_status_success;
    }
}

/********************************************************************************
 * \brief destroy matrix descriptor
 *******************************************************************************/
rocsparselt_status rocsparselt_mat_descr_destroy(const rocsparselt_mat_descr* matDescr)
{
    if(matDescr == nullptr)
        return rocsparselt_status_invalid_handle;

    auto _matDescr
        = reinterpret_cast<_rocsparselt_mat_descr*>(const_cast<rocsparselt_mat_descr*>(matDescr));

    if(_matDescr->isInit())
        return rocsparselt_status_success;
    // Destruct
    try
    {
        _matDescr->clear();
    }
    catch(const rocsparselt_status& status)
    {
        return status;
    }
    return rocsparselt_status_success;
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix descriptor
 * such as number of batches and their stride.
 *******************************************************************************/
rocsparselt_status rocsparselt_mat_descr_set_attribute(const rocsparselt_handle*       handle,
                                                       rocsparselt_mat_descr*          matDescr,
                                                       rocsparselt_mat_descr_attribute matAttribute,
                                                       const void*                     data,
                                                       size_t                          dataSize)

{

    if(handle == nullptr || matDescr == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }
    else if(data == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
            if(!_handle->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }

            auto _matDescr = reinterpret_cast<_rocsparselt_mat_descr*>(matDescr);

            if(!_matDescr->isInit())
                return rocsparselt_status_invalid_handle;

            rocsparselt_status status;
            switch(matAttribute)
            {
            case rocsparselt_mat_num_batches:
            {
                if((status = validateSetAttributeDataSize<int>(dataSize))
                   != rocsparselt_status_success)
                    return status;
                const int* num_batches = reinterpret_cast<const int*>(data);
                if(*num_batches < 1)
                {
                    hipsparselt_cerr
                        << "The number of batches must be greater or equal to 1, current: "
                        << *num_batches << std::endl;
                    return rocsparselt_status_invalid_value;
                }
                break;
            }
            case rocsparselt_mat_batch_stride:
            {
                if((status = validateSetAttributeDataSize<int64_t>(dataSize))
                   != rocsparselt_status_success)
                    return status;
                const int64_t* batch_stride = reinterpret_cast<const int64_t*>(data);
                if(*batch_stride != 0)
                {
                    int64_t expected_batch_stride = _matDescr->n * _matDescr->ld;
                    if(*batch_stride < expected_batch_stride)
                    {
                        hipsparselt_cerr << "The batch stride must be 0 or at least cols * ld ("
                                         << expected_batch_stride << "), current: " << *batch_stride
                                         << std::endl;
                        return rocsparselt_status_invalid_value;
                    }
                }
                break;
            }
            }
            _matDescr->attributes[matAttribute].set(data, dataSize);
            log_trace(_handle, "rocsparselt_mat_descr_set_attribute");
        }
        catch(const rocsparselt_status& status)
        {
            return status;
        }
        return rocsparselt_status_success;
    }
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix descriptor
 * such as number of batches and their stride.
 *******************************************************************************/
rocsparselt_status rocsparselt_mat_descr_get_attribute(const rocsparselt_handle*       handle,
                                                       const rocsparselt_mat_descr*    matDescr,
                                                       rocsparselt_mat_descr_attribute matAttribute,
                                                       void*                           data,
                                                       size_t                          dataSize)
{
    if(handle == nullptr || matDescr == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }
    else if(data == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }
    else
    {
        try
        {
            auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
            if(!_handle->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }

            auto _matDescr = reinterpret_cast<const _rocsparselt_mat_descr*>(matDescr);

            if(!_matDescr->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }

            rocsparselt_status status;
            switch(matAttribute)
            {
            case rocsparselt_mat_num_batches:
            {
                if((status = validateGetAttributeDataSize<int>(dataSize))
                   != rocsparselt_status_success)
                {
                    return status;
                }
                break;
            }
            case rocsparselt_mat_batch_stride:
            {
                if((status = validateGetAttributeDataSize<int64_t>(dataSize))
                   != rocsparselt_status_success)
                {
                    return status;
                }
                break;
            }
            default:
                return rocsparselt_status_invalid_value;
            }

            if(_matDescr->attributes[matAttribute].get(data, dataSize) == 0)
            {
                return rocsparselt_status_internal_error;
            }
            log_trace(_handle, "rocsparselt_mat_descr_get_attribute");
        }
        catch(const rocsparselt_status& status)
        {
            return status;
        }
        return rocsparselt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status rocsparselt_matmul_descr_init(const rocsparselt_handle*    handle,
                                                 rocsparselt_matmul_descr*    matmulDescr,
                                                 rocsparselt_operation        opA,
                                                 rocsparselt_operation        opB,
                                                 const rocsparselt_mat_descr* matA,
                                                 const rocsparselt_mat_descr* matB,
                                                 const rocsparselt_mat_descr* matC,
                                                 const rocsparselt_mat_descr* matD,
                                                 rocsparselt_compute_type     computeType)
{
    // Check if matmulDescr is valid
    if(handle == nullptr || matA == nullptr || matB == nullptr || matC == nullptr
       || matD == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }
    else if(matmulDescr == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
            if(!_handle->isInit())
            {
                log_trace(_handle, "handle not init");
                return rocsparselt_status_invalid_handle;
            }
            auto _matA = reinterpret_cast<_rocsparselt_mat_descr*>(
                const_cast<rocsparselt_mat_descr*>(matA));
            if(!_matA->isInit())
                return rocsparselt_status_invalid_handle;

            auto _matB = reinterpret_cast<_rocsparselt_mat_descr*>(
                const_cast<rocsparselt_mat_descr*>(matB));
            if(!_matB->isInit())
                return rocsparselt_status_invalid_handle;

            auto _matC = reinterpret_cast<_rocsparselt_mat_descr*>(
                const_cast<rocsparselt_mat_descr*>(matC));
            if(!_matC->isInit())
                return rocsparselt_status_invalid_handle;

            auto _matD = reinterpret_cast<_rocsparselt_mat_descr*>(
                const_cast<rocsparselt_mat_descr*>(matD));
            if(!_matD->isInit())
                return rocsparselt_status_invalid_handle;

            auto status = validateMatmulDescrArgs(_handle,
                                                  opA,
                                                  opB,
                                                  _matA->m,
                                                  _matA->n,
                                                  _matA->ld,
                                                  _matB->m,
                                                  _matB->n,
                                                  _matB->ld,
                                                  _matC->m,
                                                  _matC->n,
                                                  _matC->ld,
                                                  _matD->m,
                                                  _matD->n,
                                                  _matD->ld,
                                                  _matA->type,
                                                  _matB->type,
                                                  _matC->type,
                                                  _matD->type,
                                                  computeType,
                                                  _matA->m_type,
                                                  _matB->m_type,
                                                  _matC->m_type,
                                                  _matD->m_type);
            if(status != rocsparselt_status_success)
                return status;

            int64_t m, n, k;
            //don't need to check the status here, since which was done in validateMatmulDescrArgs().
            getOriginalSizes(opA, opB, _matA->m, _matA->n, _matB->m, _matB->n, m, n, k);
            _matA->c_k        = k / 2;
            _matA->c_ld       = (opA == rocsparselt_operation_transpose ? _matA->c_k : m);
            auto _matmulDescr = reinterpret_cast<_rocsparselt_matmul_descr*>(matmulDescr);
            _rocsparselt_matmul_descr tmpDescr;
            memcpy(_matmulDescr, &tmpDescr, sizeof(_rocsparselt_matmul_descr));
            _matmulDescr->op_A         = opA;
            _matmulDescr->op_B         = opB;
            _matmulDescr->matrix_A     = _matA;
            _matmulDescr->matrix_B     = _matB;
            _matmulDescr->matrix_C     = _matC;
            _matmulDescr->matrix_D     = _matD;
            _matmulDescr->compute_type = computeType;
            log_trace(_handle, "rocsparselt_matmul_descr_init");
        }
        catch(const rocsparselt_status& status)
        {
            return status;
        }
        return rocsparselt_status_success;
    }
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix multiplication
 * descriptor.
 *******************************************************************************/
rocsparselt_status
    rocsparselt_matmul_descr_set_attribute(const rocsparselt_handle*          handle,
                                           rocsparselt_matmul_descr*          matmulDescr,
                                           rocsparselt_matmul_descr_attribute matmulAttribute,
                                           const void*                        data,
                                           size_t                             dataSize)
{
    // Check if matmulDescr is valid
    if(handle == nullptr || matmulDescr == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }
    else if(data == nullptr)
    {
        hipsparselt_cerr << "The parameter number 4 (data) cannot be NULL" << std::endl;
        return rocsparselt_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
            if(!_handle->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }

            auto _matmulDescr = reinterpret_cast<_rocsparselt_matmul_descr*>(matmulDescr);

            if(!_matmulDescr->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }
            rocsparselt_status status;

            auto assign_data = [&](auto* val) {
                using val_type = typename std::remove_pointer<decltype(val)>::type;
                if((status = validateSetAttributeDataSize<val_type>(dataSize))
                   != rocsparselt_status_success)
                    return;
                *val   = *(reinterpret_cast<const val_type*>(data));
                status = rocsparselt_status_success;
            };
            switch(matmulAttribute)
            {
            case rocsparselt_matmul_activation_relu:
                assign_data(&_matmulDescr->activation_relu);
                break;
            case rocsparselt_matmul_activation_relu_upperbound:
                assign_data(&_matmulDescr->activation_relu_upperbound);
                break;
            case rocsparselt_matmul_activation_relu_threshold:
                assign_data(&_matmulDescr->activation_relu_threshold);
                break;
            case rocsparselt_matmul_activation_gelu:
                assign_data(&_matmulDescr->activation_gelu);
                break;
            case rocsparselt_matmul_activation_abs:
                assign_data(&_matmulDescr->activation_abs);
                break;
            case rocsparselt_matmul_activation_leakyrelu:
                assign_data(&_matmulDescr->activation_leakyrelu);
                break;
            case rocsparselt_matmul_activation_leakyrelu_alpha:
                assign_data(&_matmulDescr->activation_leakyrelu_alpha);
                break;
            case rocsparselt_matmul_activation_sigmoid:
                assign_data(&_matmulDescr->activation_sigmoid);
                if(_matmulDescr->activation_sigmoid
                   && _matmulDescr->matrix_D->type == rocsparselt_datatype_i8_r)
                {
                    hipsparselt_cerr << "Sigmoid activation function is not support for int8"
                                     << std::endl;
                    _matmulDescr->activation_sigmoid = 0;
                    return rocsparselt_status_not_implemented;
                }
                break;
            case rocsparselt_matmul_activation_tanh:
                assign_data(&_matmulDescr->activation_tanh);
                if(_matmulDescr->activation_tanh
                   && _matmulDescr->matrix_D->type == rocsparselt_datatype_i8_r)
                {
                    hipsparselt_cerr << "Tanh activation function is not support for int8"
                                     << std::endl;
                    _matmulDescr->activation_tanh = 0;
                    return rocsparselt_status_not_implemented;
                }
                break;
            case rocsparselt_matmul_activation_tanh_alpha:
                assign_data(&_matmulDescr->activation_tanh_alpha);
                break;
            case rocsparselt_matmul_activation_tanh_beta:
                assign_data(&_matmulDescr->activation_tanh_beta);
                break;
#if ENABLE_BIAS
            case rocsparselt_matmul_bias_pointer:
            {
                if((status = validateSetAttributeDataSize<float*>(dataSize))
                   != rocsparselt_status_success)
                    return status;
                _matmulDescr->bias_pointer = reinterpret_cast<float*>(data);

                break;
            }
            case rocsparselt_matmul_bias_stride:
            {
                if((status = validateSetAttributeDataSize<int64_t>(dataSize))
                   != rocsparselt_status_success)
                    return status;
                const int64_t* bias_stride = reinterpret_cast<const int64_t*>(data);
                if(*bias_stride != 0 && *bias_stride < _matmulDescr->matrix_D.m)
                {
                    hipsparselt_cerr << "The batch stride must be 0 or at least the number of rows "
                                        "of the output matrix (D) ("
                                     << _matmulDescr->matrix_D.m << "), current: " << *bias_stride
                                     << std::endl;
                    return rocsparselt_status_invalid_value;
                }
                _matmulDescr->bias_stride = *bias_stride;
                break;
            }
#endif
            default:
                return rocsparselt_status_not_implemented;
            }
            if(status != rocsparselt_status_success)
                return status;
            log_trace(_handle, "rocsparselt_matmul_descr_set_attribute");
        }
        catch(const rocsparselt_status& status)
        {
            return status;
        }
        return rocsparselt_status_success;
    }
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix descriptor
 * such as number of batches and their stride.
 *******************************************************************************/
rocsparselt_status
    rocsparselt_matmul_descr_get_attribute(const rocsparselt_handle*          handle,
                                           const rocsparselt_matmul_descr*    matmulDescr,
                                           rocsparselt_matmul_descr_attribute matmulAttribute,
                                           void*                              data,
                                           size_t                             dataSize)

{

    // Check if matmulDescr is valid
    if(handle == nullptr || matmulDescr == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }
    else if(data == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }
    else
    {
        try
        {
            auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
            if(!_handle->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }

            auto _matmulDescr = reinterpret_cast<const _rocsparselt_matmul_descr*>(matmulDescr);
            if(!_matmulDescr->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }

            rocsparselt_status status;

            auto retrive_data = [&](auto val) {
                if((status = validateGetAttributeDataSize<decltype(val)>(dataSize))
                   != rocsparselt_status_success)
                    return;
                *(reinterpret_cast<decltype(val)*>(data)) = val;
                status                                    = rocsparselt_status_success;
            };

            switch(matmulAttribute)
            {
            case rocsparselt_matmul_activation_relu:
                retrive_data(_matmulDescr->activation_relu);
                break;
            case rocsparselt_matmul_activation_gelu:
                retrive_data(_matmulDescr->activation_gelu);
                break;
            case rocsparselt_matmul_activation_relu_upperbound:
                retrive_data(_matmulDescr->activation_relu_upperbound);
                break;
            case rocsparselt_matmul_activation_relu_threshold:
                retrive_data(_matmulDescr->activation_relu_threshold);
                break;
            case rocsparselt_matmul_activation_abs:
                retrive_data(_matmulDescr->activation_abs);
                break;
            case rocsparselt_matmul_activation_leakyrelu:
                retrive_data(_matmulDescr->activation_leakyrelu);
                break;
            case rocsparselt_matmul_activation_leakyrelu_alpha:
                retrive_data(_matmulDescr->activation_leakyrelu_alpha);
                break;
            case rocsparselt_matmul_activation_sigmoid:
                retrive_data(_matmulDescr->activation_sigmoid);
                break;
            case rocsparselt_matmul_activation_tanh:
                retrive_data(_matmulDescr->activation_tanh);
                break;
            case rocsparselt_matmul_activation_tanh_alpha:
                retrive_data(_matmulDescr->activation_tanh_alpha);
                break;
            case rocsparselt_matmul_activation_tanh_beta:
                retrive_data(_matmulDescr->activation_tanh_beta);
                break;
#if ENABLE_BIAS
            case rocsparselt_matmul_bias_pointer:
                if((status = validateGetAttributeDataSize<float*>(dataSize))
                   != rocsparselt_status_success)
                    return status;
                if(_matmulDescr->bias_pointer.get(data, dataSize) == 0)
                    return rocsparselt_status_internal_error;
                break;
            case rocsparselt_matmul_bias_stride:
                if((status = validateGetAttributeDataSize<int64_t>(dataSize))
                   != rocsparselt_status_success)
                    return status;
                *(reinterpret_cast<int64_t*>(data)) = _matmulDescr->bias_stride;
                break;
#endif
            default:
                return rocsparselt_status_not_implemented;
            }
            if(status != rocsparselt_status_success)
                return status;
            log_trace(_handle, "rocsparselt_matmul_descr_get_attribute");
        }
        catch(const rocsparselt_status& status)
        {
            return status;
        }
        return rocsparselt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status
    rocsparselt_matmul_alg_selection_init(const rocsparselt_handle*         handle,
                                          rocsparselt_matmul_alg_selection* algSelection,
                                          const rocsparselt_matmul_descr*   matmulDescr,
                                          rocsparselt_matmul_alg            alg)
{
    // Check if algSelection is valid
    if(handle == nullptr || matmulDescr == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }
    else if(algSelection == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
            if(!_handle->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }
            auto _matmulDescr = reinterpret_cast<const _rocsparselt_matmul_descr*>(matmulDescr);
            if(!_matmulDescr->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }

            auto _algSelection = reinterpret_cast<_rocsparselt_matmul_alg_selection*>(algSelection);

            auto in_type      = _matmulDescr->matrix_A->type;
            auto out_type     = _matmulDescr->matrix_D->type;
            auto compute_type = _matmulDescr->compute_type;

            int config_max_id = 0;

            if(in_type == rocsparselt_datatype_f16_r && out_type == rocsparselt_datatype_f16_r
               && compute_type == rocsparselt_compute_f32)
                initSolutions<__half, __half, float>(
                    _handle, _matmulDescr->op_A, _matmulDescr->op_B, &config_max_id);
            else if(in_type == rocsparselt_datatype_bf16_r
                    && out_type == rocsparselt_datatype_bf16_r
                    && compute_type == rocsparselt_compute_f32)
                initSolutions<hip_bfloat16, hip_bfloat16, float>(
                    _handle, _matmulDescr->op_A, _matmulDescr->op_B, &config_max_id);
            else if(in_type == rocsparselt_datatype_i8_r && out_type == rocsparselt_datatype_i8_r
                    && compute_type == rocsparselt_compute_i32)
                initSolutions<int8_t, int8_t, int32_t>(
                    _handle, _matmulDescr->op_A, _matmulDescr->op_B, &config_max_id);

            if(!config_max_id)
            {
                hipsparselt_cerr << "There are no solutions for this problem size" << std::endl;
                return rocsparselt_status_not_implemented;
            }

            _rocsparselt_matmul_alg_selection tmpAlgSelection;
            memcpy(_algSelection, &tmpAlgSelection, sizeof(_rocsparselt_matmul_alg_selection));
            _algSelection->alg           = alg;
            _algSelection->config_max_id = config_max_id;
            log_trace(_handle, "rocsparselt_matmul_alg_selection_init");
        }
        catch(const rocsparselt_status& status)
        {
            return status;
        }
        return rocsparselt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status
    rocsparselt_matmul_alg_set_attribute(const rocsparselt_handle*         handle,
                                         rocsparselt_matmul_alg_selection* algSelection,
                                         rocsparselt_matmul_alg_attribute  attribute,
                                         const void*                       data,
                                         size_t                            dataSize)
{
    // Check if algSelection is valid
    if(handle == nullptr || algSelection == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }
    else if(data == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
            if(!_handle->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }

            auto _algSelection = reinterpret_cast<_rocsparselt_matmul_alg_selection*>(algSelection);
            if(!_algSelection->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }
            rocsparselt_status status;
            switch(attribute)
            {
            case rocsparselt_matmul_alg_config_id:
            {
                if((status = validateSetAttributeDataSize<int>(dataSize))
                   != rocsparselt_status_success)
                    return status;

                const int* config_id = reinterpret_cast<const int*>(data);
                if(*config_id >= _algSelection->config_max_id)
                {
                    hipsparselt_cerr << "The value of rocsparselt_matmul_alg_config_id data"
                                     << *config_id << "is out of the range [0, "
                                     << (_algSelection->config_max_id - 1) << "]" << std::endl;
                    return rocsparselt_status_invalid_value;
                }

                _algSelection->config_id = *config_id;
                break;
            }
            case rocsparselt_matmul_alg_config_max_id:
            {
                hipsparselt_cerr << "rocsparselt_matmul_alg_config_max_id is only for query."
                                 << std::endl;
                return rocsparselt_status_invalid_value;
            }
            case rocsparselt_matmul_search_iterations:
            {
                if((status = validateSetAttributeDataSize<int>(dataSize))
                   != rocsparselt_status_success)
                    return status;
                const int* search_iterations = reinterpret_cast<const int*>(data);
                if(*search_iterations < 1)
                {
                    hipsparselt_cerr
                        << "The search iterations must be greater or equal to 1, current: "
                        << *search_iterations << std::endl;
                    return rocsparselt_status_invalid_value;
                }
                _algSelection->search_iterations = *search_iterations;
                break;
            }
            default:
                return rocsparselt_status_not_implemented;
            }
            log_trace(_handle, "rocsparselt_matmul_alg_set_attribute");
        }
        catch(const rocsparselt_status& status)
        {
            return status;
        }
        return rocsparselt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status
    rocsparselt_matmul_alg_get_attribute(const rocsparselt_handle*               handle,
                                         const rocsparselt_matmul_alg_selection* algSelection,
                                         rocsparselt_matmul_alg_attribute        attribute,
                                         void*                                   data,
                                         size_t                                  dataSize)

{
    if(handle == nullptr || algSelection == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }
    else if(data == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }
    else if(dataSize <= 0)
    {
        return rocsparselt_status_invalid_value;
    }
    else
    {
        try
        {
            auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
            if(!_handle->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }

            auto _algSelection
                = reinterpret_cast<const _rocsparselt_matmul_alg_selection*>(algSelection);
            if(!_algSelection->isInit())
            {
                return rocsparselt_status_invalid_handle;
            }

            rocsparselt_status status;
            if((status = validateGetAttributeDataSize<int>(dataSize)) != rocsparselt_status_success)
                return status;
            switch(attribute)
            {
            case rocsparselt_matmul_alg_config_id:
                *reinterpret_cast<int*>(data) = _algSelection->config_id;
                break;
            case rocsparselt_matmul_alg_config_max_id:
                *reinterpret_cast<int*>(data) = _algSelection->config_max_id;
                break;
            case rocsparselt_matmul_search_iterations:
                *reinterpret_cast<int*>(data) = _algSelection->search_iterations;
                break;
            default:
                return rocsparselt_status_not_implemented;
            }
            log_trace(_handle, "rocsparselt_matmul_alg_get_attribute");
        }
        catch(const rocsparselt_status& status)
        {
            return status;
        }
        return rocsparselt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status
    rocsparselt_matmul_plan_init(const rocsparselt_handle*               handle,
                                 rocsparselt_matmul_plan*                plan,
                                 const rocsparselt_matmul_descr*         matmulDescr,
                                 const rocsparselt_matmul_alg_selection* algSelection,
                                 size_t                                  workspaceSize)

{
    // Check if plan is valid
    if(handle == nullptr || matmulDescr == nullptr || algSelection == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }
    else if(plan == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }
    else if(workspaceSize < 0)
    {
        return rocsparselt_status_invalid_size;
    }
    else
    {
        auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
        if(!_handle->isInit())
        {
            return rocsparselt_status_invalid_handle;
        }

        auto _matmulDescr = reinterpret_cast<const _rocsparselt_matmul_descr*>(matmulDescr);
        if(!_matmulDescr->isInit())
        {
            return rocsparselt_status_invalid_handle;
        }

        const _rocsparselt_matmul_alg_selection* _algSelection
            = reinterpret_cast<const _rocsparselt_matmul_alg_selection*>(algSelection);
        if(!_algSelection->isInit())
        {
            return rocsparselt_status_invalid_handle;
        }
        // Allocate
        auto _plan = reinterpret_cast<_rocsparselt_matmul_plan*>(plan);
        // Allocate
        try
        {
            int num_batches_a = 1, num_batches_b = 1, num_batches_c = 1, num_batches_d = 1;
            _matmulDescr->matrix_A->attributes[rocsparselt_mat_num_batches].get(&num_batches_a);
            _matmulDescr->matrix_B->attributes[rocsparselt_mat_num_batches].get(&num_batches_b);
            _matmulDescr->matrix_C->attributes[rocsparselt_mat_num_batches].get(&num_batches_c);
            _matmulDescr->matrix_D->attributes[rocsparselt_mat_num_batches].get(&num_batches_d);

            if(num_batches_a != (num_batches_b | num_batches_c | num_batches_d))
            {
                hipsparselt_cerr << " number of batches of matrics A,B,C,D must be the same"
                                 << std::endl;
                return rocsparselt_status_invalid_size;
            }

            _rocsparselt_matmul_plan tmpPlan;
            memcpy(_plan, &tmpPlan, sizeof(_rocsparselt_matmul_plan));

            (_plan)->matmul_descr   = new _rocsparselt_matmul_descr(*_matmulDescr);
            (_plan)->alg_selection  = const_cast<_rocsparselt_matmul_alg_selection*>(_algSelection);
            (_plan)->workspace_size = workspaceSize;
            log_trace(_handle, "rocsparselt_matmul_plan_init");
        }
        catch(const rocsparselt_status& status)
        {
            return status;
        }
        return rocsparselt_status_success;
    }
}

/********************************************************************************
 * \brief destroy matrix multiplication plan descriptor
 *******************************************************************************/
rocsparselt_status rocsparselt_matmul_plan_destroy(const rocsparselt_matmul_plan* plan)
{
    if(plan == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }

    auto _plan
        = reinterpret_cast<_rocsparselt_matmul_plan*>(const_cast<rocsparselt_matmul_plan*>(plan));
    if(!_plan->isInit())
    {
        return rocsparselt_status_invalid_handle;
    }

    // Destruct
    try
    {
        _plan->clear();
    }
    catch(const rocsparselt_status& status)
    {
        return status;
    }
    return rocsparselt_status_success;
}

#ifdef __cplusplus
}
#endif

/*******************************************************************************
 * GPU architecture-related functions
 ******************************************************************************/

// Emulate C++17 std::void_t
template <typename...>
using void_t = void;

// By default, use gcnArch converted to a string prepended by gfx
template <typename PROP, typename = void>
struct ArchName
{
    std::string operator()(const PROP& prop) const
    {
        return "gfx" + std::to_string(prop.gcnArch);
    }
};

// If gcnArchName exists as a member, use it instead
template <typename PROP>
struct ArchName<PROP, void_t<decltype(PROP::gcnArchName)>>
{
    std::string operator()(const PROP& prop) const
    {
        // strip out xnack/ecc from name
        std::string gcnArchName(prop.gcnArchName);
        std::string gcnArch = gcnArchName.substr(0, gcnArchName.find(":"));
        return gcnArch;
    }
};

//Get architecture name
std::string rocsparselt_internal_get_arch_name()
{
    int deviceId;
    THROW_IF_HIP_ERROR(hipGetDevice(&deviceId));
    hipDeviceProp_t deviceProperties;
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&deviceProperties, deviceId));
    return ArchName<hipDeviceProp_t>{}(deviceProperties);
}
