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

#include "hipsparselt.h"
#include "exceptions.hpp"

#include <hip/hip_runtime_api.h>
#include <rocsparselt.h>
#include <stdio.h>
#include <stdlib.h>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

#ifdef __cplusplus
extern "C" {
#endif

#define RETURN_IF_HIPSPARSELT_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                                      \
        hipsparseLtStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != HIPSPARSELT_STATUS_SUCCESS)             \
        {                                                                  \
            return TMP_STATUS_FOR_CHECK;                                   \
        }                                                                  \
    }

#define RETURN_IF_ROCSPARSELT_ERROR(INPUT_STATUS_FOR_CHECK)               \
    {                                                                     \
        rocsparselt_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocsparselt_status_success)            \
        {                                                                 \
            return RocSparseLtStatusToHIPStatus(TMP_STATUS_FOR_CHECK);    \
        }                                                                 \
    }

hipsparseLtStatus_t hipErrorToHIPSPARSELtStatus(hipError_t status)
{
    switch(status)
    {
    case hipSuccess:
        return HIPSPARSELT_STATUS_SUCCESS;
    case hipErrorMemoryAllocation:
    case hipErrorLaunchOutOfResources:
        return HIPSPARSELT_STATUS_ALLOC_FAILED;
    case hipErrorInvalidDevicePointer:
        return HIPSPARSELT_STATUS_INVALID_VALUE;
    case hipErrorInvalidDevice:
    case hipErrorInvalidResourceHandle:
        return HIPSPARSELT_STATUS_NOT_INITIALIZED;
    case hipErrorInvalidValue:
        return HIPSPARSELT_STATUS_INVALID_VALUE;
    case hipErrorNoDevice:
    case hipErrorUnknown:
        return HIPSPARSELT_STATUS_INTERNAL_ERROR;
    default:
        return HIPSPARSELT_STATUS_INTERNAL_ERROR;
    }
}

hipsparseLtStatus_t RocSparseLtStatusToHIPStatus(rocsparselt_status_ status)
{
    switch(status)
    {
    case rocsparselt_status_success:
        return HIPSPARSELT_STATUS_SUCCESS;
    case rocsparselt_status_invalid_handle:
        return HIPSPARSELT_STATUS_NOT_INITIALIZED;
    case rocsparselt_status_not_implemented:
        return HIPSPARSELT_STATUS_INTERNAL_ERROR;
    case rocsparselt_status_invalid_pointer:
        return HIPSPARSELT_STATUS_INVALID_VALUE;
    case rocsparselt_status_invalid_size:
        return HIPSPARSELT_STATUS_INVALID_VALUE;
    case rocsparselt_status_memory_error:
        return HIPSPARSELT_STATUS_ALLOC_FAILED;
    case rocsparselt_status_internal_error:
        return HIPSPARSELT_STATUS_INTERNAL_ERROR;
    case rocsparselt_status_invalid_value:
        return HIPSPARSELT_STATUS_INVALID_VALUE;
    case rocsparselt_status_arch_mismatch:
        return HIPSPARSELT_STATUS_ARCH_MISMATCH;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

rocsparselt_datatype_ HIPDatatypeToRocSparseLtDatatype(hipsparseLtDatatype_t type)
{
    switch(type)
    {
    case HIPSPARSELT_R_32F:
        return rocsparselt_datatype_f32_r;

    case HIPSPARSELT_R_16BF:
        return rocsparselt_datatype_bf16_r;

    case HIPSPARSELT_R_16F:
        return rocsparselt_datatype_f16_r;

    case HIPSPARSELT_R_8I:
        return rocsparselt_datatype_i8_r;

    case HIPSPARSELT_R_8F:
        return rocsparselt_datatype_f8_r;

    case HIPSPARSELT_R_8BF:
        return rocsparselt_datatype_bf8_r;

    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

hipsparseLtDatatype_t RocSparseLtDatatypeToHIPDatatype(rocsparselt_datatype_ type)
{
    switch(type)
    {
    case rocsparselt_datatype_f32_r:
        return HIPSPARSELT_R_32F;

    case rocsparselt_datatype_bf16_r:
        return HIPSPARSELT_R_16BF;

    case rocsparselt_datatype_f16_r:
        return HIPSPARSELT_R_16F;

    case rocsparselt_datatype_i8_r:
        return HIPSPARSELT_R_8I;

    case rocsparselt_datatype_f8_r:
        return HIPSPARSELT_R_8F;

    case rocsparselt_datatype_bf8_r:
        return HIPSPARSELT_R_8BF;

    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

rocsparselt_compute_type_ HIPComputetypeToRocSparseLtComputetype(hipsparseLtComputetype_t type)
{
    switch(type)
    {
    case HIPSPARSELT_COMPUTE_16F:
    case HIPSPARSELT_COMPUTE_TF32:
    case HIPSPARSELT_COMPUTE_TF32_FAST:
        throw HIPSPARSELT_STATUS_NOT_SUPPORTED;

    case HIPSPARSELT_COMPUTE_32F:
        return rocsparselt_compute_f32;

    case HIPSPARSELT_COMPUTE_32I:
        return rocsparselt_compute_i32;
    }
    throw HIPSPARSELT_STATUS_INVALID_ENUM;
}

hipsparseLtComputetype_t RocSparseLtComputetypeToHIPComputetype(rocsparselt_compute_type_ type)
{
    switch(type)
    {
    case rocsparselt_compute_f32:
        return HIPSPARSELT_COMPUTE_32F;

    case rocsparselt_compute_i32:
        return HIPSPARSELT_COMPUTE_32I;
    }
    throw HIPSPARSELT_STATUS_INVALID_ENUM;
}

rocsparselt_operation_ HIPOperationToHCCOperation(hipsparseLtOperation_t op)
{
    switch(op)
    {
    case HIPSPARSELT_OPERATION_NON_TRANSPOSE:
        return rocsparselt_operation_none;
    case HIPSPARSELT_OPERATION_TRANSPOSE:
        return rocsparselt_operation_transpose;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

hipsparseLtOperation_t HCCOperationToHIPOperation(rocsparselt_operation_ op)
{
    switch(op)
    {
    case rocsparselt_operation_none:
        return HIPSPARSELT_OPERATION_NON_TRANSPOSE;
    case rocsparselt_operation_transpose:
        return HIPSPARSELT_OPERATION_TRANSPOSE;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

rocsparselt_order_ HIPOrderToHCCOrder(hipsparseLtOrder_t op)
{
    switch(op)
    {
    case HIPSPARSELT_ORDER_ROW:
        return rocsparselt_order_row;
    case HIPSPARSELT_ORDER_COLUMN:
        return rocsparselt_order_column;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

hipsparseLtOrder_t HCCOrderToHIPOrder(rocsparselt_order_ op)
{
    switch(op)
    {
    case rocsparselt_order_row:
        return HIPSPARSELT_ORDER_ROW;
    case rocsparselt_order_column:
        return HIPSPARSELT_ORDER_COLUMN;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

rocsparselt_sparsity_ HIPSparsityToRocSparseLtSparsity(hipsparseLtSparsity_t sparsity)
{
    switch(sparsity)
    {
    case HIPSPARSELT_SPARSITY_50_PERCENT:
        return rocsparselt_sparsity_50_percent;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

hipsparseLtSparsity_t RocSparseLtSparsityToHIPSparsity(rocsparselt_sparsity_ sparsity)
{
    switch(sparsity)
    {
    case rocsparselt_sparsity_50_percent:
        return HIPSPARSELT_SPARSITY_50_PERCENT;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

rocsparselt_matmul_descr_attribute_
    HIPMatmulDescAttributeToRocSparseLtMatmulDescAttribute(hipsparseLtMatmulDescAttribute_t attr)
{
    switch(attr)
    {
    case HIPSPARSELT_MATMUL_ACTIVATION_RELU:
        return rocsparselt_matmul_activation_relu;
    case HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND:
        return rocsparselt_matmul_activation_relu_upperbound;
    case HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD:
        return rocsparselt_matmul_activation_relu_threshold;
    case HIPSPARSELT_MATMUL_ACTIVATION_GELU:
        return rocsparselt_matmul_activation_gelu;
    case HIPSPARSELT_MATMUL_BIAS_STRIDE:
        return rocsparselt_matmul_bias_stride;
    case HIPSPARSELT_MATMUL_BIAS_POINTER:
        return rocsparselt_matmul_bias_pointer;
    case HIPSPARSELT_MATMUL_ACTIVATION_ABS:
        return rocsparselt_matmul_activation_abs;
    case HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU:
        return rocsparselt_matmul_activation_leakyrelu;
    case HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA:
        return rocsparselt_matmul_activation_leakyrelu_alpha;
    case HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID:
        return rocsparselt_matmul_activation_sigmoid;
    case HIPSPARSELT_MATMUL_ACTIVATION_TANH:
        return rocsparselt_matmul_activation_tanh;
    case HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA:
        return rocsparselt_matmul_activation_tanh_alpha;
    case HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA:
        return rocsparselt_matmul_activation_tanh_beta;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

hipsparseLtMatmulDescAttribute_t
    RocSparseLtMatmulDescAttributeToHIPMatmulDescAttribute(rocsparselt_matmul_descr_attribute_ attr)
{
    switch(attr)
    {
    case rocsparselt_matmul_activation_relu:
        return HIPSPARSELT_MATMUL_ACTIVATION_RELU;
    case rocsparselt_matmul_activation_relu_upperbound:
        return HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND;
    case rocsparselt_matmul_activation_relu_threshold:
        return HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD;
    case rocsparselt_matmul_activation_gelu:
        return HIPSPARSELT_MATMUL_ACTIVATION_GELU;
    case rocsparselt_matmul_bias_stride:
        return HIPSPARSELT_MATMUL_BIAS_STRIDE;
    case rocsparselt_matmul_bias_pointer:
        return HIPSPARSELT_MATMUL_BIAS_POINTER;
    case rocsparselt_matmul_activation_abs:
        return HIPSPARSELT_MATMUL_ACTIVATION_ABS;
    case rocsparselt_matmul_activation_leakyrelu:
        return HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU;
    case rocsparselt_matmul_activation_leakyrelu_alpha:
        return HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA;
    case rocsparselt_matmul_activation_sigmoid:
        return HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID;
    case rocsparselt_matmul_activation_tanh:
        return HIPSPARSELT_MATMUL_ACTIVATION_TANH;
    case rocsparselt_matmul_activation_tanh_alpha:
        return HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA;
    case rocsparselt_matmul_activation_tanh_beta:
        return HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

hipsparseLtMatDescAttribute_t
    RocSparseLtMatDescAttributeToHIPMatDescAttribute(rocsparselt_mat_descr_attribute_ attr)
{
    switch(attr)
    {
    case rocsparselt_mat_num_batches:
        return HIPSPARSELT_MAT_NUM_BATCHES;
    case rocsparselt_mat_batch_stride:
        return HIPSPARSELT_MAT_BATCH_STRIDE;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

rocsparselt_mat_descr_attribute_
    HIPMatDescAttributeToRocSparseLtMatDescAttribute(hipsparseLtMatDescAttribute_t attr)
{
    switch(attr)
    {
    case HIPSPARSELT_MAT_NUM_BATCHES:
        return rocsparselt_mat_num_batches;
    case HIPSPARSELT_MAT_BATCH_STRIDE:
        return rocsparselt_mat_batch_stride;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

rocsparselt_matmul_alg_ HIPMatmulAlgToRocSparseLtMatmulAlg(hipsparseLtMatmulAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSELT_MATMUL_ALG_DEFAULT:
        return rocsparselt_matmul_alg_default;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

hipsparseLtMatmulAlg_t RocSparseLtMatmulAlgToHIPMatmulAlg(rocsparselt_matmul_alg_ alg)
{
    switch(alg)
    {
    case rocsparselt_matmul_alg_default:
        return HIPSPARSELT_MATMUL_ALG_DEFAULT;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

rocsparselt_matmul_alg_attribute_
    HIPMatmulAlgAttributeToRocSparseLtAlgAttribute(hipsparseLtMatmulAlgAttribute_t alg)
{
    switch(alg)
    {
    case HIPSPARSELT_MATMUL_ALG_CONFIG_ID:
        return rocsparselt_matmul_alg_config_id;
    case HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID:
        return rocsparselt_matmul_alg_config_max_id;
    case HIPSPARSELT_MATMUL_SEARCH_ITERATIONS:
        return rocsparselt_matmul_search_iterations;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

hipsparseLtMatmulAlgAttribute_t
    RocSparseLtAlgAttributeToHIPMatmulAlgAttribute(rocsparselt_matmul_alg_attribute_ alg)
{
    switch(alg)
    {
    case rocsparselt_matmul_alg_config_id:
        return HIPSPARSELT_MATMUL_ALG_CONFIG_ID;
    case rocsparselt_matmul_alg_config_max_id:
        return HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID;
    case rocsparselt_matmul_search_iterations:
        return HIPSPARSELT_MATMUL_SEARCH_ITERATIONS;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

rocsparselt_prune_alg_ HIPPruneAlgToRocSparseLtPruneAlg(hipsparseLtPruneAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSELT_PRUNE_SPMMA_TILE:
        return rocsparselt_prune_smfmac_tile;
    case HIPSPARSELT_PRUNE_SPMMA_STRIP:
        return rocsparselt_prune_smfmac_strip;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

hipsparseLtPruneAlg_t RocSparseLtPruneAlgToHIPPruneAlg(rocsparselt_prune_alg_ alg)
{
    switch(alg)
    {
    case rocsparselt_prune_smfmac_tile:
        return HIPSPARSELT_PRUNE_SPMMA_TILE;
    case rocsparselt_prune_smfmac_strip:
        return HIPSPARSELT_PRUNE_SPMMA_STRIP;
    default:
        throw HIPSPARSELT_STATUS_INVALID_ENUM;
    }
}

hipsparseLtStatus_t hipsparseLtInit(hipsparseLtHandle_t* handle)
try
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return HIPSPARSELT_STATUS_INVALID_VALUE;
    }

    int                 deviceId;
    hipError_t          err;
    hipsparseLtStatus_t retval = HIPSPARSELT_STATUS_SUCCESS;

    err = hipGetDevice(&deviceId);
    if(err == hipSuccess)
    {
        retval = RocSparseLtStatusToHIPStatus(rocsparselt_init((rocsparselt_handle*)handle));
    }
    return retval;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtDestroy(const hipsparseLtHandle_t* handle)
try
{
    return RocSparseLtStatusToHIPStatus(rocsparselt_destroy((const rocsparselt_handle*)handle));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

/* matrix descriptor */
// dense matrix
hipsparseLtStatus_t hipsparseLtDenseDescriptorInit(const hipsparseLtHandle_t*  handle,
                                                   hipsparseLtMatDescriptor_t* matDescr,
                                                   int64_t                     rows,
                                                   int64_t                     cols,
                                                   int64_t                     ld,
                                                   uint32_t                    alignment,
                                                   hipsparseLtDatatype_t       valueType,
                                                   hipsparseLtOrder_t          order)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_dense_descr_init((const rocsparselt_handle*)handle,
                                     (rocsparselt_mat_descr*)matDescr,
                                     rows,
                                     cols,
                                     ld,
                                     alignment,
                                     HIPDatatypeToRocSparseLtDatatype(valueType),
                                     HIPOrderToHCCOrder(order)));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

// structured matrix
hipsparseLtStatus_t hipsparseLtStructuredDescriptorInit(const hipsparseLtHandle_t*  handle,
                                                        hipsparseLtMatDescriptor_t* matDescr,
                                                        int64_t                     rows,
                                                        int64_t                     cols,
                                                        int64_t                     ld,
                                                        uint32_t                    alignment,
                                                        hipsparseLtDatatype_t       valueType,
                                                        hipsparseLtOrder_t          order,
                                                        hipsparseLtSparsity_t       sparsity)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_structured_descr_init((const rocsparselt_handle*)handle,
                                          (rocsparselt_mat_descr*)matDescr,
                                          rows,
                                          cols,
                                          ld,
                                          alignment,
                                          HIPDatatypeToRocSparseLtDatatype(valueType),
                                          HIPOrderToHCCOrder(order),
                                          HIPSparsityToRocSparseLtSparsity(sparsity)));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtMatDescriptorDestroy(const hipsparseLtMatDescriptor_t* matDescr)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_mat_descr_destroy((const rocsparselt_mat_descr*)matDescr));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtMatDescSetAttribute(const hipsparseLtHandle_t*    handle,
                                                   hipsparseLtMatDescriptor_t*   matmulDescr,
                                                   hipsparseLtMatDescAttribute_t matAttribute,
                                                   const void*                   data,
                                                   size_t                        dataSize)
try
{
    return RocSparseLtStatusToHIPStatus(rocsparselt_mat_descr_set_attribute(
        (const rocsparselt_handle*)handle,
        (rocsparselt_mat_descr*)matmulDescr,
        HIPMatDescAttributeToRocSparseLtMatDescAttribute(matAttribute),
        data,
        dataSize));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtMatDescGetAttribute(const hipsparseLtHandle_t*        handle,
                                                   const hipsparseLtMatDescriptor_t* matmulDescr,
                                                   hipsparseLtMatDescAttribute_t     matAttribute,
                                                   void*                             data,
                                                   size_t                            dataSize)
try
{
    return RocSparseLtStatusToHIPStatus(rocsparselt_mat_descr_get_attribute(
        (const rocsparselt_handle*)handle,
        (const rocsparselt_mat_descr*)matmulDescr,
        HIPMatDescAttributeToRocSparseLtMatDescAttribute(matAttribute),
        data,
        dataSize));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

/* matmul descriptor */
hipsparseLtStatus_t hipsparseLtMatmulDescriptorInit(const hipsparseLtHandle_t*        handle,
                                                    hipsparseLtMatmulDescriptor_t*    matmulDescr,
                                                    hipsparseLtOperation_t            opA,
                                                    hipsparseLtOperation_t            opB,
                                                    const hipsparseLtMatDescriptor_t* matA,
                                                    const hipsparseLtMatDescriptor_t* matB,
                                                    const hipsparseLtMatDescriptor_t* matC,
                                                    const hipsparseLtMatDescriptor_t* matD,
                                                    hipsparseLtComputetype_t          computeType)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_matmul_descr_init((const rocsparselt_handle*)handle,
                                      (rocsparselt_matmul_descr*)matmulDescr,
                                      HIPOperationToHCCOperation(opA),
                                      HIPOperationToHCCOperation(opB),
                                      (const rocsparselt_mat_descr*)matA,
                                      (const rocsparselt_mat_descr*)matB,
                                      (const rocsparselt_mat_descr*)matC,
                                      (const rocsparselt_mat_descr*)matD,
                                      HIPComputetypeToRocSparseLtComputetype(computeType)));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t
    hipsparseLtMatmulDescSetAttribute(const hipsparseLtHandle_t*       handle,
                                      hipsparseLtMatmulDescriptor_t*   matmulDescr,
                                      hipsparseLtMatmulDescAttribute_t matmulAttribute,
                                      const void*                      data,
                                      size_t                           dataSize)
try
{
    return RocSparseLtStatusToHIPStatus(rocsparselt_matmul_descr_set_attribute(
        (const rocsparselt_handle*)handle,
        (rocsparselt_matmul_descr*)matmulDescr,
        HIPMatmulDescAttributeToRocSparseLtMatmulDescAttribute(matmulAttribute),
        data,
        dataSize));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t
    hipsparseLtMatmulDescGetAttribute(const hipsparseLtHandle_t*           handle,
                                      const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                      hipsparseLtMatmulDescAttribute_t     matmulAttribute,
                                      void*                                data,
                                      size_t                               dataSize)
try
{
    return RocSparseLtStatusToHIPStatus(rocsparselt_matmul_descr_get_attribute(
        (const rocsparselt_handle*)handle,
        (const rocsparselt_matmul_descr*)matmulDescr,
        HIPMatmulDescAttributeToRocSparseLtMatmulDescAttribute(matmulAttribute),
        data,
        dataSize));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

/* algorithm selection */
hipsparseLtStatus_t
    hipsparseLtMatmulAlgSelectionInit(const hipsparseLtHandle_t*           handle,
                                      hipsparseLtMatmulAlgSelection_t*     algSelection,
                                      const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                      hipsparseLtMatmulAlg_t               alg)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_matmul_alg_selection_init((const rocsparselt_handle*)handle,
                                              (rocsparselt_matmul_alg_selection*)algSelection,
                                              (const rocsparselt_matmul_descr*)matmulDescr,
                                              HIPMatmulAlgToRocSparseLtMatmulAlg(alg)));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtMatmulAlgSetAttribute(const hipsparseLtHandle_t*       handle,
                                                     hipsparseLtMatmulAlgSelection_t* algSelection,
                                                     hipsparseLtMatmulAlgAttribute_t  attribute,
                                                     const void*                      data,
                                                     size_t                           dataSize)
try
{
    return RocSparseLtStatusToHIPStatus(rocsparselt_matmul_alg_set_attribute(
        (const rocsparselt_handle*)handle,
        (rocsparselt_matmul_alg_selection*)algSelection,
        HIPMatmulAlgAttributeToRocSparseLtAlgAttribute(attribute),
        data,
        dataSize));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t
    hipsparseLtMatmulAlgGetAttribute(const hipsparseLtHandle_t*             handle,
                                     const hipsparseLtMatmulAlgSelection_t* algSelection,
                                     hipsparseLtMatmulAlgAttribute_t        attribute,
                                     void*                                  data,
                                     size_t                                 dataSize)
try
{
    return RocSparseLtStatusToHIPStatus(rocsparselt_matmul_alg_get_attribute(
        (const rocsparselt_handle*)handle,
        (const rocsparselt_matmul_alg_selection*)algSelection,
        HIPMatmulAlgAttributeToRocSparseLtAlgAttribute(attribute),
        data,
        dataSize));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

/* matmul plan */
hipsparseLtStatus_t hipsparseLtMatmulGetWorkspace(const hipsparseLtHandle_t*     handle,
                                                  const hipsparseLtMatmulPlan_t* plan,
                                                  size_t*                        workspaceSize)
try
{
    return RocSparseLtStatusToHIPStatus(rocsparselt_matmul_get_workspace(
        (const rocsparselt_handle*)handle, (const rocsparselt_matmul_plan*)plan, workspaceSize));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtMatmulPlanInit(const hipsparseLtHandle_t*             handle,
                                              hipsparseLtMatmulPlan_t*               plan,
                                              const hipsparseLtMatmulDescriptor_t*   matmulDescr,
                                              const hipsparseLtMatmulAlgSelection_t* algSelection,
                                              size_t                                 workspaceSize)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_matmul_plan_init((const rocsparselt_handle*)handle,
                                     (rocsparselt_matmul_plan*)plan,
                                     (const rocsparselt_matmul_descr*)matmulDescr,
                                     (const rocsparselt_matmul_alg_selection*)algSelection,
                                     workspaceSize));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtMatmulPlanDestroy(const hipsparseLtMatmulPlan_t* plan)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_matmul_plan_destroy((const rocsparselt_matmul_plan*)plan));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

/* matmul execution */
hipsparseLtStatus_t hipsparseLtMatmul(const hipsparseLtHandle_t*     handle,
                                      const hipsparseLtMatmulPlan_t* plan,
                                      const void*                    alpha,
                                      const void*                    d_A,
                                      const void*                    d_B,
                                      const void*                    beta,
                                      const void*                    d_C,
                                      void*                          d_D,
                                      void*                          workspace,
                                      hipStream_t*                   streams,
                                      int32_t                        numStreams)
try
{
    return RocSparseLtStatusToHIPStatus(rocsparselt_matmul((const rocsparselt_handle*)handle,
                                                           (const rocsparselt_matmul_plan*)plan,
                                                           alpha,
                                                           d_A,
                                                           d_B,
                                                           beta,
                                                           d_C,
                                                           d_D,
                                                           workspace,
                                                           streams,
                                                           numStreams));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtMatmulSearch(const hipsparseLtHandle_t* handle,
                                            hipsparseLtMatmulPlan_t*   plan,
                                            const void*                alpha,
                                            const void*                d_A,
                                            const void*                d_B,
                                            const void*                beta,
                                            const void*                d_C,
                                            void*                      d_D,
                                            void*                      workspace,
                                            hipStream_t*               streams,
                                            int32_t                    numStreams)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_matmul_search((const rocsparselt_handle*)handle,
                                  (const rocsparselt_matmul_plan*)plan,
                                  alpha,
                                  d_A,
                                  d_B,
                                  beta,
                                  d_C,
                                  d_D,
                                  workspace,
                                  streams,
                                  numStreams));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

/* helper */
// prune
hipsparseLtStatus_t hipsparseLtSpMMAPrune(const hipsparseLtHandle_t*           handle,
                                          const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                          const void*                          d_in,
                                          void*                                d_out,
                                          hipsparseLtPruneAlg_t                pruneAlg,
                                          hipStream_t                          stream)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_prune((const rocsparselt_handle*)handle,
                                 (const rocsparselt_matmul_descr*)matmulDescr,
                                 d_in,
                                 d_out,
                                 HIPPruneAlgToRocSparseLtPruneAlg(pruneAlg),
                                 stream));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtSpMMAPruneCheck(const hipsparseLtHandle_t*           handle,
                                               const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                               const void*                          d_in,
                                               int*                                 valid,
                                               hipStream_t                          stream)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_prune_check((const rocsparselt_handle*)handle,
                                       (const rocsparselt_matmul_descr*)matmulDescr,
                                       d_in,
                                       valid,
                                       stream));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtSpMMAPrune2(const hipsparseLtHandle_t*        handle,
                                           const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                           int                               isSparseA,
                                           hipsparseLtOperation_t            op,
                                           const void*                       d_in,
                                           void*                             d_out,
                                           hipsparseLtPruneAlg_t             pruneAlg,
                                           hipStream_t                       stream)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_prune2((const rocsparselt_handle*)handle,
                                  (const rocsparselt_mat_descr*)sparseMatDescr,
                                  isSparseA,
                                  HIPOperationToHCCOperation(op),
                                  d_in,
                                  d_out,
                                  HIPPruneAlgToRocSparseLtPruneAlg(pruneAlg),
                                  stream));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtSpMMAPruneCheck2(const hipsparseLtHandle_t*        handle,
                                                const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                                int                               isSparseA,
                                                hipsparseLtOperation_t            op,
                                                const void*                       d_in,
                                                int*                              d_valid,
                                                hipStream_t                       stream)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_prune_check2((const rocsparselt_handle*)handle,
                                        (const rocsparselt_mat_descr*)sparseMatDescr,
                                        isSparseA,
                                        HIPOperationToHCCOperation(op),
                                        d_in,
                                        d_valid,
                                        stream));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

// compression
hipsparseLtStatus_t hipsparseLtSpMMACompressedSize(const hipsparseLtHandle_t*     handle,
                                                   const hipsparseLtMatmulPlan_t* plan,
                                                   size_t*                        compressedSize)
try
{
    return RocSparseLtStatusToHIPStatus(rocsparselt_smfmac_compressed_size(
        (const rocsparselt_handle*)handle, (const rocsparselt_matmul_plan*)plan, compressedSize));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtSpMMACompress(const hipsparseLtHandle_t*     handle,
                                             const hipsparseLtMatmulPlan_t* plan,
                                             const void*                    d_dense,
                                             void*                          d_compressed,
                                             hipStream_t                    stream)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_compress((const rocsparselt_handle*)handle,
                                    (const rocsparselt_matmul_plan*)plan,
                                    d_dense,
                                    d_compressed,
                                    stream));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t
    hipsparseLtSpMMACompressedSize2(const hipsparseLtHandle_t*        handle,
                                    const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                    size_t*                           compressedSize)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_compressed_size2((const rocsparselt_handle*)handle,
                                            (const rocsparselt_mat_descr*)sparseMatDescr,
                                            compressedSize));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtSpMMACompress2(const hipsparseLtHandle_t*        handle,
                                              const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                              int                               isSparseA,
                                              hipsparseLtOperation_t            op,
                                              const void*                       d_dense,
                                              void*                             d_compressed,
                                              hipStream_t                       stream)
try
{
    return RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_compress2((const rocsparselt_handle*)handle,
                                     (const rocsparselt_mat_descr*)sparseMatDescr,
                                     isSparseA,
                                     HIPOperationToHCCOperation(op),
                                     d_dense,
                                     d_compressed,
                                     stream));
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

void hipsparseLtInitialize()
{
    rocsparselt_initialize();
}

hipsparseLtStatus_t hipsparseLtGetVersion(hipsparseLtHandle_t handle, int* version)
try
{
    if(handle == nullptr)
    {
        return HIPSPARSELT_STATUS_NOT_INITIALIZED;
    }

    *version = hipsparseltVersionMajor * 100000 + hipsparseltVersionMinor * 100
               + hipsparseltVersionPatch;

    return HIPSPARSELT_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtGetGitRevision(hipsparseLtHandle_t handle, char* rev)
try
{
    // Get hipSPARSE revision
    if(handle == nullptr)
    {
        return HIPSPARSELT_STATUS_NOT_INITIALIZED;
    }

    if(rev == nullptr)
    {
        return HIPSPARSELT_STATUS_INVALID_VALUE;
    }

    static constexpr char v[] = TO_STR(hipsparseltVersionTweak);

    memcpy(rev, v, sizeof(v));

    return HIPSPARSELT_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseLtStatus_t hipsparseLtGetArchName(char** archName)
try
{
    *archName        = nullptr;
    std::string arch = rocsparselt_internal_get_arch_name();
    *archName        = (char*)malloc(arch.size() * sizeof(char));
    strncpy(*archName, arch.c_str(), arch.size());
    return HIPSPARSELT_STATUS_SUCCESS;
}
catch(...)
{
    if(archName != nullptr)
    {
        free(*archName);
        *archName = nullptr;
    }
    return exception_to_hipsparselt_status();
}

#ifdef __cplusplus
}
#endif
