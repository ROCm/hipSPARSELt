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
rocsparse_status rocsparselt_init(rocsparselt_handle* handle)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else
    {
        *handle = nullptr;
        // Allocate
        try
        {
            *handle = new _rocsparselt_handle();
            log_trace(*handle, "rocsparselt_init");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief destroy handle
 *******************************************************************************/
rocsparse_status rocsparselt_destroy(const rocsparselt_handle handle)
{
    log_trace(handle, "rocsparse_destroy");
    // Destruct
    try
    {
        delete handle;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_mat_descr is a structure holding the rocsparselt matrix
 * content. It must be initialized using rocsparselt_dense_descr_init() or
 * rocsparselt_structured_descr_init()  and the retured handle must be passed
 * to all subsequent library function calls that involve the matrix.
 * It should be destroyed at the end using rocsparselt_mat_descr_destroy().
 *******************************************************************************/
rocsparse_status rocsparselt_dense_descr_init(const rocsparselt_handle handle,
                                              rocsparselt_mat_descr*   matDescr,
                                              int64_t                  rows,
                                              int64_t                  cols,
                                              int64_t                  ld,
                                              uint32_t                 alignment,
                                              rocsparselt_datatype     valueType,
                                              rocsparse_order          order)
{
    // Check if matDescr is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(matDescr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(order != rocsparse_order_column)
    {
        return rocsparse_status_not_implemented;
    }
    else
    {
        *matDescr = nullptr;
        // Allocate
        try
        {
            auto status = validateMatrixArgs(
                handle, rows, cols, ld, alignment, valueType, order, rocsparselt_matrix_type_dense);
            if(status != rocsparse_status_success)
                throw status;

            *matDescr              = new _rocsparselt_mat_descr();
            (*matDescr)->m_type    = rocsparselt_matrix_type_dense;
            (*matDescr)->m         = rows;
            (*matDescr)->n         = cols;
            (*matDescr)->ld        = ld;
            (*matDescr)->alignment = alignment;
            (*matDescr)->type      = valueType;
            (*matDescr)->order     = order;
            log_trace(handle, "rocsparselt_dense_descr_init");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief rocsparse_mat_descr is a structure holding the rocsparselt matrix
 * content. It must be initialized using rocsparselt_dense_descr_init() or
 * rocsparselt_structured_descr_init()  and the retured handle must be passed
 * to all subsequent library function calls that involve the matrix.
 * It should be destroyed at the end using rocsparselt_mat_descr_destroy().
 *******************************************************************************/
rocsparse_status rocsparselt_structured_descr_init(const rocsparselt_handle handle,
                                                   rocsparselt_mat_descr*   matDescr,
                                                   int64_t                  rows,
                                                   int64_t                  cols,
                                                   int64_t                  ld,
                                                   uint32_t                 alignment,
                                                   rocsparselt_datatype     valueType,
                                                   rocsparse_order          order,
                                                   rocsparselt_sparsity     sparsity)

{
    // Check if matDescr is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(matDescr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else
    {
        *matDescr = nullptr;
        // Allocate
        try
        {
            auto status = validateMatrixArgs(handle,
                                             rows,
                                             cols,
                                             ld,
                                             alignment,
                                             valueType,
                                             order,
                                             rocsparselt_matrix_type_structured);
            if(status != rocsparse_status_success)
                throw status;

            *matDescr              = new _rocsparselt_mat_descr();
            (*matDescr)->m_type    = rocsparselt_matrix_type_structured;
            (*matDescr)->m         = rows;
            (*matDescr)->n         = cols;
            (*matDescr)->ld        = ld;
            (*matDescr)->alignment = alignment;
            (*matDescr)->type      = valueType;
            (*matDescr)->order     = order;
            (*matDescr)->sparsity  = sparsity;

            log_trace(handle, "rocsparselt_structured_descr_init");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief destroy matrix descriptor
 *******************************************************************************/
rocsparse_status rocsparselt_mat_descr_destroy(const rocsparselt_mat_descr matDescr)
{
    if(matDescr == nullptr)
        return rocsparse_status_invalid_pointer;
    // Destruct
    try
    {
        constexpr size_t attrs = sizeof(matDescr->attributes) / sizeof(matDescr->attributes[0]);
        for(int i = 0; i < attrs; i++)
        {
            matDescr->attributes[i].clear();
        }
        delete matDescr;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix descriptor
 * such as number of batches and their stride.
 *******************************************************************************/
rocsparse_status rocsparselt_mat_descr_set_attribute(const rocsparselt_handle        handle,
                                                     rocsparselt_mat_descr           matDescr,
                                                     rocsparselt_mat_descr_attribute matAttribute,
                                                     const void*                     data,
                                                     size_t                          dataSize)

{
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(data == nullptr || matDescr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(dataSize <= 0)
    {
        return rocsparse_status_invalid_value;
    }
    else
    {
        // Allocate
        try
        {
            matDescr->attributes[matAttribute].set(data, dataSize);
            log_trace(handle, "rocsparselt_mat_descr_set_attribute");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix descriptor
 * such as number of batches and their stride.
 *******************************************************************************/
rocsparse_status rocsparselt_mat_descr_get_attribute(const rocsparselt_handle        handle,
                                                     const rocsparselt_mat_descr     matDescr,
                                                     rocsparselt_mat_descr_attribute matAttribute,
                                                     void*                           data,
                                                     size_t                          dataSize)
{
    // Check if matmulDescr is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(data == nullptr || matDescr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(dataSize <= 0)
    {
        return rocsparse_status_invalid_value;
    }
    else
    {
        try
        {
            if(matDescr->attributes[matAttribute].get(data, dataSize) == 0)
                return rocsparse_status_invalid_value;
            log_trace(handle, "rocsparselt_mat_descr_get_attribute");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparse_status rocsparselt_matmul_descr_init(const rocsparselt_handle  handle,
                                               rocsparselt_matmul_descr* matmulDescr,
                                               rocsparse_operation       opA,
                                               rocsparse_operation       opB,
                                               rocsparselt_mat_descr     matA,
                                               rocsparselt_mat_descr     matB,
                                               rocsparselt_mat_descr     matC,
                                               rocsparselt_mat_descr     matD,
                                               rocsparselt_compute_type  computeType)
{
    // Check if matmulDescr is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(matmulDescr == nullptr || matA == nullptr || matB == nullptr || matC == nullptr
            || matD == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else
    {
        *matmulDescr = nullptr;
        // Allocate
        try
        {
            auto status = validateMatmulDescrArgs(handle,
                                                  opA,
                                                  opB,
                                                  matA->m,
                                                  matA->n,
                                                  matA->ld,
                                                  matB->m,
                                                  matB->n,
                                                  matB->ld,
                                                  matC->m,
                                                  matC->n,
                                                  matC->ld,
                                                  matD->m,
                                                  matD->n,
                                                  matD->ld,
                                                  matA->type,
                                                  matB->type,
                                                  matC->type,
                                                  matD->type,
                                                  computeType,
                                                  matA->m_type,
                                                  matB->m_type,
                                                  matC->m_type,
                                                  matD->m_type);
            if(status != rocsparse_status_success)
                return status;

            int64_t m, n, k;
            getOriginalSizes(opA, opB, matA->m, matA->n, matB->m, matB->n, m, n, k);
            matA->c_k  = k / 2;
            matA->c_ld = (opA == rocsparse_operation_transpose ? matA->c_k : m);

            *matmulDescr                 = new _rocsparselt_matmul_descr();
            (*matmulDescr)->op_A         = opA;
            (*matmulDescr)->op_B         = opB;
            (*matmulDescr)->matrix_A     = matA;
            (*matmulDescr)->matrix_B     = matB;
            (*matmulDescr)->matrix_C     = matC;
            (*matmulDescr)->matrix_D     = matD;
            (*matmulDescr)->compute_type = computeType;
            log_trace(handle, "rocsparselt_matmul_descr_init");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief destroy matrix multiplication descriptor
 *******************************************************************************/
rocsparse_status rocsparselt_matmul_descr_destroy(const rocsparselt_matmul_descr matmulDescr)
{
    if(matmulDescr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Destruct
    try
    {
        matmulDescr->bias_pointer.clear();
        delete matmulDescr;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix multiplication
 * descriptor.
 *******************************************************************************/
rocsparse_status
    rocsparselt_matmul_descr_set_attribute(const rocsparselt_handle           handle,
                                           rocsparselt_matmul_descr           matmulDescr,
                                           rocsparselt_matmul_descr_attribute matmulAttribute,
                                           const void*                        data,
                                           size_t                             dataSize)
{
    // Check if matmulDescr is valid
    if(matmulDescr == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(data == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(dataSize <= 0)
    {
        return rocsparse_status_invalid_value;
    }
    else
    {
        // Allocate
        try
        {
            switch(matmulAttribute)
            {
            case rocsparselt_matmul_activation_relu:
                if(sizeof(int) == dataSize)
                    memcpy(&matmulDescr->activation_relu, data, sizeof(int));
                else
                    return rocsparse_status_invalid_value;
                break;
            case rocsparselt_matmul_activation_relu_upperbound:
                if(sizeof(float) == dataSize)
                    memcpy(&matmulDescr->activation_relu_upperbound, data, sizeof(float));
                else
                    return rocsparse_status_invalid_value;
                break;
            case rocsparselt_matmul_activation_relu_threshold:
                if(sizeof(float) == dataSize)
                    memcpy(&matmulDescr->activation_relu_threshold, data, sizeof(float));
                else
                    return rocsparse_status_invalid_value;
                break;
            case rocsparselt_matmul_activation_gelu:
                if(sizeof(int) == dataSize)
                    memcpy(&matmulDescr->activation_gelu, data, sizeof(int));
                else
                    return rocsparse_status_invalid_value;
                break;
            case rocsparselt_matmul_bias_pointer:

                //TODO Check the bias vector size is equal to the number of rows of the output matrix D.

                matmulDescr->bias_pointer.set(data, dataSize);
                break;
            case rocsparselt_matmul_bias_stride:
                if(sizeof(int64_t) == dataSize)
                    memcpy(&matmulDescr->bias_stride, data, sizeof(int64_t));
                else
                    return rocsparse_status_invalid_value;
                break;
            }
            log_trace(handle, "rocsparselt_matmul_descr_set_attribute");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix descriptor
 * such as number of batches and their stride.
 *******************************************************************************/
rocsparse_status
    rocsparselt_matmul_descr_get_attribute(const rocsparselt_handle           handle,
                                           rocsparselt_matmul_descr           matmulDescr,
                                           rocsparselt_matmul_descr_attribute matmulAttribute,
                                           void*                              data,
                                           size_t                             dataSize)

{
    // Check if matmulDescr is valid
    if(matmulDescr == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(data == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else
    {
        try
        {
            switch(matmulAttribute)
            {
            case rocsparselt_matmul_activation_relu:
                if(dataSize < sizeof(int))
                    return rocsparse_status_invalid_value;
                memcpy(data, &matmulDescr->activation_relu, sizeof(int));
                break;
            case rocsparselt_matmul_activation_relu_upperbound:
                if(dataSize < sizeof(float))
                    return rocsparse_status_invalid_value;
                memcpy(data, &matmulDescr->activation_relu_upperbound, sizeof(float));
                break;
            case rocsparselt_matmul_activation_relu_threshold:
                if(dataSize < sizeof(float))
                    return rocsparse_status_invalid_value;
                memcpy(data, &matmulDescr->activation_relu_threshold, sizeof(float));
                break;
            case rocsparselt_matmul_activation_gelu:
                if(dataSize < sizeof(int))
                    return rocsparse_status_invalid_value;
                memcpy(data, &matmulDescr->activation_gelu, sizeof(int));
                break;
            case rocsparselt_matmul_bias_pointer:
                matmulDescr->bias_pointer.get(data, dataSize);
                break;
            case rocsparselt_matmul_bias_stride:
                if(dataSize < sizeof(int64_t))
                    return rocsparse_status_invalid_value;
                memcpy(data, &matmulDescr->bias_stride, sizeof(int64_t));
                break;
            }
            log_trace(handle, "rocsparselt_matmul_descr_get_attribute");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparse_status
    rocsparselt_matmul_alg_selection_init(const rocsparselt_handle          handle,
                                          rocsparselt_matmul_alg_selection* algSelection,
                                          const rocsparselt_matmul_descr    matmulDescr,
                                          rocsparselt_matmul_alg            alg)
{
    // Check if algSelection is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(algSelection == nullptr || matmulDescr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else
    {
        *algSelection = nullptr;
        // Allocate
        try
        {
            *algSelection        = new _rocsparselt_matmul_alg_selection();
            (*algSelection)->alg = alg;

            auto in_type      = matmulDescr->matrix_A->type;
            auto out_type     = matmulDescr->matrix_D->type;
            auto compute_type = matmulDescr->compute_type;

            int config_max_id = 0;
#if 0
            if(in_type == rocsparselt_datatype_f16_r && out_type == rocsparselt_datatype_f16_r
               && compute_type == rocsparselt_compute_f32)
                config_max_id = rocsparselt_get_matmul_alg_config_max_id<rocsparselt_half,
                                                                         rocsparselt_half,
                                                                         float>(matmulDescr->op_A,
                                                                                matmulDescr->op_B);
                generate_kernel_category_str<ocsparselt_half, rocsparselt_half, float>(matmulDescr->op_A, matmulDescr->op_B);
            else if(in_type == rocsparselt_datatype_bf16_r
                    && out_type == rocsparselt_datatype_bf16_r
                    && compute_type == rocsparselt_compute_f32)
                config_max_id = rocsparselt_get_matmul_alg_config_max_id<rocsparselt_bfloat16,
                                                                         rocsparselt_bfloat16,
                                                                         float>(matmulDescr->op_A,
                                                                                matmulDescr->op_B);
#else
            std::string str;
            if(in_type == rocsparselt_datatype_f16_r && out_type == rocsparselt_datatype_f16_r
               && compute_type == rocsparselt_compute_f32)
                str = generate_kernel_category_str<rocsparselt_half, rocsparselt_half, float>(
                    matmulDescr->op_A, matmulDescr->op_B);
            else if(in_type == rocsparselt_datatype_bf16_r
                    && out_type == rocsparselt_datatype_bf16_r
                    && compute_type == rocsparselt_compute_f32)
                str = generate_kernel_category_str<rocsparselt_bfloat16,
                                                   rocsparselt_bfloat16,
                                                   float>(matmulDescr->op_A, matmulDescr->op_B);
            config_max_id = getKernelCounts(handle, str);
#endif

            if(!config_max_id)
            {
                delete(*algSelection);
                return rocsparse_status_not_implemented;
            }

            (*algSelection)->attributes[rocsparselt_matmul_alg_config_max_id].set(&config_max_id);

            const int search_iterations = 10;
            (*algSelection)
                ->attributes[rocsparselt_matmul_search_iterations]
                .set(&search_iterations);
            log_trace(handle, "rocsparselt_matmul_alg_selection_init");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief destroy matrix multiplication descriptor
 *******************************************************************************/
rocsparse_status
    rocsparselt_matmul_alg_selection_destroy(const rocsparselt_matmul_alg_selection algSelection)
{
    if(algSelection == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Destruct
    try
    {
        constexpr size_t attrs
            = sizeof(algSelection->attributes) / sizeof(algSelection->attributes[0]);
        for(int i = 0; i < attrs; i++)
        {
            algSelection->attributes[i].clear();
        }
        delete algSelection;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparse_status rocsparselt_matmul_alg_set_attribute(const rocsparselt_handle         handle,
                                                      rocsparselt_matmul_alg_selection algSelection,
                                                      rocsparselt_matmul_alg_attribute attribute,
                                                      const void*                      data,
                                                      size_t                           dataSize)
{
    // Check if algSelection is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(data == nullptr || algSelection == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(dataSize <= 0)
    {
        return rocsparse_status_invalid_value;
    }
    else
    {
        // Allocate
        try
        {
            if(attribute == rocsparselt_matmul_alg_config_id)
            {
                int config_max_id;
                algSelection->attributes[rocsparselt_matmul_alg_config_max_id].get(&config_max_id);

                const int* config_id = reinterpret_cast<const int*>(data);
                if(*config_id >= config_max_id)
                {
                    rocsparselt_cerr << "the value of rocsparselt_matmul_alg_config_id data"
                                     << config_id << "is out of the range [0, "
                                     << (config_max_id - 1) << "]" << std::endl;
                    return rocsparse_status_invalid_value;
                }
            }
            else if(attribute == rocsparselt_matmul_alg_config_max_id)
            {
                rocsparselt_cerr << "rocsparselt_matmul_alg_config_max_id is query only."
                                 << std::endl;
                return rocsparse_status_invalid_value;
            }
            algSelection->attributes[attribute].set(data, dataSize);
            log_trace(handle, "rocsparselt_matmul_alg_set_attribute");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparse_status rocsparselt_matmul_alg_get_attribute(const rocsparselt_handle         handle,
                                                      rocsparselt_matmul_alg_selection algSelection,
                                                      rocsparselt_matmul_alg_attribute attribute,
                                                      void*                            data,
                                                      size_t                           dataSize)

{
    // Check if matmulDescr is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(data == nullptr || algSelection == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(dataSize <= 0)
    {
        return rocsparse_status_invalid_value;
    }
    else
    {
        try
        {
            if(algSelection->attributes[attribute].get(data, dataSize) == 0)
                return rocsparse_status_invalid_value;
            log_trace(handle, "rocsparselt_matmul_alg_get_attribute");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparse_status rocsparselt_matmul_plan_init(const rocsparselt_handle               handle,
                                              rocsparselt_matmul_plan*               plan,
                                              const rocsparselt_matmul_descr         matmulDescr,
                                              const rocsparselt_matmul_alg_selection algSelection,
                                              size_t                                 workspaceSize)

{
    // Check if plan is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(plan == nullptr || matmulDescr == nullptr || algSelection == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else
    {
        *plan = nullptr;
        // Allocate
        try
        {
            *plan                   = new _rocsparselt_matmul_plan();
            (*plan)->matmul_descr   = matmulDescr;
            (*plan)->alg_selection  = algSelection;
            (*plan)->workspace_size = workspaceSize;
            log_trace(handle, "rocsparselt_matmul_plan_init");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief destroy matrix multiplication plan descriptor
 *******************************************************************************/
rocsparse_status rocsparselt_matmul_plan_destroy(const rocsparselt_matmul_plan plan)
{
    if(plan == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    // Destruct
    try
    {
        delete plan;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Get rocSPARSELt version
 * version % 100        = patch level
 * version / 100 % 1000 = minor version
 * version / 100000     = major version
 *******************************************************************************/
rocsparse_status rocsparselt_get_version(rocsparselt_handle handle, int* version)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    *version = ROCSPARSELT_VERSION_MAJOR * 100000 + ROCSPARSELT_VERSION_MINOR * 100
               + ROCSPARSELT_VERSION_PATCH;

    log_trace(handle, "rocsparselt_get_version", *version);

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Get rocSPARSELt git revision
 *******************************************************************************/
rocsparse_status rocsparselt_get_git_rev(rocsparselt_handle handle, char* rev)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(rev == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    static constexpr char v[] = TO_STR(ROCSPARSELT_VERSION_TWEAK);

    memcpy(rev, v, sizeof(v));

    log_trace(handle, "rocsparselt_get_git_rev", rev);

    return rocsparse_status_success;
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

// exported. Get architecture name
std::string rocsparselt_internal_get_arch_name()
{
    int deviceId;
    hipGetDevice(&deviceId);
    hipDeviceProp_t deviceProperties;
    hipGetDeviceProperties(&deviceProperties, deviceId);
    return ArchName<hipDeviceProp_t>{}(deviceProperties);
}
