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

#include "rocsparselt_spmm.hpp"
#include "definitions.h"
#include "handle.h"
#include "rocsparselt_spmm_utils.hpp"
#include "utility.hpp"

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status
    rocsparselt_matmul_get_workspace(const rocsparselt_handle*               handle,
                                     const rocsparselt_matmul_alg_selection* algSelection,
                                     size_t*                                 workspaceSize)

{
    // Check if handle is valid
    if(handle == nullptr || algSelection == nullptr || *handle == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(workspaceSize == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }

    {
        //TODO
        *workspaceSize = 0;
        return rocsparselt_status_success;
    }
}

rocsparselt_status rocsparselt_matmul_impl(const rocsparselt_handle      handle,
                                           const rocsparselt_matmul_plan plan,
                                           const void*                   alpha,
                                           const void*                   d_A,
                                           const void*                   d_B,
                                           const void*                   beta,
                                           const void*                   d_C,
                                           void*                         d_D,
                                           void*                         workspace,
                                           hipStream_t*                  streams,
                                           int32_t                       numStreams,
                                           bool                          search = false)
{
    // Check if handle is valid
    if(handle == nullptr || plan == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(alpha == nullptr || beta == nullptr || d_A == nullptr || d_B == nullptr || d_C == nullptr
       || d_D == nullptr)
    {
        return rocsparselt_status_invalid_pointer;
    }

    if(workspace == nullptr && plan->workspace_size != 0)
    {
        hipsparselt_cerr << "The parameter number 9 (workspace) had an illegal value: "
                            "expected a device memroy with "
                         << plan->workspace_size << " bytes, but current is nullptr" << std::endl;
        return rocsparselt_status_invalid_value;
    }

    if(numStreams < 0)
    {
        hipsparselt_cerr << "The parameter number 11 (numStreams) had an illegal value: "
                         << numStreams << std::endl;
        return rocsparselt_status_invalid_value;
    }
    else if(streams == nullptr && numStreams > 0)
    {
        hipsparselt_cerr << "The parameter number 10 (streams) had an illegal value: nullptr"
                         << std::endl;
        return rocsparselt_status_invalid_value;
    }

    rocsparselt_operation    opA          = plan->matmul_descr->op_A;
    rocsparselt_operation    opB          = plan->matmul_descr->op_B;
    rocsparselt_compute_type compute_type = plan->matmul_descr->compute_type;

    // matrix A
    int64_t              num_rows_a     = plan->matmul_descr->matrix_A->m;
    int64_t              num_cols_a     = plan->matmul_descr->matrix_A->n;
    int64_t              c_k_a          = plan->matmul_descr->matrix_A->c_k;
    int64_t              c_lda          = plan->matmul_descr->matrix_A->c_ld;
    rocsparselt_datatype type_a         = plan->matmul_descr->matrix_A->type;
    int                  num_batches_a  = 1;
    int64_t              batch_stride_a = 0;
    plan->matmul_descr->matrix_A->attributes[rocsparselt_mat_num_batches].get(&num_batches_a);
    plan->matmul_descr->matrix_A->attributes[rocsparselt_mat_batch_stride].get(&batch_stride_a);
    int64_t c_batch_stride_a = (batch_stride_a == 0                   ? 0
                                : (opA == rocsparselt_operation_none) ? c_lda * c_k_a
                                                                      : c_lda * num_cols_a);

    // matrix B
    int64_t              num_rows_b     = plan->matmul_descr->matrix_B->m;
    int64_t              num_cols_b     = plan->matmul_descr->matrix_B->n;
    int64_t              ldb            = plan->matmul_descr->matrix_B->ld;
    rocsparselt_datatype type_b         = plan->matmul_descr->matrix_B->type;
    int                  num_batches_b  = 1;
    int64_t              batch_stride_b = 0;
    plan->matmul_descr->matrix_B->attributes[rocsparselt_mat_num_batches].get(&num_batches_b);
    plan->matmul_descr->matrix_B->attributes[rocsparselt_mat_batch_stride].get(&batch_stride_b);

    // matrix C
    int64_t              ldc            = plan->matmul_descr->matrix_C->ld;
    rocsparselt_datatype type_c         = plan->matmul_descr->matrix_C->type;
    int                  num_batches_c  = 1;
    int64_t              batch_stride_c = 0;
    plan->matmul_descr->matrix_C->attributes[rocsparselt_mat_num_batches].get(&num_batches_c);
    plan->matmul_descr->matrix_C->attributes[rocsparselt_mat_batch_stride].get(&batch_stride_c);

    // matrix D
    int64_t              ldd            = plan->matmul_descr->matrix_D->ld;
    rocsparselt_datatype type_d         = plan->matmul_descr->matrix_D->type;
    int                  num_batches_d  = 1;
    int64_t              batch_stride_d = 0;
    plan->matmul_descr->matrix_D->attributes[rocsparselt_mat_num_batches].get(&num_batches_d);
    plan->matmul_descr->matrix_D->attributes[rocsparselt_mat_batch_stride].get(&batch_stride_d);

    // activation
    hipsparselt_activation_type act_type    = hipsparselt_activation_type::none;
    float                       act_args[2] = {0.0f, 0.0f};
    if(plan->matmul_descr->activation_relu)
    {
        act_args[0] = plan->matmul_descr->activation_relu_threshold;
        act_args[1] = plan->matmul_descr->activation_relu_upperbound;
        if(act_args[0] == 0 && act_args[1] == std::numeric_limits<float>::infinity())
            act_type = hipsparselt_activation_type::relu;
        else
            act_type = hipsparselt_activation_type::clippedrelu;
    }
    else if(plan->matmul_descr->activation_gelu)
        act_type = hipsparselt_activation_type::gelu;
    else if(plan->matmul_descr->activation_abs)
        act_type = hipsparselt_activation_type::abs;
    else if(plan->matmul_descr->activation_leakyrelu)
    {
        act_type    = hipsparselt_activation_type::leakyrelu;
        act_args[0] = plan->matmul_descr->activation_leakyrelu_alpha;
    }
    else if(plan->matmul_descr->activation_sigmoid)
        act_type = hipsparselt_activation_type::sigmoid;
    else if(plan->matmul_descr->activation_tanh)
    {
        act_type    = hipsparselt_activation_type::tanh;
        act_args[0] = plan->matmul_descr->activation_tanh_alpha;
        act_args[1] = plan->matmul_descr->activation_tanh_beta;
    }
    float*  bias_vector = plan->matmul_descr->bias_pointer;
    int64_t bias_stride = plan->matmul_descr->bias_stride;

    // algorithm selection
    int config_id         = plan->alg_selection->config_id;
    int config_max_id     = plan->alg_selection->config_max_id;
    int search_iterations = search ? plan->alg_selection->search_iterations : 0; //default

    int64_t m, n, k;
    auto    status
        = getOriginalSizes(opA, opB, num_rows_a, num_cols_a, num_rows_b, num_cols_b, m, n, k);
    if(status != rocsparselt_status_success)
        return status;

    int64_t c_num_cols_a    = (opA == rocsparselt_operation_none ? c_k_a : num_cols_a);
    int64_t metadata_offset = rocsparselt_metadata_offset_in_compressed_matrix(
        c_num_cols_a, c_lda, (batch_stride_a == 0 ? 1 : num_batches_a), type_a);
    if(status != rocsparselt_status_success)
        return status;

    const unsigned char* metadata = reinterpret_cast<const unsigned char*>(d_A) + metadata_offset;

#define EX_PARM                                                                                  \
    handle, opA, opB, m, n, k, alpha, d_A, type_a, c_lda, c_batch_stride_a, 0, d_B, type_b, ldb, \
        batch_stride_b, 0, beta, d_C, type_c, ldc, batch_stride_c, 0, d_D, type_d, ldd,          \
        batch_stride_d, 0, num_batches_a, true, compute_type, true, metadata, act_type,          \
        act_args[0], act_args[1], bias_vector, bias_stride, streams, numStreams, &config_id,     \
        config_max_id, search_iterations

    status = rocsparselt_spmm_template(EX_PARM);
    if(search && status == rocsparselt_status_success)
    {
        plan->alg_selection->config_max_id = config_id;
    }
    return status;
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status rocsparselt_matmul(const rocsparselt_handle*      handle,
                                      const rocsparselt_matmul_plan* plan,
                                      const void*                    alpha,
                                      const void*                    d_A,
                                      const void*                    d_B,
                                      const void*                    beta,
                                      const void*                    d_C,
                                      void*                          d_D,
                                      void*                          workspace,
                                      hipStream_t*                   streams,
                                      int32_t                        numStreams)

{
    if(handle == nullptr || plan == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }

    return rocsparselt_matmul_impl(
        *handle, *plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, streams, numStreams);
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status rocsparselt_matmul_search(const rocsparselt_handle*      handle,
                                             const rocsparselt_matmul_plan* plan,
                                             const void*                    alpha,
                                             const void*                    d_A,
                                             const void*                    d_B,
                                             const void*                    beta,
                                             const void*                    d_C,
                                             void*                          d_D,
                                             void*                          workspace,
                                             hipStream_t*                   streams,
                                             int32_t                        numStreams)

{
    if(handle == nullptr || plan == nullptr)
    {
        return rocsparselt_status_invalid_handle;
    }
    return rocsparselt_matmul_impl(
        *handle, *plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, streams, numStreams, true);
}

#ifdef __cplusplus
}
#endif
