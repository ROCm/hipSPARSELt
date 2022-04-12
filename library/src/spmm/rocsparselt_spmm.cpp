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

#include "rocsparselt_spmm.hpp"
#include "definitions.h"
#include "handle.h"
#include "rocsparselt_spmm_utils.hpp"

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparse_status
    rocsparselt_matmul_get_workspace(const rocsparselt_handle               handle,
                                     const rocsparselt_matmul_alg_selection algSelection,
                                     size_t*                                workspaceSize)

{
    // Check if handle is valid
    if(handle == nullptr || algSelection == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check if pointer is valid
    if(workspaceSize == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    {
        //TODO
        *workspaceSize = 0;
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparselt_matmul_impl(const rocsparselt_handle      handle,
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
        return rocsparse_status_invalid_handle;
    }

    // Check if pointer is valid
    if(alpha == nullptr || beta == nullptr || d_A == nullptr || d_B == nullptr || d_C == nullptr
       || d_D == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(workspace == nullptr && plan->workspace_size != 0)
    {
        rocsparselt_cerr << "The parameter number 9 (workspace) had an illegal value: "
                            "expected a device memroy with "
                         << plan->workspace_size << " bytes, but current is nullptr" << std::endl;
        return rocsparse_status_invalid_value;
    }

    if(numStreams < 0)
    {
        rocsparselt_cerr << "The parameter number 11 (numStreams) had an illegal value: "
                         << numStreams << std::endl;
        return rocsparse_status_invalid_value;
    }
    else if(streams == nullptr && numStreams > 0)
    {
        rocsparselt_cerr << "The parameter number 10 (streams) had an illegal value: nullptr"
                         << std::endl;
        return rocsparse_status_invalid_value;
    }

    rocsparse_operation      opA          = plan->matmul_descr->op_A;
    rocsparse_operation      opB          = plan->matmul_descr->op_B;
    rocsparselt_compute_type compute_type = plan->matmul_descr->compute_type;

    // matrix A
    int64_t                 num_rows_a     = plan->matmul_descr->matrix_A->m;
    int64_t                 num_cols_a     = plan->matmul_descr->matrix_A->n;
    int64_t                 lda            = plan->matmul_descr->matrix_A->ld;
    int64_t                 c_k_a          = plan->matmul_descr->matrix_A->c_k;
    int64_t                 c_lda          = plan->matmul_descr->matrix_A->c_ld;
    rocsparse_order         order_a        = plan->matmul_descr->matrix_A->order;
    rocsparselt_matrix_type matrix_type_a  = plan->matmul_descr->matrix_A->m_type;
    rocsparselt_datatype    type_a         = plan->matmul_descr->matrix_A->type;
    int                     num_batches_a  = 1;
    int64_t                 batch_stride_a = 0;
    int64_t c_batch_stride_a = (opA == rocsparse_operation_none) ? c_lda * c_k_a : c_lda * num_cols_a;
    plan->matmul_descr->matrix_A->attributes[rocsparselt_mat_num_batches].get(&num_batches_a);
    plan->matmul_descr->matrix_A->attributes[rocsparselt_mat_batch_stride].get(&batch_stride_a);

    // matrix B
    int64_t                 num_rows_b     = plan->matmul_descr->matrix_B->m;
    int64_t                 num_cols_b     = plan->matmul_descr->matrix_B->n;
    int64_t                 ldb            = plan->matmul_descr->matrix_B->ld;
    rocsparse_order         order_b        = plan->matmul_descr->matrix_B->order;
    rocsparselt_matrix_type matrix_type_b  = plan->matmul_descr->matrix_B->m_type;
    rocsparselt_datatype    type_b         = plan->matmul_descr->matrix_B->type;
    int                     num_batches_b  = 1;
    int64_t                 batch_stride_b = 0;
    plan->matmul_descr->matrix_B->attributes[rocsparselt_mat_num_batches].get(&num_batches_b);
    plan->matmul_descr->matrix_B->attributes[rocsparselt_mat_batch_stride].get(&batch_stride_b);

    // matrix C
    int64_t                 num_rows_c     = plan->matmul_descr->matrix_C->m;
    int64_t                 num_cols_c     = plan->matmul_descr->matrix_C->n;
    int64_t                 ldc            = plan->matmul_descr->matrix_C->ld;
    rocsparse_order         order_c        = plan->matmul_descr->matrix_C->order;
    rocsparselt_matrix_type matrix_type_c  = plan->matmul_descr->matrix_C->m_type;
    rocsparselt_datatype    type_c         = plan->matmul_descr->matrix_C->type;
    int                     num_batches_c  = 1;
    int64_t                 batch_stride_c = 0;
    plan->matmul_descr->matrix_C->attributes[rocsparselt_mat_num_batches].get(&num_batches_c);
    plan->matmul_descr->matrix_C->attributes[rocsparselt_mat_batch_stride].get(&batch_stride_c);

    // matrix D
    int64_t                 num_rows_d     = plan->matmul_descr->matrix_D->m;
    int64_t                 num_cols_d     = plan->matmul_descr->matrix_D->n;
    int64_t                 ldd            = plan->matmul_descr->matrix_D->ld;
    rocsparse_order         order_d        = plan->matmul_descr->matrix_D->order;
    rocsparselt_matrix_type matrix_type_d  = plan->matmul_descr->matrix_D->m_type;
    rocsparselt_datatype    type_d         = plan->matmul_descr->matrix_D->type;
    int                     num_batches_d  = 1;
    int64_t                 batch_stride_d = 0;
    plan->matmul_descr->matrix_D->attributes[rocsparselt_mat_num_batches].get(&num_batches_d);
    plan->matmul_descr->matrix_D->attributes[rocsparselt_mat_batch_stride].get(&batch_stride_d);

    // activation
    int   act_relu            = plan->matmul_descr->activation_relu;
    float act_relu_upperbound = plan->matmul_descr->activation_relu_upperbound;
    float act_relu_threshold  = plan->matmul_descr->activation_relu_threshold;
    int   act_gelu            = plan->matmul_descr->activation_gelu;
    void* bias_vector         = nullptr;
    plan->matmul_descr->bias_pointer.get(&bias_vector);
    int64_t bias_stride = plan->matmul_descr->bias_stride;

    // algorithm selection
    rocsparselt_matmul_alg alg = plan->alg_selection->alg;

    int config_id         = 0;
    int config_max_id     = 0;
    int search_iterations = search ? 10 : 0; //default
    plan->alg_selection->attributes[rocsparselt_matmul_alg_config_max_id].get(&config_max_id);

    if(search)
    {
        plan->alg_selection->attributes[rocsparselt_matmul_search_iterations].get(
            &search_iterations);
    }
    else
        plan->alg_selection->attributes[rocsparselt_matmul_alg_config_id].get(&config_id);

    int64_t m, n, k;
    auto    status
        = getOriginalSizes(opA, opB, num_rows_a, num_cols_a, num_rows_b, num_cols_b, m, n, k);
    if(status != rocsparse_status_success)
        return status;

    auto validArgs = validateMatmulArgs(handle,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        d_A,
                                        d_B,
                                        beta,
                                        d_C,
                                        d_D,
                                        num_batches_a,
                                        num_batches_b,
                                        num_batches_c,
                                        num_batches_d,
                                        batch_stride_a,
                                        batch_stride_b,
                                        batch_stride_c,
                                        batch_stride_d,
                                        act_relu,
                                        act_relu_upperbound,
                                        act_relu_threshold,
                                        act_gelu,
                                        bias_vector,
                                        bias_stride);

    if(validArgs != rocsparse_status_continue)
        return validArgs;

    float alpha_f = *(reinterpret_cast<const float*>(alpha));
    float beta_f  = *(reinterpret_cast<const float*>(beta));

    int64_t c_num_cols_a    = (opA == rocsparse_operation_none ? c_k_a : num_cols_a);
    int64_t metadata_offset = rocsparselt_metadata_offset_in_compressed_matrix(
        c_num_cols_a, c_lda, num_batches_a, type_a);
    if(status != rocsparse_status_success)
        return status;

    const unsigned char* metadata = reinterpret_cast<const unsigned char*>(d_A) + metadata_offset;

#define EX_PARM                                                                                  \
    handle, opA, opB, m, n, k, alpha, d_A, type_a, c_lda, c_batch_stride_a, 0, d_B, type_b, ldb, \
        batch_stride_b, 0, beta, d_C, type_c, ldc, batch_stride_c, 0, d_D, type_d, ldd,          \
        batch_stride_d, 0, num_batches_a, true, compute_type, true, metadata, act_relu,          \
        act_relu_upperbound, act_relu_threshold, act_gelu, bias_vector, bias_stride, streams,    \
        numStreams, &config_id, config_max_id, search_iterations

    status = rocsparselt_spmm_template(EX_PARM);
    if(search && status == rocsparse_status_success)
    {
        plan->alg_selection->attributes[rocsparselt_matmul_alg_config_id].set(&config_id);
    }
    return status;
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparse_status rocsparselt_matmul(const rocsparselt_handle      handle,
                                    const rocsparselt_matmul_plan plan,
                                    const void*                   alpha,
                                    const void*                   d_A,
                                    const void*                   d_B,
                                    const void*                   beta,
                                    const void*                   d_C,
                                    void*                         d_D,
                                    void*                         workspace,
                                    hipStream_t*                  streams,
                                    int32_t                       numStreams)

{
    return rocsparselt_matmul_impl(
        handle, plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, streams, numStreams);
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparse_status rocsparselt_matmul_search(const rocsparselt_handle      handle,
                                           const rocsparselt_matmul_plan plan,
                                           const void*                   alpha,
                                           const void*                   d_A,
                                           const void*                   d_B,
                                           const void*                   beta,
                                           const void*                   d_C,
                                           void*                         d_D,
                                           void*                         workspace,
                                           hipStream_t*                  streams,
                                           int32_t                       numStreams)

{
    return rocsparselt_matmul_impl(
        handle, plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, streams, numStreams, true);
}

#ifdef __cplusplus
}
#endif
