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
rocsparselt_status rocsparselt_matmul_get_workspace(const rocsparselt_handle*      handle,
                                                    const rocsparselt_matmul_plan* plan,
                                                    size_t*                        workspaceSize)

{
    // Check if handle is valid
    if(handle == nullptr)
    {
        hipsparselt_cerr << "handle is a NULL pointer" << std::endl;
        return rocsparselt_status_invalid_handle;
    }
    auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
    if(!_handle->isInit())
    {
        hipsparselt_cerr << "handle did not initialized or already destroyed" << std::endl;
        return rocsparselt_status_invalid_handle;
    }

    if(plan == nullptr)
    {
        log_error(_handle, __func__, "plan is a NULL pointer");
        return rocsparselt_status_invalid_handle;
    }
    auto _plan = reinterpret_cast<const _rocsparselt_matmul_plan*>(plan);
    if(!_plan->isInit())
    {
        log_error(_handle, __func__, "plan did not initialized or already destroyed");
        return rocsparselt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(workspaceSize == nullptr)
    {
        log_error(_handle, __func__, "workspaceSize is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    {
        if(_plan->alg_selection->config_max_id == 0)
            *workspaceSize = 0;
        else
        {
            *workspaceSize = _plan->alg_selection->configs[_plan->alg_selection->config_id]
                                 .max_workspace_bytes;
        }
        log_api(_handle, __func__, *workspaceSize);
        return rocsparselt_status_success;
    }
}

rocsparselt_status rocsparselt_matmul_impl(const char*                    caller,
                                           const rocsparselt_handle*      handle,
                                           const rocsparselt_matmul_plan* plan,
                                           const void*                    alpha,
                                           const void*                    d_A,
                                           const void*                    d_B,
                                           const void*                    beta,
                                           const void*                    d_C,
                                           void*                          d_D,
                                           void*                          workspace,
                                           hipStream_t*                   streams,
                                           int32_t                        numStreams,
                                           bool                           search = false)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        hipsparselt_cerr << "handle is a NULL pointer" << std::endl;
        return rocsparselt_status_invalid_handle;
    }
    auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
    if(!_handle->isInit())
    {
        hipsparselt_cerr << "handle did not initialized or already destroyed" << std::endl;
        return rocsparselt_status_invalid_handle;
    }

    if(plan == nullptr)
    {
        log_error(_handle, caller, "plan is a NULL pointer");
        return rocsparselt_status_invalid_handle;
    }
    auto _plan = reinterpret_cast<const _rocsparselt_matmul_plan*>(plan);
    if(!_plan->isInit())
    {
        log_error(_handle, caller, "plan did not initialized or already destroyed");
        return rocsparselt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(alpha == nullptr)
    {
        log_error(_handle, caller, "alpha is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(d_A == nullptr)
    {
        log_error(_handle, caller, "d_A is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(d_B == nullptr)
    {
        log_error(_handle, caller, "d_B is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(beta == nullptr)
    {
        log_error(_handle, caller, "beta is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(d_C == nullptr)
    {
        log_error(_handle, caller, "d_C is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(d_D == nullptr)
    {
        log_error(_handle, caller, "d_D is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    size_t workspaceSize
        = _plan->alg_selection->config_max_id == 0
              ? 0
              : _plan->alg_selection->configs[_plan->alg_selection->config_id].max_workspace_bytes;
    if(workspace == nullptr && workspaceSize != 0)
    {
        hipsparselt_cerr << "The parameter number 9 (workspace) had an illegal value "
                            "expected a device memroy with "
                         << workspaceSize << " bytes, but current is nullptr" << std::endl;
        log_error(_handle, caller, "expected workspace is not a NULL pointer");
        return rocsparselt_status_invalid_value;
    }

    if(numStreams < 0)
    {
        hipsparselt_cerr << "The parameter number 11 (numStreams) had an illegal value: "
                         << numStreams << std::endl;
        log_error(_handle, caller, "numStreams should >= 0");
        return rocsparselt_status_invalid_value;
    }
    else if(streams == nullptr && numStreams > 0)
    {
        hipsparselt_cerr << "The parameter number 10 (streams) had an illegal value: nullptr"
                         << std::endl;
        log_error(_handle,
                  caller,
                  "streams should not be a NULL pointer because the numStreams is not 0");
        return rocsparselt_status_invalid_value;
    }

    // algorithm selection
    int config_id         = _plan->alg_selection->config_id;
    int config_max_id     = _plan->alg_selection->config_max_id;
    int search_iterations = search ? _plan->alg_selection->search_iterations : 0; //default

#define EX_PARM                                                                              \
    caller, _handle, _plan, alpha, beta, d_A, d_B, d_C, d_D, workspace, streams, numStreams, \
        &config_id, config_max_id, search_iterations

    log_api(_handle,
            caller,
            "plan[in]",
            *_plan,
            "alpha[in]",
            alpha,
            "d_A[in]",
            d_A,
            "d_B[in]",
            d_B,
            "beta[in]",
            beta,
            "d_C[in]",
            d_C,
            "d_D[in]",
            d_D,
            "workspace[in]",
            workspace,
            "workspaceSize[in]",
            workspaceSize,
            "streams[in]",
            streams,
            "numStreams[in]",
            numStreams);

    rocsparselt_status status = rocsparselt_spmm_template(EX_PARM);
    if(search && status == rocsparselt_status_success)
    {
        log_info(_handle, caller, "found the best config_id", config_id);
        _plan->alg_selection->config_id = config_id;
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
    return rocsparselt_matmul_impl(
        __func__, handle, plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, streams, numStreams);
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status rocsparselt_matmul_search(const rocsparselt_handle* handle,
                                             rocsparselt_matmul_plan*  plan,
                                             const void*               alpha,
                                             const void*               d_A,
                                             const void*               d_B,
                                             const void*               beta,
                                             const void*               d_C,
                                             void*                     d_D,
                                             void*                     workspace,
                                             hipStream_t*              streams,
                                             int32_t                   numStreams)

{
    return rocsparselt_matmul_impl(__func__,
                                   handle,
                                   plan,
                                   alpha,
                                   d_A,
                                   d_B,
                                   beta,
                                   d_C,
                                   d_D,
                                   workspace,
                                   streams,
                                   numStreams,
                                   true);
}
#ifdef __cplusplus
}
#endif

template <typename Ti, typename To, typename Tc>
rocsparselt_status ConstructRocSparseLtProblem(const char*                                 caller,
                                               RocsparseltContractionProblem<Ti, To, Tc>** prob,
                                               const _rocsparselt_matmul_descr* matmul_descr,
                                               const Tc*                        alpha,
                                               const Tc*                        beta,
                                               const Ti*                        a,
                                               const Ti*                        b,
                                               const To*                        c,
                                               To*                              d,
                                               bool                             strided_batch,
                                               void*                            workspace,
                                               size_t                           workspaceSize,
                                               hipStream_t*                     streams,
                                               int32_t                          numStreams)
{
    std::shared_ptr<Tc> _one = std::make_shared<Tc>(static_cast<Tc>(1));
    if(alpha == nullptr)
        alpha = _one.get();

    if(beta == nullptr)
        beta = _one.get();

    rocsparselt_operation opA = matmul_descr->op_A;
    rocsparselt_operation opB = matmul_descr->op_B;

    // matrix A
    int64_t              num_rows_a     = matmul_descr->matrix_A->m;
    int64_t              num_cols_a     = matmul_descr->matrix_A->n;
    int64_t              c_k_a          = matmul_descr->matrix_A->c_k;
    int64_t              c_lda          = matmul_descr->matrix_A->c_ld;
    rocsparselt_datatype type_a         = matmul_descr->matrix_A->type;
    int64_t              offset_a       = 0;
    int                  num_batches_a  = 1;
    int64_t              batch_stride_a = 0;
    num_batches_a                       = matmul_descr->matrix_A->num_batches;
    batch_stride_a                      = matmul_descr->matrix_A->batch_stride;
    int64_t c_batch_stride_a            = (batch_stride_a == 0                   ? 0
                                           : (opA == rocsparselt_operation_none) ? c_lda * c_k_a
                                                                                 : c_lda * num_cols_a);

    // matrix B
    int64_t num_rows_b     = matmul_descr->matrix_B->m;
    int64_t num_cols_b     = matmul_descr->matrix_B->n;
    int64_t ldb            = matmul_descr->matrix_B->ld;
    int64_t offset_b       = 0;
    int64_t batch_stride_b = 0;
    batch_stride_b         = matmul_descr->matrix_B->batch_stride;

    // matrix C
    int64_t ldc            = matmul_descr->matrix_C->ld;
    int64_t offset_c       = 0;
    int64_t batch_stride_c = 0;
    batch_stride_c         = matmul_descr->matrix_C->batch_stride;

    // matrix D
    int64_t num_rows_d     = matmul_descr->matrix_D->m;
    int64_t num_cols_d     = matmul_descr->matrix_D->n;
    int64_t ldd            = matmul_descr->matrix_D->ld;
    int64_t offset_d       = 0;
    int64_t batch_stride_d = 0;
    batch_stride_d         = matmul_descr->matrix_D->batch_stride;

    // activation
    hipsparselt_activation_type act_type    = hipsparselt_activation_type::none;
    float                       act_args[2] = {0.0f, 0.0f};

    if(matmul_descr->activation == rocsparselt_matmul_activation_relu)
    {
        act_args[0] = matmul_descr->activation_relu_threshold;
        act_args[1] = matmul_descr->activation_relu_upperbound;
        if(act_args[0] == 0 && act_args[1] == std::numeric_limits<float>::infinity())
            act_type = hipsparselt_activation_type::relu;
        else
            act_type = hipsparselt_activation_type::clippedrelu;
    }
    else if(matmul_descr->activation == rocsparselt_matmul_activation_gelu)
    {
        act_type    = hipsparselt_activation_type::gelu;
        act_args[0] = matmul_descr->activation_gelu_scaling;
    }
    else if(matmul_descr->activation == rocsparselt_matmul_activation_abs)
        act_type = hipsparselt_activation_type::abs;
    else if(matmul_descr->activation == rocsparselt_matmul_activation_leakyrelu)
    {
        act_type    = hipsparselt_activation_type::leakyrelu;
        act_args[0] = matmul_descr->activation_leakyrelu_alpha;
    }
    else if(matmul_descr->activation == rocsparselt_matmul_activation_sigmoid)
        act_type = hipsparselt_activation_type::sigmoid;
    else if(matmul_descr->activation == rocsparselt_matmul_activation_tanh)
    {
        act_type    = hipsparselt_activation_type::tanh;
        act_args[0] = matmul_descr->activation_tanh_alpha;
        act_args[1] = matmul_descr->activation_tanh_beta;
    }

    float*  bias_vector = matmul_descr->bias_pointer;
    int64_t bias_stride = matmul_descr->bias_stride;

    int64_t m, n, k;
    auto    status
        = getOriginalSizes(opA, opB, num_rows_a, num_cols_a, num_rows_b, num_cols_b, m, n, k);
    if(status != rocsparselt_status_success)
    {
        log_error(matmul_descr->handle, caller, "A, B matrix size are not matched");
        return status;
    }

    int64_t c_num_cols_a    = (opA == rocsparselt_operation_none ? c_k_a : num_cols_a);
    int64_t metadata_offset = rocsparselt_metadata_offset_in_compressed_matrix(
        c_num_cols_a, c_lda, (batch_stride_a == 0 ? 1 : num_batches_a), type_a);

    const unsigned char* metadata
        = (a == nullptr) ? nullptr : reinterpret_cast<const unsigned char*>(a) + metadata_offset;

    (*prob) = new RocsparseltContractionProblem<Ti, To, Tc>(matmul_descr->handle,
                                                            opA,
                                                            opB,
                                                            m,
                                                            n,
                                                            k,
                                                            alpha,
                                                            a,
                                                            nullptr,
                                                            c_lda,
                                                            c_batch_stride_a,
                                                            offset_a,
                                                            b,
                                                            nullptr,
                                                            ldb,
                                                            batch_stride_b,
                                                            offset_b,
                                                            beta,
                                                            c,
                                                            nullptr,
                                                            ldc,
                                                            batch_stride_c,
                                                            offset_c,
                                                            d,
                                                            nullptr,
                                                            ldd,
                                                            batch_stride_d,
                                                            offset_d,
                                                            num_batches_a,
                                                            strided_batch,
                                                            true,
                                                            metadata,
                                                            act_type,
                                                            act_args[0],
                                                            act_args[1],
                                                            bias_vector,
                                                            bias_stride,
                                                            matmul_descr->bias_type,
                                                            workspace,
                                                            workspaceSize,
                                                            streams,
                                                            numStreams);
    return rocsparselt_status_success;
}

#define GENERATE_DEFINITIONS(Ti, To, Tc)                                 \
    template rocsparselt_status ConstructRocSparseLtProblem<Ti, To, Tc>( \
        const char*,                                                     \
        RocsparseltContractionProblem<Ti, To, Tc>**,                     \
        const _rocsparselt_matmul_descr*,                                \
        const Tc*,                                                       \
        const Tc*,                                                       \
        const Ti*,                                                       \
        const Ti*,                                                       \
        const To*,                                                       \
        To*,                                                             \
        bool,                                                            \
        void*,                                                           \
        size_t,                                                          \
        hipStream_t*,                                                    \
        int32_t);

GENERATE_DEFINITIONS(__half, __half, float)
GENERATE_DEFINITIONS(hip_bfloat16, hip_bfloat16, float)
GENERATE_DEFINITIONS(int8_t, int8_t, float)

#undef GENERATE_DEFINITIONS
