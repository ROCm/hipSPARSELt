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

/*! \file
*  \brief rocsparselt-functions.h provides Sparse Linear Algebra Subprograms
*  of Level 1, 2 and 3, using HIP optimized for AMD GPU hardware.
*/

#pragma once
#ifndef _ROCSPARSELT_FUNCTIONS_H_
#define _ROCSPARSELT_FUNCTIONS_H_

#include "rocsparselt-export.h"
#include "rocsparselt-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
* ===========================================================================
*    SPARSE Matrix Multiplication
* ===========================================================================
*/

/*! \ingroup spmm_module
 *  \brief Determines the required workspace size.
 *  \details
 *  \p rocsparselt_matmul_get_workspace determines the required workspace size
 *  associated to the selected algorithm.
 *
 *  @param[out]
 *  workspaceSize    Workspace size in bytes
 *
 *  @param[in]
 *  handle    the rocsparselt handle
 *  algSelection     the algorithm selection descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p workspaceSize pointer is invalid.
 *  \retval rocsparse_status_invalid_value
 */
ROCSPARSELT_EXPORT
rocsparse_status
    rocsparselt_matmul_get_workspace(const rocsparselt_handle               handle,
                                     const rocsparselt_matmul_alg_selection algSelection,
                                     size_t*                                workspaceSize);

/*! \ingroup spmm_module
 *  \brief Sparse matrix dense matrix multiplication
 *
 *  \details
 *  \p rocsparselt_matmul computes the matrix multiplication of matrices A and B to
 *  produce the output matrix D, according to the following operation:
 *  \f[
 *    D := Activation(\alpha \cdot op(A) \cdot op(B) + \beta \cdot C)
 *  \f]
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only supports the case where D has the same shape of C.
 *
 *  @param[out]
 *  d_D
 *
 *  @param[in]
 *  handle      handle to the rocsparselt library context queue.
 *  @param[in]
 *  plan
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  d_A
 *  @param[in]
 *  d_B
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[in]
 *  d_C
 *  @param[in]
 *  workspace
 *  @param[in]
 *  stream
 *  @param[in]
 *  numStreams

 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p alpha, \p A, \p B, \p beta, \p C or \p D
 *              pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_not_implemented
 */
ROCSPARSELT_EXPORT
rocsparse_status rocsparselt_matmul(rocsparselt_handle      handle,
                                    rocsparselt_matmul_plan plan,
                                    const void*             alpha,
                                    const void*             d_A,
                                    const void*             d_B,
                                    const void*             beta,
                                    const void*             d_C,
                                    void*                   d_D,
                                    void*                   workspace,
                                    hipStream_t*            streams,
                                    int32_t                 numStreams);

/*! \ingroup spmm_module
 *  \brief Sparse matrix dense matrix multiplication
 *
 *  \details
 *  \p rocsparselt_matmul_search evaluates all available algorithms for the matrix multiplication
 *  and automatically updates the plan by selecting the fastest one.
 *  The functionality is intended to be used for auto-tuning purposes when the same operation
 *  is repeated multiple times over different inputs.
 *
 *  \note
 *  This function's behavior is the same of rocsparselt_matmul
 *
 *  \note
 *  This function is NOT asynchronous with respect to streams[0] (blocking call)
 *
 *  \note
 *  The number of iterations for the evaluation can be set by using
 *  rocsparselt_matmul_alg_set_attribute() with rocsparselt_matmul_search_iterations.
 *
 *  \note
o*	The selected algorithm id can be retrieved by using
 *
 *  @param[out]
 *  d_D
 *
 *  @param[in]
 *  handle      handle to the rocsparselt library context queue.
 *  @param[in]
 *  plan
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  d_A
 *  @param[in]
 *  d_B
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[in]
 *  d_C
 *  @param[in]
 *  workspace
 *  @param[in]
 *  stream
 *  @param[in]
 *  numStreams

 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p alpha, \p A, \p B, \p beta, \p C or \p D
 *              pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_not_implemented
 */
ROCSPARSELT_EXPORT
rocsparse_status rocsparse_matmul_search(rocsparselt_handle      handle,
                                         rocsparselt_matmul_plan plan,
                                         const void*             alpha,
                                         const void*             d_A,
                                         const void*             d_B,
                                         const void*             beta,
                                         const void*             d_C,
                                         void*                   d_D,
                                         void*                   workspace,
                                         hipStream_t*            streams,
                                         int32_t                 numStreams);

/*! \ingroup spmm_module
 *  \brief Purnes a dense matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_prune prunes a dense matrix d_in according to the specified
 *  algorithm pruneAlg, \ref rocsparselt_prune_smfmac_tile or \ref rocsparselt_prune_smfmac_strip.
 *
 *  \note
 o	The function requires no extra storage.
 *
 *  \note
 *  This function supports asynchronous execution with respect to stream.
 *
 *  @param[out]
 *  d_out       pointer to the pruned matrix.
 *
 *  @param[in]
 *  handle      handle to the rocsparselt library context queue.
 *  @param[in]
 *  matmulDescr matrix multiplication descriptor.
 *  @param[in]
 *  d_in        pointer to the dense matrix.
 *  @param[in]
 *  pruneAlg    pruning algorithm.
 *  @param[in]
 *  stream      HIP stream for the computation.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p d_in, \p d_out pointer is invalid.
 *  \retval     rocsparse_status_not_implemented
 */
ROCSPARSELT_EXPORT
rocsparse_status rocsparselt_smfmac_prune(rocsparselt_handle             handle,
                                          const rocsparselt_matmul_descr matmulDescr,
                                          const void*                    d_in,
                                          void*                          d_out,
                                          rocsparselt_prune_alg          pruneAlg,
                                          hipStream_t                    stream);

/*! \ingroup spmm_module
 *  \brief checks the correctness of the pruning structure for a given matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_prune_check checks the correctness of the pruning structure for a given matrix.
 *  Contents in the given matrix must be sparsity 2:4.
 *
 *
 *  @param[out]
 *  d_valid     validation results (0 correct, 1 wrong).
 *
 *  @param[in]
 *  handle      handle to the rocsparselt library context queue.
 *  @param[in]
 *  matmulDescr matrix multiplication descriptor.
 *  @param[in]
 *  d_in        pointer to the matrix to check.
 *  @param[in]
 *  stream      HIP stream for the computation.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p d_in, \p d_valid pointer is invalid.
 *  \retval     rocsparse_status_not_implemented
 */
ROCSPARSELT_EXPORT
rocsparse_status rocsparselt_smfmac_prune_check(rocsparselt_handle             handle,
                                                const rocsparselt_matmul_descr matmulDescr,
                                                const void*                    d_in,
                                                int*                           d_valid,
                                                hipStream_t                    stream);

/*! \ingroup spmm_module
 *  \brief checks the correctness of the pruning structure for a given matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_prune_check checks the correctness of the pruning structure for a given matrix.
 *  Contents in the given matrix must be sparsity 2:4.
 *
 *
 *  @param[out]
 *  d_valid     validation results (0 correct, 1 wrong).
 *
 *  @param[in]
 *  handle      handle to the rocsparselt library context queue.
 *  @param[in]
 *  matmulDescr matrix multiplication descriptor.
 *  @param[in]
 *  d_in        pointer to the matrix to check.
 *  @param[in]
 *  stream      HIP stream for the computation.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p d_in, \p d_valid pointer is invalid.
 *  \retval     rocsparse_status_not_implemented
 */
ROCSPARSELT_EXPORT
rocsparse_status rocsparselt_smfmac_prune_check(const rocsparselt_handle       handle,
                                                const rocsparselt_matmul_descr matmulDescr,
                                                const void*                    d_in,
                                                int*                           d_valid,
                                                hipStream_t                    stream);

/*! \ingroup spmm_module
 *  \brief provide the size of the compressed matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_compressed_size provides the size of the compressed matrix
 *  to be allocated before calling rocsparselt_smfmac_compress()
 *
 *
 *  @param[out]
 *  compressedSize     size in bytes of the compressed matrix.
 *
 *  @param[in]
 *  handle             handle to the rocsparselt library context queue.
 *  @param[in]
 *  plan               matrix multiplication plan descriptor.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p compressedSize pointer is invalid.
 *  \retval     rocsparse_status_not_implemented
 */
ROCSPARSELT_EXPORT
rocsparse_status rocsparselt_smfmac_compressed_size(rocsparselt_handle            handle,
                                                    const rocsparselt_matmul_plan plan,
                                                    size_t*                       compressedSize);

/*! \ingroup spmm_module
 *  \brief compresses a dense matrix to structured matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_compress compresses a dense matrix d_dense.
 *  The compressed matrix is intended to be used as the first/second operand A/B
 *  in the rocsparselt_matmul() function.
 *
 *  @param[out]
 *  d_compressed   validation results (0 correct, 1 wrong).
 *
 *  @param[in]
 *  handle         handle to the rocsparselt library context queue.
 *  @param[in]
 *  plan           matrix multiplication plan descriptor.
 *  @param[in]
 *  d_dense        pointer to the matrix to check.
 *  @param[in]
 *  stream         HIP stream for the computation.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p d_dense, \p d_compressed pointer is invalid.
 *  \retval     rocsparse_status_not_implemented
 */
ROCSPARSELT_EXPORT
rocsparse_status rocsparselt_smfmac_compress(rocsparselt_handle            handle,
                                             const rocsparselt_matmul_plan plan,
                                             const void*                   d_dense,
                                             void*                         d_compressed,
                                             hipStream_t                   stream);
#ifdef __cplusplus
}
#endif

#endif /* _ROCSPARSELT_FUNCTIONS_H_ */
