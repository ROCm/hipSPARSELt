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
#include "rocsparselt.h"
#include "utility.h"

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
    // Check if handle is valid
    if(handle == nullptr || plan == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check if pointer is valid
    if(alpha == nullptr || beta == nullptr || d_A == nullptr || d_B == nullptr || d_C == nullptr
       || d_D == nullptr || workspace == nullptr || streams == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    {
        //TODO
        return rocsparse_status_success;
    }
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
    // Check if handle is valid
    if(handle == nullptr || plan == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check if pointer is valid
    if(alpha == nullptr || beta == nullptr || d_A == nullptr || d_B == nullptr || d_C == nullptr
       || d_D == nullptr || workspace == nullptr || streams == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    {
        //TODO
        return rocsparse_status_success;
    }
}

#ifdef __cplusplus
}
#endif
