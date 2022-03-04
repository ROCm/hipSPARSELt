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
//#include "utility.hpp"

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief prunes a dense matrix according to the specified algorithm.
 *******************************************************************************/
rocsparse_status rocsparselt_smfmac_prune(const rocsparselt_handle       handle,
                                          const rocsparselt_matmul_descr matmulDescr,
                                          const void*                    d_in,
                                          void*                          d_out,
                                          rocsparselt_prune_alg          pruneAlg,
                                          hipStream_t                    stream)

{
    // Check if handle is valid
    if(handle == nullptr || matmulDescr == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check if pointer is valid
    if(d_in == nullptr || d_out == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if prune alg is valid
    if(pruneAlg != rocsparselt_prune_smfmac_strip)
    {
        return rocsparse_status_not_implemented;
    }
    else
    {
        //TODO
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparse_status rocsparselt_smfmac_prune_check(const rocsparselt_handle       handle,
                                                const rocsparselt_matmul_descr matmulDescr,
                                                const void*                    d_in,
                                                int*                           d_out,
                                                hipStream_t                    stream)
{
    // Check if handle is valid
    if(handle == nullptr || matmulDescr == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check if pointer is valid
    if(d_in == nullptr || d_out == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    //TODO
    {

        return rocsparse_status_success;
    }
}

#ifdef __cplusplus
}
#endif
