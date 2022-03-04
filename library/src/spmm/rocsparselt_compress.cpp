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
#include "rocsparselt_spmm.hpp"

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparse_status rocsparselt_smfmac_compressed_size(const rocsparselt_handle      handle,
                                                    const rocsparselt_matmul_plan plan,
                                                    size_t*                       compressedSize)

{
    // Check if handle is valid
    if(handle == nullptr || plan == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check if pointer is valid
    if(compressedSize == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    {
        //TODO
        int64_t                 num_rows_a      = plan->matmul_descr->matrix_A->m;
        int64_t                 num_cols_a      = plan->matmul_descr->matrix_A->n;
        int64_t                 lda             = plan->matmul_descr->matrix_A->ld;
        rocsparse_order         order_a         = plan->matmul_descr->matrix_A->order;
        rocsparselt_matrix_type matrix_type_a   = plan->matmul_descr->matrix_A->m_type;
        rocsparselt_datatype    type_a          = plan->matmul_descr->matrix_A->type;
        int                     num_batches_a   = 1;
        int64_t                 batch_stride_a  = lda * num_cols_a;
        size_t                  metadata_size   = num_cols_a / 8;
        size_t                  metadata_offset = 0;
        switch(type_a)
        {
        case rocsparselt_datatype_f16_r:
        case rocsparselt_datatype_bf16_r:
            metadata_offset = getMetadataOffset<_Float16>(num_batches_a, batch_stride_a);
            break;
        case rocsparselt_datatype_f8_r:
        case rocsparselt_datatype_bf8_r:
        case rocsparselt_datatype_i8_r:
            metadata_offset = getMetadataOffset<int8_t>(num_batches_a, batch_stride_a);
            break;
        }
        *compressedSize = metadata_size + metadata_offset;
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparse_status rocsparselt_smfmac_compress(const rocsparselt_handle      handle,
                                             const rocsparselt_matmul_plan plan,
                                             const void*                   d_dense,
                                             void*                         d_compressed,
                                             hipStream_t                   stream)

{
    // Check if handle is valid
    if(handle == nullptr || plan == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check if pointer is valid
    if(d_dense == nullptr || d_compressed == nullptr)
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
