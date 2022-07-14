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

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled
//! unmodified through either AMD HCC or NVCC. Key features tend to be in the spirit
//! and terminology of CUDA, but with a portable path to other accelerators as well.
//!
//! This is the master include file for hipSPARSELt, wrapping around rocSPARSELt and
//! cuSPARSELt "version 0.2".
//

#pragma once
#ifndef _HIPSPARSELT_H_
#define _HIPSPARSELT_H_

#include "hipsparselt-export.h"
#include "hipsparselt-version.h"
#include <hipsparse/hipsparse.h>

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>

#if defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif
/* Opaque structures holding information */
// clang-format off

#if defined(__HIP_PLATFORM_HCC__)
typedef void* hipsparseLtHandle_t;
typedef void* hipsparseLtMatDescriptor_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatmulDescriptor_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatmulAlgSelection_t;
typedef void* hipsparseLtMatmulPlan_t;
#elif defined(__HIP_PLATFORM_NVCC__)
typedef __nv_bfloat16 hip_bfloat16;
typedef struct {uint8_t data[11024];} hipsparseLtHandle_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatDescriptor_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatmulDescriptor_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatmulAlgSelection_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatmulPlan_t;
#endif


/* Types definitions */
typedef enum {
    HIPSPARSELT_POINTER_MODE_HOST   = 0,
    HIPSPARSELT_POINTER_MODE_DEVICE = 1
} hipsparseLtPointerMode_t;

typedef enum
{
   HIPSPARSELT_R_16F = 150, /**< 16 bit floating point, real */
   HIPSPARSELT_R_32F = 151, /**< 32 bit floating point, real */
   HIPSPARSELT_R_8I  = 160, /**<  8 bit signed integer, real */
   HIPSPARSELT_R_16BF = 168, /**< 16 bit bfloat, real */
   HIPSPARSELT_R_8F  = 170, /**<  8 bit floating point, real */
   HIPSPARSELT_R_8BF  = 171, /**<  8 bit bfloat, real */
} hipsparseLtDatatype_t;


typedef enum {
   HIPSPARSELT_SPARSITY_50_PERCENT
} hipsparseLtSparsity_t;


typedef enum {
   HIPSPARSELT_MAT_NUM_BATCHES,                // READ/WRITE
   HIPSPARSELT_MAT_BATCH_STRIDE,               // READ/WRITE
} hipsparseLtMatDescAttribute_t;

typedef enum {
   HIPSPARSE_COMPUTE_16F = 0,
   HIPSPARSE_COMPUTE_32I,
   HIPSPARSE_COMPUTE_32F,
   HIPSPARSE_COMPUTE_TF32,
   HIPSPARSE_COMPUTE_TF32_FAST
} hipsparseComputetype_t;

typedef enum {
   HIPSPARSELT_MATMUL_ACTIVATION_RELU = 0,             // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND = 1,  // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD = 2,   // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_GELU = 3,             // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING = 4,     // READ/WRITE
   HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING = 5,        // READ/WRITE
   HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING = 6,         // READ/WRITE
   HIPSPARSELT_MATMUL_BIAS_STRIDE = 7,                 // READ/WRITE
   HIPSPARSELT_MATMUL_BIAS_POINTER = 8,                // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_ABS = 9,              // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU = 10,       // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA = 11, // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID = 12,         // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_TANH = 13,            // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA = 14,      // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA = 15,       // READ/WRITE
} hipsparseLtMatmulDescAttribute_t;

typedef enum {
   HIPSPARSELT_MATMUL_ALG_DEFAULT
} hipsparseLtMatmulAlg_t;

typedef enum {
   HIPSPARSELT_MATMUL_ALG_CONFIG_ID = 0,     // READ/WRITE
   HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID = 1, // READ-ONLY
   HIPSPARSELT_MATMUL_SEARCH_ITERATIONS = 2,  // READ/WRITE
   HIPSPARSELT_MATMUL_SPLIT_K = 3,
   HIPSPARSELT_MATMUL_SPLIT_K_MODE = 4,
   HIPSPARSELT_MATMUL_SPLIT_K_BUFFERS = 5,
} hipsparseLtMatmulAlgAttribute_t;

typedef enum {
   HIPSPARSELT_PRUNE_SPMMA_TILE  = 0,
   HIPSPARSELT_PRUNE_SPMMA_STRIP = 1
} hipsparseLtPruneAlg_t;

typedef enum {
   HIPSPARSELT_SPLIT_K_MODE_ONE_KERNEL = 0,
   HIPSPARSELT_SPLIT_K_MODE_TWO_KERNELS = 1,
} hipsparseLtSplitKMode_t;

// clang-format on

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************
 * ! \brief  Initialize rocsparselt for the current HIP device, to *
 * avoid costly startup time at the first call on that device. *
 ***************************************************************/
HIPSPARSELT_EXPORT
void hipsparseLtInitialize();

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtGetVersion(hipsparseLtHandle_t handle, int* version);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtGetGitRevision(hipsparseLtHandle_t handle, char* rev);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtGetArchName(char** archName);

/* hipSPARSE initialization and management routines */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtInit(hipsparseLtHandle_t* handle);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtDestroy(const hipsparseLtHandle_t* handle);

/* matrix descriptor */
// dense matrix
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtDenseDescriptorInit(const hipsparseLtHandle_t*  handle,
                                                 hipsparseLtMatDescriptor_t* matDescr,
                                                 int64_t                     rows,
                                                 int64_t                     cols,
                                                 int64_t                     ld,
                                                 uint32_t                    alignment,
                                                 hipsparseLtDatatype_t       valueType,
                                                 hipsparseOrder_t            order);

// structured matrix
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtStructuredDescriptorInit(const hipsparseLtHandle_t*  handle,
                                                      hipsparseLtMatDescriptor_t* matDescr,
                                                      int64_t                     rows,
                                                      int64_t                     cols,
                                                      int64_t                     ld,
                                                      uint32_t                    alignment,
                                                      hipsparseLtDatatype_t       valueType,
                                                      hipsparseOrder_t            order,
                                                      hipsparseLtSparsity_t       sparsity);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatDescriptorDestroy(const hipsparseLtMatDescriptor_t* matDescr);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatDescSetAttribute(const hipsparseLtHandle_t*    handle,
                                                 hipsparseLtMatDescriptor_t*   matmulDescr,
                                                 hipsparseLtMatDescAttribute_t matAttribute,
                                                 const void*                   data,
                                                 size_t                        dataSize);
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatDescGetAttribute(const hipsparseLtHandle_t*        handle,
                                                 const hipsparseLtMatDescriptor_t* matmulDescr,
                                                 hipsparseLtMatDescAttribute_t     matAttribute,
                                                 void*                             data,
                                                 size_t                            dataSize);

/* matmul descriptor */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmulDescriptorInit(const hipsparseLtHandle_t*        handle,
                                                  hipsparseLtMatmulDescriptor_t*    matMulDescr,
                                                  hipsparseOperation_t              opA,
                                                  hipsparseOperation_t              opB,
                                                  const hipsparseLtMatDescriptor_t* matA,
                                                  const hipsparseLtMatDescriptor_t* matB,
                                                  const hipsparseLtMatDescriptor_t* matC,
                                                  const hipsparseLtMatDescriptor_t* matD,
                                                  hipsparseComputetype_t            computeType);
HIPSPARSELT_EXPORT
hipsparseStatus_t
    hipsparseLtMatmulDescSetAttribute(const hipsparseLtHandle_t*       handle,
                                      hipsparseLtMatmulDescriptor_t*   matmulDescr,
                                      hipsparseLtMatmulDescAttribute_t matmulAttribute,
                                      const void*                      data,
                                      size_t                           dataSize);
HIPSPARSELT_EXPORT
hipsparseStatus_t
    hipsparseLtMatmulDescGetAttribute(const hipsparseLtHandle_t*           handle,
                                      const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                      hipsparseLtMatmulDescAttribute_t     matmulAttribute,
                                      void*                                data,
                                      size_t                               dataSize);

/* algorithm selection */
HIPSPARSELT_EXPORT
hipsparseStatus_t
    hipsparseLtMatmulAlgSelectionInit(const hipsparseLtHandle_t*           handle,
                                      hipsparseLtMatmulAlgSelection_t*     algSelection,
                                      const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                      hipsparseLtMatmulAlg_t               alg);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmulAlgSetAttribute(const hipsparseLtHandle_t*       handle,
                                                   hipsparseLtMatmulAlgSelection_t* algSelection,
                                                   hipsparseLtMatmulAlgAttribute_t  attribute,
                                                   const void*                      data,
                                                   size_t                           dataSize);

HIPSPARSELT_EXPORT
hipsparseStatus_t
    hipsparseLtMatmulAlgGetAttribute(const hipsparseLtHandle_t*             handle,
                                     const hipsparseLtMatmulAlgSelection_t* algSelection,
                                     hipsparseLtMatmulAlgAttribute_t        attribute,
                                     void*                                  data,
                                     size_t                                 dataSize);

/* matmul plan */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmulGetWorkspace(const hipsparseLtHandle_t*     handle,
                                                const hipsparseLtMatmulPlan_t* plan,
                                                size_t*                        workspaceSize);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmulPlanInit(const hipsparseLtHandle_t*             handle,
                                            hipsparseLtMatmulPlan_t*               plan,
                                            const hipsparseLtMatmulDescriptor_t*   matmulDescr,
                                            const hipsparseLtMatmulAlgSelection_t* algSelection,
                                            size_t                                 workspaceSize);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmulPlanDestroy(const hipsparseLtMatmulPlan_t* plan);

/* matmul execution */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmul(const hipsparseLtHandle_t*     handle,
                                    const hipsparseLtMatmulPlan_t* plan,
                                    const void*                    alpha,
                                    const void*                    d_A,
                                    const void*                    d_B,
                                    const void*                    beta,
                                    const void*                    d_C,
                                    void*                          d_D,
                                    void*                          workspace,
                                    hipStream_t*                   streams,
                                    int32_t                        numStreams);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmulSearch(const hipsparseLtHandle_t* handle,
                                          hipsparseLtMatmulPlan_t*   plan,
                                          const void*                alpha,
                                          const void*                d_A,
                                          const void*                d_B,
                                          const void*                beta,
                                          const void*                d_C,
                                          void*                      d_D,
                                          void*                      workspace,
                                          hipStream_t*               streams,
                                          int32_t                    numStreams);

/* helper */
// prune
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMAPrune(const hipsparseLtHandle_t*           handle,
                                        const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                        const void*                          d_in,
                                        void*                                d_out,
                                        hipsparseLtPruneAlg_t                pruneAlg,
                                        hipStream_t                          stream);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMAPruneCheck(const hipsparseLtHandle_t*           handle,
                                             const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                             const void*                          d_in,
                                             int*                                 valid,
                                             hipStream_t                          stream);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMAPrune2(const hipsparseLtHandle_t*        handle,
                                         const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                         int                               isSparseA,
                                         hipsparseOperation_t              op,
                                         const void*                       d_in,
                                         void*                             d_out,
                                         hipsparseLtPruneAlg_t             pruneAlg,
                                         hipStream_t                       stream);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMAPruneCheck2(const hipsparseLtHandle_t*        handle,
                                              const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                              int                               isSparseA,
                                              hipsparseOperation_t              op,
                                              const void*                       d_in,
                                              int*                              d_valid,
                                              hipStream_t                       stream);

// compression
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMACompressedSize(const hipsparseLtHandle_t*     handle,
                                                 const hipsparseLtMatmulPlan_t* plan,
                                                 size_t*                        compressedSize);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMACompress(const hipsparseLtHandle_t*     handle,
                                           const hipsparseLtMatmulPlan_t* plan,
                                           const void*                    d_dense,
                                           void*                          d_compressed,
                                           hipStream_t                    stream);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMACompressedSize2(const hipsparseLtHandle_t*        handle,
                                                  const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                                  size_t*                           compressedSize);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMACompress2(const hipsparseLtHandle_t*        handle,
                                            const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                            int                               isSparseA,
                                            hipsparseOperation_t              op,
                                            const void*                       d_dense,
                                            void*                             d_compressed,
                                            hipStream_t                       stream);

#ifdef __cplusplus
}
#endif

#endif // _HIPSPARSELT_H_
