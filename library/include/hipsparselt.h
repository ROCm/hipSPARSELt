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

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>

#if defined(__HIP_PLATFORM_HCC__)
#include "hipsparselt-types.h"
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
typedef struct {uint8_t data[11024];} hipsparseLtHandle_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatDescriptor_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatmulDescriptor_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatmulAlgSelection_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatmulPlan_t;
#endif

/* hipSPARSE status types */
typedef enum
{
    HIPSPARSELT_STATUS_SUCCESS           = 0, /**< Function succeeds */
    HIPSPARSELT_STATUS_NOT_INITIALIZED   = 1, /**< hipSPARSELT library not initialized */
    HIPSPARSELT_STATUS_ALLOC_FAILED      = 2, /**< resource allocation failed */
    HIPSPARSELT_STATUS_INVALID_VALUE     = 3, /**< unsupported numerical value was passed to function */
    HIPSPARSELT_STATUS_MAPPING_ERROR     = 4, /**< access to GPU memory space failed */
    HIPSPARSELT_STATUS_EXECUTION_FAILED  = 5, /**< GPU program failed to execute */
    HIPSPARSELT_STATUS_INTERNAL_ERROR    = 6, /**< an internal HIPBLAS operation failed */
    HIPSPARSELT_STATUS_NOT_SUPPORTED     = 7, /**< function not implemented */
    HIPSPARSELT_STATUS_ARCH_MISMATCH     = 8, /**< architecture mismatch */
    HIPSPARSELT_STATUS_INVALID_ENUM      = 10, /**<  unsupported enum value was passed to function */
    HIPSPARSELT_STATUS_UNKNOWN           = 11, /**<  back-end returned an unsupported status code */
} hipsparseLtStatus_t;

/* Types definitions */
typedef enum {
    HIPSPARSELT_POINTER_MODE_HOST   = 0,
    HIPSPARSELT_POINTER_MODE_DEVICE = 1
} hipsparseLtPointerMode_t;

typedef enum {
    HIPSPARSELT_OPERATION_NON_TRANSPOSE       = 0,
    HIPSPARSELT_OPERATION_TRANSPOSE           = 1,
} hipsparseLtOperation_t;

typedef enum {
   HIPSPARSELT_ORDER_ROW = 0,
   HIPSPARSELT_ORDER_COLUMN = 1
} hipsparseLtOrder_t;

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
   HIPSPARSELT_COMPUTE_16F = 0,
   HIPSPARSELT_COMPUTE_32I,
   HIPSPARSELT_COMPUTE_32F,
   HIPSPARSELT_COMPUTE_TF32,
   HIPSPARSELT_COMPUTE_TF32_FAST
} hipsparseLtComputetype_t;

typedef enum {
   HIPSPARSELT_MATMUL_ACTIVATION_RELU = 0,            // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND = 1, // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD = 2,  // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_GELU = 3,            // READ/WRITE
   HIPSPARSELT_MATMUL_BIAS_STRIDE = 4,                // READ/WRITE
   HIPSPARSELT_MATMUL_BIAS_POINTER = 5,               // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_ABS = 6,               // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU = 7,               // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA = 8,               // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID = 9,               // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_TANH = 10,               // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA = 11,               // READ/WRITE
   HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA = 12,               // READ/WRITE
} hipsparseLtMatmulDescAttribute_t;

typedef enum {
   HIPSPARSELT_MATMUL_ALG_DEFAULT
} hipsparseLtMatmulAlg_t;

typedef enum {
   HIPSPARSELT_MATMUL_ALG_CONFIG_ID = 0,     // READ/WRITE
   HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID = 1, // READ-ONLY
   HIPSPARSELT_MATMUL_SEARCH_ITERATIONS = 2  // READ/WRITE
} hipsparseLtMatmulAlgAttribute_t;

typedef enum {
   HIPSPARSELT_PRUNE_SPMMA_TILE  = 0,
   HIPSPARSELT_PRUNE_SPMMA_STRIP = 1
} hipsparseLtPruneAlg_t;

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
hipsparseLtStatus_t hipsparseLtGetVersion(hipsparseLtHandle_t handle, int* version);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtGetGitRevision(hipsparseLtHandle_t handle, char* rev);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtGetArchName(char** archName);

/* hipSPARSE initialization and management routines */
HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtInit(hipsparseLtHandle_t* handle);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtDestroy(const hipsparseLtHandle_t* handle);

/* matrix descriptor */
// dense matrix
HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtDenseDescriptorInit(const hipsparseLtHandle_t*  handle,
                                                   hipsparseLtMatDescriptor_t* matDescr,
                                                   int64_t                     rows,
                                                   int64_t                     cols,
                                                   int64_t                     ld,
                                                   uint32_t                    alignment,
                                                   hipsparseLtDatatype_t       valueType,
                                                   hipsparseLtOrder_t          order);

// structured matrix
HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtStructuredDescriptorInit(const hipsparseLtHandle_t*  handle,
                                                        hipsparseLtMatDescriptor_t* matDescr,
                                                        int64_t                     rows,
                                                        int64_t                     cols,
                                                        int64_t                     ld,
                                                        uint32_t                    alignment,
                                                        hipsparseLtDatatype_t       valueType,
                                                        hipsparseLtOrder_t          order,
                                                        hipsparseLtSparsity_t       sparsity);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtMatDescriptorDestroy(const hipsparseLtMatDescriptor_t* matDescr);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtMatDescSetAttribute(const hipsparseLtHandle_t*    handle,
                                                   hipsparseLtMatDescriptor_t*   matmulDescr,
                                                   hipsparseLtMatDescAttribute_t matAttribute,
                                                   const void*                   data,
                                                   size_t                        dataSize);
HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtMatDescGetAttribute(const hipsparseLtHandle_t*        handle,
                                                   const hipsparseLtMatDescriptor_t* matmulDescr,
                                                   hipsparseLtMatDescAttribute_t     matAttribute,
                                                   void*                             data,
                                                   size_t                            dataSize);

/* matmul descriptor */
HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtMatmulDescriptorInit(const hipsparseLtHandle_t*        handle,
                                                    hipsparseLtMatmulDescriptor_t*    matMulDescr,
                                                    hipsparseLtOperation_t            opA,
                                                    hipsparseLtOperation_t            opB,
                                                    const hipsparseLtMatDescriptor_t* matA,
                                                    const hipsparseLtMatDescriptor_t* matB,
                                                    const hipsparseLtMatDescriptor_t* matC,
                                                    const hipsparseLtMatDescriptor_t* matD,
                                                    hipsparseLtComputetype_t          computeType);
HIPSPARSELT_EXPORT
hipsparseLtStatus_t
    hipsparseLtMatmulDescSetAttribute(const hipsparseLtHandle_t*       handle,
                                      hipsparseLtMatmulDescriptor_t*   matmulDescr,
                                      hipsparseLtMatmulDescAttribute_t matmulAttribute,
                                      const void*                      data,
                                      size_t                           dataSize);
HIPSPARSELT_EXPORT
hipsparseLtStatus_t
    hipsparseLtMatmulDescGetAttribute(const hipsparseLtHandle_t*           handle,
                                      const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                      hipsparseLtMatmulDescAttribute_t     matmulAttribute,
                                      void*                                data,
                                      size_t                               dataSize);

/* algorithm selection */
HIPSPARSELT_EXPORT
hipsparseLtStatus_t
    hipsparseLtMatmulAlgSelectionInit(const hipsparseLtHandle_t*           handle,
                                      hipsparseLtMatmulAlgSelection_t*     algSelection,
                                      const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                      hipsparseLtMatmulAlg_t               alg);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtMatmulAlgSetAttribute(const hipsparseLtHandle_t*       handle,
                                                     hipsparseLtMatmulAlgSelection_t* algSelection,
                                                     hipsparseLtMatmulAlgAttribute_t  attribute,
                                                     const void*                      data,
                                                     size_t                           dataSize);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t
    hipsparseLtMatmulAlgGetAttribute(const hipsparseLtHandle_t*             handle,
                                     const hipsparseLtMatmulAlgSelection_t* algSelection,
                                     hipsparseLtMatmulAlgAttribute_t        attribute,
                                     void*                                  data,
                                     size_t                                 dataSize);

/* matmul plan */
HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtMatmulGetWorkspace(const hipsparseLtHandle_t*     handle,
                                                  const hipsparseLtMatmulPlan_t* plan,
                                                  size_t*                        workspaceSize);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtMatmulPlanInit(const hipsparseLtHandle_t*             handle,
                                              hipsparseLtMatmulPlan_t*               plan,
                                              const hipsparseLtMatmulDescriptor_t*   matmulDescr,
                                              const hipsparseLtMatmulAlgSelection_t* algSelection,
                                              size_t                                 workspaceSize);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtMatmulPlanDestroy(const hipsparseLtMatmulPlan_t* plan);

/* matmul execution */
HIPSPARSELT_EXPORT
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
                                      int32_t                        numStreams);

HIPSPARSELT_EXPORT
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
                                            int32_t                    numStreams);

/* helper */
// prune
HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtSpMMAPrune(const hipsparseLtHandle_t*           handle,
                                          const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                          const void*                          d_in,
                                          void*                                d_out,
                                          hipsparseLtPruneAlg_t                pruneAlg,
                                          hipStream_t                          stream);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtSpMMAPruneCheck(const hipsparseLtHandle_t*           handle,
                                               const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                               const void*                          d_in,
                                               int*                                 valid,
                                               hipStream_t                          stream);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtSpMMAPrune2(const hipsparseLtHandle_t*        handle,
                                           const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                           int                               isSparseA,
                                           hipsparseLtOperation_t            op,
                                           const void*                       d_in,
                                           void*                             d_out,
                                           hipsparseLtPruneAlg_t             pruneAlg,
                                           hipStream_t                       stream);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtSpMMAPruneCheck2(const hipsparseLtHandle_t*        handle,
                                                const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                                int                               isSparseA,
                                                hipsparseLtOperation_t            op,
                                                const void*                       d_in,
                                                int*                              d_valid,
                                                hipStream_t                       stream);

// compression
HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtSpMMACompressedSize(const hipsparseLtHandle_t*     handle,
                                                   const hipsparseLtMatmulPlan_t* plan,
                                                   size_t*                        compressedSize);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtSpMMACompress(const hipsparseLtHandle_t*     handle,
                                             const hipsparseLtMatmulPlan_t* plan,
                                             const void*                    d_dense,
                                             void*                          d_compressed,
                                             hipStream_t                    stream);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t
    hipsparseLtSpMMACompressedSize2(const hipsparseLtHandle_t*        handle,
                                    const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                    size_t*                           compressedSize);

HIPSPARSELT_EXPORT
hipsparseLtStatus_t hipsparseLtSpMMACompress2(const hipsparseLtHandle_t*        handle,
                                              const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                              int                               isSparseA,
                                              hipsparseLtOperation_t            op,
                                              const void*                       d_dense,
                                              void*                             d_compressed,
                                              hipStream_t                       stream);

#ifdef __cplusplus
}
#endif

#endif // _HIPSPARSELT_H_
