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
 * \brief rocsparselt-types.h defines data types used by rocsparselt
 */

#pragma once
#ifndef _ROCSPARSELT_TYPES_H_
#define _ROCSPARSELT_TYPES_H_

#include "rocsparse-types.h"

#include <stddef.h>
#include <stdint.h>

/*! \ingroup types_module
 *  \brief Handle to the rocSPARSELt library context queue.
 *
 *  \details
 *  The rocSPARSELt handle is a structure holding the rocSPARSELt library context. It must
 *  be initialized using \ref rocsparselt_create_handle and the returned handle must be
 *  passed to all subsequent library function calls. It should be destroyed at the end
 *  using \ref rocsparselt_destroy_handle.
 */
typedef struct _rocsparselt_handle* rocsparselt_handle;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix.
 *
 *  \details
 *  The rocSPARSELt matrix descriptor is a structure holding all properties of a matrix.
 *  It must be initialized using \ref rocsparselt_create_mat_descr and the returned
 *  descriptor must be passed to all subsequent library calls that involve the matrix.
 *  It should be destroyed at the end using \ref rocsparselt_destroy_mat_descr.
 */
typedef struct _rocsparselt_mat_descr* rocsparselt_mat_descr;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication operation
 *
 *  \details
 *  The rocSPARSELt matrix multiplication descriptor is a structure holding
 *  the description of the matrix multiplication operation.
 *  It is initialized with \ref rocsparselt_matmul_descr_init function.
 */
typedef struct _rocsparselt_matmul_descr* rocsparselt_matmul_descr;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication algorithm.
 *
 *  \details
 *  It is initialized with \ref rocsparselt_matmul_alg_selection_init function.
 */
typedef struct _rocsparselt_matmul_alg_selection* rocsparselt_matmul_alg_selection;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication execution plan
 *
 *  \details
 *  The rocSPARSELt matrix multiplication execution plan descriptor is a structure holding
 *  all the information necessary to execute the rocsparselt_matmul() operation.
 *  It is initialized and destroyed with \ref rocsparselt_matmul_plan_init
 *  and \ref rocsparselt_matmul_plan_destroy functions respectively.
 */
typedef struct _rocsparselt_matmul_plan* rocsparselt_matmul_plan;

// Generic API

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Single precision floating point type */
typedef float rocsparselt_float;

#ifdef ROCM_USE_FLOAT16
typedef _Float16 rocsparselt_half;
#else
/*! \brief Structure definition for rocsparselt_half */
typedef struct rocsparselt_half
{
    uint16_t data;
} rocsparselt_half;
#endif

/*! \ingroup types_module
 *  \brief Specify the sparsity of the structured matrix.
 *
 *  \details
 *  The enumerator specifies the sparsity ratio of the structured matrix as
 *  sparsity = nnz / total elements
 *  The sparsity property is used in the rocsparselt_structured_descr_init() function.
 */
typedef enum rocsparselt_sparsity_
{
    rocsparselt_sparsity_50_percent = 0, /**< 50% sparsity ratio - 2:4 */
} rocsparselt_sparsity;

/*! \ingroup types_module
 *  \brief Specify the matrix type.
 *
 *  \details
 */
typedef enum rocsparselt_matrix_type_
{
    rocsparselt_matrix_type_dense      = 0, /**< dense matrix type. */
    rocsparselt_matrix_type_structured = 1, /**< structured matrix type. */
    rocsparselt_matrix_type_unknown    = 2
} rocsparselt_matrix_type;

/*! \ingroup types_module
 *  \brief List of rocsparselt data types.
 *
 *  \details
 *  Indicates the precision width of data stored in a rocsparselt type.
 */
typedef enum rocsparselt_datatype_
{
    rocsparselt_datatype_f16_r  = 150, /**< 16 bit floating point, real */
    rocsparselt_datatype_i8_r   = 160, /**<  8 bit signed integer, real */
    rocsparselt_datatype_bf16_r = 168, /**< 16 bit bfloat, real */
    rocsparselt_datatype_f8_r   = 170, /**< 8 bit floating point, real */
    rocsparselt_datatype_bf8_r  = 171, /**< 8 bit bfloat, real */
} rocsparselt_datatype;

/*! \ingroup types_module
 *  \brief Specify the compute precision modes of the matrix
 *
 *  \details
 */
typedef enum rocsparselt_compute_type_
{
    rocsparselt_compute_f32 = 300, /**< 32-bit floating-point precision. */
    rocsparselt_compute_i32 = 301 /**< 32-bit integer precision. */
} rocsparselt_compute_type;

/*! \ingroup types_module
 *  \brief Specify the additional attributes of a matrix descriptor
 *
 *  \details
 *  The rocsparselt_mat_descr_attribute is used in the
 *  \ref rocsparselt_mat_descr_set_attribute and \ref rocsparselt_mat_descr_get_attribute functions
 */
typedef enum rocsparselt_mat_descr_attribute_
{
    rocsparselt_mat_num_batches  = 0, /**< number of matrices in a batch. */
    rocsparselt_mat_batch_stride = 1 /**< s
    tride between consecutive matrices in a batch expressed in terms of matrix elements. */
} rocsparselt_mat_descr_attribute;

/*! \ingroup types_module
 *  \brief Specify the additional attributes of a matrix multiplication descriptor
 *
 *  \details
 *  The rocsparselt_matmul_descr_attribute_ is used in the
 *  \ref rocsparselt_matmul_descr_set_attribute and \ref rocsparselt_matmul_descr_get_attribute functions
 */
typedef enum rocsparselt_matmul_descr_attribute_
{
    rocsparselt_matmul_activation_relu = 0, /**< ReLU activation function. */
    rocsparselt_matmul_activation_relu_upperbound
    = 1, /**< Upper bound of the ReLU activation function. */
    rocsparselt_matmul_activation_relu_threshold
    = 2, /**< Lower threshold of the ReLU activation function. */
    rocsparselt_matmul_activation_gelu = 3, /**< GeLU activation function. */
    rocsparselt_matmul_bias_pointer
    = 4, /**< Bias pointer. The bias vector size must equal to the number of rows of the output matrix (D). */
    rocsparselt_matmul_bias_stride
    = 5 /**< Bias stride between consecutive bias vectors. 0 means broadcast the first bias vector. */
} rocsparselt_matmul_descr_attribute;

/*! \ingroup types_module
 *  \brief Specify the algorithm for matrix-matrix multiplication.
 *
 *  \details
 *  The \ref rocsparselt_matmul_alg is used in the \ref rocsparselt_matmul_alg_selection_init function.
 */
typedef enum rocsparselt_matmul_alg_
{
    rocsparselt_matmul_alg_default = 0, /**< Default algorithm. */
} rocsparselt_matmul_alg;

/*! \ingroup types_module
 *  \brief Specify the matrix multiplication algorithm attributes.
 *
 *  \details
 *  The \ref rocsparselt_matmul_alg_attribute is used in the
 *  \ref rocsparselt_matmul_alg_get_attribute and \ref rocsparselt_matmul_alg_set_attribute functions.
 */
typedef enum rocsparselt_matmul_alg_attribute_
{
    rocsparselt_matmul_alg_config_id     = 0, /**< Algorithm ID (set and query). */
    rocsparselt_matmul_alg_config_max_id = 1, /**< Algorithm ID limit (query only). */
    rocsparselt_matmul_search_iterations
    = 2 /**< Number of iterations (kernel launches per algorithm)
                                                  for rocsparselt_matmul_search, default=10. */
} rocsparselt_matmul_alg_attribute;

/*! \ingroup types_module
 *  \brief Specify the pruning algorithm to apply to the structured matrix before the compression.
 *
 *  \details
 *  The \ref rocsparselt_prune_alg is used in the \ref rocsparselt_smfmac_prune function.
 */
typedef enum rocsparselt_prune_alg_
{
    rocsparselt_prune_smfmac_tile = 0,
    rocsparselt_prune_smfmac_strip
    = 1, /**< - Zero-out two values in a 1x4 strip to maximize the L1-norm of the resulting strip. */
} rocsparselt_prune_alg;

#ifdef __cplusplus
}
#endif

#endif /* _ROCSPARSELT_TYPES_H_ */
