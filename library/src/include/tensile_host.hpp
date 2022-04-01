/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************/

/*********************************************************
 * Declaration of the rocSPARSELt<->Tensile interface layer. *
 *********************************************************/

#pragma once

/*****************************************************************************
 * WARNING: Tensile-specific data types, functions and macros should only be *
 * referenced from tensile_host.cpp. This header file defines the interface  *
 * that the rest of rocSPARSELt uses to access Tensile. If another Tensile       *
 * feature needs to be accessed, the API for accessing it should be defined  *
 * in this file, without referencing any Tensile-specific identifiers here.  *
 *****************************************************************************/

#include "handle.h"
#include "tuple_helper.hpp"
#include "utility.hpp"
#include <atomic>

/********************************************************************
 * RocsparseltContractionProblem captures the arguments for a GEMM-like *
 * contraction problem, to be passed to runContractionProblem.      *
 ********************************************************************/
template <typename Ti, typename To = Ti, typename Tc = To>
struct RocsparseltContractionProblem
{
    rocsparselt_handle  handle;
    rocsparse_operation trans_a;
    rocsparse_operation trans_b;

    // The RocsparseltContractionProblem data members should exactly match
    // Tensile's parameter types, even if rocSPARSELt uses differently
    // sized or signed types. The constructors should convert rocSPARSELt
    // types into the corresponding Tensile types stored in this class.
    size_t m;
    size_t n;
    size_t k;

    const Tc* alpha;

    const Ti*        A;
    const Ti* const* batch_A;
    size_t           row_stride_a;
    size_t           col_stride_a;
    size_t           batch_stride_a;
    size_t           buffer_offset_a;

    const Ti*        B;
    const Ti* const* batch_B;
    size_t           row_stride_b;
    size_t           col_stride_b;
    size_t           batch_stride_b;
    size_t           buffer_offset_b;

    const Tc* beta;

    const To*        C;
    const To* const* batch_C;
    size_t           row_stride_c;
    size_t           col_stride_c;
    size_t           batch_stride_c;
    size_t           buffer_offset_c;

    To*        D;
    To* const* batch_D;
    size_t     row_stride_d;
    size_t     col_stride_d;
    size_t     batch_stride_d;
    size_t     buffer_offset_d;

    size_t batch_count;
    bool   strided_batch;

    bool                 sparseA;
    const unsigned char* metadata;

    hipStream_t* streams;
    int32_t      numStreams;

    // gemm
    // gemm_strided_batched
    RocsparseltContractionProblem(rocsparselt_handle   handle,
                                  rocsparse_operation  trans_a,
                                  rocsparse_operation  trans_b,
                                  int64_t              m,
                                  int64_t              n,
                                  int64_t              k,
                                  const Tc*            alpha,
                                  const Ti*            A,
                                  const Ti* const*     batch_A,
                                  int64_t              ld_a,
                                  int64_t              batch_stride_a,
                                  int64_t              offset_a,
                                  const Ti*            B,
                                  const Ti* const*     batch_B,
                                  int64_t              ld_b,
                                  int64_t              batch_stride_b,
                                  int64_t              offset_b,
                                  const Tc*            beta,
                                  To*                  C,
                                  To* const*           batch_C,
                                  int64_t              ld_c,
                                  int64_t              batch_stride_c,
                                  int64_t              offset_c,
                                  int64_t              batch_count,
                                  bool                 strided_batch,
                                  bool                 sparseA,
                                  const unsigned char* metadata,
                                  hipStream_t*         streams,
                                  int32_t              numStreams)
        : handle(handle)
        , trans_a(trans_a)
        , trans_b(trans_b)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , batch_A(batch_A)
        , row_stride_a(1)
        , col_stride_a(ld_a)
        , batch_stride_a(batch_stride_a)
        , buffer_offset_a(offset_a)
        , B(B)
        , batch_B(batch_B)
        , row_stride_b(1)
        , col_stride_b(ld_b)
        , batch_stride_b(batch_stride_b)
        , buffer_offset_b(offset_b)
        , beta(beta)
        , C(C)
        , batch_C(batch_C)
        , row_stride_c(1)
        , col_stride_c(ld_c)
        , batch_stride_c(batch_stride_c)
        , buffer_offset_c(offset_c)
        , D(C)
        , batch_D(batch_C)
        , row_stride_d(1)
        , col_stride_d(ld_c)
        , batch_stride_d(batch_stride_c)
        , buffer_offset_d(offset_c)
        , batch_count(batch_count)
        , strided_batch(strided_batch)
        , sparseA(sparseA)
        , metadata(metadata)
        , streams(streams)
        , numStreams(numStreams)
    {
    }

    // gemm_ex
    // gemm_strided_batched_ex
    RocsparseltContractionProblem(rocsparselt_handle   handle,
                                  rocsparse_operation  trans_a,
                                  rocsparse_operation  trans_b,
                                  int64_t              m,
                                  int64_t              n,
                                  int64_t              k,
                                  const Tc*            alpha,
                                  const Ti*            A,
                                  const Ti* const*     batch_A,
                                  int64_t              ld_a,
                                  int64_t              batch_stride_a,
                                  int64_t              offset_a,
                                  const Ti*            B,
                                  const Ti* const*     batch_B,
                                  int64_t              ld_b,
                                  int64_t              batch_stride_b,
                                  int64_t              offset_b,
                                  const Tc*            beta,
                                  const To*            C,
                                  const To* const*     batch_C,
                                  int64_t              ld_c,
                                  int64_t              batch_stride_c,
                                  int64_t              offset_c,
                                  To*                  D,
                                  To* const*           batch_D,
                                  int64_t              ld_d,
                                  int64_t              batch_stride_d,
                                  int64_t              offset_d,
                                  int64_t              batch_count,
                                  bool                 strided_batch,
                                  bool                 sparseA,
                                  const unsigned char* metadata,
                                  hipStream_t*         streams,
                                  int32_t              numStreams)
        : handle(handle)
        , trans_a(trans_a)
        , trans_b(trans_b)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , batch_A(batch_A)
        , row_stride_a(1)
        , col_stride_a(ld_a)
        , batch_stride_a(batch_stride_a)
        , buffer_offset_a(offset_a)
        , B(B)
        , batch_B(batch_B)
        , row_stride_b(1)
        , col_stride_b(ld_b)
        , batch_stride_b(batch_stride_b)
        , buffer_offset_b(offset_b)
        , beta(beta)
        , C(C)
        , batch_C(batch_C)
        , row_stride_c(1)
        , col_stride_c(ld_c)
        , batch_stride_c(batch_stride_c)
        , buffer_offset_c(offset_c)
        , D(D)
        , batch_D(batch_D)
        , row_stride_d(1)
        , col_stride_d(ld_d)
        , batch_stride_d(batch_stride_d)
        , buffer_offset_d(offset_d)
        , batch_count(batch_count)
        , strided_batch(strided_batch)
        , sparseA(sparseA)
        , metadata(metadata)
        , streams(streams)
        , numStreams(numStreams)
    {
    }

    // gemm_ext2
    // gemm_strided_batched_ext2
    RocsparseltContractionProblem(rocsparselt_handle   handle,
                                  int64_t              m,
                                  int64_t              n,
                                  int64_t              k,
                                  const Tc*            alpha,
                                  const Ti*            A,
                                  const Ti* const*     batch_A,
                                  int64_t              row_stride_a,
                                  int64_t              col_stride_a,
                                  int64_t              batch_stride_a,
                                  int64_t              offset_a,
                                  const Ti*            B,
                                  const Ti* const*     batch_B,
                                  int64_t              row_stride_b,
                                  int64_t              col_stride_b,
                                  int64_t              batch_stride_b,
                                  int64_t              offset_b,
                                  const Tc*            beta,
                                  const To*            C,
                                  const To* const*     batch_C,
                                  int64_t              row_stride_c,
                                  int64_t              col_stride_c,
                                  int64_t              batch_stride_c,
                                  int64_t              offset_c,
                                  To*                  D,
                                  To* const*           batch_D,
                                  int64_t              row_stride_d,
                                  int64_t              col_stride_d,
                                  int64_t              batch_stride_d,
                                  int64_t              offset_d,
                                  int64_t              batch_count,
                                  bool                 strided_batch,
                                  bool                 sparseA,
                                  const unsigned char* metadata,
                                  hipStream_t*         streams,
                                  int32_t              numStreams)
        : handle(handle)
        , trans_a(rocsparse_operation_none)
        , trans_b(rocsparse_operation_none)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , batch_A(batch_A)
        , row_stride_a(row_stride_a)
        , col_stride_a(col_stride_a)
        , batch_stride_a(batch_stride_a)
        , buffer_offset_a(offset_a)
        , B(B)
        , batch_B(batch_B)
        , row_stride_b(row_stride_b)
        , col_stride_b(col_stride_b)
        , batch_stride_b(batch_stride_b)
        , buffer_offset_b(offset_b)
        , beta(beta)
        , C(C)
        , batch_C(batch_C)
        , row_stride_c(row_stride_c)
        , col_stride_c(col_stride_c)
        , batch_stride_c(batch_stride_c)
        , buffer_offset_c(offset_c)
        , D(D)
        , batch_D(batch_D)
        , row_stride_d(row_stride_d)
        , col_stride_d(col_stride_d)
        , batch_stride_d(batch_stride_d)
        , buffer_offset_d(offset_d)
        , batch_count(batch_count)
        , strided_batch(strided_batch)
        , sparseA(sparseA)
        , metadata(metadata)
        , streams(streams)
        , numStreams(numStreams)
    {
    }

    /***************************************************
     * Print a RocsparseltContractionProblem for debugging *
     ***************************************************/
    friend rocsparselt_internal_ostream& operator<<(rocsparselt_internal_ostream&        os,
                                                    const RocsparseltContractionProblem& prob)
    {
        return tuple_helper::print_tuple_pairs(
            os,
            std::make_tuple("a_type",
                            rocsparselt_precision_string<Ti>,
                            "b_type",
                            rocsparselt_precision_string<Ti>,
                            "c_type",
                            rocsparselt_precision_string<To>,
                            "d_type",
                            rocsparselt_precision_string<To>,
                            "compute_type",
                            rocsparselt_precision_string<Tc>,
                            "transA",
                            rocsparselt_transpose_letter(prob.trans_a),
                            "transB",
                            rocsparselt_transpose_letter(prob.trans_b),
                            "M",
                            prob.m,
                            "N",
                            prob.n,
                            "K",
                            prob.k,
                            "alpha",
                            *prob.alpha,
                            "row_stride_a",
                            prob.row_stride_a,
                            "col_stride_a",
                            prob.col_stride_a,
                            "row_stride_b",
                            prob.row_stride_b,
                            "col_stride_b",
                            prob.col_stride_b,
                            "row_stride_c",
                            prob.row_stride_c,
                            "col_stride_c",
                            prob.col_stride_c,
                            "row_stride_d",
                            prob.row_stride_d,
                            "col_stride_d",
                            prob.col_stride_d,
                            "beta",
                            *prob.beta,
                            "batch_count",
                            prob.batch_count,
                            "strided_batch",
                            prob.strided_batch,
                            "stride_a",
                            prob.batch_stride_a,
                            "stride_b",
                            prob.batch_stride_b,
                            "stride_c",
                            prob.batch_stride_c,
                            "stride_d",
                            prob.batch_stride_d));
    };
};

/*******************************************************************************
 * runContractionProblem() solves a RocsparseltContractionProblem                  *
 *******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocsparse_status runContractionProblem(RocsparseltContractionProblem<Ti, To, Tc> const& problem);
template <typename Ti, typename To, typename Tc, rocsparse_operation OpA, rocsparse_operation OpB>
rocsparse_status runContractionProblem(RocsparseltContractionProblem<Ti, To, Tc> const& problem,
                                       int                                              index);

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for testing) *
 ***********************************************************************************/
std::atomic_bool& rocsparselt_internal_tensile_is_initialized();

/**********************************************
 * Whether to suppress Tensile error messages *
 **********************************************/
inline bool& rocsparselt_suppress_tensile_error_messages()
{
    thread_local bool t_suppress = false;
    return t_suppress;
}
