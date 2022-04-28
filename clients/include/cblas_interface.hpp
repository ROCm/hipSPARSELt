/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#pragma once

#include "cblas.h"
#include "rocsparselt.h"
#include <type_traits>

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is only used for testing
 * not part of the GPU library
 */

// gemm
template <typename Ti, typename To, typename Tc>
void cblas_gemm(rocsparselt_operation  transA,
                rocsparselt_operation  transB,
                int64_t                m,
                int64_t                n,
                int64_t                k,
                Tc                     alpha,
                const Ti*              A,
                int64_t                lda,
                const Ti*              B,
                int64_t                ldb,
                Tc                     beta,
                std::add_pointer_t<To> C,
                int64_t                ldc,
                bool                   alt = false);
