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

#pragma once

#include "rocsparselt-types.h"
#include <vector>

struct RocSparseLtKernel
{

    RocSparseLtKernel(std::string kernel_name,
                      dim3        workGroupSize,
                      dim3        threadTile,
                      dim3        macroTile,
                      size_t      staggerU,
                      size_t      depthU,
                      size_t      globalSplitU,
                      size_t      staggerStrideShift,
                      int         workGroupMapping,
                      size_t      packBatchDims,
                      bool        useInitialStridesAB,
                      bool        useInitialStridesCD)
        : name(kernel_name)
        , workGroupSize(workGroupSize)
        , threadTile(threadTile)
        , macroTile(macroTile)
        , staggerU(staggerU)
        , depthU(depthU)
        , globalSplitU(globalSplitU)
        , staggerStrideShift(staggerStrideShift)
        , workGroupMapping(workGroupMapping)
        , packBatchDims(packBatchDims)
        , useInitialStridesAB(useInitialStridesAB)
        , useInitialStridesCD(useInitialStridesCD)
    {
    }

    ~RocSparseLtKernel() = default;

    std::string name;
    dim3        workGroupSize;
    dim3        threadTile;
    dim3        macroTile;

    size_t staggerU            = 0;
    size_t depthU              = 0;
    size_t globalSplitU        = 0;
    size_t staggerStrideShift  = 0;
    int    workGroupMapping    = 0;
    size_t packBatchDims       = 0;
    bool   useInitialStridesAB = false;
    bool   useInitialStridesCD = false;
};

template <typename Ti, typename To, typename Tc, rocsparse_operation opA, rocsparse_operation opB>
struct RocSparseLtKernelSolution
{
public:
    RocSparseLtKernelSolution()  = default;
    ~RocSparseLtKernelSolution() = default;

    void add(RocSparseLtKernel k)
    {
        kernels.push_back(k);
    }
    RocSparseLtKernel get(int index)
    {
        return kernels.at(index);
    }
    size_t size()
    {
        return kernels.size();
    }

private:
    std::vector<RocSparseLtKernel> kernels;
};

#define PARENS ()
#define PUSH_KERNEL(a) add(a);

// Rescan macro tokens 16 times, supposed we had 10 kernels.
#define EXPAND(arg) EXPAND1(EXPAND1(EXPAND1(EXPAND1(arg))))
#define EXPAND1(arg) EXPAND2(EXPAND2(EXPAND2(EXPAND2(arg))))
#define EXPAND2(arg) arg

#define FOR_EACH(macro, ...) __VA_OPT__(EXPAND(FOR_EACH_HELPER(macro, __VA_ARGS__)))

#define FOR_EACH_HELPER(macro, a, ...) \
    macro(a) __VA_OPT__(FOR_EACH_AGAIN PARENS(macro, __VA_ARGS__))

#define FOR_EACH_AGAIN() FOR_EACH_HELPER

#define GENERATE_ROCSPARSELT_KERNELS_MAP(Ti, To, Tc, opA, opB, ...)      \
    template <>                                                          \
    struct RocSparseLtKernelSolution<Ti, To, Tc, opA, opB>               \
    {                                                                    \
                                                                         \
        RocSparseLtKernelSolution(){FOR_EACH(PUSH_KERNEL, __VA_ARGS__)}; \
        ~RocSparseLtKernelSolution() = default;                          \
        void add(RocSparseLtKernel k)                                    \
        {                                                                \
            kernels.push_back(k);                                        \
        }                                                                \
        RocSparseLtKernel get(int index)                                 \
        {                                                                \
            return kernels.at(index);                                    \
        }                                                                \
        size_t size()                                                    \
        {                                                                \
            return kernels.size();                                       \
        }                                                                \
                                                                         \
    private:                                                             \
        std::vector<RocSparseLtKernel> kernels;                          \
    };

/* clang-format off */

GENERATE_ROCSPARSELT_KERNELS_MAP(
    rocsparselt_half, rocsparselt_half, float, rocsparse_operation_none, rocsparse_operation_none,
    RocSparseLtKernel("Cijk_Ailk_Bljk_HHS_BH_SA_MT128x128x32_MI32x32x16x1_SN_K1", dim3(64, 4, 1), dim3(2, 64, 0), dim3(128, 128, 0), 0, 32, 1, 0, 0, 0, false, false)
)
GENERATE_ROCSPARSELT_KERNELS_MAP(
    rocsparselt_half, rocsparselt_half, float, rocsparse_operation_none, rocsparse_operation_transpose,
    RocSparseLtKernel("Cijk_Ailk_Bjlk_HHS_BH_SA_MT128x128x32_MI32x32x16x1_SN_K1", dim3(64, 4, 1), dim3(2, 64, 0), dim3(128, 128, 0), 0, 32, 1, 0, 0, 0, false, false)
)
GENERATE_ROCSPARSELT_KERNELS_MAP(
    rocsparselt_half, rocsparselt_half, float, rocsparse_operation_transpose, rocsparse_operation_none,
    RocSparseLtKernel("Cijk_Alik_Bljk_HHS_BH_SA_MT128x128x32_MI32x32x16x1_SN_K1", dim3(64, 4, 1), dim3(2, 64, 0), dim3(128, 128, 0), 0, 32, 1, 0, 0, 0, false, false)
)
GENERATE_ROCSPARSELT_KERNELS_MAP(
    rocsparselt_half, rocsparselt_half, float, rocsparse_operation_transpose, rocsparse_operation_transpose,
    RocSparseLtKernel("Cijk_Alik_Bjlk_HHS_BH_SA_MT128x128x32_MI32x32x16x1_SN_K1", dim3(64, 4, 1), dim3(2, 64, 0), dim3(128, 128, 0), 0, 32, 1, 0, 0, 0, false, false)
)

/* clang-format on */
