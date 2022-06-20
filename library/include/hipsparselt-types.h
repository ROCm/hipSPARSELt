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

/*! \file
 * \brief hipsparselt-types.h defines data types used by rocsparselt
 */

#pragma once
#ifndef _HIPSPARSELT_TYPES_H_
#define _HIPSPARSELT_TYPES_H_

#include "hipsparselt-bfloat16.h"
#include <float.h>

// Generic API

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ROCM_USE_FLOAT16
typedef _Float16 hipsparseLtHalf;
#else
/*! \brief Structure definition for hipsparseLtHalf */
typedef struct hipsparseLtHalf
{
    uint16_t data;
} hipsparseLtHalf;
#endif

#ifdef __cplusplus
}
#endif

#endif /* _HIPSPARSELT_TYPES_H_ */
