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

#include "handle.h"
#include "definitions.h"
#include "logging.h"

#include <hip/hip_runtime.h>

ROCSPARSE_KERNEL void init_kernel(){};

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocsparselt_handle::_rocsparselt_handle()
{
    // Default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&properties, device));

    // Device wavefront size
    wavefront_size = properties.warpSize;

#if HIP_VERSION >= 307
    // ASIC revision
    asic_rev = properties.asicRevision;
#else
    asic_rev = 0;
#endif

    // Layer mode
    char* str_layer_mode;
    if((str_layer_mode = getenv("ROCSPARSE_LAYERLT")) == NULL)
    {
        layer_mode = rocsparse_layer_mode_none;
    }
    else
    {
        layer_mode = (rocsparse_layer_mode)(atoi(str_layer_mode));
    }

    // Device one
    THROW_IF_HIP_ERROR(hipMalloc(&sone, sizeof(float)));
    THROW_IF_HIP_ERROR(hipMalloc(&done, sizeof(double)));
    THROW_IF_HIP_ERROR(hipMalloc(&cone, sizeof(rocsparse_float_complex)));
    THROW_IF_HIP_ERROR(hipMalloc(&zone, sizeof(rocsparse_double_complex)));

    // Execute empty kernel for initialization
    hipLaunchKernelGGL(init_kernel, dim3(1), dim3(1), 0, stream);

    // Execute memset for initialization
    THROW_IF_HIP_ERROR(hipMemsetAsync(sone, 0, sizeof(float), stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync(done, 0, sizeof(double), stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync(cone, 0, sizeof(rocsparse_float_complex), stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync(zone, 0, sizeof(rocsparse_double_complex), stream));

    float  hsone = 1.0f;
    double hdone = 1.0;

    rocsparse_float_complex  hcone = rocsparse_float_complex(1.0f, 0.0f);
    rocsparse_double_complex hzone = rocsparse_double_complex(1.0, 0.0);

    THROW_IF_HIP_ERROR(hipMemcpyAsync(sone, &hsone, sizeof(float), hipMemcpyHostToDevice, stream));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(done, &hdone, sizeof(double), hipMemcpyHostToDevice, stream));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(
        cone, &hcone, sizeof(rocsparse_float_complex), hipMemcpyHostToDevice, stream));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(
        zone, &hzone, sizeof(rocsparse_double_complex), hipMemcpyHostToDevice, stream));

    // Wait for device transfer to finish
    THROW_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Open log file
    if(layer_mode & rocsparse_layer_mode_log_trace)
    {
        open_log_stream(&log_trace_os, &log_trace_ofs, "ROCSPARSE_LOG_TRACE_PATH");
    }

    // Open log_bench file
    if(layer_mode & rocsparse_layer_mode_log_bench)
    {
        open_log_stream(&log_bench_os, &log_bench_ofs, "ROCSPARSE_LOG_BENCH_PATH");
    }
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocsparselt_handle::~_rocsparselt_handle()
{
    PRINT_IF_HIP_ERROR(hipFree(sone));
    PRINT_IF_HIP_ERROR(hipFree(done));
    PRINT_IF_HIP_ERROR(hipFree(cone));
    PRINT_IF_HIP_ERROR(hipFree(zone));

    // Close log files
    if(log_trace_ofs.is_open())
    {
        log_trace_ofs.close();
    }
    if(log_bench_ofs.is_open())
    {
        log_bench_ofs.close();
    }
}
