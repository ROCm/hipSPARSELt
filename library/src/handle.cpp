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
    if((str_layer_mode = getenv("ROCSPARSELT_LAYER")) == NULL)
    {
        layer_mode = rocsparselt_layer_mode_none;
    }
    else
    {
        layer_mode = (rocsparselt_layer_mode)(atoi(str_layer_mode));
    }

    // Open log file
    if(layer_mode & rocsparselt_layer_mode_log_trace)
    {
        open_log_stream(&log_trace_os, &log_trace_ofs, "ROCSPARSELT_LOG_TRACE_PATH");
    }

    // Open log_bench file
    if(layer_mode & rocsparselt_layer_mode_log_bench)
    {
        open_log_stream(&log_bench_os, &log_bench_ofs, "ROCSPARSELT_LOG_BENCH_PATH");
    }
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocsparselt_handle::~_rocsparselt_handle()
{
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

_rocsparselt_attribute::~_rocsparselt_attribute()
{
    clear();
}

_rocsparselt_attribute& _rocsparselt_attribute::operator=(const _rocsparselt_attribute& rhs)
{
    clear();
    set(rhs._data, rhs._data_size);
    return *this;
};

void _rocsparselt_attribute::clear()
{
    set(nullptr, 0);
}

const void* _rocsparselt_attribute::data()
{
    return _data;
}
size_t _rocsparselt_attribute::length()
{
    return _data_size;
}

size_t _rocsparselt_attribute::get(void* out, size_t size)
{
    if(out != nullptr && _data != nullptr && _data_size >= size)
    {
        memcpy(out, _data, size);
        return size;
    }
    return 0;
}

void _rocsparselt_attribute::set(const void* in, size_t size)
{
    if(in == nullptr || (_data != nullptr && _data_size != size))
    {
        free(_data);
        _data      = nullptr;
        _data_size = 0;
    }
    if(in != nullptr)
    {
        if(_data == nullptr)
            _data = malloc(size);
        memcpy(_data, in, size);
        _data_size = size;
    }
}
