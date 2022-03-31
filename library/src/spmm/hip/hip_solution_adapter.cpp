/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
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

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <dlfcn.h>

#include "hip_solution_adapter.hpp"
#include "rocsparselt_ostream.hpp"
#include "utility.hpp"

#define HIP_CHECK_RETURN(expr)                                               \
    do                                                                       \
    {                                                                        \
        hipError_t e = (expr);                                               \
        if(e)                                                                \
        {                                                                    \
            rocsparselt_cerr << "err = " << hipGetErrorName(e) << std::endl; \
            return e;                                                        \
        }                                                                    \
    } while(0)

SolutionAdapter::SolutionAdapter(rocsparselt_handle handle)
    : m_handle(handle)
{
}

SolutionAdapter::SolutionAdapter(rocsparselt_handle handle, std::string const& name)
    : m_handle(handle)
    , m_name(name)
{
}

SolutionAdapter::~SolutionAdapter()
{
    for(auto& module : m_modules)
        hipModuleUnload(module.second);
    for(auto handle : m_lib_handles)
        dlclose(handle);
}

inline hipError_t load_lib_functions(void* handle, const char* name, void** func)
{

    *(void**)(func) = dlsym(handle, name);

    //std::map<std::string, std::vector<unsigned char>>* ktables = (std::map<std::string, std::vector<unsigned char>>*)dlsym(handle, "kernel_map");
    if((*func) == NULL)
    {
        printf("can not find get_kernel_byte.the get_kernel_byte is [%p]\n", *(void**)(func));
    }

    char* err = NULL;
    if((err = dlerror()) != NULL)
    {
        dlclose(handle);
        rocsparselt_cerr << "dlsym failed to load functon get_kernel_byte " << std::endl;
        return hipErrorInvalidContext;
    }
    return hipSuccess;
}

hipError_t SolutionAdapter::loadCodeObjectMapFile(std::string const& path)
{
    void* handle;
    char* err;

    dlerror();

    handle = dlopen(path.c_str(), RTLD_LOCAL | RTLD_LAZY);

    if(!handle || ((err = dlerror()) != NULL))
    {
        rocsparselt_cerr << "dlopn failed to load " << path << std::endl;
        return hipErrorInvalidContext;
    }

    void* get_kernel_byte;

    load_lib_functions(handle, "get_kernel_byte", (void**)&get_kernel_byte);

    {
        std::lock_guard<std::mutex> guard(m_access);
        m_lib_handles.push_back(handle);
        function_table fucs;
        fucs["get_kernel_byte"] = get_kernel_byte;
        m_lib_functions.push_back(fucs);
        m_loadedLibNames.push_back(concatenate(path));
    }
    return hipSuccess;
}

hipError_t SolutionAdapter::loadCodeObjectBytes(std::vector<uint8_t> const& bytes,
                                                std::string const&          name)
{
    return loadCodeObject(bytes.data(), name);
}

hipError_t SolutionAdapter::loadCodeObject(std::string const& name)
{
    for(auto& fucs : m_lib_functions)
    {
        auto it = fucs.find("get_kernel_byte");
        if(it == fucs.end())
            continue;

        unsigned char* (*get_kernel_byte)(const char*);
        *(void**)(&get_kernel_byte) = it->second;
        auto k_bytes                = get_kernel_byte(name.c_str());

        if(k_bytes != NULL)
        {
            return loadCodeObject(k_bytes, name);
        }
    }
    return hipErrorNotFound;
}

hipError_t SolutionAdapter::loadCodeObject(const void* image, std::string const& name)
{
    hipModule_t module;

    HIP_CHECK_RETURN(hipModuleLoadData(&module, image));
    rocsparselt_cout << "load module " << name << " success" << std::endl;
    {
        std::lock_guard<std::mutex> guard(m_access);
        auto                        it = m_modules.find(name);
        if(it == m_modules.end())
            m_modules[name] = module;
    }
    return hipSuccess;
}

hipError_t SolutionAdapter::initKernel(std::string const& name)
{
    hipFunction_t function;
    return getKernel(function, name);
}

hipError_t SolutionAdapter::getKernel(hipFunction_t& rv, std::string const& name)
{
    std::unique_lock<std::mutex> guard(m_access);
    hipError_t                   err = hipSuccess;

    auto it_k = m_kernels.find(name);
    if(it_k != m_kernels.end())
    {
        rv = it_k->second;
        printf("load function %s success\n", name.c_str());
        return err;
    }

    hipModule_t module;
    auto        it_m = m_modules.find(name);
    if(it_m != m_modules.end())
    {
        module = it_m->second;
        err    = hipModuleGetFunction(&rv, module, name.c_str());
        if(err == hipSuccess)
        {
            m_kernels[name] = rv;
            printf("load function %s success\n", name.c_str());
            return err;
        }
        else if(err != hipErrorNotFound)
        {
            return err;
        }
    }
    return err;
}

hipError_t SolutionAdapter::launchKernel(KernelInvocation const& kernel)
{
    return launchKernel(kernel, nullptr, nullptr, nullptr);
}

hipError_t SolutionAdapter::launchKernel(KernelInvocation const& kernel,
                                         hipStream_t             stream,
                                         hipEvent_t              startEvent,
                                         hipEvent_t              stopEvent)
{
    if(m_handle->layer_mode & rocsparse_layer_mode_log_trace)
    {
        rocsparselt_cout << "Kernel " << kernel.kernelName << "\n"
                         << " l"
                         << " (" << kernel.workGroupSize.x << ", " << kernel.workGroupSize.y << ". "
                         << kernel.workGroupSize.z << ")"
                         << " x g"
                         << " (" << kernel.numWorkGroups.x << ", " << kernel.numWorkGroups.y << ". "
                         << kernel.numWorkGroups.z << ")"
                         << " = "
                         << "(" << kernel.numWorkItems.x << ", " << kernel.numWorkItems.y << ". "
                         << kernel.numWorkItems.z << ") \n"
                         << kernel.args << std::endl;
    }

    hipFunction_t function;
    HIP_CHECK_RETURN(getKernel(function, kernel.kernelName));

    void*  kernelArgs = const_cast<void*>(kernel.args.data());
    size_t argsSize   = kernel.args.size();

    void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                               kernelArgs,
                               HIP_LAUNCH_PARAM_BUFFER_SIZE,
                               &argsSize,
                               HIP_LAUNCH_PARAM_END};

    if(startEvent != nullptr)
        HIP_CHECK_RETURN(hipEventRecord(startEvent, stream));
    HIP_CHECK_RETURN(hipExtModuleLaunchKernel(function,
                                              kernel.numWorkItems.x,
                                              kernel.numWorkItems.y,
                                              kernel.numWorkItems.z,
                                              kernel.workGroupSize.x,
                                              kernel.workGroupSize.y,
                                              kernel.workGroupSize.z,
                                              kernel.sharedMemBytes, // sharedMem
                                              stream, // stream
                                              nullptr,
                                              (void**)&hipLaunchParams,
                                              nullptr, // event
                                              nullptr // event
                                              ));
    if(stopEvent != nullptr)
        HIP_CHECK_RETURN(hipEventRecord(stopEvent, stream));
    return hipSuccess;
}

hipError_t SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels)
{
    for(auto const& k : kernels)
    {
        HIP_CHECK_RETURN(launchKernel(k));
    }
    return hipSuccess;
}

hipError_t SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels,
                                          hipStream_t                          stream,
                                          hipEvent_t                           startEvent,
                                          hipEvent_t                           stopEvent)
{
    auto first = kernels.begin();
    auto last  = kernels.end() - 1;

    for(auto iter = kernels.begin(); iter != kernels.end(); iter++)
    {
        hipEvent_t kStart = nullptr;
        hipEvent_t kStop  = nullptr;

        if(iter == first)
            kStart = startEvent;
        if(iter == last)
            kStop = stopEvent;

        HIP_CHECK_RETURN(launchKernel(*iter, stream, kStart, kStop));
    }
    return hipSuccess;
}

hipError_t SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels,
                                          hipStream_t                          stream,
                                          std::vector<hipEvent_t> const&       startEvents,
                                          std::vector<hipEvent_t> const&       stopEvents)
{
    if(kernels.size() != startEvents.size() || kernels.size() != stopEvents.size())
        throw std::runtime_error(concatenate("Must have an equal number of kernels (",
                                             kernels.size(),
                                             "), start events (",
                                             startEvents.size(),
                                             "), and stop events. (",
                                             stopEvents.size(),
                                             ")"));

    for(size_t i = 0; i < kernels.size(); i++)
    {
        HIP_CHECK_RETURN(launchKernel(kernels[i], stream, startEvents[i], stopEvents[i]));
    }
    return hipSuccess;
}

std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter)
{
    stream << "hip::SolutionAdapter";

    stream << " (" << adapter.name() << ", " << adapter.m_modules.size() << " total modules)"
           << std::endl;

    return stream;
}

std::ostream& operator<<(std::ostream& stream, std::shared_ptr<SolutionAdapter> const& ptr)
{
    if(ptr)
    {
        return stream << "*" << *ptr;
    }
    else
    {
        return stream << "(nullptr)";
    }
}
