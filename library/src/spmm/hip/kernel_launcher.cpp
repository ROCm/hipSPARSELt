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
#include "kernel_launcher.hpp"
#include "handle.h"
#include "utility.hpp"

/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************/

// The implementation of the rocsparselt<->Tensile interface layer.

#include "rocsparselt.h"

/*****************************************************************************
 * This is the only file in rocsparselt which should #include Tensile headers    *
 * or reference Tensile identifiers. tensile_host.hpp defines the interface. *
 *****************************************************************************/

#include "hip_solution_adapter.hpp"
#include "kernel_launcher.hpp"
#include "rocsparselt-types.h"
#include "rocsparselt_ostream.hpp"
#include "spmm_kernels.hpp"

#include <atomic>
#include <complex>
#include <exception>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include <glob.h>
#include <libgen.h>
#include <link.h>
#include <unistd.h>

#define ROCSPARSELT_LIB_PATH "/opt/rocm/rocsparselt/lib"

namespace
{
#ifndef WIN32
    std::string rocsparselt_so_path;

    int rocsparselt_dl_iterate_phdr_callback(struct dl_phdr_info* hdr_info, size_t size, void* data)
    {
        // uncomment to see all dependent .so files
        //fprintf(stderr, "rocsparselt so file: %s\n", hdr_info->dlpi_name);
        if(hdr_info->dlpi_name && strstr(hdr_info->dlpi_name, "rocsparselt."))
        {
            rocsparselt_so_path = hdr_info->dlpi_name;
        }
        return 0;
    }
#endif

    size_t totalAllcoatedElement(std::vector<size_t>& sizes,
                                 std::vector<size_t>& strides,
                                 size_t               offset)
    {

        size_t totalAllocatedElements = 1;
        for(int i = 0; i < sizes.size(); i++)
            totalAllocatedElements += strides[i] * (sizes[i] - 1);
        totalAllocatedElements += offset;
        return totalAllocatedElements;
    }

    size_t totalAllcoatedElementNonBatch(std::vector<size_t>& sizes,
                                         std::vector<size_t>& strides,
                                         BatchIndices&        batchIndex)
    {
        size_t totalAllocatedElementsNonBatch = 1;
        for(int idx = 0; idx < sizes.size(); idx++)
        {
            bool isBatch = batchIndex.end()
                           != std::find_if(batchIndex.begin(),
                                           batchIndex.end(),
                                           [idx](const BatchIndex& bi) { return bi.a == idx; });
            if(!isBatch)
                totalAllocatedElementsNonBatch += strides[idx] * (sizes[idx] - 1);
        }
        return totalAllocatedElementsNonBatch;
    };

    template <typename Ti, typename To, typename Tc>
    auto ConstructKernelInvoke(const RocsparseltContractionProblem<Ti, To, Tc>& prob,
                               const RocSparseLtKernel&                         kernel)
    {
        KernelInvocation ki;

        ki.args = KernelArguments();

        ki.args.reserve(1024, 128);

        ki.kernelName = kernel.name;

        ki.workGroupSize.x
            = kernel.workGroupSize.x * kernel.workGroupSize.y * kernel.workGroupSize.z;
        ki.workGroupSize.y = 1;
        ki.workGroupSize.z = 1;

        ki.numWorkGroups.x = 1;
        ki.numWorkGroups.y = 1;

        // Tensile Indices for contraction problem
        FreeIndices  freeIndex(2);
        BoundIndices boundIndex(1);
        BatchIndices batchIndex{{2, 2, 2, 2}};

        // Set up GEMM indices
        freeIndex[0].isA = true;
        freeIndex[1].isA = false;
        freeIndex[0].c = freeIndex[0].d = 0;
        freeIndex[1].c = freeIndex[1].d = 1;

        // We set K=0 when alpha==0.
        // This makes alpha==0 a change in the problem, and not just a change in the inputs.
        // It optimizes all problems with alpha==0 into K=0 and alpha=(don't care)
        auto k  = prob.k && *prob.alpha ? prob.k : 0;
        auto ck = prob.sparseA ? k / 2 : k;

        std::vector<size_t> sizes_a(3), sizes_b(3), sizes_c(3), sizes_d(3);
        std::vector<size_t> strides_a = {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a};
        std::vector<size_t> strides_b = {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b};
        std::vector<size_t> strides_c = {prob.row_stride_c, prob.col_stride_c, prob.batch_stride_c};
        std::vector<size_t> strides_d = {prob.row_stride_d, prob.col_stride_d, prob.batch_stride_d};

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_a != rocsparse_operation_none)
        {
            sizes_a[0] = ck;
            sizes_a[1] = prob.m;
            sizes_a[2] = prob.batch_count;

            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            sizes_a[0] = prob.m;
            sizes_a[1] = ck;
            sizes_a[2] = prob.batch_count;

            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_b != rocsparse_operation_none)
        {
            sizes_b[0] = prob.n;
            sizes_b[1] = k;
            sizes_b[2] = prob.batch_count;

            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            sizes_b[0] = k;
            sizes_b[1] = prob.n;
            sizes_b[2] = prob.batch_count;

            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        sizes_c[0] = prob.m;
        sizes_c[1] = prob.n;
        sizes_c[2] = prob.batch_count;

        sizes_d[0] = prob.m;
        sizes_d[1] = prob.n;
        sizes_d[2] = prob.batch_count;

        FreeIndices         freeIndicesA;
        FreeIndices         freeIndicesB;
        std::vector<size_t> freeSizesA;
        std::vector<size_t> freeSizesB;

        freeIndicesA.reserve(freeIndex.size());
        freeIndicesB.reserve(freeIndex.size());
        freeSizesA.reserve(freeIndex.size());
        freeSizesB.reserve(freeIndex.size());
        for(int i = 0; i < freeIndex.size(); i++)
        {
            size_t mySize = sizes_d[freeIndex[i].d];
            if(freeIndex[i].isA)
            {
                freeIndicesA.push_back(freeIndex[i]);
                freeSizesA.push_back(mySize);
            }
            else
            {
                freeIndicesB.push_back(freeIndex[i]);
                freeSizesB.push_back(mySize);
            }
        }

        for(size_t i = 0; i < freeIndicesA.size(); i++)
        {
            ki.numWorkGroups.x *= freeSizesA[i];
        }
        for(size_t i = 0; i < freeIndicesB.size(); i++)
        {
            ki.numWorkGroups.y *= freeSizesB[i];
        }

        ki.numWorkGroups.z = 1;

        std::vector<size_t> batchSizes(batchIndex.size());
        std::vector<size_t> boundSizes(boundIndex.size());
        for(int i = 0; i < batchIndex.size(); i++)
        {
            batchSizes[i] = std::max({sizes_a[batchIndex[i].a],
                                      sizes_b[batchIndex[i].b],
                                      sizes_c.empty() ? 0 : sizes_c[batchIndex[i].c],
                                      sizes_d[batchIndex[i].d]});
        }

        for(int i = 0; i < boundIndex.size(); i++)
        {
            boundSizes[i] = std::max(sizes_a[boundIndex[i].a], sizes_b[boundIndex[i].b]);
        }

        for(size_t i = 0; i < batchIndex.size(); i++)
        {
            if(kernel.packBatchDims & 0x1)
                ki.numWorkGroups.x *= batchSizes[i];
            if(kernel.packBatchDims & 0x2)
                ki.numWorkGroups.y *= batchSizes[i];
            if(!kernel.packBatchDims)
                ki.numWorkGroups.z *= batchSizes[i];
        }

        // CD always contain index0.  if this is in the B free indices, then need to
        // transposing the output tensor.
        bool transposeC01 = freeIndicesB.end()
                            != std::find_if(freeIndicesB.begin(),
                                            freeIndicesB.end(),
                                            [](const FreeIndex& fi) { return fi.c == 0 /*idx0*/; });

        if(transposeC01)
            std::swap(ki.numWorkGroups.x, ki.numWorkGroups.y);

        printf("macroTile %dx%d\n", kernel.macroTile.x, kernel.macroTile.y);
        ki.numWorkGroups.x = CeilDivide(ki.numWorkGroups.x, kernel.macroTile.x);
        ki.numWorkGroups.y = CeilDivide(ki.numWorkGroups.y, kernel.macroTile.y);

        uint32_t problemNumGroupTiles0 = ki.numWorkGroups.x;
        uint32_t problemNumGroupTiles1 = ki.numWorkGroups.y;
        // used only when persistent kernel along batch
        uint32_t problemNumGroupTiles2 = ki.numWorkGroups.z;

        ki.numWorkGroups.y *= kernel.globalSplitU;

        ki.numWorkItems.x = ki.workGroupSize.x * ki.numWorkGroups.x;
        ki.numWorkItems.y = ki.workGroupSize.y * ki.numWorkGroups.y;
        ki.numWorkItems.z = ki.workGroupSize.z * ki.numWorkGroups.z;

        ki.sharedMemBytes = 0;

        uint64_t tensor2dSizeC = totalAllcoatedElement(sizes_c, strides_c, (size_t)0);
        uint64_t tensor2dSizeA
            = (kernel.packBatchDims & 0x1)
                  ? totalAllcoatedElement(sizes_a, strides_a, (size_t)0)
                  : totalAllcoatedElementNonBatch(sizes_a, strides_a, batchIndex);
        uint64_t tensor2dSizeB
            = (kernel.packBatchDims & 0x2)
                  ? totalAllcoatedElement(sizes_b, strides_b, (size_t)0)
                  : totalAllcoatedElementNonBatch(sizes_b, strides_b, batchIndex);

        ki.args.append<uint64_t>("tensor2dSizeC", tensor2dSizeC);
        ki.args.append<uint64_t>("tensor2dSizeA", tensor2dSizeA);
        ki.args.append<uint64_t>("tensor2dSizeB", tensor2dSizeB);

        ki.args.append<To const*>("d", prob.D);
        ki.args.append<To const*>("c", prob.C);
        ki.args.append<Ti const*>("a", prob.A);
        ki.args.append<Ti const*>("b", prob.B);

        if(prob.sparseA)
            ki.args.append<unsigned char const*>("metadata", prob.metadata);

        ki.args.append<Tc const>("alpha", *prob.alpha);
        if(std::is_same<Tc, rocsparselt_half>::value)
            ki.args.append<Tc const>("alpha_2", *prob.alpha);

        {
            ki.args.append<Tc const>("beta", *prob.beta);
            if(std::is_same<Tc, rocsparselt_half>::value)
                ki.args.append<Tc const>("beta_2", *prob.beta);
        }

        size_t startStrideCD = kernel.useInitialStridesCD ? 0 : 1;
        size_t startStrideAB = kernel.useInitialStridesAB ? 0 : 1;

        for(size_t i = startStrideCD; i < sizes_d.size(); i++)
            ki.args.append<uint32_t>(concatenate_if<true>("strideD", i), strides_d[i]);

        for(size_t i = startStrideCD; i < sizes_c.size(); i++)
            ki.args.append<uint32_t>(concatenate_if<true>("strideC", i), strides_c[i]);

        for(size_t i = startStrideAB; i < sizes_a.size(); i++)
            ki.args.append<uint32_t>(concatenate_if<true>("strideA", i), strides_a[i]);

        for(size_t i = startStrideAB; i < sizes_b.size(); i++)
            ki.args.append<uint32_t>(concatenate_if<true>("strideB", i), strides_b[i]);

        std::vector<size_t> problemSizes;
        problemSizes.resize(0);
        problemSizes.reserve(sizes_c.size() + boundSizes.size());
        problemSizes.insert(problemSizes.end(), sizes_c.begin(), sizes_c.end());
        problemSizes.insert(problemSizes.end(), boundSizes.begin(), boundSizes.end());

        int idx = 0;
        for(auto size : problemSizes)
        {
            ki.args.append<uint32_t>(concatenate_if<true>("size_", idx), size);
            idx++;
        }

        // Caculate staggerU
        uint32_t sizeL = boundSizes[0];

        // how many stride-sized clicks to stagger start offset
        unsigned int staggerUIter = kernel.staggerU;

        // /DepthU/GSU
        int unrollLoopIters = sizeL / kernel.depthU / kernel.globalSplitU;

        unsigned int shifted = 1 << kernel.staggerStrideShift;

        while(staggerUIter > 1)
        {
            if(unrollLoopIters >= (staggerUIter * shifted))
                break;

            staggerUIter /= 2; // step down to smaller stagger
        }

        if(staggerUIter >= 1)
            staggerUIter -= 1;

        ki.args.append<int32_t>("staggerUIter", staggerUIter);
        ki.args.append<uint32_t>("problemNumGroupTiles0", problemNumGroupTiles0);
        ki.args.append<uint32_t>("problemNumGroupTiles1", problemNumGroupTiles1);

        uint32_t numFullBlocks            = problemNumGroupTiles1;
        uint32_t wgmRemainder1            = 0;
        uint32_t magicNumberWgmRemainder1 = 0;

        if(kernel.workGroupMapping != 0)
        {
            numFullBlocks = problemNumGroupTiles1 / kernel.workGroupMapping;
            wgmRemainder1 = problemNumGroupTiles1 % kernel.workGroupMapping;
            if(wgmRemainder1 == 0)
                wgmRemainder1 = kernel.workGroupMapping;

            uint64_t  magicNum;
            const int smallMagicShift = 31;
            magicNum                  = (1L << smallMagicShift) / wgmRemainder1 + 1;
            assert(magicNum >> 32 == 0); // ensure magic number fits
            magicNumberWgmRemainder1 = static_cast<uint32_t>(magicNum);
        }

        ki.args.append<uint32_t>("numFullBlocks", numFullBlocks);
        ki.args.append<uint32_t>("wgmRemainder1", wgmRemainder1);
        ki.args.append<uint32_t>("magicNumberWgmRemainder1", magicNumberWgmRemainder1);

        ki.args.append<uint32_t>("offsetD", prob.buffer_offset_b);
        ki.args.append<uint32_t>("offsetC", prob.buffer_offset_c);
        ki.args.append<uint32_t>("offsetA", prob.buffer_offset_a);
        ki.args.append<uint32_t>("offsetB", prob.buffer_offset_b);

        ki.args.append<uint32_t>("pad", 0);
        return ki;
    }

    /**************************************************
     * The KernelLauncher struct interfaces with Tensile *
     **************************************************/
    class KernelLauncher
    {
        // The library object
        //std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> m_library;
        std::shared_ptr<hipDeviceProp_t> m_deviceProp;

        // The adapter object. mutable is used to allow adapters to be modified
        // even when they are stored in a const vector which is immutable in size
        struct adapter_s
        {
            mutable std::atomic<SolutionAdapter*> adapter{nullptr};
            mutable std::mutex                    mutex;
        };

        // Each device contains an adapter
        std::vector<adapter_s> const m_adapters;

    public:
        KernelLauncher()
            : m_adapters(GetDeviceCount())
        {
            // We mark KernelLauncher as initialized. This is so that CI tests can
            // verify that the initialization occurs in the "multiheaded" tests
            rocsparselt_internal_kl_is_initialized() = true;
        }

        // KernelLauncher is not copyable or assignable
        KernelLauncher(const KernelLauncher&) = delete;
        KernelLauncher& operator=(const KernelLauncher&) = delete;

        // Get the number of devices
        static int GetDeviceCount()
        {
            int count;
            if(hipGetDeviceCount(&count) != hipSuccess)
            {
                rocsparselt_cerr
                    << "\nrocsparselt error: Could not initialize Tensile host: No devices found"
                    << std::endl;
                //rocsparselt_abort();
            }
            return count;
        }

        ~KernelLauncher()
        {
            for(auto& a : m_adapters)
                delete a.adapter;
        }

        auto& get_device_property() const
        {
            return m_deviceProp;
        }

        auto& get_adapters() const
        {
            return m_adapters;
        }

        /*******************************************************
         * Testpath() tests that a path exists and is readable *
         *******************************************************/
        static bool TestPath(const std::string& path)
        {
#ifdef WIN32
            return ((_access(path.c_str(), 4) != -1) || (_access(path.c_str(), 6) != -1));
#else
            return access(path.c_str(), R_OK) == 0;
#endif
        }

        /*********************************************************************
         * Initialize adapter and library according to environment variables *
         * and default paths based on librocsparselt.so location and GPU         *
         *********************************************************************/
        void initialize(SolutionAdapter& adapter, int32_t deviceId)
        {
            std::string path;
#ifndef WIN32
            path.reserve(PATH_MAX);
#endif

            // The name of the current GPU platform
            std::string processor = rocsparselt_internal_get_arch_name();

            const char* env = getenv("ROCSPARSELT_TENSILE_LIBPATH");
            if(env)
            {
                path = env;
            }
            else
            {
                path = ROCSPARSELT_LIB_PATH;

                // Find the location of librocsparselt.so
                // Fall back on hard-coded path if static library or not found

#ifndef ROCSPARSELT_STATIC_LIB
                dl_iterate_phdr(rocsparselt_dl_iterate_phdr_callback, NULL);
                if(rocsparselt_so_path.size())
                    path = std::string{dirname(&rocsparselt_so_path[0])};
#endif // ifndef ROCSPARSELT_STATIC_LIB

                // Find the location of the libraries
                if(TestPath(path + "/library"))
                    path += "/library";

                //if(TestPath(path + "/" + processor))
                //    path += "/" + processor;
            }

            auto dir      = path + "/libspmm_kernels_" + processor + ".so";
            bool no_match = false;
            if(TestPath(dir))
            {
                if(adapter.loadCodeObjectMapFile(dir) != hipSuccess)
                    no_match = true;
            }
            else
                no_match = true;

            if(no_match)
            {
                static rocsparselt_internal_ostream& once
                    = rocsparselt_cerr
                      << "\nrocsparselt warning: No paths matched " << dir
                      << ". Make sure that ROCSPARSELT_TENSILE_LIBPATH is set correctly."
                      << std::endl;
            }

            hipDeviceProp_t prop;
            //HIP_CHECK_EXC(hipGetDeviceProperties(&prop, deviceId));
            hipGetDeviceProperties(&prop, deviceId);

            m_deviceProp = std::make_shared<hipDeviceProp_t>(prop);
        }
    };

    // Return the library and adapter for the current HIP device
    auto& get_adapter(rocsparselt_handle                handle,
                      std::shared_ptr<hipDeviceProp_t>* deviceProp = nullptr,
                      int                               device     = -1)
    {
        try
        {
            // KernelLauncher is initialized on the first call
            static KernelLauncher host;

            if(device == -1)
                hipGetDevice(&device);

            // Adapter entry for the current HIP device ID
            auto& a       = host.get_adapters().at(device);
            auto* adapter = a.adapter.load(std::memory_order_acquire);

            // Once set, a.adapter contains the adapter for the current HIP device ID
            if(!adapter)
            {
                // Lock so that only one thread performs initialization of the adapter
                std::lock_guard<std::mutex> lock(a.mutex);

                adapter = a.adapter.load(std::memory_order_relaxed);
                if(!adapter)
                {
                    // Allocate a new adapter using the current HIP device
                    adapter = new SolutionAdapter(handle);

                    // Initialize the adapter and possibly the library
                    host.initialize(*adapter, device);

                    // Atomically change the adapter stored for this device ID
                    a.adapter.store(adapter, std::memory_order_release);
                }
            }

            if(deviceProp)
                *deviceProp = host.get_device_property();

            return *adapter;
        }
        catch(const std::exception& e)
        {
            rocsparselt_cerr << "\nrocsparselt error: Could not initialize Tensile host:\n"
                             << e.what() << std::endl;
            rocsparselt_abort();
        }
        catch(...)
        {
            rocsparselt_cerr << "\nrocsparselt error: Could not initialize Tensile host:\nUnknown "
                                "exception thrown"
                             << std::endl;
            rocsparselt_abort();
        }
    }

    /**************************************************************************
    * We normally print error messages only once, to avoid excessive logging *
    **************************************************************************/
    void print_once(const rocsparselt_internal_ostream& msg)
    {
        if(rocsparselt_suppress_kl_error_messages())
            return;
        static constexpr char varname[] = "ROCSPARSELT_VERBOSE_KL_ERROR";
        static const char*    verbose   = getenv(varname);
        if(!verbose)
        {
            static auto& once = rocsparselt_cerr
                                << msg
                                << "\nThis message will be only be displayed once, unless the "
                                << varname << " environment variable is set." << std::endl;
        }
        else
            rocsparselt_cerr << msg << std::endl;
    }

} // namespace

/******************************************************************************
 * runContractionProblem calls Tensile to run a contraction problem described *
 * by RocsparseltContractionProblem                                               *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc, rocsparse_operation OpA, rocsparse_operation OpB>
rocsparse_status runContractionProblem(const RocsparseltContractionProblem<Ti, To, Tc>& prob,
                                       int*                                             config_id,
                                       const int config_max_id,
                                       const int search_iterations)
{
    rocsparse_status status = rocsparse_status_internal_error;

    auto solution = RocSparseLtKernelSolution<Ti, To, Tc, OpA, OpB>();
    try
    {
        std::shared_ptr<hipDeviceProp_t> deviceProp;

        auto& adapter = get_adapter(prob.handle, &deviceProp, prob.handle->device);

        if(!solution.size())
        {
            rocsparselt_internal_ostream msg;
            print_once(msg << "\nrocsparselt error: No solution found for " << prob);
            status = rocsparse_status_not_implemented;
        }
        else
        {
            if(adapter.loadCodeObject(solution.get(*config_id).name) != hipSuccess)
            {
                rocsparselt_internal_ostream msg;
                print_once(msg << "\nrocsparselt error: failed to load kernel:  "
                               << solution.get(*config_id).name);
            }
            else
            {
                if(!search_iterations)
                {
                    adapter.launchKernel(
                        ConstructKernelInvoke<Ti, To, Tc>(prob, solution.get(*config_id)),
                        prob.streams[0],
                        nullptr,
                        nullptr);
                }
                else
                {
                    float      min_ms = std::numeric_limits<float>::max();
                    hipEvent_t startEvent, stopEvent;
                    float      ms;
                    hipEventCreate(&startEvent);
                    hipEventCreate(&stopEvent);
                    for(int id = 0; id < config_max_id; id++)
                    {
                        auto ki = ConstructKernelInvoke<Ti, To, Tc>(prob, solution.get(id));
                        //warm up
                        adapter.launchKernel(ki, prob.streams[0], nullptr, nullptr);

                        hipEventRecord(startEvent, prob.streams[0]);
                        for(int it = 0; it < search_iterations; it++)
                        {
                            adapter.launchKernel(ki, prob.streams[0], nullptr, nullptr);
                        }
                        hipEventRecord(stopEvent, prob.streams[0]);
                        hipEventSynchronize(stopEvent);

                        hipEventElapsedTime(&ms, startEvent, stopEvent);
                        if(ms < min_ms)
                            *config_id = id;
                    }
                    hipEventDestroy(startEvent);
                    hipEventDestroy(stopEvent);
                }
                status = rocsparse_status_success;
            }
        }
    }
    catch(const std::exception& e)
    {
        rocsparselt_internal_ostream msg;
        print_once(msg << "\nrocsparselt error: " << (solution.size() ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
    }
    catch(...)
    {
        rocsparselt_internal_ostream msg;
        print_once(msg << "\nrocsparselt error: " << (solution.size() ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
    }

    return status;
}

/***************************************************************
 * ! \brief  Initialize rocsparselt for the current HIP device, to *
 * avoid costly startup time at the first call on that device. *
 ***************************************************************/
extern "C" void rocsparselt_initialize(rocsparselt_handle handle)
{
    get_adapter(handle);
}

/******************************************************************************
 * Intantiate the cases of runContractionProblem which are needed to satisfy  *
 * rocsparselt dependencies. This file's template functions are not defined in a  *
 * header file, in order to keep Tensile and rocsparselt separate.                *
 ******************************************************************************/

#define GENERATE_RUN_CONTRACTION_PROBLEM(Ti, To, Tc)                                           \
    template rocsparse_status                                                                  \
        runContractionProblem<Ti, To, Tc, rocsparse_operation_none, rocsparse_operation_none>( \
            const RocsparseltContractionProblem<Ti, To, Tc>&, int*, const int, const int);     \
    template rocsparse_status runContractionProblem<Ti,                                        \
                                                    To,                                        \
                                                    Tc,                                        \
                                                    rocsparse_operation_none,                  \
                                                    rocsparse_operation_transpose>(            \
        const RocsparseltContractionProblem<Ti, To, Tc>&, int*, const int, const int);         \
    template rocsparse_status runContractionProblem<Ti,                                        \
                                                    To,                                        \
                                                    Tc,                                        \
                                                    rocsparse_operation_transpose,             \
                                                    rocsparse_operation_none>(                 \
        const RocsparseltContractionProblem<Ti, To, Tc>&, int*, const int, const int);         \
    template rocsparse_status runContractionProblem<Ti,                                        \
                                                    To,                                        \
                                                    Tc,                                        \
                                                    rocsparse_operation_transpose,             \
                                                    rocsparse_operation_transpose>(            \
        const RocsparseltContractionProblem<Ti, To, Tc>&, int*, const int, const int);

GENERATE_RUN_CONTRACTION_PROBLEM(rocsparselt_half, rocsparselt_half, float)

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for testing) *
 ***********************************************************************************/
std::atomic_bool& rocsparselt_internal_kl_is_initialized()
{
    static std::atomic_bool init;
    return init;
}
