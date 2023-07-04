/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
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

// The implementation of the rocsparselt<->Tensile interface layer.

#include "tensile_host.hpp"
#include "activation.hpp"
#include "definitions.h"
#include "rocsparselt_spmm_utils.hpp"
#include "status.h"
#include "utility.hpp"
/*****************************************************************************
 * This is the only file in rocsparselt which should #include Tensile headers    *
 * or reference Tensile identifiers. tensile_host.hpp defines the interface. *
 *****************************************************************************/

//#include <Tensile/AMDGPU.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>
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

#define ROCSPARSELT_LIB_PATH "/opt/rocm/hipsparselt/lib"

namespace
{
#ifndef WIN32
    std::string rocsparselt_so_path;

    int rocsparselt_dl_iterate_phdr_callback(struct dl_phdr_info* hdr_info, size_t size, void* data)
    {
        // uncomment to see all dependent .so files
        //fprintf(stderr, "rocsparselt so file: %s\n", hdr_info->dlpi_name);
        if(hdr_info->dlpi_name && strstr(hdr_info->dlpi_name, "hipsparselt."))
        {
            rocsparselt_so_path = hdr_info->dlpi_name;
        }
        return 0;
    }
#endif

    /******************************************************
     * Map a rocsparselt type to a corresponding Tensile type *
     ******************************************************/
    template <typename T>
    struct rocsparselt_to_tensile_type
    {
        using tensile_type = T;
    };

    template <>
    struct rocsparselt_to_tensile_type<__half>
    {
        using tensile_type = Tensile::Half;
    };

    template <>
    struct rocsparselt_to_tensile_type<hip_bfloat16>
    {
        using tensile_type = Tensile::BFloat16;
    };

    // int8_t -> int8_t (supported for MI-kernel) / rocsparselt_int8x4 -> PackedInt8x4
    template <>
    struct rocsparselt_to_tensile_type<int8_t>
    {
        using tensile_type = int8_t;
    };

    /********************************************************************
     * Variable template to map a rocsparselt type into a Tensile::DataType *
     ********************************************************************/
    template <typename>
    constexpr auto tensile_datatype = nullptr;

    // int8_t -> int8_t (supported for MI-kernel) / rocsparselt_int8x4 -> PackedInt8x4
    template <>
    constexpr auto tensile_datatype<int8_t> = Tensile::DataType::Int8;

    template <>
    constexpr auto tensile_datatype<__half> = Tensile::DataType::Half;

    template <>
    constexpr auto tensile_datatype<hip_bfloat16> = Tensile::DataType::BFloat16;

    template <>
    constexpr auto tensile_datatype<float> = Tensile::DataType::Float;

    /*************************************************************************
     * Class for converting alpha and beta between rocsparselt and Tensile types *
     * By default, alpha and beta are the same type as Tc compute_type       *
     *************************************************************************/
    template <typename Ti, typename To = Ti, typename Tc = To>
    struct AlphaBeta
    {
        using tensile_type = typename rocsparselt_to_tensile_type<Tc>::tensile_type;
        static void copy(tensile_type* dst, const Tc* src)
        {
            static_assert(sizeof(*src) == sizeof(*dst),
                          "Tensile and rocsparselt types are not the same size");
            static_assert(std::is_standard_layout<tensile_type>{} && std::is_standard_layout<Tc>{},
                          "Tensile or rocsparselt types are not standard layout types");
            memcpy(dst, src, sizeof(*dst));
        }
    };

    /****************************************************************
     * Construct a Tensile Problem from a RocsparseltContractionProblem *
     ****************************************************************/
    template <typename Ti, typename To, typename Tc>
    auto ConstructTensileProblem(const RocsparseltContractionProblem<Ti, To, Tc>& prob)
    {
        // Tensile DataTypes corresponding to rocsparselt data types
        static constexpr Tensile::DataType Tensile_Ti = tensile_datatype<Ti>;
        static constexpr Tensile::DataType Tensile_To = tensile_datatype<To>;
        static constexpr Tensile::DataType Tensile_Tc = tensile_datatype<Tc>;

        // Tensor descriptors for a, b
        Tensile::TensorDescriptor a, b;

        // Tensile Indices for contraction problem
        Tensile::ContractionProblemGemm::FreeIndices  freeIndex(2);
        Tensile::ContractionProblemGemm::BoundIndices boundIndex(1);
        Tensile::ContractionProblemGemm::BatchIndices batchIndex{{2, 2, 2, 2}};

        // Set up GEMM indices
        freeIndex[0].isA = true;
        freeIndex[1].isA = false;
        freeIndex[0].c = freeIndex[0].d = 0;
        freeIndex[1].c = freeIndex[1].d = 1;

        // We set K=0 when alpha==0.
        // This makes alpha==0 a change in the problem, and not just a change in the inputs.
        // It optimizes all problems with alpha==0 into K=0 and alpha=(don't care)
        auto k = prob.k && *prob.alpha ? prob.k : 0;

        // clang-format off

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_a != rocsparselt_operation_none)
        {
            a = {
                    "a",
                    Tensile_Ti,
                    {k, prob.m, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a}
                };
            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            a = {
                    "a",
                    Tensile_Ti,
                    {prob.m, k, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a}
                };
            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_b != rocsparselt_operation_none)
        {
            b = {
                    "b",
                    Tensile_Ti,
                    {prob.n, k, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b}
                };
            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            b = {
                    "b",
                    Tensile_Ti,
                    {k, prob.n, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b}
                };
            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        // clang-format on

        // Descriptor for input matrix C
        Tensile::TensorDescriptor c{"c",
                                    Tensile_To,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_c, prob.col_stride_c, prob.batch_stride_c}};

        // Descriptor for output matrix D
        Tensile::TensorDescriptor d{"d",
                                    Tensile_To,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_d, prob.col_stride_d, prob.batch_stride_d}};

        size_t workspace_size = prob.workspaceSize;

        Tensile::TensorDescriptor e{"e"};
        Tensile::TensorDescriptor bias{"bias"};
        Tensile::TensorDescriptor scaleD{"scaleD"};

        // The ContractionProblemGemm
        Tensile::ContractionProblemGemm tensileProblem{a,
                                                       b,
                                                       c,
                                                       d,
                                                       e,
                                                       bias,
                                                       scaleD,
                                                       freeIndex,
                                                       batchIndex,
                                                       boundIndex,
                                                       value_category(*prob.beta),
                                                       workspace_size};

        tensileProblem.setAlphaType(Tensile_Tc);
        tensileProblem.setBetaType(Tensile_Tc);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(sizeof(Tc) > sizeof(Ti));

        // set batch mode
        tensileProblem.setStridedBatched(prob.strided_batch);

        // alpha and beta are stored by value in Tensile::TypedContractionInputs
        // alpha and beta are copied from host to Tensile::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set tensileAlpha=0
        // Not positive if this is necessary here as well
        typename AlphaBeta<Ti, To, Tc>::tensile_type tensileAlpha;
        if(prob.k)
            AlphaBeta<Ti, To, Tc>::copy(&tensileAlpha, prob.alpha);
        else
            memset(&tensileAlpha, 0, sizeof(tensileAlpha));
        tensileProblem.setAlphaRestriction(Tensile::toScalarValueEnum(tensileAlpha));

        // Add problem predicates for CEqualsD
        tensileProblem.setCEqualsD(prob.C == prob.D);

        tensileProblem.setSparseA(prob.sparseA);

        // set Actvation
        tensileProblem.setActivationType(Tensile::ActivationType::All);
        tensileProblem.setActivationHPA(sizeof(Tc) > sizeof(Ti));
        Tensile::ActivationType tensileAct = Tensile::ActivationType::None;

        switch(prob.act_type)
        {
        case hipsparselt_activation_type::abs:
            tensileAct = Tensile::ActivationType::Abs;
            break;
        case hipsparselt_activation_type::clippedrelu:
            tensileAct = Tensile::ActivationType::Clippedrelu;
            break;
        case hipsparselt_activation_type::gelu:
            tensileAct = Tensile::ActivationType::Gelu;
            break;
        case hipsparselt_activation_type::leakyrelu:
            tensileAct = Tensile::ActivationType::Leakyrelu;
            break;
        case hipsparselt_activation_type::relu:
            tensileAct = Tensile::ActivationType::Relu;
            break;
        case hipsparselt_activation_type::sigmoid:
            tensileAct = Tensile::ActivationType::Sigmoid;
            break;
        case hipsparselt_activation_type::tanh:
            tensileAct = Tensile::ActivationType::Tanh;
            break;
        default:
            break;
        }
        tensileProblem.setActivationEnumArg(tensileAct);

        return tensileProblem;
    }

    /***************************************************************
     * Construct the inputs to a Tensile ContractionProblemGemm        *
     ***************************************************************/
    template <typename Ti, typename To, typename Tc>
    auto GetTensileInputs(const RocsparseltContractionProblem<Ti, To, Tc>& prob)
    {
        // Tensile types corresponding to Ti, To, Tc
        using Tensile_Ti          = typename rocsparselt_to_tensile_type<Ti>::tensile_type;
        using Tensile_To          = typename rocsparselt_to_tensile_type<To>::tensile_type;
        using Tensile_Talpha_beta = typename AlphaBeta<Ti, To, Tc>::tensile_type;

        // Make sure rocsparselt and Tensile types are compatible
        // (Even if Ti=rocsparselt_int8x4, Tensile_Ti=Int8x4, they are both 32-byte)
        static_assert(sizeof(Tensile_Ti) == sizeof(Ti) && sizeof(Tensile_To) == sizeof(To),
                      "Tensile and rocsparselt types are not the same size");

        static_assert(std::is_standard_layout<Ti>{} && std::is_standard_layout<Tensile_Ti>{}
                          && std::is_standard_layout<To>{} && std::is_standard_layout<Tensile_To>{},
                      "Tensile or rocsparselt types are not standard layout types");

        // Structure describing the inputs (A, B, C, D, alpha, beta)
        Tensile::ContractionInputs inputs;

        // Set the A, B, C, D matrices pointers in Tensile
        inputs.a = reinterpret_cast<const void*>(prob.A);
        inputs.b = reinterpret_cast<const void*>(prob.B);
        inputs.c = reinterpret_cast<const void*>(prob.C);
        inputs.d = reinterpret_cast<void*>(prob.D);

        inputs.batchA = reinterpret_cast<void const* const*>(prob.batch_A);
        inputs.batchB = reinterpret_cast<void const* const*>(prob.batch_B);
        inputs.batchC = reinterpret_cast<void const* const*>(prob.batch_C);
        inputs.batchD = reinterpret_cast<void* const*>(prob.batch_D);

        // Set the GSU workspace
        inputs.ws = prob.workspace;

        // alpha and beta are stored by value in Tensile::TypedContractionInputs
        // alpha and beta are copied from host to Tensile::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set inputs.alpha=0
        if(prob.k)
            inputs.alpha = static_cast<Tensile_Talpha_beta>((*prob.alpha));
        else
            inputs.alpha = static_cast<Tensile_Talpha_beta>(0);
        inputs.beta = static_cast<Tensile_Talpha_beta>((*prob.beta));

        if(prob.sparseA)
            inputs.metadata = reinterpret_cast<const unsigned char*>(prob.metadata);

        // push 2 activation arguments
        inputs.activationArgs.push_back(static_cast<Tensile_Talpha_beta>(prob.act_arg0));
        inputs.activationArgs.push_back(static_cast<Tensile_Talpha_beta>(prob.act_arg1));

        return inputs;
    }

    /**************************************************
     * The TensileHost struct interfaces with Tensile *
     **************************************************/
    class TensileHost
    {
        // The library object
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> m_library;
        std::shared_ptr<hipDeviceProp_t> m_deviceProp;

        // The adapter object. mutable is used to allow adapters to be modified
        // even when they are stored in a const vector which is immutable in size
        struct adapter_s
        {
            mutable std::atomic<Tensile::hip::SolutionAdapter*> adapter{nullptr};
            mutable std::mutex                                  mutex;
        };

        // Each device contains an adapter
        std::vector<adapter_s> const m_adapters;

    public:
        TensileHost()
            : m_adapters(GetDeviceCount())
        {
            // We mark TensileHost as initialized. This is so that CI tests can
            // verify that the initialization occurs in the "multiheaded" tests
            rocsparselt_internal_tensile_is_initialized() = true;
        }

        // TensileHost is not copyable or assignable
        TensileHost(const TensileHost&)            = delete;
        TensileHost& operator=(const TensileHost&) = delete;

        // Get the number of devices
        static int GetDeviceCount()
        {
            int count;
            if(hipGetDeviceCount(&count) != hipSuccess)
            {
                hipsparselt_cerr
                    << "\nhipsparselt_error: Could not initialize Tensile host: No devices found"
                    << std::endl;
                hipsparselt_abort();
            }
            return count;
        }

        ~TensileHost()
        {
            for(auto& a : m_adapters)
                delete a.adapter;
        }

        auto& get_library() const
        {
            return m_library;
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
        void initialize(Tensile::hip::SolutionAdapter& adapter, int32_t deviceId)
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

#ifndef HIPSPARSELT_STATIC_LIB
                dl_iterate_phdr(rocsparselt_dl_iterate_phdr_callback, NULL);
                if(rocsparselt_so_path.size())
                    path = std::string{dirname(&rocsparselt_so_path[0])};
#endif // ifndef HIPSPARSELT_STATIC_LIB

                // Find the location of the libraries
                if(TestPath(path + "/../Tensile/library"))
                    path += "/../Tensile/library";
                else
                    path += "/hipsparselt/library";

                if(TestPath(path + "/" + processor))
                    path += "/" + processor;
            }

            // only load modules for the current architecture
            auto dir = path + "/*" + processor + "*co";

            bool no_match = false;
#ifdef WIN32
            std::replace(dir.begin(), dir.end(), '/', '\\');
            WIN32_FIND_DATAA finddata;
            HANDLE           hfine = FindFirstFileA(dir.c_str(), &finddata);
            if(hfine != INVALID_HANDLE_VALUE)
            {
                do
                {
                    std::string codeObjectFile = path + "\\" + finddata.cFileName;
                    adapter.loadCodeObjectFile(codeObjectFile.c_str());
                } while(FindNextFileA(hfine, &finddata));
            }
            else
            {
                no_match = true;
            }
            FindClose(hfine);
#else
            glob_t glob_result{};
            int    g = glob(dir.c_str(), GLOB_NOSORT, nullptr, &glob_result);
            if(!g)
            {
                for(size_t i = 0; i < glob_result.gl_pathc; ++i)
                    (void)adapter.loadCodeObjectFile(glob_result.gl_pathv[i]);
            }
            else if(g == GLOB_NOMATCH)
            {
                no_match = true;
            }
            else
            {
                // clang-format off
                static hipsparselt_internal_ostream& once = hipsparselt_cerr
                                    << "\nrocsparselt warning: glob(\"" << dir << "\", ...) returned "
                                    << (g == GLOB_ABORTED ? "GLOB_ABORTED"
                                                          : g == GLOB_NOSPACE ? "GLOB_NOSPACE"
                                                                              : "an unknown error")
                                    << "." << std::endl;
                (void)once;
                // clang-format on
            }
            globfree(&glob_result);
#endif
            if(no_match)
            {
                static hipsparselt_internal_ostream& once
                    = hipsparselt_cerr
                      << "\nrocsparselt warning: No paths matched " << dir
                      << ". Make sure that ROCSPARSELT_TENSILE_LIBPATH is set correctly."
                      << std::endl;
                (void)once;
            }

            // We initialize a local static variable with a lambda function call to avoid
            // race conditions when multiple threads with different device IDs try to
            // initialize library. This ensures that only one thread initializes library,
            // and other threads trying to initialize library wait for it to complete.
            static int once = [&] {
#ifdef TENSILE_YAML
                path += "/TensileLibrary.yaml";
#else
                path += "/TensileLibrary.dat";
#endif
                if(!TestPath(path))
                {
                    hipsparselt_cerr << "\nhipsparselt_error: Cannot read " << path << ": "
                                     << strerror(errno) << std::endl;
                    //rocsparselt_abort();
                }

                auto lib = Tensile::LoadLibraryFile<Tensile::ContractionProblemGemm>(path);
                if(!lib)
                {
                    hipsparselt_cerr << "\nhipsparselt_error: Could not load " << path << std::endl;
                    return -1;
                }
                else
                {
                    using MSL = Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>;
                    m_library = std::dynamic_pointer_cast<MSL>(lib);
                }
                return 0;
            }();

            if(!m_library && once != 0)
            {
                hipsparselt_cerr << "\nhipsparselt_error: Could not initialize Tensile library"
                                 << std::endl;
                //rocsparselt_abort();
            }

            hipDeviceProp_t prop;
            THROW_IF_HIP_ERROR(hipGetDeviceProperties(&prop, deviceId));

            m_deviceProp = std::make_shared<hipDeviceProp_t>(prop);
        }
    };

    // Return the library and adapter for the current HIP device
    auto& get_library_and_adapter(
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>>* library
        = nullptr,
        std::shared_ptr<hipDeviceProp_t>* deviceProp = nullptr,
        int                               device     = -1)
    try
    {
        // TensileHost is initialized on the first call
        static TensileHost host;

        if(device == -1)
            if(hipGetDevice(&device) != hipSuccess)
                throw "Invalid Device";

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
                adapter = new Tensile::hip::SolutionAdapter;

                // Initialize the adapter and possibly the library
                host.initialize(*adapter, device);

                // Atomically change the adapter stored for this device ID
                a.adapter.store(adapter, std::memory_order_release);
            }
        }

        // If an adapter is found, it is assumed that the library is initialized
        if(library)
            *library = host.get_library();
        if(deviceProp)
            *deviceProp = host.get_device_property();

        return *adapter;
    }
    catch(const std::exception& e)
    {
        hipsparselt_cerr << "\nhipsparselt_error: Could not initialize Tensile host:\n"
                         << e.what() << std::endl;
        hipsparselt_abort();
    }
    catch(...)
    {
        hipsparselt_cerr
            << "\nhipsparselt_error: Could not initialize Tensile host:\nUnknown exception thrown"
            << std::endl;
        hipsparselt_abort();
    }

    /**************************************************************************
    * We normally print error messages only once, to avoid excessive logging *
    **************************************************************************/
    void print_once(const hipsparselt_internal_ostream& msg)
    {
        if(rocsparselt_suppress_tensile_error_messages())
            return;
        static constexpr char varname[] = "ROCSPARSELT_VERBOSE_TENSILE_ERROR";
        static const char*    verbose   = getenv(varname);
        if(!verbose)
        {
            static auto& once = hipsparselt_cerr
                                << msg
                                << "\nThis message will be only be displayed once, unless the "
                                << varname << " environment variable is set." << std::endl;
            (void)once;
        }
        else
            hipsparselt_cerr << msg << std::endl;
    }

} // namespace

/******************************************************************************
 * runContractionProblem calls Tensile to run a contraction problem described *
 * by RocsparseltContractionProblem                                               *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocsparselt_status runContractionProblem(const RocsparseltContractionProblem<Ti, To, Tc>& prob,
                                         _rocsparselt_matmul_config*                      configs,
                                         int*                                             config_id,
                                         const int config_max_id,
                                         const int search_iterations)
{
    rocsparselt_status                            status = rocsparselt_status_internal_error;
    std::shared_ptr<Tensile::ContractionSolution> solution;

    try
    {
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
        std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
        std::shared_ptr<Tensile::Hardware>                                               hardware;

        auto& adapter = get_library_and_adapter(&library, &deviceProp, prob.handle->device);

        hardware          = Tensile::hip::GetDevice(*deviceProp);
        auto tensile_prob = ConstructTensileProblem(prob);

        if(!config_max_id || configs == nullptr)
        {
            hipsparselt_internal_ostream msg;
            print_once(msg << "\nhipsparselt_error: No Tensile solution found for " << prob);
            status = rocsparselt_status_not_implemented;
        }
        else
        {
            auto tensile_inputs = GetTensileInputs(prob);

            if(!search_iterations)
            {
                if(configs[*config_id].max_workspace_bytes > prob.workspaceSize
                   || (configs[*config_id].max_workspace_bytes > 0 && prob.workspace == nullptr))
                {
                    hipsparselt_cerr << "config " << *config_id << " need extra workspace "
                                     << configs[*config_id].max_workspace_bytes << " bytes - skip."
                                     << std::endl;
                    return rocsparselt_status_internal_error;
                }

                solution = library->getSolutionByIndex(
                    tensile_prob, *hardware, configs[*config_id].index);
                if(!solution)
                {
                    hipsparselt_cerr << "Solution of config:" << *config_id
                                     << " does not exists - skip" << std::endl;
                    return rocsparselt_status_not_implemented;
                }

                RETURN_IF_HIP_ERROR(
                    adapter.launchKernels(solution->solve(tensile_prob, tensile_inputs, *hardware),
                                          prob.streams[0],
                                          nullptr,
                                          nullptr));
            }
            else
            {
                float      min_ms = std::numeric_limits<float>::max();
                hipEvent_t startEvent, stopEvent;
                float      ms, sum_ms;
                RETURN_IF_HIP_ERROR(hipEventCreate(&startEvent));
                RETURN_IF_HIP_ERROR(hipEventCreate(&stopEvent));
                for(int id = 0; id < config_max_id; id++)
                {
                    if(configs[id].max_workspace_bytes > prob.workspaceSize
                       || (configs[id].max_workspace_bytes > 0 && prob.workspace == nullptr))
                    {
                        hipsparselt_cerr << "config " << id << " need extra workspace "
                                         << configs[id].max_workspace_bytes << " bytes - skip."
                                         << std::endl;
                        continue;
                    }

                    solution
                        = library->getSolutionByIndex(tensile_prob, *hardware, configs[id].index);
                    if(!solution)
                    {
                        hipsparselt_cerr << "Solution of config:" << id << " does not exists - skip"
                                         << std::endl;
                        continue;
                    }

                    //warm up
                    RETURN_IF_HIP_ERROR(adapter.launchKernels(
                        solution->solve(tensile_prob, tensile_inputs, *hardware),
                        prob.streams[0],
                        nullptr,
                        nullptr));

                    sum_ms = 0.0f;
                    for(int i = 0; i < search_iterations; i++)
                    {
                        RETURN_IF_HIP_ERROR(adapter.launchKernels(
                            solution->solve(tensile_prob, tensile_inputs, *hardware),
                            prob.streams[0],
                            startEvent,
                            stopEvent));
                        RETURN_IF_HIP_ERROR(hipEventSynchronize(stopEvent));
                        RETURN_IF_HIP_ERROR(hipEventElapsedTime(&ms, startEvent, stopEvent));
                        sum_ms += ms;
                    }

                    if(sum_ms < min_ms)
                    {
                        min_ms     = sum_ms;
                        *config_id = id;
                    }
                }
                RETURN_IF_HIP_ERROR(hipEventDestroy(startEvent));
                RETURN_IF_HIP_ERROR(hipEventDestroy(stopEvent));

                if(min_ms == std::numeric_limits<float>::max())
                    return rocsparselt_status_internal_error;
            }

            status = rocsparselt_status_success;
        }
    }
    catch(const std::exception& e)
    {
        hipsparselt_internal_ostream msg;
        print_once(msg << "\nhipsparselt_error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
    }
    catch(...)
    {
        hipsparselt_internal_ostream msg;
        print_once(msg << "\nhipsparselt_error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
    }

    return status;
}

/******************************************************************************
 * getBestSolutions calls Tensile's findTopSolutions and converts to          *
 * _rocsparselt_matmul_config.                                                *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocsparselt_status getBestSolutions(const RocsparseltContractionProblem<Ti, To, Tc>& prob,
                                    int                                              requestConfigs,
                                    _rocsparselt_matmul_config*                      configs,
                                    int*                                             foundConfigs)
{
    std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> library;
    std::shared_ptr<hipDeviceProp_t>                                                 deviceProp;
    std::shared_ptr<Tensile::Hardware>                                               hardware;

    // auto &adapter =
    get_library_and_adapter(&library, &deviceProp, prob.handle->device);

    hardware          = Tensile::hip::GetDevice(*deviceProp);
    auto tensile_prob = ConstructTensileProblem(prob);
    // auto handle = prob.handle;

    auto solutions = library->findTopSolutions(tensile_prob, *hardware, requestConfigs);

    *foundConfigs = std::min((int)solutions.size(), requestConfigs);
    for(size_t i = 0; i < *foundConfigs; i++)
    {
        auto solution                  = solutions[i];
        configs[i].index               = solution->index;
        configs[i].max_workspace_bytes = solution->requiredWorkspaceSize(tensile_prob);
    }
    return rocsparselt_status_success;
}

/***************************************************************
 * ! \brief  Initialize rocsparselt for the current HIP device, to *
 * avoid costly startup time at the first call on that device. *
 ***************************************************************/
extern "C" void rocsparselt_initialize()
{
    get_library_and_adapter();
}

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for testing) *
 ***********************************************************************************/
std::atomic_bool& rocsparselt_internal_tensile_is_initialized()
{
    static std::atomic_bool init;
    return init;
}

/******************************************************************************
 * Intantiate the cases of runContractionProblem which are needed to satisfy  *
 * rocsparselt dependencies. This file's template functions are not defined in a  *
 * header file, in order to keep Tensile and rocsparselt separate.                *
 ******************************************************************************/
#define GENERATE_DEFINITIONS(Ti, To, Tc)                           \
    template rocsparselt_status runContractionProblem<Ti, To, Tc>( \
        const RocsparseltContractionProblem<Ti, To, Tc>&,          \
        _rocsparselt_matmul_config*,                               \
        int*,                                                      \
        const int,                                                 \
        const int);                                                \
    template rocsparselt_status getBestSolutions<Ti, To, Tc>(      \
        const RocsparseltContractionProblem<Ti, To, Tc>&, int, _rocsparselt_matmul_config*, int*);

GENERATE_DEFINITIONS(__half, __half, float)
GENERATE_DEFINITIONS(hip_bfloat16, hip_bfloat16, float)
GENERATE_DEFINITIONS(int8_t, int8_t, float)

#undef GENERATE_DEFINITIONS
