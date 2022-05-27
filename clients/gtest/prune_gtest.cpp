/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocsparselt_data.hpp"
#include "rocsparselt_datatype2string.hpp"
#include "rocsparselt_test.hpp"
#include "spmm/testing_prune.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{

    // ----------------------------------------------------------------------------
    // prune
    // ----------------------------------------------------------------------------

    // In the general case of <Ti, To, Tc>, these tests do not apply, and if this
    // functor is called, an internal error message is generated. When converted
    // to bool, this functor returns false.
    template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
    struct prune_testing : rocsparselt_test_invalid
    {
    };

    // When Ti = To = Tc != void, this test applies.
    // When converted to bool, this functor returns true.
    template <typename Ti, typename To, typename Tc>
    struct prune_testing<
        Ti,
        To,
        Tc,
        std::enable_if_t<std::is_same<Ti, rocsparselt_half>{}
                         || std::is_same<Ti, rocsparselt_bfloat16>{} || std::is_same<Ti, int8_t>{}>>
        : rocsparselt_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "prune"))
                testing_prune<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "prune_batched"))
                testing_prune<Ti, To, Tc, rocsparselt_batch_type::batched>(arg);
            else if(!strcmp(arg.function, "prune_strided_batched"))
                testing_prune<Ti, To, Tc, rocsparselt_batch_type::strided_batched>(arg);
            else if(!strcmp(arg.function, "prune_bad_arg"))
                testing_prune_bad_arg<Ti, To, Tc>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct prune_test : RocSparseLt_Test<prune_test, prune_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocsparselt_spmm_dispatch<type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "prune") || !strcmp(arg.function, "prune_batched")
                   || !strcmp(arg.function, "prune_strided_batched")
                   || !strcmp(arg.function, "prune_bad_arg");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocSparseLt_TestName<prune_test> name(arg.name);
            switch(arg.prune_algo)
            {
            case rocsparselt_prune_smfmac_tile:
                name << "tile";
                break;
            case rocsparselt_prune_smfmac_strip:
                name << "strip";
                break;
            default:
                name << "invalid";
                break;
            }

            name << "_" << rocsparselt_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {

                name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB);

                name << '_' << arg.M << '_' << arg.N << '_' << arg.K << '_' << arg.lda;

                name << '_' << arg.batch_count;

                name << '_' << arg.stride_a;
            }
            return std::move(name);
        }
    };

    TEST_P(prune_test, conversion)
    {
        RUN_TEST_ON_THREADS_STREAMS(rocsparselt_spmm_dispatch<prune_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(prune_test);

} // namespace
