/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocsparselt_data.hpp"
#include "rocsparselt_datatype2string.hpp"
#include "rocsparselt_test.hpp"
#include "spmm/testing_spmm.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{

    // ----------------------------------------------------------------------------
    // spmm
    // ----------------------------------------------------------------------------

    // In the general case of <Ti, To, Tc>, these tests do not apply, and if this
    // functor is called, an internal error message is generated. When converted
    // to bool, this functor returns false.
    template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
    struct spmm_testing : rocsparselt_test_invalid
    {
    };

    // When Ti = To = Tc != void, this test applies.
    // When converted to bool, this functor returns true.
    template <typename Ti, typename To, typename Tc>
    struct spmm_testing<
        Ti,
        To,
        Tc,
        std::enable_if_t<std::is_same<Ti, rocsparselt_half>{}
                         || std::is_same<Ti, rocsparselt_bfloat16>{} || std::is_same<Ti, int8_t>{}>>
        : rocsparselt_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "spmm"))
                testing_spmm<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "spmm_batched"))
                testing_spmm<Ti, To, Tc, rocsparselt_batch_type::batched>(arg);
            else if(!strcmp(arg.function, "spmm_strided_batched"))
                testing_spmm<Ti, To, Tc, rocsparselt_batch_type::strided_batched>(arg);
            else if(!strcmp(arg.function, "spmm_bad_arg"))
                testing_spmm_bad_arg<Ti, To, Tc>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct spmm_test : RocSparseLt_Test<spmm_test, spmm_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocsparselt_spmm_dispatch<type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "spmm") || !strcmp(arg.function, "spmm_batched")
                   || !strcmp(arg.function, "spmm_strided_batched")
                   || !strcmp(arg.function, "spmm_bad_arg");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocSparseLt_TestName<spmm_test> name(arg.name);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "bad_arg";
            }
            else
            {
                name << rocsparselt_datatype2string(arg.a_type)
                     << rocsparselt_datatype2string(arg.b_type)
                     << rocsparselt_datatype2string(arg.c_type)
                     << rocsparselt_datatype2string(arg.d_type)
                     << rocsparselt_computetype2string(arg.compute_type);

                if(arg.activation_type != rocsparselt_activation_type::none)
                {
                    name << '_' << rocsparselt_activation_type_string(arg.activation_type);
                    switch(arg.activation_type)
                    {
                    case rocsparselt_activation_type::clippedrelu:
                    case rocsparselt_activation_type::tanh:
                        name << '_' << arg.activation_arg1 << '_' << arg.activation_arg2;
                        break;
                    case rocsparselt_activation_type::leakyrelu:
                        name << '_' << arg.activation_arg1;
                        break;
                    default:
                        break;
                    }
                }

                name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB);

                name << '_' << arg.M << '_' << arg.N << '_' << arg.K << '_' << arg.alpha << '_'
                     << arg.lda << '_' << arg.ldb << '_' << arg.beta << '_' << arg.ldc << '_'
                     << arg.ldd;

                if(strstr(arg.function, "_batched") != nullptr)
                    name << '_' << arg.batch_count;

                if(strstr(arg.function, "_strided_batched") != nullptr)
                    name << '_' << arg.stride_a << '_' << arg.stride_b << '_' << arg.stride_c;
            }

            return std::move(name);
        }
    };

    TEST_P(spmm_test, spmm)
    {
        RUN_TEST_ON_THREADS_STREAMS(rocsparselt_spmm_dispatch<spmm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(spmm_test);

} // namespace
