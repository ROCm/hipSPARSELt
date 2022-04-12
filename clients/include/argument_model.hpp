/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocsparselt_arguments.hpp"

namespace ArgumentLogging
{
    const double NA_value = -1.0; // invalid for time, GFlop, GB
}

void ArgumentModel_set_log_function_name(bool f);
bool ArgumentModel_get_log_function_name();

// ArgumentModel template has a variadic list of argument enums
template <rocsparselt_argument... Args>
class ArgumentModel
{
    // Whether model has a particular parameter
    // TODO: Replace with C++17 fold expression ((Args == param) || ...)
    static constexpr bool has(rocsparselt_argument param)
    {
        for(auto x : {Args...})
            if(x == param)
                return true;
        return false;
    }

public:
    void log_perf(rocsparselt_internal_ostream& name_line,
                  rocsparselt_internal_ostream& val_line,
                  const Arguments&              arg,
                  double                        gpu_us,
                  double                        gflops,
                  double                        gbytes,
                  double                        cpu_us,
                  double                        norm1,
                  double                        norm2,
                  double                        norm3,
                  double                        norm4)
    {
        constexpr bool has_batch_count = has(e_batch_count);
        int64_t        batch_count     = has_batch_count ? arg.batch_count : 1;
        int64_t        hot_calls       = arg.iters < 1 ? 1 : arg.iters;

        // gpu time is total cumulative over hot calls, cpu is not
        if(hot_calls > 1)
            gpu_us /= hot_calls;

        // per/us to per/sec *10^6
        double rocsparselt_gflops = gflops * batch_count / gpu_us * 1e6;
        double rocsparselt_GBps   = gbytes * batch_count / gpu_us * 1e6;

        // append performance fields
        if(gflops != ArgumentLogging::NA_value)
        {
            name_line << ",rocsparselt-Gflops";
            val_line << ", " << rocsparselt_gflops;
        }

        if(gbytes != ArgumentLogging::NA_value)
        {
            // GB/s not usually reported for non-memory bound functions
            name_line << ",rocsparselt-GB/s";
            val_line << ", " << rocsparselt_GBps;
        }

        name_line << ",us";
        val_line << ", " << gpu_us;

        if(arg.unit_check || arg.norm_check)
        {
            if(cpu_us != ArgumentLogging::NA_value)
            {
                if(gflops != ArgumentLogging::NA_value)
                {
                    double cblas_gflops = gflops * batch_count / cpu_us * 1e6;
                    name_line << ",CPU-Gflops";
                    val_line << "," << cblas_gflops;
                }

                name_line << ",CPU-us";
                val_line << "," << cpu_us;
            }
            if(arg.norm_check)
            {
                if(norm1 != ArgumentLogging::NA_value)
                {
                    name_line << ",norm_error_1";
                    val_line << "," << norm1;
                }
                if(norm2 != ArgumentLogging::NA_value)
                {
                    name_line << ",norm_error_2";
                    val_line << "," << norm2;
                }
                if(norm3 != ArgumentLogging::NA_value)
                {
                    name_line << ",norm_error_3";
                    val_line << "," << norm3;
                }
                if(norm4 != ArgumentLogging::NA_value)
                {
                    name_line << ",norm_error_4";
                    val_line << "," << norm4;
                }
            }
        }
    }

    template <typename T>
    void log_args(rocsparselt_internal_ostream& str,
                  const Arguments&              arg,
                  double                        gpu_us,
                  double                        gflops,
                  double                        gpu_bytes = ArgumentLogging::NA_value,
                  double                        cpu_us    = ArgumentLogging::NA_value,
                  double                        norm1     = ArgumentLogging::NA_value,
                  double                        norm2     = ArgumentLogging::NA_value,
                  double                        norm3     = ArgumentLogging::NA_value,
                  double                        norm4     = ArgumentLogging::NA_value)
    {
        rocsparselt_internal_ostream name_list;
        rocsparselt_internal_ostream value_list;

        if(ArgumentModel_get_log_function_name())
        {
            auto delim = ",";
            name_list << "function" << delim;
            value_list << arg.function << delim;
        }

        // Output (name, value) pairs to name_list and value_list
        auto print = [&, delim = ""](const char* name, auto&& value) mutable {
            name_list << delim << name;
            value_list << delim << value;
            delim = ",";
        };

#if __cplusplus >= 201703L
        // C++17
        (ArgumentsHelper::apply<Args>(print, arg, T{}), ...);
#else
        // C++14. TODO: Remove when C++17 is used
        (void)(int[]){(ArgumentsHelper::apply<Args>{}()(print, arg, T{}), 0)...};
#endif

        if(arg.timing)
            log_perf(name_list,
                     value_list,
                     arg,
                     gpu_us,
                     gflops,
                     gpu_bytes,
                     cpu_us,
                     norm1,
                     norm2,
                     norm3,
                     norm4);

        str << name_list << "\n" << value_list << std::endl;
    }
};
