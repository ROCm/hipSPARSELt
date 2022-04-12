/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparselt_arguments.hpp"
#include "../../library/src/include/tuple_helper.hpp"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <istream>
#include <ostream>
#include <utility>

void Arguments::init()
{
    // match python in rocsparselt_common.py

    function[0] = 0;
    strcpy(name, "rocsparselt-bench");
    category[0]            = 0;
    known_bug_platforms[0] = 0;

    alpha = 1.0;
    beta  = 0.0;

    stride_a = 0;
    stride_b = 0;
    stride_c = 0;
    stride_d = 0;

    user_allocated_workspace = 0;

    M = 128;
    N = 128;
    K = 128;

    lda = 0;
    ldb = 0;
    ldc = 0;
    ldd = 0;

    batch_count = 1;

    iters      = 10;
    cold_iters = 2;

    algo           = 0;
    solution_index = 0;

    a_type       = rocsparselt_datatype_f16_r;
    b_type       = rocsparselt_datatype_f16_r;
    c_type       = rocsparselt_datatype_f16_r;
    d_type       = rocsparselt_datatype_f16_r;
    compute_type = rocsparselt_compute_f32;

    prune_algo = rocsparselt_prune_smfmac_strip;

    initialization = rocsparselt_initialization::hpl;

    // memory padding for testing write out of bounds
    pad = 4096;

    // 16 bit
    threads = 0;
    streams = 0;

    // bytes
    devices = 0;

    norm_check = 0;
    unit_check = 1;
    timing     = 0;

    transA = '*';
    transB = '*';

    c_noalias_d = false;
    HMM         = false;
}

// Function to print Arguments out to stream in YAML format
rocsparselt_internal_ostream& operator<<(rocsparselt_internal_ostream& os, const Arguments& arg)
{
    // delim starts as "{ " and becomes ", " afterwards
    auto print_pair = [&, delim = "{ "](const char* name, const auto& value) mutable {
        os << delim << std::make_pair(name, value);
        delim = ", ";
    };

    // Print each (name, value) tuple pair
#define NAME_VALUE_PAIR(NAME) print_pair(#NAME, arg.NAME)
    // cppcheck-suppress unknownMacro
    FOR_EACH_ARGUMENT(NAME_VALUE_PAIR, ;);

    // Closing brace
    return os << " }\n";
}

// Google Tests uses this automatically with std::ostream to dump parameters
std::ostream& operator<<(std::ostream& os, const Arguments& arg)
{
    rocsparselt_internal_ostream oss;
    // Print to rocsparselt_internal_ostream, then transfer to std::ostream
    return os << (oss << arg);
}

// Function to read Structures data from stream
std::istream& operator>>(std::istream& is, Arguments& arg)
{
    is.read(reinterpret_cast<char*>(&arg), sizeof(arg));
    return is;
}

// Error message about incompatible binary file format
static void validation_error [[noreturn]] (const char* name)
{
    rocsparselt_cerr << "Arguments field \"" << name
                     << "\" does not match format.\n\n"
                        "Fatal error: Binary test data does match input format.\n"
                        "Ensure that rocsparselt_arguments.hpp and rocsparselt_common.yaml\n"
                        "define exactly the same Arguments, that rocsparselt_gentest.py\n"
                        "generates the data correctly, and that endianness is the same."
                     << std::endl;
    rocsparselt_abort();
}

// rocsparselt_gentest.py is expected to conform to this format.
// rocsparselt_gentest.py uses rocsparselt_common.yaml to generate this format.
void Arguments::validate(std::istream& ifs)
{
    char      header[12]{}, trailer[12]{};
    Arguments arg{};

    ifs.read(header, sizeof(header));
    ifs >> arg;
    ifs.read(trailer, sizeof(trailer));
    if(strcmp(header, "rocSPARSELt"))
        validation_error("header");

    if(strcmp(trailer, "ROCsparselT"))
        validation_error("trailer");

    auto check_func = [sig = 0u](const char* name, const auto& value) mutable {
        static_assert(sizeof(value) <= 256,
                      "Fatal error: Arguments field is too large (greater than 256 bytes).");
        for(size_t i = 0; i < sizeof(value); ++i)
        {
            if(reinterpret_cast<const unsigned char*>(&value)[i] ^ sig ^ i)
                validation_error(name);
        }
        sig = (sig + 89) % 256;
    };

    // Apply check_func to each pair (name, value) of Arguments as a tuple
#define CHECK_FUNC(NAME) check_func(#NAME, arg.NAME)
    FOR_EACH_ARGUMENT(CHECK_FUNC, ;);
}
