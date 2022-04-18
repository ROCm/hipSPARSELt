/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocsparselt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_ROCSPARSE_ERROR
#define CHECK_ROCSPARSE_ERROR(error)                              \
    if(error != rocsparse_status_success)                         \
    {                                                             \
        fprintf(stderr, "rocSPARSELt error: ");                   \
        if(error == rocsparse_status_invalid_handle)              \
            fprintf(stderr, "rocsparse_status_invalid_handle");   \
        if(error == rocsparse_status_not_implemented)             \
            fprintf(stderr, " rocsparse_status_not_implemented"); \
        if(error == rocsparse_status_invalid_pointer)             \
            fprintf(stderr, "rocsparse_status_invalid_pointer");  \
        if(error == rocsparse_status_invalid_size)                \
            fprintf(stderr, "rocsparse_status_invalid_size");     \
        if(error == rocsparse_status_memory_error)                \
            fprintf(stderr, "rocsparse_status_memory_error");     \
        if(error == rocsparse_status_internal_error)              \
            fprintf(stderr, "rocsparse_status_internal_error");   \
        fprintf(stderr, "\n");                                    \
        exit(EXIT_FAILURE);                                       \
    }
#endif

// default sizes
#define DIM1 127
#define DIM2 128
#define DIM3 129
#define BATCH_COUNT 10

template <typename Ti, typename Tc>
inline float norm2(Ti a, Ti b)
{
    auto ac = static_cast<Tc>(a);
    auto bc = static_cast<Tc>(b);

    return static_cast<Tc>(sqrt(ac * ac + bc * bc));
}

template <typename Ti, typename Tc>
void prune_strip(const Ti* in,
                 Ti*       out,
                 int64_t   m,
                 int64_t   n,
                 int64_t   stride1,
                 int64_t   stride2,
                 int       num_batches,
                 int64_t   stride_b)
{
    for(int b = 0; b < num_batches; b++)
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j += 4)
            {
                size_t pos[4];
                for(int k = 0; k < 4; k++)
                {
                    pos[k] = b * stride_b + i * stride1 + (j + k) * stride2;
                }

                auto max_norm2 = static_cast<Tc>(-1.0f);
                int  pos_a, pos_b;
                for(int a = 0; a < 4; a++)
                {
                    for(int b = a + 1; b < 4; b++)
                    {
                        auto norm2_v = norm2<Ti, double>(in[pos[a]], in[pos[b]]);
                        if(norm2_v > max_norm2)
                        {
                            pos_a     = a;
                            pos_b     = b;
                            max_norm2 = norm2_v;
                        }
                    }
                }

                for(int k = 0; k < 4; k++)
                {
                    if(k == pos_a || k == pos_b)
                    {
                        if(in != out)
                            out[pos[k]] = in[pos[k]];
                    }
                    else
                        out[pos[k]] = static_cast<Ti>(0.0f);
                }
            }
        }
}

template <typename T>
void validate(T* A, T* B, int64_t n1, int64_t n2, int64_t n3, int64_t s1, int64_t s2, int64_t s3)
{
    bool correct = true;
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    for(int i3 = 0; i3 < n3; i3++)
    {
        for(int i1 = 0; i1 < n1; i1++)
        {
            for(int i2 = 0; i2 < n2; i2++)
            {
                auto pos     = (i1 * s1) + (i2 * s2) + (i3 * s3);
                auto value_a = A[pos];
                auto value_b = B[pos];
                if(value_a != value_b)
                {
                    // direct floating point comparison is not reliable
                    std::printf("(%d, %d, %d):\t%f vs. %f\n",
                                i3,
                                i1,
                                i2,
                                static_cast<double>(value_a),
                                static_cast<double>(value_b));
                    correct = false;
                    //break;
                }
            }
        }
    }
    if(correct)
        std::printf("prune strip test PASSED\n");
    else
        std::printf("prune strip test FAILED: wrong result\n");
}

template <typename T>
void test_prune_check(rocsparselt_handle       handle,
                      rocsparselt_matmul_descr matmul,
                      T*                       d,
                      hipStream_t              stream,
                      bool                     expected)
{
    int* d_valid;
    int  h_valid = 0;
    CHECK_HIP_ERROR(hipMalloc(&d_valid, sizeof(int)));
    CHECK_ROCSPARSE_ERROR(rocsparselt_smfmac_prune_check(handle, matmul, d, d_valid, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(&h_valid, d_valid, sizeof(int), hipMemcpyDeviceToHost, stream));
    hipStreamSynchronize(stream);
    auto s = expected ? "passed" : "falied";
    if((h_valid == 0) == expected)
        std::printf("expected %s, prune check PASSED\n", s);
    else
        std::printf("expected %s, prune check FAILED: wrong result\n", s);
    CHECK_HIP_ERROR(hipFree(d_valid));
}

template <typename T>
void print_strided_batched(
    const char* name, T* A, int64_t n1, int64_t n2, int64_t n3, int64_t s1, int64_t s2, int64_t s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    printf("---------- %s ----------\n", name);
    int max_size = 128;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                printf("%8.1f ", static_cast<float>(A[(i1 * s1) + (i2 * s2) + (i3 * s3)]));
            }
            printf("\n");
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            printf("\n");
    }
}

// cppcheck-suppress constParameter
static void show_usage(char* argv[])
{
    std::cerr << "Usage: " << argv[0] << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\t\tShow this help message\n"
              << "\t-v, --verbose\t\t\t\tverbose output\n"
              << "\t-m \t\t\tm\t\tGEMM_STRIDED_BATCHED argument m\n"
              << "\t-n \t\t\tn\t\tGEMM_STRIDED_BATCHED argument n\n"
              << "\t--ld \t\t\tld \t\tGEMM_STRIDED_BATCHED argument lda\n"
              << "\t--trans \t\ttrans \tGEMM_STRIDED_BATCHED argument trans_a\n"
              << "\t--stride \t\tstride \tGEMM_STRIDED_BATCHED argument stride_a\n"
              << "\t--batch_count \t\tbatch_count \tGEMM_STRIDED_BATCHED argument batch count\n"
              << "\t--header \t\theader \t\tprint header for output\n"
              << std::endl;
}

static int parse_arguments(int                   argc,
                           char*                 argv[],
                           int64_t&              m,
                           int64_t&              n,
                           int64_t&              ld,
                           int64_t&              stride,
                           int&                  batch_count,
                           rocsparse_operation&  trans,
                           rocsparselt_datatype& type,
                           bool&                 header,
                           bool&                 verbose)
{
    if(argc >= 2)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
            {
                if((arg == "-h") || (arg == "--help"))
                {
                    return EXIT_FAILURE;
                }
                if((arg == "-v") || (arg == "--verbose"))
                {
                    verbose = true;
                }
                else if(arg == "--header")
                {
                    header = true;
                }
                else if((arg == "-m") && (i + 1 < argc))
                {
                    m = atoi(argv[++i]);
                }
                else if((arg == "-n") && (i + 1 < argc))
                {
                    n = atoi(argv[++i]);
                }
                else if((arg == "--batch_count") && (i + 1 < argc))
                {
                    batch_count = atoi(argv[++i]);
                }
                else if((arg == "--ld") && (i + 1 < argc))
                {
                    ld = atoi(argv[++i]);
                }
                else if((arg == "--stride") && (i + 1 < argc))
                {
                    stride = atoi(argv[++i]);
                }
                else if((arg == "--trans") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans = rocsparse_operation_none;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans = rocsparse_operation_transpose;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "-r") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "h", 1) == 0)
                    {
                        type = type;
                    }
                    else if(strncmp(argv[i], "b", 1) == 0)
                    {
                        type = rocsparselt_datatype_bf16_r;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else
                {
                    std::cerr << "error with " << arg << std::endl;
                    std::cerr << "do not recognize option" << std::endl << std::endl;
                    return EXIT_FAILURE;
                }
            }
            else
            {
                std::cerr << "error with " << arg << std::endl;
                std::cerr << "option must start with - or --" << std::endl << std::endl;
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

bool bad_argument(rocsparse_operation trans,
                  int64_t             m,
                  int64_t             n,
                  int64_t             ld,
                  int64_t             stride,
                  int64_t             batch_count)
{
    bool argument_error = false;
    if((trans == rocsparse_operation_none) && (ld < m))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << ld << " < " << m << std::endl;
    }
    if((trans == rocsparse_operation_transpose) && (ld < n))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << ld << " < " << n << std::endl;
    }
    if(stride < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride < 0" << std::endl;
    }
    if(batch_count < 1)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument batch_count = " << batch_count << " < 1" << std::endl;
    }

    return argument_error;
}

template <typename T>
void initialize_a(std::vector<T>& ha, int64_t size_a)
{
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = static_cast<T>(rand() % 17);
    }
}

template <typename T>
void run(int64_t              m,
         int64_t              n,
         int64_t              ld,
         int64_t              stride,
         int                  batch_count,
         rocsparse_operation  trans,
         rocsparselt_datatype type,
         bool                 verbose)
{
    int64_t stride_1, stride_2;
    int64_t row, col;
    int     size_1;
    if(trans == rocsparse_operation_none)
    {
        std::cout << ", N";
        row      = m;
        col      = n;
        stride_1 = 1;
        stride_2 = ld;
        size_1   = ld * n;
    }
    else
    {
        std::cout << ", T";
        row      = n;
        col      = m;
        stride_1 = ld;
        stride_2 = 1;
        size_1   = ld * m;
    }

    std::cout << ", " << m << ", " << n << ", " << ld << ", " << stride << ", " << batch_count
              << std::endl;

    int size = stride == 0 ? size_1 * batch_count : stride * batch_count;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<T> hp(size);
    std::vector<T> hp_test(size);
    std::vector<T> hp_gold(size);

    // initial data on host
    initialize_a(hp, size);

    if(verbose)
    {
        printf("\n");
        if(trans == rocsparse_operation_none)
        {
            print_strided_batched("host initial", &hp[0], m, n, batch_count, 1, ld, stride);
        }
        else
        {
            print_strided_batched("host initial", &hp[0], m, n, batch_count, ld, 1, stride);
        }
    }

    // allocate memory on device
    T*          d;
    T*          d_test;
    void*       d_wworkspace;
    int         num_streams = 0;
    hipStream_t stream      = nullptr;
    hipStreamCreate(&stream);

    CHECK_HIP_ERROR(hipMalloc(&d, size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_test, size * sizeof(T)));
    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(d, hp.data(), sizeof(T) * size, hipMemcpyHostToDevice));

    rocsparselt_handle       handle;
    rocsparselt_mat_descr    matA, matB, matC, matD;
    rocsparselt_matmul_descr matmul;

    CHECK_ROCSPARSE_ERROR(rocsparselt_init(&handle));

    CHECK_ROCSPARSE_ERROR(rocsparselt_structured_descr_init(handle,
                                                            &matA,
                                                            row,
                                                            col,
                                                            ld,
                                                            16,
                                                            type,
                                                            rocsparse_order_column,
                                                            rocsparselt_sparsity_50_percent));
    CHECK_ROCSPARSE_ERROR(
        rocsparselt_dense_descr_init(handle, &matB, n, m, ld, 16, type, rocsparse_order_column));
    CHECK_ROCSPARSE_ERROR(
        rocsparselt_dense_descr_init(handle, &matC, m, n, ld, 16, type, rocsparse_order_column));
    CHECK_ROCSPARSE_ERROR(
        rocsparselt_dense_descr_init(handle, &matD, m, n, ld, 16, type, rocsparse_order_column));

    CHECK_ROCSPARSE_ERROR(rocsparselt_mat_descr_set_attribute(
        handle, matA, rocsparselt_mat_num_batches, &batch_count, sizeof(batch_count)));
    CHECK_ROCSPARSE_ERROR(rocsparselt_mat_descr_set_attribute(
        handle, matA, rocsparselt_mat_batch_stride, &stride, sizeof(stride)));
    CHECK_ROCSPARSE_ERROR(rocsparselt_mat_descr_set_attribute(
        handle, matB, rocsparselt_mat_num_batches, &batch_count, sizeof(batch_count)));
    CHECK_ROCSPARSE_ERROR(rocsparselt_mat_descr_set_attribute(
        handle, matB, rocsparselt_mat_batch_stride, &stride, sizeof(stride)));
    CHECK_ROCSPARSE_ERROR(rocsparselt_mat_descr_set_attribute(
        handle, matC, rocsparselt_mat_num_batches, &batch_count, sizeof(batch_count)));
    CHECK_ROCSPARSE_ERROR(rocsparselt_mat_descr_set_attribute(
        handle, matC, rocsparselt_mat_batch_stride, &stride, sizeof(stride)));
    CHECK_ROCSPARSE_ERROR(rocsparselt_mat_descr_set_attribute(
        handle, matD, rocsparselt_mat_num_batches, &batch_count, sizeof(batch_count)));
    CHECK_ROCSPARSE_ERROR(rocsparselt_mat_descr_set_attribute(
        handle, matD, rocsparselt_mat_batch_stride, &stride, sizeof(stride)));

    CHECK_ROCSPARSE_ERROR(rocsparselt_matmul_descr_init(
        handle, &matmul, trans, trans, matA, matB, matC, matD, rocsparselt_compute_f32));

    CHECK_ROCSPARSE_ERROR(rocsparselt_smfmac_prune(
        handle, matmul, d, d_test, rocsparselt_prune_smfmac_strip, stream));
    hipStreamSynchronize(stream);

    CHECK_HIP_ERROR(hipMemcpy(hp_test.data(), d_test, sizeof(T) * size, hipMemcpyDeviceToHost));

    prune_strip<T, float>(hp.data(), hp_gold.data(), m, n, stride_1, stride_2, batch_count, stride);

    if(verbose)
    {
        auto stride_r = stride == 0 ? size_1 : stride;
        print_strided_batched(
            "hp_gold calculated", &hp_gold[0], m, n, batch_count, stride_1, stride_2, stride_r);
        print_strided_batched(
            "hp_test calculated", &hp_test[0], m, n, batch_count, stride_1, stride_2, stride_r);
    }

    validate<T>(&hp_gold[0], &hp_test[0], m, n, batch_count, stride_1, stride_2, stride);

    test_prune_check<T>(handle, matmul, d_test, stream, true);
    test_prune_check<T>(handle, matmul, d, stream, false);

    CHECK_HIP_ERROR(hipFree(d));
    CHECK_HIP_ERROR(hipFree(d_test));

    CHECK_ROCSPARSE_ERROR(rocsparselt_matmul_descr_destroy(matmul));
    CHECK_ROCSPARSE_ERROR(rocsparselt_mat_descr_destroy(matA));
    CHECK_ROCSPARSE_ERROR(rocsparselt_mat_descr_destroy(matB));
    CHECK_ROCSPARSE_ERROR(rocsparselt_mat_descr_destroy(matC));
    CHECK_ROCSPARSE_ERROR(rocsparselt_mat_descr_destroy(matD));
    CHECK_ROCSPARSE_ERROR(rocsparselt_destroy(handle));
}
int main(int argc, char* argv[])
{
    // initialize parameters with default values
    rocsparse_operation trans = rocsparse_operation_none;

    // invalid int and float for rocsparselt spmm int and float arguments
    int64_t invalid_int64 = std::numeric_limits<int64_t>::min() + 1;
    int     invalid_int   = std::numeric_limits<int>::min() + 1;
    float   invalid_float = std::numeric_limits<float>::quiet_NaN();

    // initialize to invalid value to detect if values not specified on command line
    int64_t m = invalid_int64, n = invalid_int64, ld = invalid_int64, stride = invalid_int64;

    int                  batch_count = invalid_int;
    rocsparselt_datatype type        = rocsparselt_datatype_f16_r;

    bool verbose = false;
    bool header  = false;

    if(parse_arguments(argc, argv, m, n, ld, stride, batch_count, trans, type, header, verbose))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    // when arguments not specified, set to default values
    if(m == invalid_int64)
        m = DIM1;
    if(n == invalid_int64)
        n = DIM2;
    if(ld == invalid_int64)
        ld = trans == rocsparse_operation_none ? m : n;
    if(stride == invalid_int64)
        stride = trans == rocsparse_operation_none ? ld * n : ld * m;
    if(batch_count == invalid_int)
        batch_count = BATCH_COUNT;

    if(bad_argument(trans, m, n, ld, stride, batch_count))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    if(header)
    {
        std::cout << "type,trans,M,N,ld,stride,batch_count,"
                     "result,error";
        std::cout << std::endl;
    }

    switch(type)
    {
    case rocsparselt_datatype_f16_r:
        std::cout << "H";
        run<rocsparselt_half>(m, n, ld, stride, batch_count, trans, type, verbose);
        break;
    case rocsparselt_datatype_bf16_r:
        std::cout << "BF16";
        run<rocsparselt_bfloat16>(m, n, ld, stride, batch_count, trans, type, verbose);
        break;
    default:
        break;
    }

    return EXIT_SUCCESS;
}
