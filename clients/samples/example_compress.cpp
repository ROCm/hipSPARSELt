/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include "hipsparselt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>
#include <stdlib.h>
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

#ifndef CHECK_HIPSPARSELT_ERROR
#define CHECK_HIPSPARSELT_ERROR(error)                           \
    if(error != HIPSPARSE_STATUS_SUCCESS)                        \
    {                                                            \
        fprintf(stderr, "hipSPARSELt error(Err=%d) : ", error);  \
        if(error == HIPSPARSE_STATUS_NOT_INITIALIZED)            \
            fprintf(stderr, "HIPSPARSE_STATUS_NOT_INITIALIZED"); \
        if(error == HIPSPARSE_STATUS_INTERNAL_ERROR)             \
            fprintf(stderr, " HIPSPARSE_STATUS_INTERNAL_ERROR"); \
        if(error == HIPSPARSE_STATUS_INVALID_VALUE)              \
            fprintf(stderr, "HIPSPARSE_STATUS_INVALID_VALUE");   \
        if(error == HIPSPARSE_STATUS_ALLOC_FAILED)               \
            fprintf(stderr, "HIPSPARSE_STATUS_ALLOC_FAILED");    \
        if(error == HIPSPARSE_STATUS_ARCH_MISMATCH)              \
            fprintf(stderr, "HIPSPARSE_STATUS_ARCH_MISMATCH");   \
        fprintf(stderr, "\n");                                   \
        exit(EXIT_FAILURE);                                      \
    }
#endif

inline unsigned char generate_metadata(int a, int b, int c, int d)
{
    unsigned char metadata = (a)&0x03;
    metadata |= (b << 2) & 0x0C;
    metadata |= ((c - 4) << 4) & 0x30;
    metadata |= (((d - 4) << 6)) & 0xC0;
    return metadata;
}

inline void extract_metadata(unsigned metadata, int& a, int& b, int& c, int& d)
{
    a = metadata & 0x03;
    b = (metadata >> 2) & 0x03;
    c = ((metadata >> 4) & 0x03);
    d = ((metadata >> 6) & 0x03);
}

std::string metadata_to_bits_str(unsigned char meta)
{
    std::string str;
    for(int b = 0; b < 8; b++)
    {
        str.append(std::to_string(meta >> (7 - b) & 0x01));
    }
    return str;
}

template <typename Ti, typename Tc>
void compress(const Ti*      in,
              Ti*            out,
              unsigned char* metadata,
              int64_t        m,
              int64_t        n,
              int64_t        stride1,
              int64_t        stride2,
              int64_t        stride_b,
              int64_t        c_stride1,
              int64_t        c_stride2,
              int64_t        c_stride_b,
              int64_t        m_stride1,
              int64_t        m_stride2,
              int64_t        m_stride_b,
              int            num_batches)
{
    constexpr int tiles_y = 8;

    for(int b = 0; b < num_batches; b++)
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j += tiles_y)
            {
                Ti  values[4];
                int idx[4];
                int m_idx = 0;

                auto valid_data = [&](int midx, int index, Ti value) {
                    idx[midx]    = index;
                    values[midx] = value;
                };

                for(int k = 0; k < tiles_y; k++)
                {
                    auto offset = b * stride_b + i * stride1 + (j + k) * stride2;
                    Ti   value  = in[offset];

                    if(m_idx > 4)
                    {
                        printf("Err - The given matrix is not a 2:4 sparse matrix\n");
                        return;
                    }

                    if((k == 3 && m_idx == 0) || (k == 7 && m_idx == 2))
                    {
                        offset = b * stride_b + i * stride1 + (j + k - 1) * stride2;
                        value  = in[offset];
                        valid_data(m_idx++, k - 1, value);
                    }
                    if((k == 3 && m_idx == 1) || (k == 7 && m_idx == 3)
                       || static_cast<Tc>(value) != static_cast<Tc>(0))
                    {
                        offset = b * stride_b + i * stride1 + (j + k) * stride2;
                        value  = in[offset];
                        valid_data(m_idx++, k, value);
                    }
                }
                for(int k = 0; k < 4; k++)
                {
                    auto c_offset = b * c_stride_b + i * c_stride1 + (j / 2 + k) * c_stride2;
                    out[c_offset] = values[k];
                }

                unsigned char md = generate_metadata(idx[0], idx[1], idx[2], idx[3]);

                auto metadata_offset      = b * m_stride_b + i * m_stride1 + (j / 8) * m_stride2;
                metadata[metadata_offset] = md;
            }
        }
}

template <typename T>
void test_prune_check(hipsparseLtHandle_t*           handle,
                      hipsparseLtMatmulDescriptor_t* matmul,
                      T*                             d,
                      hipStream_t                    stream,
                      bool                           expected)
{
    int* d_valid;
    int  h_valid = 0;
    CHECK_HIP_ERROR(hipMalloc(&d_valid, sizeof(int)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtSpMMAPruneCheck(handle, matmul, d, d_valid, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(&h_valid, d_valid, sizeof(int), hipMemcpyDeviceToHost, stream));
    hipStreamSynchronize(stream);
    auto s = expected ? "passed" : "falied";
    if((h_valid == 0) == expected)
        std::printf("expected %s, prune check PASSED\n", s);
    else
        std::printf("expected %s, prune check FAILED: wrong result\n", s);
    CHECK_HIP_ERROR(hipFree(d_valid));
}

// default sizes
#define DIM1 127
#define DIM2 128
#define DIM3 129
#define BATCH_COUNT 10

template <typename T>
void validate_gold(T*             A,
                   T*             C,
                   unsigned char* M,
                   int64_t        n1,
                   int64_t        n2,
                   int64_t        n3,
                   int64_t        s1,
                   int64_t        s2,
                   int64_t        s3,
                   int64_t        c_n1,
                   int64_t        c_n2,
                   int64_t        c_n3,
                   int64_t        c_s1,
                   int64_t        c_s2,
                   int64_t        c_s3,
                   int64_t        m_n1,
                   int64_t        m_n2,
                   int64_t        m_n3,
                   int64_t        m_s1,
                   int64_t        m_s2,
                   int64_t        m_s3)
{
    bool correct = true;
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    for(int i3 = 0; i3 < m_n3; i3++)
    {
        for(int i1 = 0; i1 < m_n1; i1++)
        {
            for(int i2 = 0; i2 < m_n2; i2++)
            {
                auto compare = [&](T a, T b, int x, int64_t apos, int64_t bpos) {
                    using c_type = std::conditional_t<std::is_same<__half, T>::value, float, T>;
                    if(static_cast<c_type>(a) != static_cast<c_type>(b))
                    {
                        // direct floating point comparison is not reliable
                        std::printf("(%d, %d, %d, %d):\t[%ld]%f vs. [%ld]%f\n",
                                    i3,
                                    i1,
                                    i2,
                                    x,
                                    apos,
                                    static_cast<double>(a),
                                    bpos,
                                    static_cast<double>(b));
                        return false;
                    }
                    return true;
                };

                auto m_pos = (i1 * m_s1) + (i2 * m_s2) + (i3 * m_s3);
                int  idx[4];
                extract_metadata(M[m_pos], idx[0], idx[1], idx[2], idx[3]);
                idx[2] += 4;
                idx[3] += 4;
                int m_idx = 0;
                for(int i = 0; i < 8; i++)
                {
                    auto a_pos = (i1 * s1) + ((i2 * 8 + i) * s2) + (i3 * s3);
                    auto c_pos = (i1 * c_s1) + ((i2 * 4 + m_idx) * c_s2) + (i3 * c_s3);
                    T    a     = A[a_pos];
                    T    b     = static_cast<T>(0.0f);
                    if(i == idx[m_idx])
                    {

                        b = C[c_pos];
                        m_idx++;
                    }
                    correct &= compare(a, b, i, a_pos, c_pos);
                }
            }
        }
    }
    if(correct)
        std::printf("compressed gold test PASSED\n");
    else
        std::printf("compressed gold test FAILED: wrong result\n");
}

template <typename T>
void validate_compressed(
    T* A, T* B, int64_t n1, int64_t n2, int64_t n3, int64_t s1, int64_t s2, int64_t s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    using c_type = std::conditional_t<std::is_same<__half, T>::value, float, T>;
    bool correct = true;
    for(int i3 = 0; i3 < n3; i3++)
    {
        for(int i1 = 0; i1 < n1; i1++)
        {
            for(int i2 = 0; i2 < n2; i2++)
            {
                T a = A[(i1 * s1) + (i2 * s2) + (i3 * s3)];
                T b = B[(i1 * s1) + (i2 * s2) + (i3 * s3)];
                if(static_cast<c_type>(a) != static_cast<c_type>(b))
                {
                    // direct floating point comparison is not reliable
                    std::printf("(%d, %d, %d):\t%f vs. %f\n",
                                i3,
                                i1,
                                i2,
                                static_cast<double>(a),
                                static_cast<double>(b));
                    correct = false;
                }
            }
        }
    }
    if(correct)
        std::printf("compressed PASSED\n");
    else
        std::printf("compressed FAILED: wrong result\n");
}

void validate_metadata(unsigned char* A,
                       unsigned char* B,
                       int64_t        n1,
                       int64_t        n2,
                       int64_t        n3,
                       int64_t        s1,
                       int64_t        s2,
                       int64_t        s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    bool correct = true;
    for(int i3 = 0; i3 < n3; i3++)
    {
        for(int i1 = 0; i1 < n1; i1++)
        {
            for(int i2 = 0; i2 < n2; i2++)
            {
                auto a = A[(i1 * s1) + (i2 * s2) + (i3 * s3)];
                auto b = B[(i1 * s1) + (i2 * s2) + (i3 * s3)];
                if(a != b)
                {
                    auto a_str = metadata_to_bits_str(A[(i1 * s1) + (i2 * s2) + (i3 * s3)]);
                    auto b_str = metadata_to_bits_str(B[(i1 * s1) + (i2 * s2) + (i3 * s3)]);
                    std::printf(
                        "(%d, %d, %d):\t%s vs. %s\n", i3, i1, i2, a_str.c_str(), b_str.c_str());
                    correct = false;
                }
            }
        }
    }
    if(correct)
        std::printf("metadata PASSED\n");
    else
        std::printf("metadata FAILED: wrong result\n");
}

template <typename T>
void print_strided_batched(
    const char* name, T* A, int64_t n1, int64_t n2, int64_t n3, int64_t s1, int64_t s2, int64_t s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    printf("---------- %s (MxN=%ldx%ld,batch=%ld,stride0=%ld, stride1=%ld)----------\n",
           name,
           n1,
           n2,
           n3,
           s1,
           s2);
    int max_size = 128;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                printf("[%ld]%8.1f\t ",
                       (i1 * s1) + (i2 * s2) + (i3 * s3),
                       static_cast<float>(A[(i1 * s1) + (i2 * s2) + (i3 * s3)]));
            }
            printf("\n");
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            printf("\n");
    }
}

void print_strided_batched_meta(const char*    name,
                                unsigned char* A,
                                int64_t        n1,
                                int64_t        n2,
                                int64_t        n3,
                                int64_t        s1,
                                int64_t        s2,
                                int64_t        s3)
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
                auto meta = A[(i1 * s1) + (i2 * s2) + (i3 * s3)];
                auto str  = metadata_to_bits_str(meta);
                int  a, b, c, d;
                extract_metadata(meta, a, b, c, d);
                std::printf("[%ld][bits=%s]%02x%02x%02x%02x\t",
                            (i1 * s1) + (i2 * s2) + (i3 * s3),
                            str.c_str(),
                            a,
                            b,
                            c + 4,
                            d + 4);
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
    std::cerr
        << "Usage: " << argv[0] << " <options>\n"
        << "options:\n"
        << "\t-h, --help\t\t\t\tShow this help message\n"
        << "\t-v, --verbose\t\t\t\tverbose output\n"
        << "\t-m \t\t\tm\t\tGEMM_STRIDED_BATCHED argument m\n"
        << "\t-n \t\t\tn\t\tGEMM_STRIDED_BATCHED argument n\n"
        << "\t--ld \t\t\tld \t\tGEMM_STRIDED_BATCHED argument lda\n"
        << "\t--trans \t\ttrans \tGEMM_STRIDED_BATCHED argument trans_a\n"
        << "\t--stride \t\tstride \tGEMM_STRIDED_BATCHED argument stride_a\n"
        << "\t--batch_count \t\tbatch_count \tGEMM_STRIDED_BATCHED argument batch count\n"
        << "\t-r \t\tprecision \tGEMM_STRIDED_BATCHED argument precsion, etc., h=half, b=bfloat16\n"
        << "\t--header \t\theader \t\tprint header for output\n"
        << std::endl;
}

static int parse_arguments(int                    argc,
                           char*                  argv[],
                           int64_t&               m,
                           int64_t&               n,
                           int64_t&               ld,
                           int64_t&               stride,
                           int&                   batch_count,
                           hipsparseOperation_t&  trans,
                           hipsparseLtDatatype_t& type,
                           bool&                  header,
                           bool&                  verbose)
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
                        trans = HIPSPARSE_OPERATION_NON_TRANSPOSE;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans = HIPSPARSE_OPERATION_TRANSPOSE;
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
                        type = HIPSPARSELT_R_16BF;
                    }
                    else if(strncmp(argv[i], "i8", 1) == 0)
                    {
                        type = HIPSPARSELT_R_8I;
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

bool bad_argument(hipsparseOperation_t trans,
                  int64_t              m,
                  int64_t              n,
                  int64_t              ld,
                  int64_t              stride,
                  int64_t              batch_count)
{
    bool argument_error = false;
    if((trans == HIPSPARSE_OPERATION_NON_TRANSPOSE) && (ld < m))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << ld << " < " << m << std::endl;
    }
    if((trans == HIPSPARSE_OPERATION_TRANSPOSE) && (ld < n))
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
void run(int64_t               m,
         int64_t               n,
         int64_t               ld,
         int64_t               stride,
         int                   batch_count,
         hipsparseOperation_t  trans,
         hipsparseLtDatatype_t type,
         bool                  verbose)
{

    int64_t stride_1, stride_2;
    int64_t c_stride_1, c_stride_2, c_stride_b, c_stride_b_r;
    int64_t m_stride_1, m_stride_2, m_stride_b, m_stride_b_r;
    int64_t row, col;
    int     size_1;
    if(trans == HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        std::cout << "N";
        row          = m;
        col          = n;
        stride_1     = 1;
        stride_2     = ld;
        size_1       = ld * n;
        c_stride_1   = 1;
        c_stride_2   = m;
        c_stride_b_r = n / 2 * c_stride_2;

        m_stride_1   = n / 8;
        m_stride_2   = 1;
        m_stride_b_r = m * m_stride_1;
    }
    else
    {
        std::cout << "T";
        row          = n;
        col          = m;
        stride_1     = ld;
        stride_2     = 1;
        size_1       = ld * m;
        c_stride_1   = n / 2;
        c_stride_2   = 1;
        c_stride_b_r = m * c_stride_1;

        m_stride_1   = n / 8;
        m_stride_2   = 1;
        m_stride_b_r = m * m_stride_1;
    }
    c_stride_b         = stride == 0 ? 0 : c_stride_b_r;
    m_stride_b         = stride == 0 ? 0 : m_stride_b_r;
    auto batch_count_f = stride == 0 ? 1 : batch_count;

    std::cout << ", " << m << ", " << n << ", " << ld << ", " << stride << ", " << batch_count
              << std::endl;

    int size = stride == 0 ? size_1 * batch_count : stride * batch_count;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<T> hp(size);
    std::vector<T> hp_test(size);

    // initial data on host
    initialize_a(hp, size);

    if(verbose)
    {
        printf("\n");
        if(trans == HIPSPARSE_OPERATION_NON_TRANSPOSE)
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

    hipsparseLtHandle_t             handle;
    hipsparseLtMatDescriptor_t      matA, matB, matC, matD;
    hipsparseLtMatmulDescriptor_t   matmul;
    hipsparseLtMatmulAlgSelection_t alg_sel;
    hipsparseLtMatmulPlan_t         plan;

    CHECK_HIPSPARSELT_ERROR(hipsparseLtInit(&handle));

    CHECK_HIPSPARSELT_ERROR(hipsparseLtStructuredDescriptorInit(&handle,
                                                                &matA,
                                                                row,
                                                                col,
                                                                ld,
                                                                16,
                                                                type,
                                                                HIPSPARSE_ORDER_COLUMN,
                                                                HIPSPARSELT_SPARSITY_50_PERCENT));
    CHECK_HIPSPARSELT_ERROR(
        hipsparseLtDenseDescriptorInit(&handle, &matB, n, m, n, 16, type, HIPSPARSE_ORDER_COLUMN));
    CHECK_HIPSPARSELT_ERROR(
        hipsparseLtDenseDescriptorInit(&handle, &matC, m, m, m, 16, type, HIPSPARSE_ORDER_COLUMN));
    CHECK_HIPSPARSELT_ERROR(
        hipsparseLtDenseDescriptorInit(&handle, &matD, m, m, m, 16, type, HIPSPARSE_ORDER_COLUMN));

    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matA, HIPSPARSELT_MAT_NUM_BATCHES, &batch_count, sizeof(batch_count)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matA, HIPSPARSELT_MAT_BATCH_STRIDE, &stride, sizeof(stride)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matB, HIPSPARSELT_MAT_NUM_BATCHES, &batch_count, sizeof(batch_count)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matB, HIPSPARSELT_MAT_BATCH_STRIDE, &stride, sizeof(stride)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matC, HIPSPARSELT_MAT_NUM_BATCHES, &batch_count, sizeof(batch_count)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matC, HIPSPARSELT_MAT_BATCH_STRIDE, &stride, sizeof(stride)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matD, HIPSPARSELT_MAT_NUM_BATCHES, &batch_count, sizeof(batch_count)));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matD, HIPSPARSELT_MAT_BATCH_STRIDE, &stride, sizeof(stride)));

    auto compute_type = type == HIPSPARSELT_R_8I ? HIPSPARSE_COMPUTE_32I : HIPSPARSE_COMPUTE_32F;

    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmulDescriptorInit(&handle,
                                                            &matmul,
                                                            trans,
                                                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                            &matA,
                                                            &matB,
                                                            &matC,
                                                            &matD,
                                                            compute_type));

    CHECK_HIPSPARSELT_ERROR(
        hipsparseLtSpMMAPrune(&handle, &matmul, d, d_test, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream));
    hipStreamSynchronize(stream);

    CHECK_HIP_ERROR(hipMemcpy(hp_test.data(), d_test, sizeof(T) * size, hipMemcpyDeviceToHost));

    if(verbose)
    {
        print_strided_batched(
            "hp_test calculated", &hp_test[0], m, n, batch_count, stride_1, stride_2, stride);
    }

    test_prune_check<T>(&handle, &matmul, d_test, stream, true);

    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmulAlgSelectionInit(
        &handle, &alg_sel, &matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT));

    size_t workspace_size, compressed_size;
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size));

    CHECK_HIPSPARSELT_ERROR(
        hipsparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size));

    CHECK_HIPSPARSELT_ERROR(hipsparseLtSpMMACompressedSize(&handle, &plan, &compressed_size));

    printf("compressed_size = %ld\n", compressed_size);

    T*             d_compressed;
    std::vector<T> hp_gold(compressed_size / sizeof(T));
    std::vector<T> hp_compressed(compressed_size / sizeof(T));
    CHECK_HIP_ERROR(hipMalloc(&d_compressed, compressed_size));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtSpMMACompress(&handle, &plan, d_test, d_compressed, stream));
    hipStreamSynchronize(stream);
    CHECK_HIP_ERROR(
        hipMemcpy(hp_compressed.data(), d_compressed, compressed_size, hipMemcpyDeviceToHost));

    compress<T, float>(&hp_test[0],
                       &hp_gold[0],
                       reinterpret_cast<unsigned char*>(&hp_gold[c_stride_b_r * batch_count_f]),
                       m,
                       n,
                       stride_1,
                       stride_2,
                       stride,
                       c_stride_1,
                       c_stride_2,
                       c_stride_b,
                       m_stride_1,
                       m_stride_2,
                       m_stride_b,
                       batch_count);

    if(verbose)
    {
        printf("\n");
        print_strided_batched("host gold compress calculated",
                              &hp_gold[0],
                              m,
                              n / 2,
                              batch_count,
                              c_stride_1,
                              c_stride_2,
                              c_stride_b_r);

        print_strided_batched_meta(
            "host gold metadata calculated",
            reinterpret_cast<unsigned char*>(&hp_gold[c_stride_b_r * batch_count_f]),
            m,
            n / 8,
            batch_count,
            m_stride_1,
            m_stride_2,
            m_stride_b_r);

        print_strided_batched("device compress calculated",
                              &hp_compressed[0],
                              m,
                              n / 2,
                              batch_count,
                              c_stride_1,
                              c_stride_2,
                              c_stride_b_r);
        print_strided_batched_meta(
            "device metadata calculated",
            reinterpret_cast<unsigned char*>(&hp_compressed[c_stride_b_r * batch_count_f]),
            m,
            n / 8,
            batch_count,
            m_stride_1,
            m_stride_2,
            m_stride_b_r);
    }

    validate_gold(&hp_test[0],
                  &hp_gold[0],
                  reinterpret_cast<unsigned char*>(&hp_gold[c_stride_b_r * batch_count_f]),
                  m,
                  n,
                  batch_count,
                  stride_1,
                  stride_2,
                  stride,
                  m,
                  n / 2,
                  batch_count,
                  c_stride_1,
                  c_stride_2,
                  c_stride_b,
                  m,
                  n / 8,
                  batch_count,
                  m_stride_1,
                  m_stride_2,
                  m_stride_b);
    validate_compressed(
        &hp_gold[0], &hp_compressed[0], m, n / 2, batch_count, c_stride_1, c_stride_2, c_stride_b);
    validate_metadata(
        reinterpret_cast<unsigned char*>(&hp_gold[c_stride_b_r * batch_count_f]),
        reinterpret_cast<unsigned char*>(&hp_compressed[c_stride_b_r * batch_count_f]),
        m,
        n / 8,
        batch_count,
        m_stride_1,
        m_stride_2,
        m_stride_b);

    //validate(&hp_test[0], &hp_compressed[0], reinterpret_cast<unsigned char*>(&hp_compressed[c_stride_b * batch_count]), m, n, batch_count, stride_1, stride_2, stride, m, n/2, batch_count, c_stride_1, c_stride_2, c_stride_b, m, n/8, batch_count, m_stride_1, m_stride_2, m_stride_b);

    //CHECK_HIP_ERROR(hipMalloc(&d_compressed, compressed_size));

    CHECK_HIP_ERROR(hipFree(d));
    CHECK_HIP_ERROR(hipFree(d_test));

    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmulPlanDestroy(&plan));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescriptorDestroy(&matA));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescriptorDestroy(&matB));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescriptorDestroy(&matC));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescriptorDestroy(&matD));
    CHECK_HIPSPARSELT_ERROR(hipsparseLtDestroy(&handle));
}

int main(int argc, char* argv[])
{
    // initialize parameters with default values
    hipsparseOperation_t trans = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    // invalid int and float for hipsparselt spmm int and float arguments
    int64_t invalid_int64 = std::numeric_limits<int64_t>::min() + 1;
    int     invalid_int   = std::numeric_limits<int>::min() + 1;
    float   invalid_float = std::numeric_limits<float>::quiet_NaN();

    // initialize to invalid value to detect if values not specified on command line
    int64_t m = invalid_int64, n = invalid_int64, ld = invalid_int64, stride = invalid_int64;

    int                   batch_count = invalid_int;
    hipsparseLtDatatype_t type        = HIPSPARSELT_R_16F;

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
        ld = trans == HIPSPARSE_OPERATION_NON_TRANSPOSE ? m : n;
    if(stride == invalid_int64)
        stride = trans == HIPSPARSE_OPERATION_NON_TRANSPOSE ? ld * n : ld * m;
    if(batch_count == invalid_int)
        batch_count = BATCH_COUNT;

    if(bad_argument(trans, m, n, ld, stride, batch_count))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    if(header)
    {
        std::cout << "type,trans,M,N,K,ld,stride,batch_count,"
                     "result,error";
        std::cout << std::endl;
    }

    switch(type)
    {
    case HIPSPARSELT_R_16F:
        std::cout << "H_";
        run<__half>(m, n, ld, stride, batch_count, trans, type, verbose);
        break;
    case HIPSPARSELT_R_16BF:
        std::cout << "BF16_";
        run<hip_bfloat16>(m, n, ld, stride, batch_count, trans, type, verbose);
        break;
    case HIPSPARSELT_R_8I:
        std::cout << "I8_";
        run<int8_t>(m, n, ld, stride, batch_count, trans, type, verbose);
        break;
    default:
        break;
    }

    return EXIT_SUCCESS;
}
