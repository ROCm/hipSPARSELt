/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../../library/src/include/utility.hpp"
#include "rocsparselt.h"
#include "rocsparselt_vector.hpp"
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

/*!\file
 * \brief provide common utilities
 */

// We use rocsparselt_cout and rocsparselt_cerr instead of std::cout, std::cerr, stdout and stderr,
// for thread-safe IO.
//
// All stdio and std::ostream functions related to stdout and stderr are poisoned, as are
// functions which can create buffer overflows, or which are inherently thread-unsafe.
//
// This must come after the header #includes above, to avoid poisoning system headers.
//
// This is only enabled for rocsparselt-test and rocsparselt-bench.
//
// If you are here because of a poisoned identifier error, here is the rationale for each
// included identifier:
//
// cout, stdout: rocsparselt_cout should be used instead, for thread-safe and atomic line buffering
// cerr, stderr: rocsparselt_cerr should be used instead, for thread-safe and atomic line buffering
// clog: C++ stream which should not be used
// gets: Always unsafe; buffer-overflows; removed from later versions of the language; use fgets
// puts, putchar, fputs, printf, fprintf, vprintf, vfprintf: Use rocsparselt_cout or rocsparselt_cerr
// sprintf, vsprintf: Possible buffer overflows; us snprintf or vsnprintf instead
// strerror: Thread-unsafe; use snprintf / dprintf with %m or strerror_* alternatives
// strsignal: Thread-unsafe; use sys_siglist[signal] instead
// strtok: Thread-unsafe; use strtok_r
// gmtime, ctime, asctime, localtime: Thread-unsafe
// tmpnam: Thread-unsafe; use mkstemp or related functions instead
// putenv: Use setenv instead
// clearenv, fcloseall, ecvt, fcvt: Miscellaneous thread-unsafe functions
// sleep: Might interact with signals by using alarm(); use nanosleep() instead
// abort: Does not abort as cleanly as rocsparselt_abort, and can be caught by a signal handler

#if defined(GOOGLE_TEST) || defined(ROCSPARSELT_BENCH)
#undef stdout
#undef stderr
#pragma GCC poison cout cerr clog stdout stderr gets puts putchar fputs fprintf printf sprintf    \
    vfprintf vprintf vsprintf perror strerror strtok gmtime ctime asctime localtime tmpnam putenv \
        clearenv fcloseall ecvt fcvt sleep abort strsignal
#else
// Suppress warnings about hipMalloc(), hipFree() except in rocsparselt-test and rocsparselt-bench
#undef hipMalloc
#undef hipFree
#endif

#define LIMITED_MEMORY_STRING "Error: Attempting to allocate more memory than available."
#define TOO_MANY_DEVICES_STRING "Error: Too many devices requested."
#define HMM_NOT_SUPPORTED "Error: HMM not supported."

// TODO: This is dependent on internal gtest behaviour.
// Compared with result.message() when a test ended. Note that "Succeeded\n" is
// added to the beginning of the message automatically by gtest, so this must be compared.
#define LIMITED_MEMORY_STRING_GTEST "Succeeded\n" LIMITED_MEMORY_STRING
#define TOO_MANY_DEVICES_STRING_GTEST "Succeeded\n" TOO_MANY_DEVICES_STRING
#define HMM_NOT_SUPPORTED_GTEST "Succeeded\n" HMM_NOT_SUPPORTED

enum class rocsparselt_batch_type
{
    none = 0,
    batched,
    strided_batched
};

/* ============================================================================================ */
/*! \brief  local handle which is automatically created and destroyed  */
class rocsparselt_local_handle
{
    rocsparselt_handle m_handle;
    void*              m_memory = nullptr;

public:
    rocsparselt_local_handle();

    explicit rocsparselt_local_handle(const Arguments& arg);

    ~rocsparselt_local_handle();

    rocsparselt_local_handle(const rocsparselt_local_handle&) = delete;
    rocsparselt_local_handle(rocsparselt_local_handle&&)      = delete;
    rocsparselt_local_handle& operator=(const rocsparselt_local_handle&) = delete;
    rocsparselt_local_handle& operator=(rocsparselt_local_handle&&) = delete;

    // Allow rocsparselt_local_handle to be used anywhere rocsparselt_handle is expected
    operator rocsparselt_handle&()
    {
        return m_handle;
    }
    operator const rocsparselt_handle&() const
    {
        return m_handle;
    }
};

/* ============================================================================================ */
/*! \brief  local matrix descriptor which is automatically created and destroyed  */
class rocsparselt_local_mat_descr
{
    rocsparselt_mat_descr m_descr;
    rocsparselt_status    m_status  = rocsparselt_status_not_initialized;
    static constexpr int  alignment = 16;

public:
    rocsparselt_local_mat_descr(rocsparselt_matrix_type mtype,
                                rocsparselt_handle      handle,
                                int64_t                 row,
                                int64_t                 col,
                                int64_t                 ld,
                                rocsparselt_datatype    type,
                                rocsparselt_order       order)
    {
        if(mtype == rocsparselt_matrix_type_structured)
            this->m_status = rocsparselt_structured_descr_init(handle,
                                                               &this->m_descr,
                                                               row,
                                                               col,
                                                               ld,
                                                               this->alignment,
                                                               type,
                                                               order,
                                                               rocsparselt_sparsity_50_percent);
        else
            this->m_status = rocsparselt_dense_descr_init(
                handle, &this->m_descr, row, col, ld, this->alignment, type, order);
    }

    ~rocsparselt_local_mat_descr()
    {
        if(this->m_status == rocsparselt_status_success)
            rocsparselt_mat_descr_destroy(this->m_descr);
    }

    rocsparselt_local_mat_descr(const rocsparselt_local_mat_descr&) = delete;
    rocsparselt_local_mat_descr(rocsparselt_local_mat_descr&&)      = delete;
    rocsparselt_local_mat_descr& operator=(const rocsparselt_local_mat_descr&) = delete;
    rocsparselt_local_mat_descr& operator=(rocsparselt_local_mat_descr&&) = delete;

    rocsparselt_status status()
    {
        return m_status;
    }

    // Allow rocsparselt_local_mat_descr to be used anywhere rocsparselt_mat_descr is expected
    operator rocsparselt_mat_descr&()
    {
        return m_descr;
    }
    operator const rocsparselt_mat_descr&() const
    {
        return m_descr;
    }
};

/* ============================================================================================ */
/*! \brief  local matrix multiplication descriptor which is automatically created and destroyed  */
class rocsparselt_local_matmul_descr
{
    rocsparselt_matmul_descr m_descr  = nullptr;
    rocsparselt_status       m_status = rocsparselt_status_not_initialized;

public:
    rocsparselt_local_matmul_descr(rocsparselt_handle       handle,
                                   rocsparselt_operation    opA,
                                   rocsparselt_operation    opB,
                                   rocsparselt_mat_descr    matA,
                                   rocsparselt_mat_descr    matB,
                                   rocsparselt_mat_descr    matC,
                                   rocsparselt_mat_descr    matD,
                                   rocsparselt_compute_type compute_type)
    {
        this->m_status = rocsparselt_matmul_descr_init(
            handle, &this->m_descr, opA, opB, matA, matB, matC, matD, compute_type);
    }

    ~rocsparselt_local_matmul_descr()
    {
        if(this->m_status == rocsparselt_status_success)
            rocsparselt_matmul_descr_destroy(this->m_descr);
    }

    rocsparselt_local_matmul_descr(const rocsparselt_local_matmul_descr&) = delete;
    rocsparselt_local_matmul_descr(rocsparselt_local_matmul_descr&&)      = delete;
    rocsparselt_local_matmul_descr& operator=(const rocsparselt_local_matmul_descr&) = delete;
    rocsparselt_local_matmul_descr& operator=(rocsparselt_local_matmul_descr&&) = delete;

    rocsparselt_status status()
    {
        return m_status;
    }

    // Allow rocsparselt_local_matmul_descr to be used anywhere rocsparselt_matmul_descr is expected
    operator rocsparselt_matmul_descr&()
    {
        return m_descr;
    }
    operator const rocsparselt_matmul_descr&() const
    {
        return m_descr;
    }
};

/* ================================================================================================================= */
/*! \brief  local matrix multiplication algorithm selection descriptor which is automatically created and destroyed  */
class rocsparselt_local_matmul_alg_selection
{
    rocsparselt_matmul_alg_selection m_alg_sel;
    rocsparselt_status               m_status = rocsparselt_status_not_initialized;

public:
    rocsparselt_local_matmul_alg_selection(rocsparselt_handle       handle,
                                           rocsparselt_matmul_descr matmul,
                                           rocsparselt_matmul_alg   alg)
    {

        this->m_status
            = rocsparselt_matmul_alg_selection_init(handle, &this->m_alg_sel, matmul, alg);
    }

    ~rocsparselt_local_matmul_alg_selection()
    {
        if(this->m_status == rocsparselt_status_success)
            rocsparselt_matmul_alg_selection_destroy(this->m_alg_sel);
    }

    rocsparselt_local_matmul_alg_selection(const rocsparselt_local_matmul_alg_selection&) = delete;
    rocsparselt_local_matmul_alg_selection(rocsparselt_local_matmul_alg_selection&&)      = delete;
    rocsparselt_local_matmul_alg_selection& operator=(const rocsparselt_local_matmul_alg_selection&)
        = delete;
    rocsparselt_local_matmul_alg_selection& operator=(rocsparselt_local_matmul_alg_selection&&)
        = delete;

    rocsparselt_status status()
    {
        return m_status;
    }

    operator rocsparselt_matmul_alg_selection&()
    {
        return m_alg_sel;
    }
    operator const rocsparselt_matmul_alg_selection&() const
    {
        return m_alg_sel;
    }
};

/* ================================================================================================================= */
/*! \brief  local matrix multiplication plan descriptor which is automatically created and destroyed  */
class rocsparselt_local_matmul_plan
{
    rocsparselt_matmul_plan m_plan;
    rocsparselt_status      m_status = rocsparselt_status_not_initialized;

public:
    rocsparselt_local_matmul_plan(rocsparselt_handle               handle,
                                  rocsparselt_matmul_descr         matmul,
                                  rocsparselt_matmul_alg_selection alg_sel,
                                  size_t                           workspace_size)
    {

        this->m_status
            = rocsparselt_matmul_plan_init(handle, &this->m_plan, matmul, alg_sel, workspace_size);
    }

    ~rocsparselt_local_matmul_plan()
    {
        if(this->m_status == rocsparselt_status_success)
            rocsparselt_matmul_plan_destroy(this->m_plan);
    }

    rocsparselt_local_matmul_plan(const rocsparselt_local_matmul_plan&) = delete;
    rocsparselt_local_matmul_plan(rocsparselt_local_matmul_plan&&)      = delete;
    rocsparselt_local_matmul_plan& operator=(const rocsparselt_local_matmul_plan&) = delete;
    rocsparselt_local_matmul_plan& operator=(rocsparselt_local_matmul_plan&&) = delete;

    rocsparselt_status status()
    {
        return this->m_status;
    }

    operator rocsparselt_matmul_plan&()
    {
        return m_plan;
    }
    operator const rocsparselt_matmul_plan&() const
    {
        return m_plan;
    }
};

/* ============================================================================================ */
/*  device query and print out their ID and name */
int64_t query_device_property();

/*  set current device to device_id */
void set_device(int64_t device_id);

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            rocsparselt sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us_sync_device();

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

/*! \brief  CPU Timer(in microsecond): no GPU synchronization and return wall time */
double get_time_us_no_sync();

/* ============================================================================================ */
// Return path of this executable
std::string rocsparselt_exepath();

/* ============================================================================================ */
// Temp directory rooted random path
std::string rocsparselt_tempname();

/* ============================================================================================ */
/* Read environment variable */
const char* read_env_var(const char* env_var);

/* ============================================================================================ */
/* Compute strided batched matrix allocation size allowing for strides smaller than full matrix */
size_t strided_batched_matrix_size(int rows, int cols, int lda, int64_t stride, int batch_count);

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T>
inline void rocsparselt_print_matrix(
    std::vector<T> CPU_result, std::vector<T> GPU_result, size_t m, size_t n, size_t lda)
{
    for(size_t i = 0; i < m; i++)
        for(size_t j = 0; j < n; j++)
        {
            rocsparselt_cout << "matrix  col " << i << ", row " << j
                             << ", CPU result=" << CPU_result[j + i * lda]
                             << ", GPU result=" << GPU_result[j + i * lda] << "\n";
        }
}

template <typename T>
void rocsparselt_print_matrix(const char* name, T* A, size_t m, size_t n, size_t lda)
{
    rocsparselt_cout << "---------- " << name << " ----------\n";
    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
            rocsparselt_cout << std::setprecision(0) << std::setw(5) << A[i + j * lda] << " ";
        rocsparselt_cout << std::endl;
    }
}

/* ============================================================================= */
/*! \brief For testing purposes, to convert a regular matrix to a banded matrix. */
template <typename T>
inline void regular_to_banded(
    bool upper, const T* A, int64_t lda, T* AB, int64_t ldab, int64_t n, int64_t k)
{
    // convert regular hA matrix to banded hAB matrix
    for(int j = 0; j < n; j++)
    {
        int64_t min1 = upper ? std::max(0, static_cast<int>(j - k)) : j;
        int64_t max1 = upper ? j : std::min(static_cast<int>(n - 1), static_cast<int>(j + k));
        int64_t m    = upper ? k - j : -j;

        // Move bands of hA into new banded hAB format.
        for(int i = min1; i <= max1; i++)
            AB[j * ldab + (m + i)] = A[j * lda + i];

        min1 = upper ? k + 1 : std::min(k + 1, n - j);
        max1 = ldab - 1;

        // fill in bottom with random data to ensure we aren't using it.
        // for !upper, fill in bottom right triangle as well.
        for(int i = min1; i <= max1; i++)
            rocsparselt_init<T>(AB + j * ldab + i, 1, 1, 1);

        // for upper, fill in top left triangle with random data to ensure
        // we aren't using it.
        if(upper)
        {
            for(int i = 0; i < m; i++)
                rocsparselt_init<T>(AB + j * ldab + i, 1, 1, 1);
        }
    }
}

/* =============================================================================== */
/*! \brief For testing purposes, zeros out elements not needed in a banded matrix. */
template <typename T>
inline void banded_matrix_setup(bool upper, T* A, int64_t lda, int64_t n, int64_t k)
{
    // Made A a banded matrix with k sub/super-diagonals
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(upper && (j > k + i || i > j))
                A[j * n + i] = T(0);
            else if(!upper && (i > k + j || j > i))
                A[j * n + i] = T(0);
        }
    }
}

/* ============================================================================================= */
/*! \brief For testing purposes, to convert a regular matrix to a packed matrix.                  */
template <typename T>
inline void regular_to_packed(bool upper, const T* A, T* AP, int64_t n)
{
    int index = 0;
    if(upper)
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                AP[index++] = A[j + i * n];
            }
        }
    }
    else
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = i; j < n; j++)
            {
                AP[index++] = A[j + i * n];
            }
        }
    }
}

template <typename T>
void print_strided_batched(
    const char* name, T* A, int64_t n1, int64_t n2, int64_t n3, int64_t s1, int64_t s2, int64_t s3)
{
    constexpr bool is_int = std::is_same<T, int8_t>();
    using Tp              = std::conditional_t<is_int, int32_t, float>;
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    rocsparselt_cout << "---------- " << name << " ----------\n";
    int max_size = 128;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                rocsparselt_cout << static_cast<Tp>(A[(i1 * s1) + (i2 * s2) + (i3 * s3)]) << "|";
            }
            rocsparselt_cout << "\n";
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            rocsparselt_cout << "\n";
    }
    rocsparselt_cout << std::flush;
}
