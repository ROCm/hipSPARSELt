/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocsparselt.h"
#include "rocsparselt_math.hpp"
#include <cinttypes>
#include <random>
#include <thread>
#include <type_traits>

/* ============================================================================================ */
// Random number generator
using rocsparselt_rng_t = std::mt19937;

extern rocsparselt_rng_t g_rocsparselt_seed;
extern std::thread::id   g_main_thread_id;

extern thread_local rocsparselt_rng_t t_rocsparselt_rng;
extern thread_local int               t_rocsparselt_rand_idx;

// optimized helper
float rocsparselt_uniform_int_1_10();

void rocsparselt_uniform_int_1_10_run_float(float* ptr, size_t num);
void rocsparselt_uniform_int_1_10_run_double(double* ptr, size_t num);

// For the main thread, we use g_rocsparselt_seed; for other threads, we start with a different seed but
// deterministically based on the thread id's hash function.
inline rocsparselt_rng_t get_seed()
{
    auto tid = std::this_thread::get_id();
    return tid == g_main_thread_id ? g_rocsparselt_seed
                                   : rocsparselt_rng_t(std::hash<std::thread::id>{}(tid));
}

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void rocsparselt_seedrand()
{
    t_rocsparselt_rng      = get_seed();
    t_rocsparselt_rand_idx = 0;
}

/* ============================================================================================ */
/*! \brief  Random number generator which generates NaN values */
class rocsparselt_nan_rng
{
    // Generate random NaN values
    template <typename T, typename UINT_T, int SIG, int EXP>
    static T random_nan_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
        union
        {
            UINT_T u;
            T      fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>{}(t_rocsparselt_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>{}(t_rocsparselt_rng);
    }

    // Random signed char
    explicit operator signed char()
    {
        return static_cast<signed char>(std::uniform_int_distribution<int>{}(t_rocsparselt_rng));
    }

    // Random NaN double
    explicit operator double()
    {
        return random_nan_data<double, uint64_t, 52, 11>();
    }

    // Random NaN float
    explicit operator float()
    {
        return random_nan_data<float, uint32_t, 23, 8>();
    }

    // Random NaN half
    explicit operator rocsparselt_half()
    {
        return random_nan_data<rocsparselt_half, uint16_t, 10, 5>();
    }

    // Random NaN bfloat16
    explicit operator rocsparselt_bfloat16()
    {
        return random_nan_data<rocsparselt_bfloat16, uint16_t, 7, 8>();
    }
};

/* ============================================================================================ */
/*! \brief  Random number generator which generates Inf values */
class rocsparselt_inf_rng
{
    // Generate random Inf values
    static unsigned rand2()
    {
        return std::uniform_int_distribution<unsigned>(0, 1)(t_rocsparselt_rng);
    }

public:
    // Random integer
    template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return rand2() ? std::numeric_limits<T>::min : std::numeric_limits<T>::max;
    }
    // Random float
    template <typename T, std::enable_if_t<!std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return T(rand2() ? -std::numeric_limits<double>::infinity()
                         : std::numeric_limits<double>::infinity());
    }
};

/* ============================================================================================ */
/*! \brief  Random number generator which generates zero values */
class rocsparselt_zero_rng
{
    // Generate random zero values
    static unsigned rand2()
    {
        return std::uniform_int_distribution<unsigned>(0, 1)(t_rocsparselt_rng);
    }

public:
    // Random integer
    template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return 0;
    }
    // Random float
    template <typename T, std::enable_if_t<!std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return T(rand2() ? -0.0 : 0.0);
    }
};

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random NaN number */
template <typename T>
inline T random_nan_generator()
{
    return T(rocsparselt_nan_rng{});
}

/*! \brief  generate a random Inf number */
template <typename T>
inline T random_inf_generator()
{
    return T(rocsparselt_inf_rng{});
}

/*! \brief  generate a random Inf number */
template <typename T>
inline T random_zero_generator()
{
    return T(rocsparselt_zero_rng{});
}

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
inline T random_generator()
{
    return T(rocsparselt_uniform_int_1_10());
}

template <>
inline float random_generator()
{
    return rocsparselt_uniform_int_1_10();
}

// for rocsparselt_half, generate float, and convert to rocsparselt_half
/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
inline rocsparselt_half random_generator<rocsparselt_half>()
{
    return rocsparselt_half(std::uniform_int_distribution<int>(-2, 2)(t_rocsparselt_rng));
};

// for rocsparselt_bfloat16, generate float, and convert to rocsparselt_bfloat16
/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
inline rocsparselt_bfloat16 random_generator<rocsparselt_bfloat16>()
{
    return rocsparselt_bfloat16(std::uniform_int_distribution<int>(-2, 2)(t_rocsparselt_rng));
};

/*! \brief  generate a random number in range [1,2,3] */
template <>
inline int8_t random_generator<int8_t>()
{
    return static_cast<int8_t>(
        std::uniform_int_distribution<unsigned short>(1, 3)(t_rocsparselt_rng));
};

/*! \brief  generate a sequence of random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
inline void random_run_generator(T* ptr, size_t num)
{
    for(size_t i = 0; i < num; i++)
    {
        ptr[i] = random_generator<T>();
    }
}

template <>
inline void random_run_generator<float>(float* ptr, size_t num)
{
    rocsparselt_uniform_int_1_10_run_float(ptr, num);
};

template <>
inline void random_run_generator<double>(double* ptr, size_t num)
{
    rocsparselt_uniform_int_1_10_run_double(ptr, num);
};

// HPL

/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <typename T>
inline T random_hpl_generator()
{
    return std::uniform_real_distribution<double>(-0.5, 0.5)(t_rocsparselt_rng);
}

// for rocsparselt_bfloat16, generate float, and convert to rocsparselt_bfloat16
/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <>
inline rocsparselt_bfloat16 random_hpl_generator()
{
    return rocsparselt_bfloat16(
        std::uniform_real_distribution<float>(-0.5, 0.5)(t_rocsparselt_rng));
}

/*! \brief  generate a random ASCII string of up to length n */
inline std::string random_string(size_t n)
{
    std::string str;
    if(n)
    {
        size_t len = std::uniform_int_distribution<size_t>(1, n)(t_rocsparselt_rng);
        str.reserve(len);
        for(size_t i = 0; i < len; ++i)
            str.push_back(static_cast<char>(
                std::uniform_int_distribution<unsigned short>(0x20, 0x7E)(t_rocsparselt_rng)));
    }
    return str;
}
