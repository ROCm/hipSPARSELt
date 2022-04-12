/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparselt_random.hpp"

// Random number generator
// Note: We do not use random_device to initialize the RNG, because we want
// repeatability in case of test failure. TODO: Add seed as an optional CLI
// argument, and print the seed on output, to ensure repeatability.
rocsparselt_rng_t g_rocsparselt_seed(69069); // A fixed seed to start at

// This records the main thread ID at startup
std::thread::id g_main_thread_id = std::this_thread::get_id();

// For the main thread, we use g_rocsparselt_seed; for other threads, we start with a different seed but
// deterministically based on the thread id's hash function.
thread_local rocsparselt_rng_t t_rocsparselt_rng = get_seed();

thread_local int t_rocsparselt_rand_idx;

// length to allow use as bitmask to wraparound
#define RANDLEN 1024
#define RANDWIN 256
#define RANDBUF RANDLEN + RANDWIN
static thread_local int    t_rand_init = 0;
static thread_local float  t_rand_f_array[RANDBUF];
static thread_local double t_rand_d_array[RANDBUF];

/* ============================================================================================ */

float rocsparselt_uniform_int_1_10()
{
    if(!t_rand_init)
    {
        for(int i = 0; i < RANDBUF; i++)
        {
            t_rand_f_array[i]
                = (float)std::uniform_int_distribution<unsigned>(1, 10)(t_rocsparselt_rng);
            t_rand_d_array[i] = (double)t_rand_f_array[i];
        }
        t_rand_init = 1;
    }
    t_rocsparselt_rand_idx = (t_rocsparselt_rand_idx + 1) & (RANDLEN - 1);
    return t_rand_f_array[t_rocsparselt_rand_idx];
}

inline int pseudo_rand_ptr_offset()
{
    t_rocsparselt_rand_idx = (t_rocsparselt_rand_idx + 1) & (RANDWIN - 1);
    return t_rocsparselt_rand_idx;
}

void rocsparselt_uniform_int_1_10_run_float(float* ptr, size_t num)
{
    if(!t_rand_init)
        rocsparselt_uniform_int_1_10();

    for(size_t i = 0; i < num; i += RANDLEN)
    {
        float* rptr = t_rand_f_array + pseudo_rand_ptr_offset();
        size_t n    = i + RANDLEN < num ? RANDLEN : num - i;
        memcpy(ptr, rptr, sizeof(float) * n);
        ptr += RANDLEN;
    }
}

void rocsparselt_uniform_int_1_10_run_double(double* ptr, size_t num)
{
    if(!t_rand_init)
        rocsparselt_uniform_int_1_10();

    for(size_t i = 0; i < num; i += RANDLEN)
    {
        double* rptr = t_rand_d_array + pseudo_rand_ptr_offset();
        size_t  n    = i + RANDLEN < num ? RANDLEN : num - i;
        memcpy(ptr, rptr, sizeof(double) * n);
        ptr += RANDLEN;
    }
}

#undef RANDLEN
#undef RANDWIN
#undef RANDBUF
