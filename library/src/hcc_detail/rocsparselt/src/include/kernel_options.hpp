#pragma once
#ifndef KERNEL_OPTIONS_HPP
#define KERNEL_OPTIONS_HPP

#include "handle.h"

_rocsparselt_kernel_options get_kernel_options(int64_t               m,
                                               int64_t               n,
                                               bool                  isSparseA,
                                               rocsparselt_operation op,
                                               rocsparselt_order_    order,
                                               int                   tt0i = 1,
                                               int                   tt1j = 8);

#endif // KERNEL_OPTIONS_HPP
