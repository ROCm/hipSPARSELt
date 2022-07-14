# ########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

set(hipsparselt_source_common  src/hipsparselt_ostream.cpp
                               src/auxiliary.cpp)
if(NOT BUILD_CUDA)
  # hipSPARSE source
  include(src/hcc_detail/rocsparselt/CMakeLists.txt)
  set(hipsparselt_source src/hcc_detail/hipsparselt.cpp
                         ${hipsparselt_source_common}
                         ${rocsparselt_source})
else()
  # hipSPARSE CUDA source
  set(hipsparselt_source src/nvcc_detail/hipsparselt.cpp
                         ${hipsparselt_source_common})
endif()
