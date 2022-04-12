/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <string>

// Parse --data and --yaml command-line arguments
bool rocsparselt_parse_data(int& argc, char** argv, const std::string& default_file = "");
